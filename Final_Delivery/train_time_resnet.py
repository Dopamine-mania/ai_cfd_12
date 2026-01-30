import os
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ================= 配置 =================
_DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "processed_data", "26ms")
if not os.path.isdir(_DEFAULT_DATA_DIR):
    _DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "processed_data")
DATA_DIR = os.getenv("DATA_DIR", _DEFAULT_DATA_DIR)
SAVE_DIR = os.getenv("SAVE_DIR", os.path.join(REPO_ROOT, "checkpoints_time"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_FACTOR = float(os.getenv("NORM_FACTOR", "30.0"))
DT = float(os.getenv("DT", "0.002"))  # s, 仅用于记录/可视化

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
LR = float(os.getenv("LR", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY = int(os.getenv("PIN_MEMORY", "1"))
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "1.0"))  # 0 to disable

SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))
RESUME = int(os.getenv("RESUME", "0"))
RESUME_PATH = os.getenv("RESUME_PATH", "")
FINETUNE_FROM = os.getenv("FINETUNE_FROM", "")  # 仅加载权重（不加载优化器/epoch）

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))  # 调试用

# 输入/输出定义
# 输入: (u0,v0,w0,t_norm) -> 4 通道
# 输出: x_t，通过网络预测 Δx 并加回 x0（更易收敛）
TIME_NORM_MODE = os.getenv("TIME_NORM_MODE", "index").lower()  # index | seconds

# ROI 加权（可选）：基于 u0 的梯度强度强调尾流结构
ROI_ENABLE = int(os.getenv("ROI_ENABLE", "0"))
ROI_WEIGHT = float(os.getenv("ROI_WEIGHT", "4.0"))  # mask 区域额外权重
ROI_THRESH_Q = float(os.getenv("ROI_THRESH_Q", "95.0"))  # 梯度阈值分位数（越大 ROI 越小）
ROI_BLUR = int(os.getenv("ROI_BLUR", "1"))  # 0/1：是否对 mask 做一次 3x3 平滑

# ====== 物理/结构损失（第二阶段微调）======
PHYS_ENABLE = int(os.getenv("PHYS_ENABLE", "0"))
VORTICITY_WEIGHT = float(os.getenv("VORTICITY_WEIGHT", "0.2"))
DIVERGENCE_WEIGHT = float(os.getenv("DIVERGENCE_WEIGHT", "0.2"))

TEXTURE_ENABLE = int(os.getenv("TEXTURE_ENABLE", "0"))
GRADIENT_WEIGHT = float(os.getenv("GRADIENT_WEIGHT", "0.05"))
LAPLACIAN_WEIGHT = float(os.getenv("LAPLACIAN_WEIGHT", "0.02"))
TEXTURE_L1 = int(os.getenv("TEXTURE_L1", "1"))  # 1: L1（更锐），0: MSE

# ====== v2: Loss 只在 Wake ROI + 时间加权 ======
LOSS_ROI_ONLY = int(os.getenv("LOSS_ROI_ONLY", "0"))  # 1: 仅在 ROI 内计算所有损失

TIME_WEIGHT_ENABLE = int(os.getenv("TIME_WEIGHT_ENABLE", "0"))
TIME_WEIGHT_BASE = float(os.getenv("TIME_WEIGHT_BASE", "0.2"))   # 最小权重
TIME_WEIGHT_SCALE = float(os.getenv("TIME_WEIGHT_SCALE", "0.8")) # 增量权重
TIME_WEIGHT_POWER = float(os.getenv("TIME_WEIGHT_POWER", "2.0")) # t^p, p>1 强调后期

# v2: 从 loss 中剔除船体/边界（与推理硬约束对齐）
EXCLUDE_HULL_IN_LOSS = int(os.getenv("EXCLUDE_HULL_IN_LOSS", "1"))
HULL_MASK_MODE = os.getenv("HULL_MASK_MODE", "u").lower()  # u | speed
HULL_U_THRESHOLD = float(os.getenv("HULL_U_THRESHOLD", "0.2"))
HULL_SPEED_THRESHOLD = float(os.getenv("HULL_SPEED_THRESHOLD", "0.2"))
HULL_LEFT_COLS = int(os.getenv("HULL_LEFT_COLS", "24"))

EXCLUDE_BC_IN_LOSS = int(os.getenv("EXCLUDE_BC_IN_LOSS", "1"))
BC_PAD_RIGHT = int(os.getenv("BC_PAD_RIGHT", "4"))
BC_PAD_TOP = int(os.getenv("BC_PAD_TOP", "4"))
BC_PAD_BOTTOM = int(os.getenv("BC_PAD_BOTTOM", "4"))


def weighted_mse(pred, target, weight_map):
    # weight_map: (B,1,H,W) or (B,3,H,W)
    if weight_map.dim() == 4 and weight_map.shape[1] == 1:
        weight_map = weight_map.expand_as(pred)
    loss = (pred - target) ** 2
    loss = loss * weight_map
    return loss.mean() / (weight_map.mean().clamp_min(1e-8))


def sobel_grad_mag(u):
    # u: (B,1,H,W)
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 2.0, 1.0]], device=u.device, dtype=u.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(u, kx, padding=1)
    gy = F.conv2d(u, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def central_diff_y(tensor_):
    # tensor_: (B,1,H,W) -> dy along H (y)
    ky = torch.tensor([[-0.5], [0.0], [0.5]], device=tensor_.device, dtype=tensor_.dtype).view(1, 1, 3, 1)
    return F.conv2d(tensor_, ky, padding=(1, 0))

def central_diff_z(tensor_):
    # tensor_: (B,1,H,W) -> dz along W (z)
    kz = torch.tensor([[-0.5, 0.0, 0.5]], device=tensor_.device, dtype=tensor_.dtype).view(1, 1, 1, 3)
    return F.conv2d(tensor_, kz, padding=(0, 1))

def vorticity_x(field):
    """
    ω_x = ∂W/∂y - ∂V/∂z
    field: (B,3,H,W) channel order [U,V,W]
    """
    v = field[:, 1:2]
    w = field[:, 2:3]
    dw_dy = central_diff_y(w)
    dv_dz = central_diff_z(v)
    return dw_dy - dv_dz

def divergence_slice(field):
    """
    不可压约束的切片近似：∂V/∂y + ∂W/∂z（缺少 ∂U/∂x）
    field: (B,3,H,W)
    """
    v = field[:, 1:2]
    w = field[:, 2:3]
    dv_dy = central_diff_y(v)
    dw_dz = central_diff_z(w)
    return dv_dy + dw_dz

def laplacian(field):
    # field: (B,C,H,W) -> per-channel laplacian via depthwise conv
    k = torch.tensor([[0.0, 1.0, 0.0],
                      [1.0, -4.0, 1.0],
                      [0.0, 1.0, 0.0]], device=field.device, dtype=field.dtype).view(1, 1, 3, 3)
    c = field.shape[1]
    k = k.repeat(c, 1, 1, 1)
    return F.conv2d(field, k, padding=1, groups=c)

def sobel(field):
    # field: (B,C,H,W) -> per-channel sobel gradients magnitude via depthwise conv
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=field.device, dtype=field.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 2.0, 1.0]], device=field.device, dtype=field.dtype).view(1, 1, 3, 3)
    c = field.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(field, kx, padding=1, groups=c)
    gy = F.conv2d(field, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def compute_hull_mask_from_x0(x0):
    """
    x0: (B,3,H,W) normalized
    return: (B,1,H,W) float mask in {0,1}
    """
    if HULL_LEFT_COLS <= 0:
        return torch.zeros((x0.shape[0], 1, x0.shape[2], x0.shape[3]), device=x0.device, dtype=x0.dtype)
    b, _, h, w = x0.shape
    left_cols = max(0, min(int(HULL_LEFT_COLS), w))
    region = torch.zeros((b, 1, h, w), device=x0.device, dtype=x0.dtype)
    region[:, :, :, :left_cols] = 1.0

    if HULL_MASK_MODE == "u":
        u0 = x0[:, 0:1]
        low = (u0 < HULL_U_THRESHOLD).float()
    elif HULL_MASK_MODE == "speed":
        speed0 = torch.linalg.vector_norm(x0, dim=1, keepdim=True)
        low = (speed0 < HULL_SPEED_THRESHOLD).float()
    else:
        raise ValueError(f"Unknown HULL_MASK_MODE={HULL_MASK_MODE}")

    return low * region

def compute_bc_exclude_mask_like(x0):
    """
    x0: (B,3,H,W) -> mask (B,1,H,W) where 1 means valid (not boundary strip)
    """
    b, _, h, w = x0.shape
    m = torch.ones((b, 1, h, w), device=x0.device, dtype=x0.dtype)
    if BC_PAD_RIGHT > 0:
        m[:, :, :, -BC_PAD_RIGHT:] = 0.0
    if BC_PAD_TOP > 0:
        m[:, :, :BC_PAD_TOP, :] = 0.0
    if BC_PAD_BOTTOM > 0:
        m[:, :, -BC_PAD_BOTTOM:, :] = 0.0
    return m

def compute_time_weight(inp):
    if not TIME_WEIGHT_ENABLE:
        return 1.0
    # t_norm channel is constant map; take mean -> scalar per sample
    t = inp[:, 3:4].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
    w = TIME_WEIGHT_BASE + TIME_WEIGHT_SCALE * torch.pow(t, TIME_WEIGHT_POWER)
    return w


class TimeConditionedCFDDataset(Dataset):
    """
    One-shot/time-conditioned:
      input = [x0(3), t_norm(1)]
      target = x_t(3)
    每个 .npy: (T,H,W,3). 默认用 t=0 作为 x0，t=1..T-1 作为监督。
    """

    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.data_maps = []
        self.indices = []

        for file_idx, path in enumerate(self.files):
            data = np.load(path, mmap_mode="r")
            self.data_maps.append(data)
            T = data.shape[0]
            for t in range(1, T):
                self.indices.append((file_idx, t, T))

        if MAX_SAMPLES and MAX_SAMPLES > 0:
            self.indices = self.indices[:MAX_SAMPLES]

        print(f"✅ Dataset: {len(self.files)} files, {len(self.indices)} samples (x0->xt)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_id, t, T = self.indices[idx]
        data = self.data_maps[file_id]

        x0 = data[0].astype(np.float32) / NORM_FACTOR  # (H,W,3)
        xt = data[t].astype(np.float32) / NORM_FACTOR  # (H,W,3)

        if TIME_NORM_MODE == "index":
            t_norm = float(t) / float(T - 1)
        elif TIME_NORM_MODE == "seconds":
            t_seconds = float(t) * DT
            t_max = float(T - 1) * DT
            t_norm = t_seconds / max(t_max, 1e-12)
        else:
            raise ValueError(f"Unknown TIME_NORM_MODE={TIME_NORM_MODE}")

        x0_t = torch.from_numpy(x0).permute(2, 0, 1)  # (3,H,W)
        xt_t = torch.from_numpy(xt).permute(2, 0, 1)  # (3,H,W)

        time_map = torch.full((1, x0_t.shape[1], x0_t.shape[2]), float(t_norm), dtype=torch.float32)
        inp = torch.cat([x0_t, time_map], dim=0)  # (4,H,W)

        return inp, xt_t


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class TimeCondResNet(nn.Module):
    """
    输入 4 通道：(u0,v0,w0,t_norm)，输出 x_t (3 通道)。
    网络内部学习 Δx，再加回 x0：x_t = x0 + Δx
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Δx
        )

    def forward(self, x):
        x0 = x[:, :3]
        feats = self.encoder(x)
        feats = self.bottleneck(feats)
        delta = self.decoder(feats)
        return x0 + delta


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "loss": float(loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "time_norm_mode": TIME_NORM_MODE,
                "roi_enable": ROI_ENABLE,
                "roi_weight": ROI_WEIGHT,
                "roi_thresh_q": ROI_THRESH_Q,
            },
        },
        path,
    )


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("epoch", -1)) + 1


if __name__ == "__main__":
    print(f"🚀 Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = TimeConditionedCFDDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=bool(PIN_MEMORY),
        drop_last=True,
    )

    model = TimeCondResNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

    start_epoch = 0
    if FINETUNE_FROM:
        path = FINETUNE_FROM
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        start_epoch = 0
        print(f"📥 Fine-tune weights loaded from {path} (start_epoch=0)")
    elif RESUME and RESUME_PATH and os.path.exists(RESUME_PATH):
        start_epoch = load_checkpoint(model, optimizer, RESUME_PATH)
        print(f"📥 Resumed from {RESUME_PATH}, start_epoch={start_epoch}")

    print(
        f"Config: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, time_norm={TIME_NORM_MODE}, "
        f"roi={ROI_ENABLE}, phys={PHYS_ENABLE}, texture={TEXTURE_ENABLE}, grad_clip={GRAD_CLIP_NORM}"
    )
    if PHYS_ENABLE:
        print(f"PhysLoss: vort_w={VORTICITY_WEIGHT}, div_w={DIVERGENCE_WEIGHT}")
    if TEXTURE_ENABLE:
        print(f"TextureLoss: grad_w={GRADIENT_WEIGHT}, lap_w={LAPLACIAN_WEIGHT}, l1={TEXTURE_L1}")
    if LOSS_ROI_ONLY:
        print(
            f"v2 Loss ROI-only: time_weight={TIME_WEIGHT_ENABLE}(base={TIME_WEIGHT_BASE},scale={TIME_WEIGHT_SCALE},p={TIME_WEIGHT_POWER}), "
            f"exclude_hull={EXCLUDE_HULL_IN_LOSS}, exclude_bc={EXCLUDE_BC_IN_LOSS}"
        )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inp, target in pbar:
            inp = inp.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            pred = model(inp)

            base_loss = None
            # ===== v2: 构造“只在 wake ROI 内、且排除 hull/bc 的有效区域”权重图 =====
            # wake ROI 基于 u0 的梯度强度（以 u0 的剪切层为主）
            weights_for_loss = None
            if LOSS_ROI_ONLY:
                x0 = inp[:, :3]
                valid = torch.ones((inp.shape[0], 1, inp.shape[2], inp.shape[3]), device=inp.device, dtype=inp.dtype)
                if EXCLUDE_BC_IN_LOSS:
                    valid = valid * compute_bc_exclude_mask_like(x0)
                if EXCLUDE_HULL_IN_LOSS:
                    hull = compute_hull_mask_from_x0(x0)
                    valid = valid * (1.0 - hull).clamp(0.0, 1.0)

                u0 = x0[:, 0:1]
                gmag = sobel_grad_mag(u0) * valid
                q = torch.quantile(gmag.reshape(gmag.shape[0], -1), ROI_THRESH_Q / 100.0, dim=1).view(-1, 1, 1, 1)
                wake = (gmag >= q).float()
                if ROI_BLUR:
                    wake = F.avg_pool2d(wake, kernel_size=3, stride=1, padding=1)
                    wake = (wake > 0).float()
                wake = wake * valid  # 确保背景/边界权重为 0

                t_w = compute_time_weight(inp)  # (B,1,1,1) or scalar
                if isinstance(t_w, float):
                    t_w = torch.tensor(t_w, device=inp.device, dtype=inp.dtype).view(1, 1, 1, 1)
                weights_for_loss = wake * t_w

                # 若 ROI 太小导致全零，则退化为全场（避免 NaN）
                if float(weights_for_loss.mean().item()) <= 0.0:
                    weights_for_loss = valid * 1.0

            if ROI_ENABLE and not LOSS_ROI_ONLY:
                u0 = inp[:, 0:1]  # (B,1,H,W)
                gmag = sobel_grad_mag(u0)
                # per-batch阈值（分位数），避免阈值写死
                q = torch.quantile(gmag.reshape(gmag.shape[0], -1), ROI_THRESH_Q / 100.0, dim=1)
                q = q.view(-1, 1, 1, 1)
                mask = (gmag >= q).float()
                if ROI_BLUR:
                    mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
                    mask = (mask > 0).float()
                weights = 1.0 + ROI_WEIGHT * mask
                base_loss = weighted_mse(pred, target, weights)
            else:
                if weights_for_loss is not None:
                    base_loss = weighted_mse(pred, target, weights_for_loss)
                else:
                    base_loss = mse(pred, target)

            loss = base_loss

            # 物理损失（在归一化空间计算）
            if PHYS_ENABLE:
                vort_p = vorticity_x(pred)
                vort_t = vorticity_x(target)
                div_p = divergence_slice(pred)
                div_t = divergence_slice(target)
                # vorticity 一致性：匹配涡结构
                if weights_for_loss is not None:
                    loss = loss + VORTICITY_WEIGHT * weighted_mse(vort_p, vort_t, weights_for_loss)
                else:
                    loss = loss + VORTICITY_WEIGHT * mse(vort_p, vort_t)
                # divergence penalty：抑制非物理压缩（切片近似）
                if weights_for_loss is not None:
                    loss = loss + DIVERGENCE_WEIGHT * weighted_mse(div_p, div_t, weights_for_loss)
                else:
                    loss = loss + DIVERGENCE_WEIGHT * mse(div_p, div_t)

            # 纹理/结构损失：让红蓝交界更锐、更接近真值纹理
            if TEXTURE_ENABLE:
                grad_p = sobel(pred[:, 0:1])  # 先对 U 做锐化约束（更符合可视化需求）
                grad_t = sobel(target[:, 0:1])
                lap_p = laplacian(pred[:, 0:1])
                lap_t = laplacian(target[:, 0:1])
                if TEXTURE_L1:
                    if weights_for_loss is not None:
                        loss = loss + GRADIENT_WEIGHT * weighted_mse(grad_p, grad_t, weights_for_loss)
                        loss = loss + LAPLACIAN_WEIGHT * weighted_mse(lap_p, lap_t, weights_for_loss)
                    else:
                        loss = loss + GRADIENT_WEIGHT * torch.mean(torch.abs(grad_p - grad_t))
                        loss = loss + LAPLACIAN_WEIGHT * torch.mean(torch.abs(lap_p - lap_t))
                else:
                    if weights_for_loss is not None:
                        loss = loss + GRADIENT_WEIGHT * weighted_mse(grad_p, grad_t, weights_for_loss)
                        loss = loss + LAPLACIAN_WEIGHT * weighted_mse(lap_p, lap_t, weights_for_loss)
                    else:
                        loss = loss + GRADIENT_WEIGHT * mse(grad_p, grad_t)
                        loss = loss + LAPLACIAN_WEIGHT * mse(lap_p, lap_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6e}", "base": f"{base_loss.item():.3e}"})

        avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} | avg_loss={avg:.8e}")

        if SAVE_EVERY > 0 and ((epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg,
                os.path.join(SAVE_DIR, f"time_resnet_epoch_{epoch+1}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "time_resnet_final.pth"))
    print(f"✅ Done. Final weights: {os.path.join(SAVE_DIR, 'time_resnet_final.pth')}")
