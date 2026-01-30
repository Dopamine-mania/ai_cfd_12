import os
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# ================= 配置 =================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "processed_data", "26ms")
if not os.path.isdir(_DEFAULT_DATA_DIR):
    _DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "processed_data")
DATA_DIR = os.getenv("DATA_DIR", _DEFAULT_DATA_DIR)
SAVE_DIR = os.getenv("SAVE_DIR", os.path.join(REPO_ROOT, "checkpoints_fno"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_FACTOR = float(os.getenv("NORM_FACTOR", "30.0"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY = int(os.getenv("PIN_MEMORY", "1"))

EPOCHS = int(os.getenv("EPOCHS", "100"))
LR = float(os.getenv("LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "1.0"))  # 0 to disable

MODES = int(os.getenv("MODES", "12"))
WIDTH = int(os.getenv("WIDTH", "32"))
PADDING = int(os.getenv("PADDING", "20"))

UNROLL_STEPS = int(os.getenv("UNROLL_STEPS", "1"))  # FNO 也可做短 unroll（建议先 1）
STEP_WEIGHT_GAMMA = float(os.getenv("STEP_WEIGHT_GAMMA", "1.0"))

LOSS_TYPE = os.getenv("LOSS_TYPE", "rel_l2").lower()  # rel_l2 | mse

SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))
RESUME = int(os.getenv("RESUME", "0"))
RESUME_PATH = os.getenv("RESUME_PATH", "")

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))  # 调试用：>0 则截断数据量

# ================= 训练期硬物理约束（对齐推理端） =================
PHYS_ENABLE = int(os.getenv("PHYS_ENABLE", "1"))
MASK_MODE = os.getenv("MASK_MODE", "u").lower()  # u | speed
MASK_U_THRESHOLD = float(os.getenv("MASK_U_THRESHOLD", "0.2"))
MASK_SPEED_THRESHOLD = float(os.getenv("MASK_SPEED_THRESHOLD", "0.2"))
MASK_LEFT_COLS = int(os.getenv("MASK_LEFT_COLS", "24"))
MASK_LOCK = os.getenv("MASK_LOCK", "initial").lower()  # initial | zero

BC_ENABLE = int(os.getenv("BC_ENABLE", "1"))
BC_PAD_RIGHT = int(os.getenv("BC_PAD_RIGHT", "4"))
BC_PAD_TOP = int(os.getenv("BC_PAD_TOP", "4"))
BC_PAD_BOTTOM = int(os.getenv("BC_PAD_BOTTOM", "4"))

CLAMP_ENABLE = int(os.getenv("CLAMP_ENABLE", "1"))
CLAMP_MIN = float(os.getenv("CLAMP_MIN", "-0.5"))
CLAMP_MAX = float(os.getenv("CLAMP_MAX", "1.5"))


def rel_l2_loss(pred, target, eps=1e-8):
    diff = pred - target
    diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1)
    return torch.mean(diff_norm / (target_norm + eps))


class CFDDataset(Dataset):
    """
    复用原 .npy 数据，不重建数据：
    - 每个文件 shape: (T, H, W, 3)
    - 返回: x_t (3,H,W), targets_seq (K,3,H,W)
    """

    def __init__(self, data_dir, unroll_steps):
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        self.unroll_steps = int(unroll_steps)

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.data_maps = []
        self.valid_indices = []

        for file_idx, path in enumerate(self.files):
            data = np.load(path, mmap_mode="r")
            self.data_maps.append(data)
            num_frames = data.shape[0]
            for t in range(num_frames - self.unroll_steps):
                self.valid_indices.append((file_idx, t))

        if MAX_SAMPLES and MAX_SAMPLES > 0:
            self.valid_indices = self.valid_indices[:MAX_SAMPLES]

        print(f"✅ Dataset: {len(self.files)} files, {len(self.valid_indices)} samples, unroll={self.unroll_steps}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_id, t = self.valid_indices[idx]
        current = self.data_maps[file_id][t].astype(np.float32) / NORM_FACTOR
        future = [
            self.data_maps[file_id][t + i].astype(np.float32) / NORM_FACTOR
            for i in range(1, self.unroll_steps + 1)
        ]

        x = torch.from_numpy(current).permute(2, 0, 1)  # (3,H,W)
        y_seq = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in future], dim=0)  # (K,3,H,W)
        return x, y_seq


# ================= FNO2d with Padding =================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x: (B, C, H, W)
        batchsize, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(h, w))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes, width, padding=20):
        super().__init__()
        self.modes1 = int(modes)
        self.modes2 = int(modes)
        self.width = int(width)
        self.padding = int(padding)

        self.fc0 = nn.Linear(3 + 2, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    @staticmethod
    def get_grid(shape, device, dtype):
        # shape: (B, C, H, W)
        batchsize, _, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, device=device, dtype=dtype).view(1, 1, size_x, 1)
        gridx = gridx.repeat(batchsize, 1, 1, size_y)
        gridy = torch.linspace(0, 1, size_y, device=device, dtype=dtype).view(1, 1, 1, size_y)
        gridy = gridy.repeat(batchsize, 1, size_x, 1)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        # x: (B,3,H,W) -> output: (B,3,H,W) 直接预测 x_{t+1}
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        grid = self.get_grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=1)  # (B,5,H,W)
        x = x.permute(0, 2, 3, 1)  # (B,H,W,5)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B,width,H,W)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (B,3,H,W)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        return x


def compute_hull_mask_batch(initial_tensor):
    """
    initial_tensor: (B,3,H,W) 归一化
    return: (B,1,H,W) bool
    """
    if not (PHYS_ENABLE and MASK_LEFT_COLS > 0):
        return None
    b, _, h, w = initial_tensor.shape
    left_cols = max(0, min(int(MASK_LEFT_COLS), w))
    if left_cols <= 0:
        return None

    if MASK_MODE == "u":
        u0 = initial_tensor[:, 0:1]
        low = u0 < MASK_U_THRESHOLD
    elif MASK_MODE == "speed":
        speed0 = torch.linalg.vector_norm(initial_tensor, dim=1, keepdim=True)
        low = speed0 < MASK_SPEED_THRESHOLD
    else:
        raise ValueError(f"Unknown MASK_MODE={MASK_MODE}")

    region = torch.zeros((b, 1, h, w), device=initial_tensor.device, dtype=torch.bool)
    region[:, :, :, :left_cols] = True
    return low & region


def apply_hard_constraints_batch(x, initial_tensor, hull_mask):
    """
    x: (B,3,H,W) 当前预测
    initial_tensor: (B,3,H,W) unroll 初始帧（对齐推理：每步重置边界/锁定mask到初值）
    """
    if not PHYS_ENABLE:
        return x

    if BC_ENABLE:
        if BC_PAD_RIGHT > 0:
            x[:, :, :, -BC_PAD_RIGHT:] = initial_tensor[:, :, :, -BC_PAD_RIGHT:]
        if BC_PAD_TOP > 0:
            x[:, :, :BC_PAD_TOP, :] = initial_tensor[:, :, :BC_PAD_TOP, :]
        if BC_PAD_BOTTOM > 0:
            x[:, :, -BC_PAD_BOTTOM:, :] = initial_tensor[:, :, -BC_PAD_BOTTOM:, :]

    if hull_mask is not None:
        if MASK_LOCK == "zero":
            lock_val = torch.zeros_like(x)
        elif MASK_LOCK == "initial":
            lock_val = initial_tensor
        else:
            raise ValueError(f"Unknown MASK_LOCK={MASK_LOCK}")
        x = torch.where(hull_mask.expand_as(x), lock_val, x)

    return x


def apply_clamp(x):
    if not (PHYS_ENABLE and CLAMP_ENABLE):
        return x
    return torch.clamp(x, min=CLAMP_MIN, max=CLAMP_MAX)


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
                "modes": MODES,
                "width": WIDTH,
                "padding": PADDING,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "unroll_steps": UNROLL_STEPS,
                "loss_type": LOSS_TYPE,
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

    dataset = CFDDataset(DATA_DIR, unroll_steps=UNROLL_STEPS)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=bool(PIN_MEMORY),
        drop_last=True,
    )

    model = FNO2d(modes=MODES, width=WIDTH, padding=PADDING).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

    start_epoch = 0
    if RESUME:
        if RESUME_PATH:
            resume_path = RESUME_PATH
        else:
            existing = sorted(glob.glob(os.path.join(SAVE_DIR, "fno_epoch_*.pth")))
            resume_path = existing[-1] if existing else ""
        if resume_path and os.path.exists(resume_path):
            start_epoch = load_checkpoint(model, optimizer, resume_path)
            print(f"📥 Resumed from {resume_path}, start_epoch={start_epoch}")

    print(
        f"Config: modes={MODES}, width={WIDTH}, padding={PADDING}, epochs={EPOCHS}, lr={LR}, "
        f"batch={BATCH_SIZE}, unroll={UNROLL_STEPS}, loss={LOSS_TYPE}, grad_clip={GRAD_CLIP_NORM}"
    )
    if PHYS_ENABLE:
        print(
            f"Phys: clamp={CLAMP_ENABLE}[{CLAMP_MIN},{CLAMP_MAX}], "
            f"mask={MASK_MODE}@{MASK_U_THRESHOLD if MASK_MODE=='u' else MASK_SPEED_THRESHOLD} left_cols={MASK_LEFT_COLS} lock={MASK_LOCK}, "
            f"bc=R{BC_PAD_RIGHT}/T{BC_PAD_TOP}/B{BC_PAD_BOTTOM}"
        )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y_seq in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y_seq = y_seq.to(DEVICE, non_blocking=True)  # (B,K,3,H,W)

            optimizer.zero_grad(set_to_none=True)

            loss = 0.0
            pred_prev = x
            initial_tensor = x  # 对齐推理：unroll 期间使用初始帧做边界与 mask 锁定
            hull_mask = compute_hull_mask_batch(initial_tensor) if PHYS_ENABLE else None

            steps = min(UNROLL_STEPS, y_seq.shape[1])
            for s in range(steps):
                gt = y_seq[:, s]
                pred = model(pred_prev)
                pred = apply_clamp(pred)
                pred = apply_hard_constraints_batch(pred, initial_tensor, hull_mask)
                pred = apply_clamp(pred)

                step_w = STEP_WEIGHT_GAMMA ** s
                if LOSS_TYPE == "mse":
                    loss = loss + step_w * mse(pred, gt)
                elif LOSS_TYPE == "rel_l2":
                    loss = loss + step_w * rel_l2_loss(pred, gt)
                else:
                    raise ValueError(f"Unknown LOSS_TYPE={LOSS_TYPE}")

                pred_prev = pred

            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} | avg_loss={avg:.8f}")

        if SAVE_EVERY > 0 and ((epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg,
                os.path.join(SAVE_DIR, f"fno_epoch_{epoch+1}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fno_final.pth"))
    print(f"✅ Done. Final weights: {os.path.join(SAVE_DIR, 'fno_final.pth')}")
