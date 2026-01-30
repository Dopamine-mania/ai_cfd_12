import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ================= 核心配置区域 =================
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))  # 显存不够可调小，如 2/4/8
LEARNING_RATE = 1e-5     # 学习率
EPOCHS = 100             # 总训练轮数(首次训练用)；微调时用 EXTRA_EPOCHS 控制额外轮数
DATA_DIR = os.getenv('DATA_DIR', './processed_data/26ms' if os.path.isdir('./processed_data/26ms') else './processed_data')  # 数据路径
SAVE_DIR = './checkpoints'     # 模型保存路径

# ================= 微调(不重训)配置 =================
# 务必从已有 checkpoint 恢复权重与优化器，不要从头开始训练
FINETUNE_FROM = os.getenv('FINETUNE_FROM', './checkpoints/resnet_epoch_100.pth')
EXTRA_EPOCHS = int(os.getenv('EXTRA_EPOCHS', '50'))  # 建议 30-50
RESUME_LATEST = int(os.getenv('RESUME_LATEST', '1')) # 中断后默认续跑最新 checkpoint

# ================= 多步 Unroll 训练配置(核心) =================
# 短 unroll (2-4 步) 让模型在训练期就暴露在“自回归误差”下，从根源缓解长时漂移
UNROLL_STEPS = int(os.getenv('UNROLL_STEPS', '3'))  # 2-4 推荐
if UNROLL_STEPS < 1:
    raise ValueError("UNROLL_STEPS 必须 >= 1")

# 多步损失：同时约束状态和值的变化量(差分)，减少“漂移”同时保留细节
STATE_LOSS_WEIGHT = float(os.getenv('STATE_LOSS_WEIGHT', '0.25'))  # 绝对场 MSE 权重
DELTA_LOSS_WEIGHT = float(os.getenv('DELTA_LOSS_WEIGHT', '1.0'))   # 差分 MSE 权重
DELTA_SCALE = float(os.getenv('DELTA_SCALE', '100.0'))             # 差分放大倍数(沿用原策略)
STEP_WEIGHT_GAMMA = float(os.getenv('STEP_WEIGHT_GAMMA', '1.0'))   # 每步权重衰减(=1 表示等权)

# 物理钳制：训练期也加双重保险，避免多步反传时数值炸掉
TRAIN_CLAMP = int(os.getenv('TRAIN_CLAMP', '1'))
CLAMP_MIN = float(os.getenv('CLAMP_MIN', '-0.5'))
CLAMP_MAX = float(os.getenv('CLAMP_MAX', '1.5'))
CLAMP_MODE = os.getenv('CLAMP_MODE', 'hard').lower()  # hard | hard_ste | smooth(softplus) | tanh | none
SMOOTH_CLAMP_BETA = float(os.getenv('SMOOTH_CLAMP_BETA', '200.0'))  # 越大越接近 hard clamp，且区间内更接近恒等

# 可选：梯度裁剪，提升多步反传稳定性
CLIP_GRAD_NORM = float(os.getenv('CLIP_GRAD_NORM', '1.0'))  # 设为 0 关闭

# 显存不够时用梯度累积模拟大 batch（不会增加显存占用，代价是更慢）
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', '1'))

# 物理一致性/结构一致性损失（更适合写进 POF）
# 1) Range penalty: 让网络学会“不要去撞 clamp”，减少长期饱和导致的形态失真
#    soft clamp 场景下一般不需要；默认关闭。
RANGE_PENALTY_WEIGHT = float(os.getenv('RANGE_PENALTY_WEIGHT', '0.0'))
# 2) Vorticity loss: 用切片涡量保持涡结构（ω_x = ∂W/∂y - ∂V/∂z，假设 H=Y, W=Z）
VORTICITY_LOSS_WEIGHT = float(os.getenv('VORTICITY_LOSS_WEIGHT', '0.05'))

def range_penalty(x, min_val, max_val):
    over = torch.relu(x - max_val)
    under = torch.relu(min_val - x)
    return (over * over + under * under).mean()

def soft_clamp_tanh(x, min_val, max_val):
    """
    不推荐的“压缩式”软钳制：把输出整体压缩映射到 [min_val, max_val]。
    注意：tanh 映射不是区间内恒等，可能导致动力学被“拉平/变形”，长序列更明显。
    """
    mid = (max_val + min_val) * 0.5
    half = (max_val - min_val) * 0.5
    # half>0，tanh 将实数映射到 (-1,1)
    return mid + half * torch.tanh((x - mid) / max(half, 1e-6))

def smooth_clamp_softplus(x, min_val, max_val, beta):
    """
    推荐的“恒等式”软钳制：区间内近似恒等，越界后平滑饱和到边界。
    y = x + softplus(min-x) - softplus(x-max)
    """
    beta = max(float(beta), 1e-6)
    return x + F.softplus(min_val - x, beta=beta, threshold=20.0) - F.softplus(x - max_val, beta=beta, threshold=20.0)

def hard_clamp_ste(x, min_val, max_val):
    """
    Hard clamp 的“直通估计器”(STE)：前向等价于 clamp，反向把梯度当成恒等传递，避免越界时梯度饱和。
    """
    y = torch.clamp(x, min=min_val, max=max_val)
    return x + (y - x).detach()

def vorticity_x(field):
    """
    计算切片涡量 ω_x = ∂W/∂y - ∂V/∂z
    field: (B,3,H,W) 通道顺序 [U,V,W]
    """
    v = field[:, 1:2]
    w = field[:, 2:3]

    # 中心差分核
    ky = torch.tensor([[-0.5], [0.0], [0.5]], device=field.device, dtype=field.dtype).view(1, 1, 3, 1)
    kz = torch.tensor([[-0.5, 0.0, 0.5]], device=field.device, dtype=field.dtype).view(1, 1, 1, 3)

    dw_dy = F.conv2d(w, ky, padding=(1, 0))
    dv_dz = F.conv2d(v, kz, padding=(0, 1))
    return dw_dy - dv_dz

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用计算设备: {device}")
if device.type == 'cuda':
    print(f"🔥 显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"📊 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ================= 1. 定义数据集 =================
class CFDDataset(Dataset):
    def __init__(self, data_dir, unroll_steps):
        """
        加载所有 .npy 数据文件
        使用内存映射方式避免OOM
        """
        if unroll_steps < 1:
            raise ValueError("unroll_steps 必须 >= 1")
        self.unroll_steps = int(unroll_steps)

        # 获取所有数据文件（按数字排序）
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))

        if len(self.files) == 0:
            raise ValueError(f"在 {data_dir} 中没有找到 .npy 文件！")

        self.data_maps = []
        self.valid_indices = []

        print(f"📂 正在建立数据索引 (使用内存映射，不会爆内存)...")

        for file_idx, f in enumerate(self.files):
            # mmap_mode='r' 关键！只建立映射不读入内存
            try:
                data = np.load(f, mmap_mode='r')
                self.data_maps.append(data)
                num_frames = data.shape[0]

                # 生成样本索引: 输入t -> 预测 t+1..t+K
                # 需要保证有足够的未来帧，因此最后 K 帧不参与训练样本
                for t in range(num_frames - self.unroll_steps):
                    self.valid_indices.append((file_idx, t))

                print(f"  ✓ {os.path.basename(f)}: {data.shape}, {num_frames-self.unroll_steps} 个训练样本 (unroll={self.unroll_steps})")

            except Exception as e:
                print(f"  ❌ 加载 {f} 失败: {e}")

        print(f"✅ 数据集就绪！共 {len(self.files)} 个切片，包含 {len(self.valid_indices)} 个训练样本。")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_id, t = self.valid_indices[idx]

        # 读取当前帧和未来 K 帧
        # 数据格式: (H, W, C) = (200, 128, 3)
        current_frame = self.data_maps[file_id][t].astype(np.float32)
        future_frames = [
            self.data_maps[file_id][t + i].astype(np.float32)
            for i in range(1, self.unroll_steps + 1)
        ]

        # 归一化策略：
        # X方向速度约27，我们除以 30 让它归一化到 0-1 之间
        # 这样模型收敛更快
        norm_factor = 30.0
        current_frame = current_frame / norm_factor
        future_frames = [f / norm_factor for f in future_frames]

        # 转为 PyTorch Tensor: (H, W, C) -> (C, H, W)
        input_tensor = torch.from_numpy(current_frame).permute(2, 0, 1)
        target_tensors = torch.stack(
            [torch.from_numpy(f).permute(2, 0, 1) for f in future_frames],
            dim=0,
        )  # (K, C, H, W)

        return input_tensor, target_tensors

# ================= 2. 定义模型 (ResNet架构) =================
class ResidualBlock(nn.Module):
    """残差块：学习输入到输出的变化量"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),  # 加入BN层加速收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv(x)  # 学习残差

class CFDPredictor(nn.Module):
    """CFD 速度场预测器 - ResNet 架构"""
    def __init__(self):
        super(CFDPredictor, self).__init__()

        # 编码器 (下采样，提取特征)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 128x200 -> 64x100
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 64x100 -> 32x50
            nn.ReLU(inplace=True)
        )

        # 瓶颈层 (学习物理规律)
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # 解码器 (上采样，恢复分辨率)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x50 -> 64x100
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64x100 -> 128x200
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 输出层
        )

    def forward(self, x):
        # 编码
        features = self.encoder(x)

        # 学习物理规律
        features = self.bottleneck(features)

        # 解码得到变化量
        delta = self.decoder(features)

        # 物理约束：下一时刻 = 当前时刻 + 变化量
        return x + delta

# ================= 3. 检查点管理 =================
def save_checkpoint(model, optimizer, epoch, loss, filename):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    filepath = os.path.join(SAVE_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"💾 检查点已保存: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """加载训练检查点"""
    if not os.path.exists(filepath):
        return 0

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print(f"📥 已加载检查点: {filepath}")
    print(f"   从 Epoch {start_epoch} 继续训练")

    return start_epoch

def get_latest_checkpoint():
    """获取最新的检查点文件"""
    checkpoints = glob.glob(os.path.join(SAVE_DIR, 'resnet_epoch_*.pth'))
    if not checkpoints:
        return None

    # 按 epoch 数字排序
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
    return checkpoints[-1]

def get_resume_checkpoint():
    """
    获取用于恢复训练的 checkpoint:
    - 默认优先使用最新的 checkpoint（支持微调中断后继续）
    - 优先 resnet_interrupted.pth（若比最新 epoch checkpoint 更新）
    - 若未启用 RESUME_LATEST，则优先 FINETUNE_FROM
    """
    latest_epoch_ckpt = get_latest_checkpoint()
    interrupted = os.path.join(SAVE_DIR, 'resnet_interrupted.pth')
    if RESUME_LATEST:
        if os.path.exists(interrupted):
            if (not latest_epoch_ckpt) or (not os.path.exists(latest_epoch_ckpt)):
                return interrupted
            if os.path.getmtime(interrupted) >= os.path.getmtime(latest_epoch_ckpt):
                return interrupted
        if latest_epoch_ckpt and os.path.exists(latest_epoch_ckpt):
            return latest_epoch_ckpt

    if os.path.exists(FINETUNE_FROM):
        return FINETUNE_FROM
    if latest_epoch_ckpt and os.path.exists(latest_epoch_ckpt):
        return latest_epoch_ckpt
    if os.path.exists(interrupted):
        return interrupted

    return None

def get_checkpoint_start_epoch(filepath):
    """
    读取 checkpoint 的起始 epoch（= ckpt['epoch'] + 1）。
    若不是包含元数据的 checkpoint（仅权重），返回 None。
    """
    ckpt = torch.load(filepath, map_location='cpu')
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        return int(ckpt['epoch']) + 1
    return None

# ================= 4. 训练主程序 =================
if __name__ == '__main__':
    # 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("="*60)
    print("CFD 深度学习训练系统")
    print("="*60)

    # 1. 准备数据
    print("\n[1/4] 准备数据集...")
    dataset = CFDDataset(DATA_DIR, unroll_steps=UNROLL_STEPS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速 CPU->GPU 传输
    )

    # 2. 初始化模型
    print("\n[2/4] 初始化模型...")
    model = CFDPredictor().to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 3. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 均方误差损失

    # 4. 检查是否有断点可以恢复
    print("\n[3/4] 检查断点...")
    start_epoch = 0
    resume_ckpt = get_resume_checkpoint()
    if resume_ckpt:
        start_epoch = load_checkpoint(model, optimizer, resume_ckpt)
    else:
        raise RuntimeError(
            f"未找到可用 checkpoint（FINETUNE_FROM={FINETUNE_FROM}，且 {SAVE_DIR}/resnet_epoch_*.pth 不存在）。"
            "按交接要求不要从头训练，请先确认权重文件在 checkpoints/ 下。"
        )

    # 5. 开始训练
    print(f"\n[4/4] 开始训练！")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   学习率: {LEARNING_RATE}")
    finetune_origin_ckpt = FINETUNE_FROM if os.path.exists(FINETUNE_FROM) else resume_ckpt
    finetune_origin_start_epoch = get_checkpoint_start_epoch(finetune_origin_ckpt) or start_epoch
    target_end_epoch = finetune_origin_start_epoch + EXTRA_EPOCHS
    end_epoch = target_end_epoch
    if start_epoch >= end_epoch:
        print(f"✅ 已达到目标轮数：当前 start_epoch={start_epoch}，目标 end_epoch={end_epoch}，无需继续训练。")
        raise SystemExit(0)

    print(f"   训练轮数: {end_epoch} (起始 {start_epoch}, 目标结束 {end_epoch})")
    print(f"   起始 Epoch: {start_epoch}")
    print(f"   恢复训练: {resume_ckpt}")
    print(f"   微调基准: {finetune_origin_ckpt} (origin_start={finetune_origin_start_epoch}, extra={EXTRA_EPOCHS})")
    print(f"   Unroll: steps={UNROLL_STEPS}, gamma={STEP_WEIGHT_GAMMA}")
    print(f"   Loss: state_w={STATE_LOSS_WEIGHT}, delta_w={DELTA_LOSS_WEIGHT}, delta_scale={DELTA_SCALE}")
    print(f"   Clamp: enable={TRAIN_CLAMP}, mode={CLAMP_MODE}, min={CLAMP_MIN}, max={CLAMP_MAX}, smooth_beta={SMOOTH_CLAMP_BETA}")
    print(f"   GradClip: max_norm={CLIP_GRAD_NORM}")
    print("="*60)

    try:
        avg_loss = 0.0
        for epoch in range(start_epoch, end_epoch):
            model.train()
            total_loss = 0

            # 使用 tqdm 显示进度条
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{end_epoch}")

            for batch_idx, (inputs, targets_seq) in enumerate(pbar):
                # 数据移到 GPU
                inputs = inputs.to(device)
                targets_seq = targets_seq.to(device)  # (B, K, C, H, W)

                # 前向传播
                if batch_idx % GRAD_ACCUM_STEPS == 0:
                    optimizer.zero_grad(set_to_none=True)
                loss = 0.0

                pred_prev = inputs
                gt_prev = inputs

                steps = min(UNROLL_STEPS, targets_seq.shape[1])
                for s in range(steps):
                    pred_raw = model(pred_prev)

                    if TRAIN_CLAMP and CLAMP_MODE != 'none':
                        if CLAMP_MODE in ('smooth', 'softplus'):
                            pred = smooth_clamp_softplus(pred_raw, CLAMP_MIN, CLAMP_MAX, SMOOTH_CLAMP_BETA)
                        elif CLAMP_MODE in ('soft', 'tanh'):
                            pred = soft_clamp_tanh(pred_raw, CLAMP_MIN, CLAMP_MAX)
                        elif CLAMP_MODE == 'hard':
                            pred = torch.clamp(pred_raw, min=CLAMP_MIN, max=CLAMP_MAX)
                        elif CLAMP_MODE in ('hard_ste', 'ste'):
                            pred = hard_clamp_ste(pred_raw, CLAMP_MIN, CLAMP_MAX)
                        else:
                            raise ValueError(f"未知 CLAMP_MODE: {CLAMP_MODE} (期望 hard|hard_ste|smooth|tanh|none)")
                    else:
                        pred = pred_raw

                    gt = targets_seq[:, s]
                    step_w = STEP_WEIGHT_GAMMA ** s

                    if STATE_LOSS_WEIGHT > 0:
                        loss = loss + step_w * STATE_LOSS_WEIGHT * criterion(pred, gt)

                    if DELTA_LOSS_WEIGHT > 0:
                        delta_pred = pred - pred_prev
                        delta_gt = gt - gt_prev
                        loss = loss + step_w * DELTA_LOSS_WEIGHT * criterion(delta_pred * DELTA_SCALE, delta_gt * DELTA_SCALE)

                    if VORTICITY_LOSS_WEIGHT > 0:
                        vort_pred = vorticity_x(pred)
                        vort_gt = vorticity_x(gt)
                        loss = loss + step_w * VORTICITY_LOSS_WEIGHT * criterion(vort_pred, vort_gt)

                    if TRAIN_CLAMP and CLAMP_MODE == 'hard' and RANGE_PENALTY_WEIGHT > 0:
                        loss = loss + step_w * RANGE_PENALTY_WEIGHT * range_penalty(pred_raw, CLAMP_MIN, CLAMP_MAX)

                    pred_prev = pred
                    gt_prev = gt

                # 反向传播和优化
                (loss / GRAD_ACCUM_STEPS).backward()
                is_step = ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0) or (batch_idx + 1 == len(dataloader))
                if is_step:
                    if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                    optimizer.step()

                # 统计
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{end_epoch} 完成 | 平均 Loss: {avg_loss:.8f}")

            # 每10轮保存一次模型
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    f"resnet_epoch_{epoch+1}.pth"
                )

        # 保存最终结果
        final_path = os.path.join(SAVE_DIR, "resnet_unroll_softclamp_final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"\n🎉🎉🎉 恭喜！训练全部完成！")
        print(f"💾 最终模型已保存: {final_path}")
        print(f"📁 检查点文件夹: {SAVE_DIR}")

    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        save_checkpoint(model, optimizer, epoch, avg_loss, "resnet_interrupted.pth")
        print("💾 已保存中断时的检查点")

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        save_checkpoint(model, optimizer, epoch, 0, "resnet_error.pth")
        print("💾 已保存错误时的检查点")
        raise
