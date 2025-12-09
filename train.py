import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ================= 核心配置区域 =================
BATCH_SIZE = 16          # A40显存很大，16没问题，如果爆显存改成8
LEARNING_RATE = 1e-5     # 学习率
EPOCHS = 100             # 训练轮数
DATA_DIR = './processed_data'  # 数据路径
SAVE_DIR = './checkpoints'     # 模型保存路径

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用计算设备: {device}")
if device.type == 'cuda':
    print(f"🔥 显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"📊 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ================= 1. 定义数据集 =================
class CFDDataset(Dataset):
    def __init__(self, data_dir):
        """
        加载所有 .npy 数据文件
        使用内存映射方式避免OOM
        """
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

                # 生成样本索引: 输入t -> 预测t+1
                # 最后一帧没有后续帧，所以不包含在训练集中
                for t in range(num_frames - 1):
                    self.valid_indices.append((file_idx, t))

                print(f"  ✓ {os.path.basename(f)}: {data.shape}, {num_frames-1} 个训练样本")

            except Exception as e:
                print(f"  ❌ 加载 {f} 失败: {e}")

        print(f"✅ 数据集就绪！共 {len(self.files)} 个切片，包含 {len(self.valid_indices)} 个训练样本。")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_id, t = self.valid_indices[idx]

        # 读取当前帧和下一帧
        # 数据格式: (H, W, C) = (128, 200, 3)
        current_frame = self.data_maps[file_id][t].astype(np.float32)
        next_frame = self.data_maps[file_id][t+1].astype(np.float32)

        # 归一化策略：
        # X方向速度约27，我们除以 30 让它归一化到 0-1 之间
        # 这样模型收敛更快
        norm_factor = 30.0
        current_frame = current_frame / norm_factor
        next_frame = next_frame / norm_factor

        # 转为 PyTorch Tensor: (H, W, C) -> (C, H, W)
        input_tensor = torch.from_numpy(current_frame).permute(2, 0, 1)
        target_tensor = torch.from_numpy(next_frame).permute(2, 0, 1)

        return input_tensor, target_tensor

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

    checkpoint = torch.load(filepath)
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
    dataset = CFDDataset(DATA_DIR)
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

    # 加载之前的权重继续跑
    checkpoint_path = './checkpoints/resnet_epoch_10.pth'
    if os.path.exists(checkpoint_path):
         model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
         print("✅ 已加载 Epoch 10 权重，继续训练...")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 3. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 均方误差损失

    # 4. 检查是否有断点可以恢复
    print("\n[3/4] 检查断点...")
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, latest_checkpoint)
    else:
        start_epoch = 0
        print("   未找到检查点，从头开始训练")

    # 5. 开始训练
    print(f"\n[4/4] 开始训练！")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   训练轮数: {EPOCHS}")
    print(f"   起始 Epoch: {start_epoch}")
    print("="*60)

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            total_loss = 0

            # 使用 tqdm 显示进度条
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for batch_idx, (inputs, targets) in enumerate(pbar):
                # 数据移到 GPU
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)

                # 计算损失 - 改进版：关注细节变化
                # 1. 算出模型预测的"变化量" (Prediction Delta)
                diff_pred = outputs - inputs

                # 2. 算出真实的"变化量" (Ground Truth Delta)
                diff_gt = targets - inputs

                # 3. 把细节放大 100 倍再算 Loss！
                # 这样模型就必须关注那些微小的涡流变化，否则 Loss 会很大
                loss = criterion(diff_pred * 100.0, diff_gt * 100.0)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 统计
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

            # 计算平均损失
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{EPOCHS} 完成 | 平均 Loss: {avg_loss:.8f}")

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
        final_path = os.path.join(SAVE_DIR, "resnet_final.pth")
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
