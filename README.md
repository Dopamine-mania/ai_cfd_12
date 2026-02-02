# CFD AI 训练项目

舰船尾流 3 通道速度场切片（U/V/W）时序预测项目。

当前交付主路线：`time_conditioned/` 下的 **Time-Conditioned ResNet（One-shot）**，用 `x_t = G(x0, t)` 避免 4500 步自回归误差累积。
备选/实验：FNO（Fourier Neural Operator + padding 处理非周期边界）。
历史对照：老版 ResNet 自回归路线已归档在 `resnet_legacy/`。

## 项目概述

本项目使用深度学习技术预测流体动力学中的速度场演化，目标是长时滚动预测（9s / 4500 steps）。

## 环境信息

- **GPU**: NVIDIA A40 (46GB VRAM)
- **Python**: 3.11.7
- **PyTorch**: 2.3.1 (CUDA 12.1)
- **数据格式**: NumPy arrays (6000, 200, 128, 3) - [时间步, Y, Z, 通道(U,V,W)]

## 项目结构

```
ai_cfd/
├── activate_cfd.sh       # 环境激活脚本
├── train.py              # 主训练脚本（ResNet架构）
├── predict.py            # 预测脚本
├── time_conditioned/     # One-shot/time-conditioned ResNet 训练/推理（推荐）
├── fno/                  # FNO 训练/推理脚本（当前主路线）
├── resnet_legacy/        # ResNet 归档/对照（tag）
├── run_train.ipynb       # Jupyter Notebook 启动脚本
├── check_data.py         # 数据验证脚本
├── .gitignore            # Git 忽略文件配置
├── cfd_env/              # 虚拟环境（不上传）
├── processed_data/       # 训练数据（不上传）
│   ├── 26ms/             # 26m/s 工况（基座）
│   └── 18ms/             # 18m/s 工况（迁移学习）
├── checkpoints/          # 模型检查点（不上传）
├── checkpoints_fno/      # FNO 模型检查点（不上传）
├── logs/                 # 训练日志（不上传）
└── tensorboard/          # TensorBoard 日志（不上传）
```

## 快速开始

### 1. 环境搭建

```bash
# 创建虚拟环境
python -m venv cfd_env

# 激活环境
source cfd_env/bin/activate

# 安装依赖
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas matplotlib seaborn tensorboard tqdm psutil ipykernel
```

### 2. 启动训练

#### Time-Conditioned ResNet（推荐/交付路线）

26m/s one-shot 训练：
```bash
source activate_cfd.sh
python -u time_conditioned/train_time_resnet.py | tee logs/time_resnet_26ms_$(date +%Y%m%d_%H%M%S).log
```

18m/s 迁移学习（从 26m/s 权重微调 + 产出 9 秒交付物）：
```bash
source activate_cfd.sh
bash time_conditioned/run_18ms_transfer.sh
```

#### FNO（推荐/当前主路线）

训练（与项目组确认一致：`MODES=12, WIDTH=32, PADDING=20, LR=1e-3, EPOCHS=100`）：
```bash
source activate_cfd.sh
MODES=12 WIDTH=32 PADDING=20 EPOCHS=100 LR=1e-3 BATCH_SIZE=16 \
python -u fno/train_fno.py | tee logs/fno_train_$(date +%Y%m%d_%H%M%S).log
```

推理并生成 9 秒 GIF（输出默认在 `results_fno/`）：
```bash
MODEL_PATH=./checkpoints_fno/fno_final.pth \
python -u fno/predict_fno.py
```

#### ResNet（对照/已归档）

**方式 1: 命令行（简单）**
```bash
source activate_cfd.sh
python train.py
```

**方式 2: 后台运行（推荐长时间训练）**
```bash
# 使用 tmux
tmux new-session -s cfd_training
source activate_cfd.sh
python train.py | tee logs/train_$(date +%Y%m%d_%H%M%S).log
# 按 Ctrl+b, d 分离会话

# 使用 nohup
nohup python train.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**方式 3: Jupyter Notebook（交互式）**
1. 打开 JupyterLab
2. 选择 Kernel: `Python (CFD Environment)`
3. 打开 `run_train.ipynb`
4. 运行 `%run train.py`

### 3. 监控训练

```bash
# 实时日志
tail -f logs/train_*.log

# GPU 监控
watch -n 1 nvidia-smi

# 检查检查点
ls -lh checkpoints/
```

## 模型架构

### ResNet CFD Predictor

```
输入 (3, 128, 200) → 编码器 → 残差块×4 → 解码器 → 输出 (3, 128, 200)
```

**编码器** (下采样)
- Conv2d(3→64) → Conv2d(64→128, stride=2) → Conv2d(128→256, stride=2)

**残差块** (学习物理规律)
- 4 个 ResidualBlock(256)，每个包含 2 层卷积 + BatchNorm + skip connection

**解码器** (上采样)
- ConvTranspose2d(256→128, stride=2) → ConvTranspose2d(128→64, stride=2) → Conv2d(64→3)

**损失函数** (关注细节变化)
```python
diff_pred = outputs - inputs      # 预测的变化量
diff_gt = targets - inputs        # 真实的变化量
loss = MSE(diff_pred * 100.0, diff_gt * 100.0)  # 放大100倍强调细节
```

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 16 | A40 显存充足 |
| Learning Rate | 1e-4 | Adam 优化器 |
| Epochs | 100 | 可根据收敛调整 |
| Loss Function | MSE | 均方误差 |
| 归一化 | /30.0 | 基于速度范围 |
| 检查点保存 | 每10 epochs | 自动保存 |

## 预期结果

- **训练时间**: ~10-12 小时 (100 epochs)
- **初始 Loss**: 0.01 - 0.05
- **收敛 Loss**: < 0.001
- **GPU 利用率**: 80-100%
- **显存使用**: ~10-15 GB / 46 GB

## 数据格式

训练数据为 NumPy 数组，每个文件包含：
- **形状**: (6000, 200, 128, 3)
- **维度**: [时间步, 宽度, 高度, 通道]
- **通道**:
  - Channel 0: U 速度分量
  - Channel 1: V 速度分量
  - Channel 2: W 速度分量

## 关键特性

✅ **内存映射加载** - 使用 `np.load(mmap_mode='r')` 避免 OOM
✅ **断点续训** - 自动检测最新检查点并继续训练
✅ **关注细节** - 损失函数放大 100 倍强调微小变化
✅ **进度监控** - tqdm 进度条 + 实时 Loss 显示
✅ **多种启动方式** - 命令行 / 后台 / Jupyter Notebook

## 故障排查

### CUDA Out of Memory
```python
# 修改 train.py 中的 BATCH_SIZE
BATCH_SIZE = 8  # 从 16 改为 8
```

### 数据加载慢
```python
# 增加 DataLoader 的 num_workers
DataLoader(..., num_workers=8)  # 从 4 改为 8
```

### 训练不收敛
- 检查数据归一化
- 降低学习率: `LEARNING_RATE = 1e-5`
- 添加学习率调度器


## 许可证

本项目仅用于学术研究和教育目的。

## 致谢

- NVIDIA A40 GPU 计算支持
- PyTorch 深度学习框架
