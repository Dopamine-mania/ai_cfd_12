import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train import CFDPredictor  # 从训练代码导入模型结构
import os

# ================= 配置 =================
# 可用环境变量 MODEL_PATH 指定权重文件，便于切换微调后的 checkpoint
MODEL_PATH = os.getenv('MODEL_PATH', './checkpoints/resnet_epoch_100.pth')  # 使用100轮训练的完整checkpoint
_DEFAULT_DATA_PATH = './processed_data/26ms/340.npy' if os.path.exists('./processed_data/26ms/340.npy') else './processed_data/340.npy'
DATA_PATH = os.getenv('DATA_PATH', _DEFAULT_DATA_PATH) # X=340位置的截面数据
SAVE_GIF = os.getenv('SAVE_GIF', './results/prediction_9s.gif')  # 9秒预测结果
SAVE_ERROR_CURVE = os.getenv('SAVE_ERROR_CURVE', './results/error_curve_9s.png')  # 误差曲线
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 9秒预测 (Δt=2ms). 可用环境变量 PRED_STEPS 覆盖，便于快速调试
PRED_STEPS = int(os.getenv('PRED_STEPS', '4500'))

# 归一化后的合理物理范围 (用于钳制防止溢出/发散)
CLAMP_MIN = float(os.getenv('CLAMP_MIN', '-0.5'))
CLAMP_MAX = float(os.getenv('CLAMP_MAX', '1.5'))
CLAMP_MODE = os.getenv('CLAMP_MODE', 'hard').lower()  # hard | smooth(softplus) | soft(tanh)
# alpha 控制 soft clamp 在中部的“斜率”：导数约为 1/alpha。
# - alpha=1：中部近似恒等映射（不压缩细节），边界处平滑饱和（推荐，且与训练端一致）
# - alpha>1：会把场拉向 mid（可能造成“发灰/变平”）
SOFT_CLAMP_ALPHA = float(os.getenv('SOFT_CLAMP_ALPHA', '1.0'))
SMOOTH_CLAMP_BETA = float(os.getenv('SMOOTH_CLAMP_BETA', '200.0'))  # 越大越接近 hard clamp，且区间内更接近恒等

# 空间滤波 (Gaussian Smoothing): 强效低通，抑制花屏高频震荡
# 微调方案下默认关闭；需要时再通过 GAUSS_ENABLE=1 打开
GAUSS_ENABLE = int(os.getenv('GAUSS_ENABLE', '0'))
GAUSS_BLEND = float(os.getenv('GAUSS_BLEND', '0.8'))  # 0.8*平滑 + 0.2*原图
GAUSS_START_STEP = int(os.getenv('GAUSS_START_STEP', '3500'))  # 从第几步开始启用(默认后段启用以保细节)

# ================= 物理硬约束（推理端） =================
# 1) 船体/死水区 Mask：从 t=0 自动估计，并在滚动预测中锁死（不让模型预测船体）。
MASK_ENABLE = int(os.getenv('MASK_ENABLE', '1'))
MASK_MODE = os.getenv('MASK_MODE', 'u').lower()  # u | speed
MASK_U_THRESHOLD = float(os.getenv('MASK_U_THRESHOLD', '0.2'))  # 在归一化空间 (u/30)
MASK_SPEED_THRESHOLD = float(os.getenv('MASK_SPEED_THRESHOLD', '0.2'))  # 在归一化空间 (|v|/30)
MASK_LEFT_COLS = int(os.getenv('MASK_LEFT_COLS', '24'))  # 只在左侧若干列里找 mask，避免误锁 wake
MASK_LOCK = os.getenv('MASK_LOCK', 'initial').lower()  # initial | zero

# 2) 开放边界 Dirichlet：每一步把远场边缘像素重置为 t=0 的背景流场值（类似海绵层硬边界）
BC_ENABLE = int(os.getenv('BC_ENABLE', '1'))
BC_PAD_RIGHT = int(os.getenv('BC_PAD_RIGHT', '4'))
BC_PAD_TOP = int(os.getenv('BC_PAD_TOP', '4'))
BC_PAD_BOTTOM = int(os.getenv('BC_PAD_BOTTOM', '4'))

# 数值阻尼 (Numerical Damping): next = (1-d)*pred + d*prev
# 可用环境变量 DAMPING 覆盖 (0~1)，d 越大越平滑/更稳定但更“粘”
# 微调方案下默认关闭；需要时再通过 DAMPING=0.05~0.2 打开
DAMPING = float(os.getenv('DAMPING', '0.0'))

# 可导软钳制：避免硬截断造成“大色块/死锁感”，同时仍保证落在物理区间
def soft_clamp_tanh(x, min_val, max_val, alpha):
    mid = (max_val + min_val) * 0.5
    half = (max_val - min_val) * 0.5
    denom = max(half * max(alpha, 1e-6), 1e-6)
    return mid + half * torch.tanh((x - mid) / denom)

def smooth_clamp_softplus(x, min_val, max_val, beta):
    """
    更适合“视觉保真”的软钳制：区间内近似恒等，越界后平滑饱和到边界。
    y = x + softplus(min-x) - softplus(x-max)
    """
    beta = max(float(beta), 1e-6)
    return x + F.softplus(min_val - x, beta=beta, threshold=20.0) - F.softplus(x - max_val, beta=beta, threshold=20.0)

def apply_physical_bound(x):
    if CLAMP_MODE in ('soft', 'tanh'):
        return soft_clamp_tanh(x, CLAMP_MIN, CLAMP_MAX, SOFT_CLAMP_ALPHA)
    if CLAMP_MODE == 'smooth':
        return smooth_clamp_softplus(x, CLAMP_MIN, CLAMP_MAX, SMOOTH_CLAMP_BETA)
    if CLAMP_MODE == 'hard':
        return torch.clamp(x, min=CLAMP_MIN, max=CLAMP_MAX)
    raise ValueError(f"未知 CLAMP_MODE={CLAMP_MODE}，期望 hard|soft|smooth")

def compute_hull_mask(initial_tensor):
    """
    initial_tensor: (1,3,H,W) 归一化后的初始帧
    返回: (1,1,H,W) bool mask（True 表示锁定区域）
    """
    if not MASK_ENABLE:
        return None

    _, _, h, w = initial_tensor.shape
    left_cols = max(0, min(int(MASK_LEFT_COLS), w))
    if left_cols <= 0:
        return None

    if MASK_MODE == 'u':
        u0 = initial_tensor[:, 0:1, :, :]  # (1,1,H,W)
        low = u0 < MASK_U_THRESHOLD
    elif MASK_MODE == 'speed':
        speed0 = torch.linalg.vector_norm(initial_tensor, dim=1, keepdim=True)  # (1,1,H,W)
        low = speed0 < MASK_SPEED_THRESHOLD
    else:
        raise ValueError(f"未知 MASK_MODE={MASK_MODE}，期望 u|speed")

    region = torch.zeros((1, 1, h, w), device=initial_tensor.device, dtype=torch.bool)
    region[:, :, :, :left_cols] = True
    return low & region

def apply_hard_constraints(x, initial_tensor, hull_mask):
    """
    x: (1,3,H,W) 当前预测
    initial_tensor: (1,3,H,W) t=0 归一化初始帧（用于 Dirichlet BC 与 mask 锁定）
    hull_mask: (1,1,H,W) bool 或 None
    """
    # 远场边界 Dirichlet：右/上/下
    if BC_ENABLE:
        if BC_PAD_RIGHT > 0:
            x[:, :, :, -BC_PAD_RIGHT:] = initial_tensor[:, :, :, -BC_PAD_RIGHT:]
        if BC_PAD_TOP > 0:
            x[:, :, :BC_PAD_TOP, :] = initial_tensor[:, :, :BC_PAD_TOP, :]
        if BC_PAD_BOTTOM > 0:
            x[:, :, -BC_PAD_BOTTOM:, :] = initial_tensor[:, :, -BC_PAD_BOTTOM:, :]

    # 船体/死水区锁定
    if hull_mask is not None:
        if MASK_LOCK == 'zero':
            lock_val = torch.zeros_like(x)
        elif MASK_LOCK == 'initial':
            lock_val = initial_tensor
        else:
            raise ValueError(f"未知 MASK_LOCK={MASK_LOCK}，期望 initial|zero")
        x = torch.where(hull_mask.expand_as(x), lock_val, x)

    return x

# 固定 3x3 高斯核 (低通滤波器)，对 3 通道做 depthwise conv
GAUSS_KERNEL_3x3 = torch.tensor(
    [[1.0, 2.0, 1.0],
     [2.0, 4.0, 2.0],
     [1.0, 2.0, 1.0]],
    device=DEVICE,
    dtype=torch.float32,
) / 16.0
GAUSS_KERNEL_3x3 = GAUSS_KERNEL_3x3.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

if not os.path.exists('./results'):
    os.makedirs('./results')

# 1. 加载模型
print("🔄 正在加载模型...")
model = CFDPredictor().to(DEVICE)
# # 如果 final 还没跑完，你可以手动改这里加载中间的 checkpoint，比如 resnet_epoch_10.pth
# if os.path.exists(MODEL_PATH):
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     print("✅ 模型权重加载成功！")
# else:
#     print(f"⚠️ 找不到 {MODEL_PATH}，请等待训练结束或修改路径加载中间权重。")
#     exit()
print(f"🔄 正在加载模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# --- 核心修改开始 ---
if 'model_state_dict' in checkpoint:
    # 情况 A: 如果是包含了 epoch, loss 等信息的“大礼包”
    print("📦 检测到包含元数据的模型文件，正在提取权重...")
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # 情况 B: 如果只是单纯的权重文件
    print("📄 检测到纯权重文件，直接加载...")
    model.load_state_dict(checkpoint)
# --- 核心修改结束 ---

print("✅ 模型权重加载成功！")

model.eval()

# 2. 加载测试数据
print("📂 读取测试数据...")
# 使用 mmap 避免一次性读入内存
data = np.load(DATA_PATH, mmap_mode='r') # (6001, 200, 128, 3) -> [时间, Y, Z, (U,V,W)]
# 取前 (PRED_STEPS+1) 帧用于预测 (初始帧 + PRED_STEPS 步预测)
test_seq = data[: PRED_STEPS + 1]
print(f"✅ 测试数据形状: {test_seq.shape}") 

# 3. 滚动预测 (Rolling Prediction)
# 也就是：用第1帧预测第2帧，用预测出的第2帧预测第3帧... (这是最难的，看AI会不会崩)
print("🔮 开始滚动预测 (自回归测试)...")
current_frame = test_seq[0] # 初始条件 (H, W, 3)

# 归一化
current_tensor = torch.from_numpy(current_frame / 30.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
initial_tensor = current_tensor.clone()
hull_mask = compute_hull_mask(initial_tensor)
if hull_mask is not None:
    mask_ratio = hull_mask.float().mean().item()
    print(f"🧱 Hull mask: mode={MASK_MODE}, lock={MASK_LOCK}, left_cols={MASK_LEFT_COLS}, ratio={mask_ratio:.4f}")
if BC_ENABLE:
    print(f"🧷 Dirichlet BC: right={BC_PAD_RIGHT}px, top={BC_PAD_TOP}px, bottom={BC_PAD_BOTTOM}px (reset to t=0)")

# 为了避免存下 4501 帧(占用数GB内存)，这里按 stride 只缓存 GIF 需要的帧；
# 误差曲线则按步在线计算。
GIF_STRIDE = int(os.getenv('GIF_STRIDE', '25'))  # 约 9s@20fps -> 180帧左右
gif_steps = list(range(0, PRED_STEPS + 1, max(GIF_STRIDE, 1)))
if gif_steps[-1] != PRED_STEPS:
    gif_steps.append(PRED_STEPS)
gif_step_set = set(gif_steps)
gif_truth_u = {0: test_seq[0, :, :, 0]}
gif_pred_u = {0: test_seq[0, :, :, 0]}
errors = [0.0]  # t=0: 预测等于初值

print(f"开始 {PRED_STEPS} 步滚动预测 (9秒 @ Δt=2ms)...")
with torch.no_grad():
    for t in range(PRED_STEPS):
        # 1. 预测
        next_tensor = model(current_tensor)
        step_idx = t + 1

        # 2. 物理约束与阻尼
        # A) 先做范围约束（防止数值爆掉影响后续卷积/混合）
        next_tensor = apply_physical_bound(next_tensor)

        # B) 空间高斯滤波: 强效低通，滤除花屏高频噪点（类似 LES 过滤）
        if GAUSS_ENABLE and step_idx >= GAUSS_START_STEP:
            kernel = GAUSS_KERNEL_3x3.to(device=next_tensor.device, dtype=next_tensor.dtype)
            next_smoothed = F.conv2d(next_tensor, kernel, padding=1, groups=3)
            next_tensor = GAUSS_BLEND * next_smoothed + (1.0 - GAUSS_BLEND) * next_tensor

        # C) 数值阻尼: 时间方向惯性平滑（可选，默认 0.1）
        if DAMPING > 0:
            next_tensor = (1.0 - DAMPING) * next_tensor + DAMPING * current_tensor

        # D) 船体 mask 锁定 + 远场 Dirichlet 边界（硬约束，抑制误差从边缘倒灌）
        next_tensor = apply_hard_constraints(next_tensor, initial_tensor, hull_mask)

        # E) 再约束一次，避免混合/重置后仍越界
        next_tensor = apply_physical_bound(next_tensor)

        # 3. 转回 numpy (物理量单位)
        pred_np = next_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 30.0
        # 4. 在线计算误差 (L2 相对误差)
        truth_np = test_seq[step_idx]
        diff_norm = np.linalg.norm(pred_np - truth_np)
        truth_norm = np.linalg.norm(truth_np)
        errors.append(diff_norm / truth_norm)

        # 5. 仅缓存 GIF 需要的帧 (U 分量)
        if step_idx in gif_step_set:
            gif_truth_u[step_idx] = truth_np[:, :, 0]
            gif_pred_u[step_idx] = pred_np[:, :, 0]

        # 6. 更新输入
        current_tensor = next_tensor

        # 7. 进度显示
        if (t + 1) % 500 == 0:
            print(f"  Step {t+1}/{PRED_STEPS}: Max={next_tensor.max().item():.3f}, Min={next_tensor.min().item():.3f}")

errors = np.array(errors)
print(f"✅ 预测完成！")
print(f"✅ 误差计算完成！")
print(f"   平均相对误差: {errors.mean():.6f}")
print(f"   最大相对误差: {errors.max():.6f}")
print(f"   最终时刻误差: {errors[-1]:.6f}")

# 5. 画图制作 GIF
print("🎨 正在渲染 GIF (左: 真实, 右: 预测)...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

vmin, vmax = test_seq[:, :, :, 0].min(), test_seq[:, :, :, 0].max()

def update(frame_i):
    for ax in axes: ax.clear()

    step_idx = gif_steps[frame_i]
    # 取 U 速度 (X方向) 展示
    real_u = gif_truth_u[step_idx]
    pred_u = gif_pred_u[step_idx]
    
    axes[0].imshow(real_u, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (step={step_idx}, t={step_idx*0.002:.3f}s)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_u, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"AI Prediction (step={step_idx}, t={step_idx*0.002:.3f}s)")
    axes[1].axis('off')

ani = animation.FuncAnimation(fig, update, frames=len(gif_steps), interval=50)
ani.save(SAVE_GIF, writer='pillow', fps=20)
plt.close(fig)

print(f"🎉 动图已保存至: {SAVE_GIF}")

# 6. 绘制误差曲线图
print("📈 绘制误差曲线...")
fig, ax = plt.subplots(figsize=(10, 6))

# 时间轴 (秒)
time_axis = np.arange(len(errors)) * 0.002  # Δt = 2ms = 0.002s

# 绘制误差曲线
ax.plot(time_axis, errors, linewidth=2, color='#E74C3C', label='L2 Relative Error')
ax.axhline(y=errors.mean(), color='#3498DB', linestyle='--', linewidth=1.5,
           label=f'Mean Error = {errors.mean():.6f}')

# 设置图表
ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative L2 Error', fontsize=14, fontweight='bold')
ax.set_title('Long-term Prediction Error Evolution (9s @ Δt=2ms)',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper left')

# 添加统计信息文本框
textstr = f'Statistics:\n' \
          f'Mean Error: {errors.mean():.6f}\n' \
          f'Max Error: {errors.max():.6f}\n' \
          f'Final Error: {errors[-1]:.6f}\n' \
          f'Prediction Steps: {PRED_STEPS}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(SAVE_ERROR_CURVE, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"🎉 误差曲线已保存至: {SAVE_ERROR_CURVE}")
print("\n" + "="*60)
print("✅ 所有任务完成！")
print("="*60)
print(f"📂 输出文件:")
print(f"   1. 预测动图: {SAVE_GIF}")
print(f"   2. 误差曲线: {SAVE_ERROR_CURVE}")
print("="*60)
