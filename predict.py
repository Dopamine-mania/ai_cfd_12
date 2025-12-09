import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train import CFDPredictor  # 从训练代码导入模型结构
import os

# ================= 配置 =================
MODEL_PATH = './checkpoints/resnet_final.pth' # 也可以改成 epoch_100.pth
DATA_PATH = './processed_data/340.npy' # 拿一个切片来测试
SAVE_GIF = './results/prediction_340.gif'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
data = np.load(DATA_PATH) # (6000, 128, 200, 3)
# 取前 200 帧做个演示
test_seq = data[:1000] 

# 3. 滚动预测 (Rolling Prediction)
# 也就是：用第1帧预测第2帧，用预测出的第2帧预测第3帧... (这是最难的，看AI会不会崩)
print("🔮 开始滚动预测 (自回归测试)...")
preds = []
current_frame = test_seq[0] # 初始条件 (H, W, 3)

# 归一化
current_tensor = torch.from_numpy(current_frame / 30.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

preds.append(current_frame) # 存入第0帧

with torch.no_grad():
    for t in range(999):
         # 1. 预测
        next_tensor = model(current_tensor)
        
        # 2. 【核心修改】强制物理约束 (Clamp)
        # 我们知道归一化后的速度不可能超过 1.1 (33m/s) 也不可能低于 -0.5 (-15m/s)
        # 强行把数值按在这个范围内，防止它飞到天上去
        next_tensor = torch.clamp(next_tensor, min=-0.5, max=1.2)
        
        # 3. 存结果
        pred_np = next_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 30.0
        preds.append(pred_np)
        
        # 4. 更新输入
        current_tensor = next_tensor
        
        

        if t % 100 == 0:
            print(f"Step {t}: Max value = {next_tensor.max().item()}")

preds = np.array(preds) # (200, 128, 200, 3)

# 4. 画图制作 GIF
print("🎨 正在渲染 GIF (左: 真实, 右: 预测)...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

def update(frame_idx):
    for ax in axes: ax.clear()
    
    # 取 U 速度 (X方向) 展示
    real_u = test_seq[frame_idx, :, :, 0]
    pred_u = preds[frame_idx, :, :, 0]
    
    # 统一色标范围 (用真实数据的最大最小值)
    vmin, vmax = test_seq[:,:,:,0].min(), test_seq[:,:,:,0].max()
    
    axes[0].imshow(real_u, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (t={frame_idx})")
    axes[0].axis('off')
    
    axes[1].imshow(pred_u, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"AI Prediction (t={frame_idx})")
    axes[1].axis('off')

ani = animation.FuncAnimation(fig, update, frames=len(preds), interval=50)
ani.save(SAVE_GIF, writer='pillow', fps=20)

print(f"🎉 动图已保存至: {SAVE_GIF}")
print("快下载下来发给客户看！")