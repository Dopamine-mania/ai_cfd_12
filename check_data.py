import numpy as np
import os

# 检查 340.npy 是否存在
file_path = './processed_data/340.npy' 

if not os.path.exists(file_path):
    print(f"❌ 找不到文件: {file_path}")
    print("当前目录下的文件有：")
    print(os.listdir('.'))
else:
    print(f"🔄 正在加载 {file_path} ...")
    try:
        data = np.load(file_path)
        print(f"✅ 加载成功！Shape: {data.shape}")
        # 应该是 (6000, 128, 200, 3) 
        
        # 检查数值
        u = data[0, :, :, 0] # 取第一帧的 U 速度
        print(f"🚀 U速度最大值: {u.max():.2f} m/s (应在 20-30 之间)")
        
    except Exception as e:
        print(f"❌ 文件可能损坏: {e}")