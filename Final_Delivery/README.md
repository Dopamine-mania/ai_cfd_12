# 最终交付（Time-Conditioned ResNet, One-shot）

本目录用于对外交付/打包，已将关键产物集中收敛在这里。

## 结果文件（仅 2 个切片：X=340, X=520）

### 26m/s
- `26ms/prediction_26ms_340.gif`
- `26ms/error_curve_26ms_340.png`
- `26ms/prediction_26ms_520.gif`
- `26ms/error_curve_26ms_520.png`

### 18m/s（迁移学习）
- `18ms/prediction_18ms_340.gif`
- `18ms/error_curve_18ms_340.png`
- `18ms/prediction_18ms_520.gif`
- `18ms/error_curve_18ms_520.png`

## 权重（只保留最终权重）
- `weights/time_resnet_26ms_timecond_physv2_final.pth`：26m/s 最终权重（用于生成上面的 26m/s 结果）
- `weights/time_resnet_18ms_finetune_final.pth`：18m/s 迁移学习后的最终权重

## 代码（可复现推理/训练）
- `predict_time_resnet.py`：one-shot 推理并生成 9 秒 GIF + 误差曲线
- `train_time_resnet.py`：one-shot 训练/微调脚本
- `run_18ms_transfer.sh`：18m/s 迁移学习一键流程（从 26m/s 权重微调 + 产出 9 秒结果）

## 快速复现（示例）

从项目根目录运行（推荐）：

```bash
# 26m/s 推理（X=340）
MODEL_PATH=./Final_Delivery/weights/time_resnet_26ms_timecond_physv2_final.pth \
DATA_PATH=./processed_data/26ms/340.npy \
SAVE_GIF=./Final_Delivery/26ms/prediction_26ms_340.gif \
SAVE_ERROR_CURVE=./Final_Delivery/26ms/error_curve_26ms_340.png \
python ./Final_Delivery/predict_time_resnet.py

# 18m/s 推理（X=520）
MODEL_PATH=./Final_Delivery/weights/time_resnet_18ms_finetune_final.pth \
DATA_PATH=./processed_data/18ms/520.npy \
SAVE_GIF=./Final_Delivery/18ms/prediction_18ms_520.gif \
SAVE_ERROR_CURVE=./Final_Delivery/18ms/error_curve_18ms_520.png \
python ./Final_Delivery/predict_time_resnet.py
```
