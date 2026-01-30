# ResNet Legacy Tag

该目录用于冻结/归档原 ResNet 路线（CNN 局部感受野）以便回溯对比。

## 现状结论（项目组确认）
- ResNet（含多步 unroll 与推理端硬约束）在长时滚动预测 `t > ~4s (step>2000)` 后出现不可逆动力学漂移。
- Clamping/Dirichlet BC 等只能抑制数值溢出，无法修正相位误差与长期漂移。
- 决策：切换到具有全局感受野的 FNO（Fourier Neural Operator）。

## 关键权重（用于对照）
- `../checkpoints/resnet_epoch_160.pth`：多步 unroll(=3) 训练后相对“还能用”的对照权重。

## 关键推理物理硬约束（已并入当时的 ResNet 推理脚本）
- 船体/死水区 mask：从 `t=0` 自动估计并锁死为初值。
- 开放边界 Dirichlet：右/上/下边缘每步重置为 `t=0` 背景流。

