#!/bin/bash
# CFD 训练环境激活脚本
# 使用方法: source activate_cfd.sh

PROJECT_ROOT="/home/jovyan/teaching_material/Work/December/ai_cfd"
ENV_PATH="${PROJECT_ROOT}/cfd_env"

echo "🚀 激活 CFD 训练环境..."

# 激活虚拟环境
source "${ENV_PATH}/bin/activate"

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

# 显示环境信息
echo "✅ 环境已激活！"
echo "   Python: $(python --version)"
echo "   工作目录: ${PROJECT_ROOT}"
echo "   GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

cd "${PROJECT_ROOT}"
