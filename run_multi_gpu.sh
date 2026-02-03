#!/bin/bash
#SBATCH --job-name=gmae_multi
#SBATCH --partition=suma_rtx4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2  # GPU 개수 (2-4개)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Multi-GPU Training Script for Gaussian Merging AutoEncoder
# ============================================================

# Configuration
NUM_GPUS=${NUM_GPUS:-2}  # 기본 2개, 환경변수로 변경 가능
MIXED_PRECISION=${MIXED_PRECISION:-fp16}  # fp16, bf16, no
CONFIG_FILE=${CONFIG_FILE:-configs/default.yaml}
PLY_FILE=${PLY_FILE:-""}

# Environment Setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base  # 또는 gaussian_autoencoder

# Working Directory
cd /home/rchkl2380/Workspace/gaussian-autoencoder

# Create logs directory if not exists
mkdir -p logs

# Set CUDA devices (uncomment to specify specific GPUs)
# export CUDA_VISIBLE_DEVICES=0,1  # GPU 0, 1 사용
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU 0, 1, 2, 3 사용

# Print info
echo "============================================================"
echo "Multi-GPU Training Configuration"
echo "============================================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Mixed Precision: ${MIXED_PRECISION}"
echo "Config File: ${CONFIG_FILE}"
echo "PLY File: ${PLY_FILE}"
echo "============================================================"

# ============================================================
# Option 1: Accelerate 사용 (권장) - 가장 쉽고 안정적
# ============================================================
if [ -n "${PLY_FILE}" ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes=${NUM_GPUS} \
        --mixed_precision=${MIXED_PRECISION} \
        --gradient_accumulation_steps=1 \
        scripts/train.py \
        --config ${CONFIG_FILE} \
        --ply ${PLY_FILE} \
        --use_accelerate
else
    echo "ERROR: PLY_FILE is required. Set PLY_FILE environment variable."
    echo "Usage: PLY_FILE=/path/to/point_cloud.ply ./run_multi_gpu.sh"
    exit 1
fi

# ============================================================
# Option 2: torchrun 사용 (PyTorch native)
# ============================================================
# torchrun --standalone --nproc_per_node=${NUM_GPUS} \
#     scripts/train.py \
#     --config ${CONFIG_FILE} \
#     --ply ${PLY_FILE} \
#     --use_accelerate

# ============================================================
# Option 3: accelerate config 파일 사용
# ============================================================
# accelerate launch --config_file accelerate_config.yaml \
#     scripts/train.py \
#     --config ${CONFIG_FILE} \
#     --ply ${PLY_FILE}

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
