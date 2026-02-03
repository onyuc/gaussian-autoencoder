#!/bin/bash
# ============================================================
# Quick Training Script - Single Command Execution
# ============================================================
# 
# Usage Examples:
#   # Single GPU
#   ./run_train.sh /path/to/point_cloud.ply
#
#   # Multi-GPU (2 GPUs)
#   NUM_GPUS=2 ./run_train.sh /path/to/point_cloud.ply
#
#   # 4 GPUs with bf16 precision
#   NUM_GPUS=4 MIXED_PRECISION=bf16 ./run_train.sh /path/to/point_cloud.ply
#
# ============================================================

set -e

# Configuration (can be overridden by environment variables)
NUM_GPUS=${NUM_GPUS:-1}  # 기본값 1개 (Single GPU)
MIXED_PRECISION=${MIXED_PRECISION:-fp16}  # fp16, bf16, no
CONFIG_FILE=${CONFIG_FILE:-configs/default.yaml}

# PLY file from argument
PLY_FILE=${1:-""}

# Check if PLY file is provided
if [ -z "${PLY_FILE}" ]; then
    echo "Error: PLY file path is required"
    echo "Usage: ./run_train.sh /path/to/point_cloud.ply"
    echo ""
    echo "Environment variables:"
    echo "  NUM_GPUS=2          - Number of GPUs (default: 1)"
    echo "  MIXED_PRECISION=fp16 - Mixed precision type (fp16, bf16, no)"
    echo "  CONFIG_FILE=...      - Config file path"
    exit 1
fi

# Check if PLY file exists
if [ ! -f "${PLY_FILE}" ]; then
    echo "Error: PLY file not found: ${PLY_FILE}"
    exit 1
fi

# Working directory
cd "$(dirname "$0")"

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "Training Configuration"
echo "============================================================"
echo "PLY File: ${PLY_FILE}"
echo "Config: ${CONFIG_FILE}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Mixed Precision: ${MIXED_PRECISION}"
echo "============================================================"

# Training command
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo ""
    echo "Starting Multi-GPU training with ${NUM_GPUS} GPUs..."
    echo ""
    
    accelerate launch \
        --multi_gpu \
        --num_processes=${NUM_GPUS} \
        --mixed_precision=${MIXED_PRECISION} \
        scripts/train.py \
        --config ${CONFIG_FILE} \
        --ply ${PLY_FILE} \
        --use_accelerate
else
    echo ""
    echo "Starting Single GPU training..."
    echo ""
    
    python scripts/train.py \
        --config ${CONFIG_FILE} \
        --ply ${PLY_FILE}
fi

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
