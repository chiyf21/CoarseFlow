#!/bin/bash
# ============================================================
# Stage 1: Swin3D training — 8× A100, DDP
# ============================================================
set -e

cd /home/cyf/wbi/CoarseFlow

# Activate conda env
source /home/cyf/miniconda3/etc/profile.d/conda.sh
conda activate wbi_cuda124

# Log dir
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/swin3d_stage1_${TIMESTAMP}.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Stage 1: Swin3D training — 8× A100 DDP" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/02.1_train_stage1.py \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Stage 1 finished at: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
