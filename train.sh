#!/bin/bash
# ==============================================================================
# WaveDetect Two-Stage Training Script
#
# Usage:
#   ./train.sh [BASE_DIR] [RUN_NAME]
# 
# Example:
#   ./train.sh . wavedetect_experiment_1
# ==============================================================================

# Uncomment and modify the following lines if you need to activate a specific Conda environment
# source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# 1. Setup Directories and Run Name
BASE_DIR=${1:-"."}                                    # Default to current directory
NAME=${2:-"wavedetect_default_run"}                   # Configurable run name
DATA_DIR="$BASE_DIR/data"                             # Base data directory
CKPT_DIR="$BASE_DIR/checkpoints/wavedetect/$NAME"
CONFIG_DIR="$BASE_DIR/model/config"

# 2. Model and Data Paths
# You can use a local path or a Hugging Face model repository (e.g., "Qwen/Qwen2.5-0.5B")
BASE_MODEL_NAME="path/to/qwen2.5-0.5b-base"           
TRAIN_PATH="$DATA_DIR/train/raid-all_train.jsonl"
VAL_PATH="$DATA_DIR/binary/all/val.jsonl"
SAVE_PATH="$CKPT_DIR"
CNN_CONFIG_PATH="$CONFIG_DIR/resnet18_config.json"

echo "=================================================="
echo "Starting training with base_dir=$BASE_DIR"
echo "Run name: $NAME"
echo "Model: $BASE_MODEL_NAME"
echo "Train data: $TRAIN_PATH"
echo "Validation data: $VAL_PATH"
echo "Save path: $SAVE_PATH"
echo "CNN config: $CNN_CONFIG_PATH"
echo "=================================================="

# 3. Prepare Logging Directory (../log relative to CKPT_DIR)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create checkpoints base directory if it doesn't exist to avoid cd errors
mkdir -p "$BASE_DIR/checkpoints"
LOG_DIR="$(cd "$CKPT_DIR/.." 2>/dev/null || mkdir -p "$BASE_DIR/checkpoints/log" && cd "$BASE_DIR/checkpoints/log"; pwd)"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/${NAME}_${TIMESTAMP}.log"

echo "Logging to: $LOG_PATH"

# 4. Launch Accelerate (run with nohup, redirect stdout+stderr to log, run in background)
nohup accelerate launch \
  --num_processes 4 \
  model/train.py \
  --base_model_name "$BASE_MODEL_NAME" \
  --train_path "$TRAIN_PATH" \
  --val_path "$VAL_PATH" \
  --save_path "$SAVE_PATH" \
  --cnn_config_path "$CNN_CONFIG_PATH" \
  --batch_size 8 \
  --epochs_stage1 3 \
  --epochs_stage2 3 \
  --lr_stage1 1e-3 \
  --lr_stage2 1e-5 \
  --num_scales 16 \
  --wavelet_type morl \
  --contrastive_weight 0.8 \
  --pos_ratio 0.9 \
  --max_seq_len 1024 \
  > "$LOG_PATH" 2>&1 &

PID=$!
echo "Training started (PID: $PID). Logs: $LOG_PATH"
echo "You can check progress with: tail -f $LOG_PATH"