#!/usr/bin/env bash
# Quick start for vision/multimodal eval example
# Usage: chmod +x scripts/run_vision_eval.sh && ./scripts/run_vision_eval.sh

set -e

PY=${PYTHON:-python}

DATA_PATH=${DATA_PATH:-"./data/vision_samples"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/vision_eval"}
MODEL_NAME=${MODEL_NAME:-"qwen3-vl-8b"}  # adjust to your checkpoint

mkdir -p "$OUTPUT_DIR"

echo "Running vision/multimodal evaluation..."
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODEL_NAME=$MODEL_NAME"

$PY src/vision/Qwen3_vl_8b_vision.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME"

echo "Done. Check outputs in $OUTPUT_DIR"
