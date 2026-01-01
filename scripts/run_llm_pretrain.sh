#!/usr/bin/env bash
# Quick start for LLM pretrain/finetune example
# Usage: chmod +x scripts/run_llm_pretrain.sh && ./scripts/run_llm_pretrain.sh

set -e

PY=${PYTHON:-python}

DATA_PATH=${DATA_PATH:-"./data/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_pretrain"}
CONFIG=${CONFIG:-"./configs/finetune_llm.yaml"}  # optional, if you wire configs

mkdir -p "$OUTPUT_DIR"

echo "Running LLM pretrain/finetune..."
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

$PY src/llm/pretrain_llm.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG"

echo "Done. Check outputs in $OUTPUT_DIR"
