#!/usr/bin/env bash
# End-to-end safety-critical layer scan (paper Sec. 2.1 / B.1).
#   1. capture all-layer MLP activations        (get_layer_logits.py)
#   2. train a linear probe per layer           (get_layer_if.py)
#   3. rank layers by SS-Score                  (get_layer_sorted.py)
#
# Usage: bash scan/run_scan.sh ./models/Llama-3.1-8B-Instruct scan/features_llama-8b
set -euo pipefail
CKPT="${1:?Usage: run_scan.sh <model_path> <feature_dir>}"
OUT="${2:-scan/features}"

HERE="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$HERE/get_layer_logits.py" \
    --ckpt "$CKPT" --csv "$HERE/merged_dataset.csv" --out_dir "$OUT"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$HERE/get_layer_if.py" "$OUT" \
    --csv "$HERE/merged_dataset.csv"

python "$HERE/get_layer_sorted.py" "$OUT/res.log"
