#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
LAYERS="${LAYERS:-2,3,4,5,6,7}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
PRECISION="${PRECISION:-float32}"

# These are the six MLP layers used by the released Llama-3.2-3B MEMIT setup.
# Override LAYERS only when evaluating a custom hyperparameter configuration.
conda run -n mf-memit python -m rome.layer_stats \
  --model_name "$MODEL_NAME" \
  --layers "$LAYERS" \
  --sample_size "$SAMPLE_SIZE" \
  --precision "$PRECISION"
