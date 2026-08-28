#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source /data1/cheongwoong/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-EasyEdit}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
DATASET="${DATASET:-multiformat_counterfact_1000}"
MODEL_DIR="${MODEL//\//_}"
DIRECTION_DIR="${DIRECTION_DIR:-results/edit_directions/single_format_transfer}"
DIRECTION_BASE="${DIRECTION_DIR}/${MODEL_DIR}/${DATASET}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/xf_direction_alignment/lora_local_factor_directions_${RUN_ID}}"
BACKUP_DIR="${BACKUP_DIR:-results/edit_directions/backups/lora_local_representation_${RUN_ID}}"

mkdir -p "$LOG_DIR"

backup_existing_direction_dir() {
  local method="$1"
  local path="${DIRECTION_BASE}/${method}"
  if [[ -d "$path" ]]; then
    mkdir -p "$BACKUP_DIR"
    echo "[backup] ${path} -> ${BACKUP_DIR}/${method}"
    mv "$path" "${BACKUP_DIR}/${method}"
  fi
}

run_lora_local() {
  local result_alg_name="$1"
  local log_name="$2"
  shift 2

  echo "[start] ${result_alg_name} ${log_name} gpu=${CUDA_VISIBLE_DEVICES}"
  python mf-memit/evaluate_baselines_with_easyedit.py \
    --alg_name LoRA \
    --result_alg_name "$result_alg_name" \
    --model "$MODEL" \
    --ds_name "$DATASET" \
    --lora_layers_override 7 \
    --lora_target_modules_override down_proj \
    --save_edit_directions \
    --direction_dir "$DIRECTION_DIR" \
    "$@" \
    > "${LOG_DIR}/${log_name}.log" 2>&1
  echo "[done] ${result_alg_name} ${log_name}"
}

backup_existing_direction_dir LoRA-local-down
backup_existing_direction_dir LoRA-local-down-XF
backup_existing_direction_dir LoRA-local-down-Para

for fmt in completion triplet ODQA MC TF YN; do
  run_lora_local LoRA-local-down "LoRA-local-down_${fmt}" --edit_formats "$fmt"
done

run_lora_local LoRA-local-down-XF "LoRA-local-down-XF" \
  --edit_formats completion triplet ODQA MC TF YN

run_lora_local LoRA-local-down-Para "LoRA-local-down-Para" \
  --edit_formats completion \
  --edit_paraphrases \
  --edit_paraphrase_subject_count 1

echo "[all-done] LoRA-local factor direction reruns completed. logs=${LOG_DIR}"
