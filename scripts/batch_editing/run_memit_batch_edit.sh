#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${MF_MEMIT_PYTHON:-python}"
cd "$ROOT" || exit 1

MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="multiformat_counterfact_1000"
LOG_DIR="logs/memit_batch_edit"
QUEUE_FILE="$LOG_DIR/queue.tsv"
LOCK_FILE="$LOG_DIR/queue.lock"
DONE_FILE="$LOG_DIR/done.tsv"
read -r -a GPU_IDS <<< "${GPU_IDS_STRING:-0 1 2 3}"
mkdir -p "$LOG_DIR"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

write_queue() {
  : > "$QUEUE_FILE"
  printf 'memit_completion_b1_lambda1\tMEMIT\t1\t1\tcompletion\n' >> "$QUEUE_FILE"
  printf 'memit_xf_k4_b1_lambda1\tMEMIT-XF\t1\t1\tcompletion TF MC triplet\n' >> "$QUEUE_FILE"
  printf 'memit_xf_k6_b1_lambda1\tMEMIT-XF\t1\t1\tcompletion triplet ODQA MC TF YN\n' >> "$QUEUE_FILE"
  for lambda_value in 1 15000; do
    for num_edits in 10 100 1000; do
      printf 'memit_completion_b%s_lambda%s\tMEMIT\t%s\t%s\tcompletion\n' \
        "$num_edits" "$lambda_value" "$num_edits" "$lambda_value" >> "$QUEUE_FILE"
      printf 'memit_xf_k4_b%s_lambda%s\tMEMIT-XF\t%s\t%s\tcompletion TF MC triplet\n' \
        "$num_edits" "$lambda_value" "$num_edits" "$lambda_value" >> "$QUEUE_FILE"
      printf 'memit_xf_k6_b%s_lambda%s\tMEMIT-XF\t%s\t%s\tcompletion triplet ODQA MC TF YN\n' \
        "$num_edits" "$lambda_value" "$num_edits" "$lambda_value" >> "$QUEUE_FILE"
    done
  done
}

gpu_mem_used() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' '
}

wait_for_gpu() {
  local gpu="$1"
  local threshold="${GPU_FREE_MEM_THRESHOLD_MB:-1000}"
  while true; do
    local used
    used="$(gpu_mem_used "$gpu")"
    if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$used" -lt "$threshold" ]]; then
      return 0
    fi
    echo "[$(date '+%F %T')] gpu=${gpu} busy (${used} MiB), waiting" >> "$LOG_DIR/scheduler.log"
    sleep 120
  done
}

pop_task() {
  local task
  task="$(
    flock "$LOCK_FILE" bash -c '
      if [[ ! -s "$0" ]]; then
        exit 1
      fi
      head -n 1 "$0"
      tail -n +2 "$0" > "$0.tmp"
      mv "$0.tmp" "$0"
    ' "$QUEUE_FILE"
  )" || return 1
  printf '%s' "$task"
}

run_task() {
  local gpu="$1"
  local label="$2"
  local alg="$3"
  local num_edits="$4"
  local lambda_value="$5"
  local formats="$6"
  local log="$LOG_DIR/${label}.gpu${gpu}.log"
  local status="$LOG_DIR/${label}.status"

  local memit_merge_args=()
  if [[ "$alg" == "MEMIT-XF" ]]; then
    memit_merge_args=(--memit_merge)
  fi
  read -r -a format_args <<< "$formats"

  echo "[$(date '+%F %T')] START ${label} gpu=${gpu} alg=${alg} num_edits=${num_edits} lambda=${lambda_value} formats=${formats}" | tee -a "$LOG_DIR/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 \
    "$PYTHON" mf-memit/evaluate.py \
      --alg_name "$alg" \
      --model "$MODEL" \
      --ds_name "$DATASET" \
      "${memit_merge_args[@]}" \
      --edit_formats "${format_args[@]}" \
      --num_edits "$num_edits" \
      --mom2_update_weight_override "$lambda_value" \
      > "$log" 2>&1
  local code=$?
  echo "$code" > "$status"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "$gpu" "$code" "$num_edits" "$lambda_value" >> "$DONE_FILE"
  echo "[$(date '+%F %T')] END ${label} gpu=${gpu} exit=${code}" | tee -a "$LOG_DIR/scheduler.log"
  return "$code"
}

worker() {
  local gpu="$1"
  while true; do
    wait_for_gpu "$gpu"
    local task
    task="$(pop_task)" || {
      echo "[$(date '+%F %T')] worker gpu=${gpu}: queue empty" >> "$LOG_DIR/scheduler.log"
      return 0
    }

    local label alg num_edits lambda_value formats
    IFS=$'\t' read -r label alg num_edits lambda_value formats <<< "$task"
    run_task "$gpu" "$label" "$alg" "$num_edits" "$lambda_value" "$formats"
  done
}

if [[ "${RESET_QUEUE:-1}" == "1" ]]; then
  write_queue
  : > "$DONE_FILE"
  : > "$LOG_DIR/scheduler.log"
fi

echo "$$" > "$LOG_DIR/scheduler.pid"
echo "[$(date '+%F %T')] MEMIT batch scheduler started" | tee -a "$LOG_DIR/scheduler.log"
echo "[$(date '+%F %T')] queue size: $(wc -l < "$QUEUE_FILE"); GPUs: ${GPU_IDS[*]}" | tee -a "$LOG_DIR/scheduler.log"

for gpu in "${GPU_IDS[@]}"; do
  worker "$gpu" &
  echo $! > "$LOG_DIR/worker_gpu${gpu}.pid"
done

wait
echo "[$(date '+%F %T')] MEMIT batch scheduler completed" | tee -a "$LOG_DIR/scheduler.log"
