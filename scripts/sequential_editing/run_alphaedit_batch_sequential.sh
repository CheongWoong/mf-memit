#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${EASYEDIT_PYTHON:-python}"
LOG_DIR="$ROOT/logs/alphaedit_batch_sequential"
K1_GPU_ID="${K1_GPU:-1}"
K4_GPU_ID="${K4_GPU:-2}"
K6_GPU_ID="${K6_GPU:-3}"

if [[ "$K1_GPU_ID" == "$K4_GPU_ID" || "$K1_GPU_ID" == "$K6_GPU_ID" || "$K4_GPU_ID" == "$K6_GPU_ID" ]]; then
  echo "K1_GPU, K4_GPU, and K6_GPU must identify distinct GPUs." >&2
  exit 2
fi

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

run_one() {
  local gpu="$1"
  local label="$2"
  shift 2
  local formats=("$@")
  local log="$LOG_DIR/${label}.gpu${gpu}.log"

  echo "[$(date '+%F %T')] START ${label} gpu=${gpu} formats=${formats[*]}" \
    | tee -a "$LOG_DIR/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    -m mf-memit.evaluate_alphaedit_batch_sequential \
    --case_limit 1000 \
    --batch_size 100 \
    --fact_chunk_size 2 \
    --edit_formats "${formats[@]}" \
    --run_name "$label" \
    --resume \
    > "$log" 2>&1
  code=$?
  echo "$code" > "$LOG_DIR/${label}.status"
  echo "[$(date '+%F %T')] END ${label} gpu=${gpu} exit=${code}" \
    | tee -a "$LOG_DIR/scheduler.log"
  return "$code"
}

: > "$LOG_DIR/scheduler.log"
echo "$$" > "$LOG_DIR/scheduler.pid"

run_one "$K1_GPU_ID" alphaedit_k1_batch100_seq1000 completion &
pid_k1=$!
echo "$pid_k1" > "$LOG_DIR/worker_gpu${K1_GPU_ID}.pid"
run_one "$K4_GPU_ID" alphaedit_k4_batch100_seq1000 completion TF MC triplet &
pid_k4=$!
echo "$pid_k4" > "$LOG_DIR/worker_gpu${K4_GPU_ID}.pid"
run_one "$K6_GPU_ID" alphaedit_k6_batch100_seq1000 completion triplet ODQA MC TF YN &
pid_k6=$!
echo "$pid_k6" > "$LOG_DIR/worker_gpu${K6_GPU_ID}.pid"

exit_code=0
for pid in "$pid_k1" "$pid_k4" "$pid_k6"; do
  wait "$pid" || exit_code=1
done
exit "$exit_code"
