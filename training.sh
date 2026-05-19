#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/train.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/train.pid}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"

CMD=(
  torchrun
  --nproc_per_node="$NPROC_PER_NODE"
  train.py
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
)

usage() {
  cat <<'EOF'
Usage:
  ./training.sh             Start training in background with a new session
  ./training.sh --foreground
                            Run training in the current terminal

Environment overrides:
  NPROC_PER_NODE=2
  DEVICE=cuda
  BATCH_SIZE=4
  LOG_FILE=./train.log
  PID_FILE=./train.pid
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Training is already running (PID: $existing_pid)."
    echo "Log: $LOG_FILE"
    exit 1
  fi
fi

cd "$ROOT_DIR"

if [[ "${1:-}" == "--foreground" ]]; then
  exec "${CMD[@]}"
fi

echo "Starting training in background..."
echo "Log: $LOG_FILE"

(
  exec setsid "${CMD[@]}" >>"$LOG_FILE" 2>&1 < /dev/null
) &

pid=$!
echo "$pid" > "$PID_FILE"

echo "PID: $pid"
echo "Follow log with: tail -f $LOG_FILE"
