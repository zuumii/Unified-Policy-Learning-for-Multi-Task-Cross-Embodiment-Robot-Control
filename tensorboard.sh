#!/usr/bin/env bash
set -euo pipefail

# 用法：./run_tb.sh [LOGDIR] [PORT] [HOST]
# 例子：./run_tb.sh logs 6006 0.0.0.0
LOGDIR="${1:-logs}"
PORT="${2:-6006}"
HOST="${3:-0.0.0.0}"

# 在非交互 shell 中加载 conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then

  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "✗ Error" >&2
  exit 1
fi

conda activate ma

echo "✓ Open TensorBoard：logdir=$LOGDIR  host=$HOST  port=$PORT"
exec tensorboard --logdir "$LOGDIR" --host "$HOST"