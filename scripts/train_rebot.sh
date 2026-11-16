#!/usr/bin/env bash
# Train a Rebot config; the same command resumes the experiment after an interruption.
#
#   scripts/train_rebot.sh <experiment> [config] [train.py options]
#   scripts/train_rebot.sh packing_unscrew_1116
#   scripts/train_rebot.sh smoke pi05_rebot_post_train_smoke
set -eo pipefail
experiment=${1:?usage: scripts/train_rebot.sh <experiment> [config] [train.py options]}
config=${2:-pi05_rebot_post_train}
shift $(($# > 1 ? 2 : 1))

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate openpi_rebot
cd "$ROOT"
mkdir -p logs/wandb logs/jax
export OPENPI_DATA_HOME="$ROOT/models"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR="$ROOT/logs/jax"
# Training nodes are offline.
export WANDB_MODE=offline WANDB_DIR="$ROOT/logs/wandb" HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u scripts/train.py "$config" --exp-name "$experiment" --resume "$@" 2>&1 | tee -a "logs/$config-$experiment.log"
exit "${PIPESTATUS[0]}"
