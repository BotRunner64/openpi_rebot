#!/usr/bin/env bash
set -eo pipefail
BASE=/inspire/hdd/global_user/czxs253130598/projects/real_projects/openpi
source "$BASE/activate.sh"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR="$BASE/cache/jax-training"
export WANDB_MODE=offline WANDB_DIR="$BASE/logs/wandb" HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
git diff --quiet
python - "$BASE" <<'CHECK'
import hashlib,json,pathlib,subprocess,sys
import jax
base=pathlib.Path(sys.argv[1])
proof=json.loads((base/'logs/rebot-smoke-passed.json').read_text())
choice=json.loads((base/'logs/rebot-loader-selection.json').read_text())
assert proof['passed'] and proof['checkpoint_resume_verified']
# Only runtime/diagnostic launchers may change relative to the tested model.
subprocess.run(['git','diff','--exit-code',proof['code_commit'],'--','src/openpi','scripts/train.py','assets/pi05_rebot_post_train'],check=True)
assert proof['manifest_sha256']==hashlib.sha256((base/'cache/rebot_post_train_h50/manifest.json').read_bytes()).hexdigest()
assert choice['model_inputs_unchanged'] and choice['selected_workers'] in (16,24,32)
assert jax.default_backend()=='gpu' and jax.device_count()==8,jax.devices()
run=base/'runs/pi05_rebot_post_train/post_train_20261113'
checkpoints=[p for p in run.iterdir() if p.name.isdigit() and (p/'train_state').is_dir() and (p/'params').is_dir()]
assert checkpoints,'Optimization must resume existing training, not initialize from base'
print('Resume checkpoint:',max(int(p.name) for p in checkpoints),flush=True)
print('Training devices:',jax.devices(),flush=True)
print('Selected data workers:',choice['selected_workers'],flush=True)
CHECK
workers=$(python -c "import json; print(json.load(open('$BASE/logs/rebot-loader-selection.json'))['selected_workers'])")
set +e
python -u scripts/train.py pi05_rebot_post_train --exp-name post_train_20261113 --resume --num-workers "$workers" 2>&1 | tee "$BASE/logs/rebot-sft-8gpu-opt.log"
code=${PIPESTATUS[0]}
echo "$code" > "$BASE/logs/rebot-sft-8gpu-opt.exit"
exit "$code"
