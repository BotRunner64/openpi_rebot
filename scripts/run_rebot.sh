#!/usr/bin/env bash
set -eo pipefail
BASE=/inspire/hdd/global_user/czxs253130598/projects/real_projects/openpi
source "$BASE/activate.sh"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR="$BASE/cache/jax-training"
export WANDB_MODE=offline WANDB_DIR="$BASE/logs/wandb" HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"
stage=${1:?Expected smoke or train}
python - "$stage" <<'CHECK'
import jax,sys
expected=2 if sys.argv[1]=='smoke' else 8
assert jax.default_backend()=='gpu' and jax.device_count()==expected,(expected,jax.devices())
print('Training devices:',jax.devices(),flush=True)
CHECK
run_stage() {
    label=$1
    shift
    set +e
    python -u scripts/train.py "$@" 2>&1 | tee "$BASE/logs/$label.log"
    code=${PIPESTATUS[0]}
    set -e
    echo "$code" > "$BASE/logs/$label.exit"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
}
if [ "$stage" = smoke ]; then
    run_stage rebot-smoke pi05_rebot_umi_smoke --exp-name smoke_20261113 --resume
    run_stage rebot-smoke-resume pi05_rebot_umi_smoke --exp-name smoke_20261113 --resume --num-train-steps 55
    python - "$BASE" <<'PY'
import json,pathlib,sys,subprocess,hashlib,numpy as np
base=pathlib.Path(sys.argv[1]);run=base/'runs/pi05_rebot_umi_smoke/smoke_20261113'
metrics=[json.loads(line) for line in (run/'training_metrics.jsonl').read_text().splitlines()]
validation=[json.loads(line) for line in (run/'validation_metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['step']==55, metrics[-1]
assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) and r['grad_norm']>0 for r in metrics)
assert all(np.isfinite(r['validation_loss']) for r in validation)
assert (run/'50').is_dir() and (run/'55').is_dir()
result={'passed':True,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'manifest_sha256':hashlib.sha256((base/'cache/rebot_umi_h50/manifest.json').read_bytes()).hexdigest(),'completed_step':55,'checkpoint_resume_verified':True,
        'first_loss':metrics[0]['loss'],'last_loss':metrics[-1]['loss'],
        'last_validation':validation[-1], 'last_seconds_per_step':metrics[-1]['seconds_per_step']}
(base/'logs/rebot-smoke-passed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
PY
elif [ "$stage" = train ]; then
    test -f "$BASE/logs/rebot-smoke-passed.json"
    git diff --quiet
    python - "$BASE" <<'CHECK'
import json,pathlib,subprocess,hashlib,sys
base=pathlib.Path(sys.argv[1]);proof=json.loads((base/'logs/rebot-smoke-passed.json').read_text())
assert proof['passed'] and proof['code_commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert proof['manifest_sha256']==hashlib.sha256((base/'cache/rebot_umi_h50/manifest.json').read_bytes()).hexdigest()
CHECK
    run_stage rebot-sft-8gpu pi05_rebot_umi --exp-name sft_20261113_umi --resume
else
    echo 'Unknown stage' >&2
    exit 2
fi
