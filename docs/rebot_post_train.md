# Rebot pi05 post-train

Everything lives inside this repository; `data/`, `models/` and `logs/` are git-ignored:

- `data/<task>`: unchanged LeRobot v3 snapshots from `BingqianWu/rebot-manipulation`
  (`data/packing` is the source of the current preparation).
- `data/post_train_h50`: prepared metadata and relative action chunks.
- `models/pi05_base`: base weights; `models/big_vision`: PaliGemma tokenizer.
- `models/<config>/<experiment>`: training checkpoints and metrics.
- `logs/`: launcher logs, offline W&B and the JAX compilation cache.

On the Inspire cluster `data/<task>` and `models` are symlinks into the project storage
(`public/dataset/rebot/<task>` and `public/model`); `data/post_train_h50` stays in the checkout.

The Python environment is the conda environment `openpi_rebot`.

The local adapter reads original videos using the pinned LeRobot PyAV reader;
normal openpi image resizing and augmentation remain active.

For each arm and observation t, target k is inv(T[t]) @ T[t+k+1], k=0..49.
The source action[t] exactly equals state[t+1] inside all episodes; do not shift it twice.
Targets are 3 local position values, the first two rotation columns (6D), and
absolute future gripper openness: 10 dimensions per arm, padded from 20 to 32.
The observed state retains its absolute 26-dimensional source representation.
Source rotations use column-major R9 order. Policy output transforms undo
normalization and the relative transform, returning absolute 26D target states.

Windows use stride one and require 50 real future states within the same episode.
No terminal padding or cross-episode labels are used. Every episode is used for
training; there is no validation split. Position/gripper normalization uses q01/q99
over all episodes, including all future targets; rotations retain identity normalization.

## Checks and preparation

- `python -m pytest src/openpi/training/rebot_dataset_test.py`
- `python scripts/check_rebot_data.py`
- `python scripts/prepare_rebot.py --help` describes one-time preparation.

Images are read on demand by default. Optional image predecoding is disabled;
`cache_images=false` in the preparation manifest records this choice.

## Training

Both stages use local pi05_base weights, JAX full fine-tuning, bfloat16 compute,
no frozen parameters, AdamW, peak LR 2.5e-5, 1000-step warmup, standard cosine schedule,
EMA 0.99 and 50-step action horizon. FSDP groups contain two GPUs.

- Two-GPU smoke: global batch 16, 50 steps, checkpoints at 25 and 50;
  restore the same optimizer/training state and run to step 55.
- Eight-GPU production: one node, global batch 64, 20000 steps; checkpoint every
  1000 steps.
- Keep the newest checkpoint for resume and every 5000-step checkpoint. Offline W&B
  logs and JSONL metrics are on shared storage.

Launch through `scripts/run_rebot.sh smoke|train`.
Production requires a successful smoke marker matching both the code commit and data
manifest hash. Production uses a fresh experiment initialized from pi05_base; it does
not continue from the smoke experiment. The launcher always requests resume, so a
platform restart resumes the same production experiment instead of overwriting it.

## Data loading optimization

Production resumes with `scripts/run_rebot_optimized.sh`. It selects the tested
worker count from `logs/rebot-loader-selection.json`.
Only CPU data-reading parallelism changes; model, action construction, optimizer,
batch size and learning-rate schedule remain identical to the smoke-tested baseline.
The optimized launcher verifies those core source files against that baseline and
requires a complete existing training checkpoint.

A CPU-only loader test on the training node, while the original job was running,
measured 67 / 130 / 168 / 244 samples per second with 8 / 16 / 24 / 32 workers.
All first-batch model-input hashes matched exactly. Each timing uses a warmup
wave and 64 measured batches. These numbers describe the input pipeline, not the
end-to-end training speedup; actual GPU utilization is measured after restarting.
No whole-video predecoding or image cache was enabled.
