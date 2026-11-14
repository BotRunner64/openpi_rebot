"""Measure the unchanged training input pipeline on CPU, without using GPUs."""

import gc
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
import torch
import tyro

from openpi.training import config, data_loader


def main(workers: tuple[int, ...] = (8, 16, 24, 32), batches: int = 64):
    torch.set_num_threads(1)
    cfg = config.get_config("pi05_rebot_post_train")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.transform_dataset(
        data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model), dc
    )
    root = Path(dc.rebot_cache_dir).parent.parent
    results = []
    expected = None
    for count in workers:
        kwargs = dict(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(42),
            num_workers=count,
            collate_fn=data_loader._collate_fn,
            worker_init_fn=data_loader._worker_init_fn,
        )
        if count:
            kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        loader = torch.utils.data.DataLoader(**kwargs)
        started = time.perf_counter()
        iterator = iter(loader)
        first = next(iterator)
        startup = time.perf_counter() - started
        digest = hashlib.sha256()
        for array in jax.tree.leaves(first):
            digest.update(np.asarray(array).tobytes())
        fingerprint = digest.hexdigest()
        if expected is None:
            expected = fingerprint
        assert fingerprint == expected, "Worker count changed model inputs"
        # Drain one wave before timing two or more waves, avoiding a queue-only result.
        for _ in range(max(2, count) - 1):
            next(iterator)
        times = []
        for _ in range(batches):
            start = time.perf_counter()
            batch = next(iterator)
            times.append(time.perf_counter() - start)
        result = dict(
            workers=count,
            batches=batches,
            startup_seconds=startup,
            mean_batch_seconds=float(np.mean(times)),
            p50_batch_seconds=float(np.median(times)),
            p95_batch_seconds=float(np.percentile(times, 95)),
            samples_per_second=64 / float(np.mean(times)),
            input_sha256=fingerprint,
            model_inputs_unchanged=True,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        (root / "logs/rebot-loader-benchmark.json").write_text(json.dumps(results, indent=2))
        if count:
            iterator._shutdown_workers()
        del batch, first, iterator, loader
        gc.collect()
    print("LOADER BENCHMARK PASSED", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
