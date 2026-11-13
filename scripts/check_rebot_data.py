"""CPU integration checks on the actual prepared dataset and policy transforms."""

import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from openpi import transforms
from openpi.training import config, data_loader
from openpi.training.rebot_dataset import RebotDataset, absolute_actions


def main():
    cfg = config.get_config("pi05_rebot_umi")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    train, val = RebotDataset(dc.rebot_cache_dir, 50), RebotDataset(dc.rebot_cache_dir, 50, "val")
    assert not set(train.manifest["train_episodes"]) & set(train.manifest["val_episodes"])
    assert not set(train.indices) & set(val.indices)
    assert len(train) + len(val) == train.manifest["rows"] - 50 * 54
    train._open()
    for i in np.random.default_rng(42).choice(len(train), 32, replace=False):
        j = train.indices[i]
        anchor = train._arrays["anchors"][j]
        expected = train._arrays["states"][anchor + np.arange(1, 51)]
        restored = absolute_actions(train._arrays["states"][anchor], train._arrays["actions"][j])
        np.testing.assert_allclose(restored, expected, atol=2e-6)
    start = time.monotonic()
    raw = train[0]
    decode_time = time.monotonic() - start
    transformed = data_loader.transform_dataset(train, dc)[0]
    assert transformed["actions"].shape == (50, 32)
    assert transformed["state"].shape == (32,)
    assert transformed["tokenized_prompt"].shape == (200,)
    assert all(x.shape == (224, 224, 3) for x in transformed["image"].values())
    assert np.isfinite(transformed["actions"]).all()
    assert np.isfinite(transformed["state"]).all()
    physical = transforms.Unnormalize(dc.norm_stats, use_quantiles=True)(
        {"state": transformed["state"], "actions": transformed["actions"]}
    )
    outputs = transforms.compose(dc.data_transforms.outputs)(physical)
    np.testing.assert_allclose(outputs["actions"], absolute_actions(raw["state"], raw["actions"]), atol=2e-6)
    assert dataclasses.replace(cfg.data, split="val").create(cfg.assets_dirs, cfg.model).rebot_split == "val"
    result = {
        "train_chunks": len(train),
        "val_chunks": len(val),
        "horizon": 50,
        "train_episodes": train.manifest["train_episodes"],
        "val_episodes": train.manifest["val_episodes"],
        "first_raw_decode_seconds": decode_time,
        "all_checks_passed": True,
    }
    path = Path(dc.rebot_cache_dir).parent.parent / "logs/rebot-data-check.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
