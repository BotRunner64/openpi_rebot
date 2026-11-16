"""Check the prepared Rebot data through the real training transforms (CPU only).

python scripts/check_rebot_data.py
"""

import json

import numpy as np

from openpi import transforms
from openpi.rebot.actions import absolute_actions
from openpi.training import config
from openpi.training import data_loader


def main(config_name: str = "pi05_rebot_post_train", samples: int = 32):
    cfg = config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError("Normalization statistics are missing; run scripts/prepare_rebot.py")
    horizon = cfg.model.action_horizon
    dataset = data_loader.create_torch_dataset(data_config, horizon, cfg.model)
    arrays = dataset.arrays()
    assert len(dataset) == len(arrays["anchors"]) == len(arrays["actions"]) == len(arrays["image_indices"])

    # Stored chunks restore the exact future states.
    picks = np.random.default_rng(0).choice(len(dataset), min(samples, len(dataset)), replace=False)
    for i in picks:
        anchor = arrays["anchors"][i]
        future = arrays["states"][anchor + np.arange(1, horizon + 1)]
        np.testing.assert_allclose(absolute_actions(arrays["states"][anchor], arrays["actions"][i]), future, atol=2e-6)

    # One sample per task through decoding, model transforms and back to absolute targets.
    transformed_dataset = data_loader.transform_dataset(dataset, data_config)
    firsts = np.cumsum([0] + [source["chunks"] for source in dataset.manifest["sources"]])[:-1]
    for i in firsts:
        raw, sample = dataset[int(i)], transformed_dataset[int(i)]
        assert sample["actions"].shape == (horizon, cfg.model.action_dim)
        assert sample["state"].shape == (cfg.model.action_dim,)
        assert all(image.shape == (224, 224, 3) for image in sample["image"].values())
        assert np.isfinite(sample["actions"]).all() and np.isfinite(sample["state"]).all()
        physical = transforms.Unnormalize(data_config.norm_stats, use_quantiles=True)(
            {"state": sample["state"], "actions": sample["actions"]}
        )
        outputs = transforms.compose(data_config.data_transforms.outputs)(physical)
        np.testing.assert_allclose(outputs["actions"], absolute_actions(raw["state"], raw["actions"]), atol=2e-6)

    print(
        json.dumps(
            {"windows": len(dataset), "sources": dataset.manifest["sources"], "prompts": dataset.manifest["prompts"]},
            indent=2,
        )
    )
    print("REBOT DATA CHECK PASSED")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
