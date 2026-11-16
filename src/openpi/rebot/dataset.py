"""Training dataset over the arrays written by ``scripts/prepare_rebot.py``.

The prepared directory sits next to the task datasets (``data/post_train_h50`` beside
``data/<task>``) and holds, for every training window: the observed state, the relative
action chunk, the prompt, and which video frame each camera shows. Images are decoded
from the original videos on demand.
"""

import json
from pathlib import Path

import numpy as np

# LeRobot video key -> model image key, in the order stored in image_indices.
CAMERAS = {
    "observation.images.top": "base_0_rgb",
    "observation.images.left_wrist": "left_wrist_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}
ARRAYS = ("states", "actions", "anchors", "image_indices", "prompt_indices")
MANIFEST_VERSION = 2


class RebotDataset:
    def __init__(self, prepared_dir, horizon):
        self.root = Path(prepared_dir)
        if not (self.root / "manifest.json").is_file():
            raise FileNotFoundError(f"{self.root} is not prepared; run scripts/prepare_rebot.py first")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["version"] != MANIFEST_VERSION:
            raise ValueError("Prepared data comes from another code version; run scripts/prepare_rebot.py again")
        if self.manifest["horizon"] != horizon:
            raise ValueError("Prepared action horizon differs from model configuration")
        self._arrays = None

    def arrays(self):
        # Opened lazily so that every data loader worker maps the files itself.
        if self._arrays is None:
            self._arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
        return self._arrays

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __len__(self):
        return self.manifest["chunks"]

    def __getitem__(self, index):
        # Use the exact video reader shipped with openpi's pinned LeRobot.
        from lerobot.common.datasets.video_utils import decode_video_frames

        arrays, i = self.arrays(), int(index)
        images = {}
        for key, (video, frame) in zip(CAMERAS.values(), arrays["image_indices"][i], strict=True):
            metadata = self.manifest["videos"][int(video)]
            frames = decode_video_frames(
                self.root.parent / metadata["path"], [int(frame) / metadata["fps"]], 0.5 / metadata["fps"], "pyav"
            )
            images[key] = np.rint(frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return {
            "image": images,
            "image_mask": dict.fromkeys(images, np.bool_(True)),
            "state": np.array(arrays["states"][arrays["anchors"][i]], copy=True),
            "actions": np.array(arrays["actions"][i], copy=True),
            "prompt": self.manifest["prompts"][int(arrays["prompt_indices"][i])],
        }
