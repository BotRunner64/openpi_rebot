"""Local LeRobot-v3 Rebot data with UMI chunk-relative next-state targets.

Pose matrices in the source are flattened column-major. Every future pose in a
chunk uses the observed pose as its anchor: inv(T_observed) @ T_future.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

CAMERAS = ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist")
MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def rotation_from_state(state):
    return np.asarray(state)[..., 3:12].reshape(*np.shape(state)[:-1], 3, 3).swapaxes(-1, -2)


def rotation_to_6d(rotation):
    """First two columns, in column order; inverse uses Gram-Schmidt."""
    return np.swapaxes(rotation[..., :, :2], -1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_from_6d(value):
    value = np.asarray(value)
    x, y = value[..., :3], value[..., 3:6]
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-8)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def relative_actions(anchor, future):
    """anchor (...,26), future (...,H,26) -> (...,H,20). Gripper stays absolute."""
    anchor, future = np.asarray(anchor), np.asarray(future)
    result = []
    for offset in (0, 13):
        base, targets = anchor[..., offset : offset + 13], future[..., offset : offset + 13]
        inverse = np.swapaxes(rotation_from_state(base), -1, -2)
        pos = np.einsum("...ij,...hj->...hi", inverse, targets[..., :3] - base[..., None, :3])
        rot = np.einsum("...ij,...hjk->...hik", inverse, rotation_from_state(targets))
        result.extend((pos, rotation_to_6d(rot), targets[..., 12:13]))
    return np.concatenate(result, axis=-1).astype(np.float32)


def absolute_actions(anchor, relative):
    """Undo the chunk transform, returning source-order 26-dimensional targets."""
    anchor, relative = np.asarray(anchor), np.asarray(relative)
    result = []
    for source, target in ((0, 0), (13, 10)):
        base = anchor[..., source : source + 13]
        action = relative[..., target : target + 10]
        rotation = rotation_from_state(base)
        pos = np.einsum("...ij,...hj->...hi", rotation, action[..., :3]) + base[..., None, :3]
        rot = np.einsum("...ij,...hjk->...hik", rotation, rotation_from_6d(action[..., 3:9]))
        result.extend((pos, rot.swapaxes(-1, -2).reshape(*rot.shape[:-2], 9), action[..., 9:10]))
    return np.concatenate(result, axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class RebotOutputs:
    def __call__(self, data):
        # Policy output state has already been unnormalized by openpi.
        return {**data, "actions": absolute_actions(data["state"][..., :26], data["actions"][..., :20])}


class RebotDataset:
    """Read prepared uint8 images and exact, episode-bounded action chunks."""

    def __init__(self, cache_dir, horizon, split="train"):
        self.root = Path(cache_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["horizon"] != horizon:
            raise ValueError("Prepared action horizon differs from model configuration")
        self.split = split
        self._arrays = None
        self._videos = {}
        self.indices = np.load(self.root / f"{split}_indices.npy")
        if not len(self.indices):
            raise ValueError(f"Empty {split} split")

    def _open(self):
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.root / f"{name}.npy", mmap_mode="r")
                for name in ("states", "actions", "anchors", "image_indices", "task_indices")
            }

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None, "_videos": {}}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        self._open()
        i = int(self.indices[index])
        images = {}
        for camera, (video, frame) in zip(MODEL_CAMERAS, self._arrays["image_indices"][i], strict=True):
            video = int(video)
            metadata = self.manifest["videos"][video]
            cached = self.root / metadata["cache"]
            if cached.exists():
                if video not in self._videos:
                    self._videos[video] = np.load(cached, mmap_mode="r")
                images[camera] = np.array(self._videos[video][int(frame)], copy=True)
            else:
                # Use the exact video reader shipped with openpi's pinned LeRobot.
                from lerobot.common.datasets.video_utils import decode_video_frames

                frames = decode_video_frames(
                    Path(self.manifest["source"]) / metadata["source"],
                    [int(frame) / metadata["fps"]],
                    0.5 / metadata["fps"],
                    backend="pyav",
                )
                images[camera] = np.rint(frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return {
            "image": images,
            "image_mask": dict.fromkeys(MODEL_CAMERAS, np.bool_(True)),
            "state": np.array(self._arrays["states"][self._arrays["anchors"][i]], copy=True),
            "actions": np.array(self._arrays["actions"][i], copy=True),
            "prompt": self.manifest["tasks"][str(int(self._arrays["task_indices"][i]))],
        }
