"""State and action representation shared by training and serving.

A state is 26 values, left arm then right arm: xyz, the rotation matrix flattened
column-major (R9), gripper openness. Model targets are relative to the observed pose:
every future pose in a chunk is ``inv(T_observed) @ T_future``, stored per arm as xyz,
the first two rotation columns (6D) and the absolute future gripper openness.
"""

import dataclasses

import numpy as np

STATE_DIM = 26
ACTION_DIM = 20
FPS = 30
# Indices that keep identity normalization: rotations are not range-rescaled.
STATE_ROTATION = np.r_[3:12, 16:25]
ACTION_ROTATION = np.r_[3:9, 13:19]


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
    """Policy output transform: relative model actions -> absolute 26D target states."""

    def __call__(self, data):
        # The state has already been unnormalized by openpi.
        return {
            **data,
            "actions": absolute_actions(data["state"][..., :STATE_DIM], data["actions"][..., :ACTION_DIM]),
        }
