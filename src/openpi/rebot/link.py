"""Adapt the policy-agnostic rebot-policy-link messages to the Rebot pi05 model.

The controller sends recording-shaped streams (state/<side>_eef, state/<side>_gripper,
rgb/<camera> as JPEG) and expects future rows of the state streams back. Everything
model-specific lives here: camera names, image decoding, the 26-value state layout and
the action frequency. This is the online counterpart of the offline LeRobot export.
"""

import io

import numpy as np

from openpi.rebot.actions import FPS
from openpi.rebot.actions import STATE_DIM
from openpi.rebot.actions import relative_actions
from openpi.rebot.dataset import CAMERAS as _DATASET_CAMERAS

SIDES = ("left", "right")
STATE_STREAMS = ("left_eef", "right_eef", "left_gripper", "right_gripper")
# Recorded camera name -> model image key.
CAMERAS = {key.removeprefix("observation.images."): image for key, image in _DATASET_CAMERAS.items()}


def model_observation(observation):
    """Latest row of every stream -> the observation dictionary the trained policy consumes."""
    from PIL import Image

    state = []
    for side in SIDES:
        state.extend(observation["state"][f"{side}_eef"]["pose"][-1])
        state.append(observation["state"][f"{side}_gripper"]["openness"][-1])
    images = {}
    for name, key in CAMERAS.items():
        with Image.open(io.BytesIO(observation["rgb"][name]["jpeg"][-1])) as image:
            # Native resolution; the model transforms resize with padding as in training.
            images[key] = np.asarray(image.convert("RGB"))
    return {
        "state": np.asarray(state, dtype=np.float32),
        "image": images,
        "image_mask": dict.fromkeys(images, np.bool_(True)),
        "prompt": observation["instruction"],
    }


def action_reply(actions, timestamp_ns):
    """(H, 26) absolute future states -> state streams; row k is timestamp_ns + (k + 1) / 30 s."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != STATE_DIM or not np.all(np.isfinite(actions)):
        raise ValueError("Policy must return finite absolute EEF states with shape (H, 26).")
    times = [timestamp_ns + (k + 1) * 1_000_000_000 // FPS for k in range(len(actions))]
    streams = {}
    for index, side in enumerate(SIDES):
        rows = actions[:, index * 13 : index * 13 + 13]
        streams[f"{side}_eef"] = {"timestamp_ns": times, "pose": rows[:, :12].tolist()}
        streams[f"{side}_gripper"] = {"timestamp_ns": times, "openness": rows[:, 12].tolist()}
    return {"state": streams}


class RebotLinkPolicy:
    """Callable for rebot_policy_link.serve wrapping an openpi policy."""

    def __init__(self, policy, prefix_steps=0):
        self.policy = policy
        self.prefix_steps = prefix_steps

    def __call__(self, observation):
        inputs = model_observation(observation)
        kwargs = {}
        if "action_prefix" in observation:
            import rebot_policy_link as wire

            streams = wire.validate_actions(observation["action_prefix"])["state"]
            expected = wire.future_timestamps(observation["timestamp_ns"], self.prefix_steps, FPS)
            if (not expected or set(streams) != set(STATE_STREAMS)
                    or any(table["timestamp_ns"] != expected for table in streams.values())):
                raise ValueError("RTC prefix must match the declared length, streams and observation time grid")
            columns = []
            for side in SIDES:
                columns.append(np.asarray(streams[f"{side}_eef"]["pose"]))
                columns.append(np.asarray(streams[f"{side}_gripper"]["openness"])[:, None])
            targets = np.concatenate(columns, axis=1)
            kwargs["action_prefix"] = relative_actions(inputs["state"], targets)
        result = self.policy.infer(inputs, **kwargs)
        return action_reply(result["actions"], observation["timestamp_ns"])
