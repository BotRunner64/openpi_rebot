"""Adapt the policy-agnostic rebot-policy-link messages to the Rebot pi05 model.

The controller sends recording-shaped streams (state/<side>_eef, state/<side>_gripper,
rgb/<camera> as JPEG) and expects future rows of the state streams back. Everything
model-specific lives here: camera names, image decoding, the 26-value state layout and
the action frequency. This is the online counterpart of the offline LeRobot export.
"""

import io

import numpy as np

SIDES = ("left", "right")
STATE_STREAMS = ("left_eef", "right_eef", "left_gripper", "right_gripper")
# Recorded camera name -> model image key.
CAMERAS = {"top": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}
ACTION_HZ = 30


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
    if actions.ndim != 2 or actions.shape[1] != 26 or not np.all(np.isfinite(actions)):
        raise ValueError("Policy must return finite absolute EEF states with shape (H, 26).")
    times = [timestamp_ns + (k + 1) * 1_000_000_000 // ACTION_HZ for k in range(len(actions))]
    streams = {}
    for index, side in enumerate(SIDES):
        rows = actions[:, index * 13 : index * 13 + 13]
        streams[f"{side}_eef"] = {"timestamp_ns": times, "pose": rows[:, :12].tolist()}
        streams[f"{side}_gripper"] = {"timestamp_ns": times, "openness": rows[:, 12].tolist()}
    return {"state": streams}


class RebotLinkPolicy:
    """Callable for rebot_policy_link.serve wrapping an openpi policy."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, observation):
        result = self.policy.infer(model_observation(observation))
        return action_reply(result["actions"], observation["timestamp_ns"])
