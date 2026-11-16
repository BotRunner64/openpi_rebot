import io

import numpy as np
from PIL import Image
import pytest

from openpi.rebot import link

T0 = 5_000_000_000


def _jpeg(color, size=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _state(offset):
    state = np.zeros(13)
    state[:3] = offset + np.arange(3)
    state[3:12] = np.eye(3).ravel(order="F")
    state[12] = offset / 10
    return state


def _observation():
    left, right = _state(1.0), _state(2.0)
    stale = [0.0] * 12
    return {
        "timestamp_ns": T0,
        "instruction": "pack the box",
        "state": {
            # The latest row is used when a stream carries history.
            "left_eef": {"timestamp_ns": [T0 - 9, T0], "pose": [stale, left[:12].tolist()]},
            "right_eef": {"timestamp_ns": [T0], "pose": [right[:12].tolist()]},
            "left_gripper": {"timestamp_ns": [T0], "openness": [float(left[12])]},
            "right_gripper": {"timestamp_ns": [T0], "openness": [float(right[12])]},
        },
        "rgb": {
            "top": {"timestamp_ns": [T0], "jpeg": [_jpeg((250, 0, 0))]},
            "left_wrist": {"timestamp_ns": [T0], "jpeg": [_jpeg((0, 250, 0))]},
            "right_wrist": {"timestamp_ns": [T0], "jpeg": [_jpeg((0, 0, 250))]},
        },
    }


def test_streams_become_the_training_observation_layout():
    result = link.model_observation(_observation())
    np.testing.assert_allclose(result["state"], np.concatenate([_state(1.0), _state(2.0)]), rtol=1e-6)
    assert result["state"].dtype == np.float32
    assert result["prompt"] == "pack the box"
    assert list(result["image"]) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert result["image"]["base_0_rgb"].shape == (48, 64, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["base_0_rgb"][24, 32].argmax() == 0
    assert result["image"]["right_wrist_0_rgb"][24, 32].argmax() == 2
    assert all(result["image_mask"].values())


def test_actions_become_future_rows_of_the_state_streams():
    class Hold:
        def infer(self, observation):
            return {"actions": np.repeat(observation["state"][None], 50, axis=0)}

    reply = link.RebotLinkPolicy(Hold())(_observation())
    assert set(reply["state"]) == set(link.STATE_STREAMS)
    times = reply["state"]["right_eef"]["timestamp_ns"]
    assert len(times) == 50
    assert times[0] == T0 + 33_333_333
    assert times[29] == T0 + 1_000_000_000
    assert all(isinstance(value, int) for value in times)
    np.testing.assert_allclose(reply["state"]["right_eef"]["pose"][7], _state(2.0)[:12], rtol=1e-6)
    np.testing.assert_allclose(reply["state"]["left_gripper"]["openness"], [0.1] * 50, rtol=1e-6)


def test_bad_model_output_is_rejected():
    for actions in (np.zeros((50, 20)), np.full((50, 26), np.nan)):
        with pytest.raises(ValueError, match="26"):
            link.action_reply(actions, T0)
