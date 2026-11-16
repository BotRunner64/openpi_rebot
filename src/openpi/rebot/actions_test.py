import numpy as np
from scipy.spatial.transform import Rotation

from openpi.rebot.actions import absolute_actions
from openpi.rebot.actions import relative_actions
from openpi.rebot.actions import rotation_from_state
from openpi.rebot.actions import rotation_to_6d


def make_states(seed, shape):
    rng = np.random.default_rng(seed)
    arms = []
    for _ in range(2):
        rotation = Rotation.random(int(np.prod(shape)), random_state=rng).as_matrix().reshape(*shape, 3, 3)
        arms.append(
            np.concatenate(
                (
                    rng.normal(size=(*shape, 3)),
                    rotation.swapaxes(-1, -2).reshape(*shape, 9),
                    rng.uniform(size=(*shape, 1)),
                ),
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1).astype(np.float32)


def test_pose_roundtrip_and_column_order():
    anchor, future = make_states(1, (4,)), make_states(2, (4, 50))
    relative = relative_actions(anchor, future)
    assert relative.shape == (4, 50, 20)
    np.testing.assert_allclose(absolute_actions(anchor, relative), future, atol=2e-6)
    for source, target in ((0, 0), (13, 10)):
        np.testing.assert_array_equal(relative[..., target + 9], future[..., source + 12])


def test_every_target_uses_observation_anchor():
    anchor = make_states(3, ())
    future = np.repeat(anchor[None], 3, axis=0)
    rotation = rotation_from_state(anchor[:13])
    for k in range(3):
        future[k, :3] += rotation @ np.array([k + 1, 0, 0])
    relative = relative_actions(anchor, future)
    np.testing.assert_allclose(relative[:, :3], [[1, 0, 0], [2, 0, 0], [3, 0, 0]], atol=1e-6)
    # In particular it must not become successive [1,0,0] increments.
    np.testing.assert_allclose(relative[:, 3:9], np.tile(rotation_to_6d(np.eye(3)), (3, 1)), atol=1e-6)


def test_common_world_transform_does_not_change_relative_action():
    anchor, future = make_states(4, (2,)), make_states(5, (2, 7))
    original = relative_actions(anchor, future)
    world = Rotation.from_euler("xyz", [0.7, -0.9, 1.2]).as_matrix()
    translation = np.array([1.0, -2.0, 0.4])

    def transform(states):
        states = states.copy()
        for offset in (0, 13):
            states[..., offset : offset + 3] = states[..., offset : offset + 3] @ world.T + translation
            rotations = world @ rotation_from_state(states[..., offset : offset + 13])
            states[..., offset + 3 : offset + 12] = rotations.swapaxes(-1, -2).reshape(*states.shape[:-1], 9)
        return states

    np.testing.assert_allclose(relative_actions(transform(anchor), transform(future)), original, atol=2e-6)
