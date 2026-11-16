"""Turn LeRobot v3 task datasets under data/ into the arrays read by RebotDataset.

    python scripts/prepare_rebot.py                      # every dataset under data/
    python scripts/prepare_rebot.py --tasks packing unscrew

Writes data/post_train_h50 (states, relative action chunks, frame indices, prompts and
normalization statistics). The source datasets are only read. Every episode of every
task is training data.
"""

import concurrent.futures
import hashlib
import json
import shutil

import av
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi import rebot
from openpi.rebot import actions as _actions
from openpi.rebot import dataset as _dataset
from openpi.shared import normalize

ARM = ["x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "R02", "R12", "R22", "gripper_openness"]


def video_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count, fps = stream.frames, float(stream.average_rate)
    if count <= 0:
        raise ValueError(f"{path} does not report its frame count")
    return count, fps


def read_task(root, horizon, row_offset, video_offset):
    """Validate one LeRobot dataset and index its training windows."""
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != _actions.FPS:
        raise ValueError(f"{root.name}: expected a {_actions.FPS} FPS LeRobot v3.0 dataset")
    if info["features"]["observation.state"]["names"] != [f"{arm}_{name}" for arm in ("left", "right") for name in ARM]:
        raise ValueError(f"{root.name}: unexpected observation.state layout")
    table = pq.read_table(root / "data")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    labels = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episode_ids, timestamps = np.asarray(table["episode_index"]), np.asarray(table["timestamp"])
    task_ids = np.asarray(table["task_index"])
    assert np.array_equal(np.asarray(table["index"]), np.arange(len(states))), f"{root.name}: unexpected row order"
    assert np.isfinite(states).all() and np.isfinite(labels).all(), f"{root.name}: non-finite values"
    # action[t] is state[t + 1]; chunks are built from states, so it must not be shifted twice.
    same_episode = episode_ids[:-1] == episode_ids[1:]
    assert np.array_equal(labels[:-1][same_episode], states[1:][same_episode]), f"{root.name}: action != next state"
    for offset in (0, 13):
        rotation = _actions.rotation_from_state(states[:, offset : offset + 13])
        assert np.max(abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3))) < 1e-4, f"{root.name}: bad rotation"
        assert np.allclose(np.linalg.det(rotation), 1, atol=1e-4), f"{root.name}: bad rotation"

    paths = sorted(p.relative_to(root).as_posix() for p in (root / "videos").rglob("*.mp4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        video_infos = list(pool.map(video_info, [root / p for p in paths]))
    videos = [
        {"path": f"{root.name}/{path}", "frames": count, "fps": fps}
        for path, (count, fps) in zip(paths, video_infos, strict=True)
    ]
    video_id = {path: i for i, path in enumerate(paths)}

    episodes = sorted(pq.read_table(root / "meta/episodes").to_pylist(), key=lambda e: e["episode_index"])
    anchors, image_indices = [], []
    for ep in episodes:
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        assert end - start == ep["length"] and np.all(episode_ids[start:end] == ep["episode_index"])
        # Stride one; every window needs `horizon` real future states of the same episode.
        chosen = np.arange(start, end - horizon, dtype=np.int64)
        if not len(chosen):
            continue
        frames = []
        for camera in _dataset.CAMERAS:
            path = info["video_path"].format(
                video_key=camera,
                chunk_index=ep[f"videos/{camera}/chunk_index"],
                file_index=ep[f"videos/{camera}/file_index"],
            )
            video = videos[video_id[path]]
            frame = np.rint((timestamps[chosen] + ep[f"videos/{camera}/from_timestamp"]) * video["fps"]).astype(
                np.int64
            )
            assert frame.min() >= 0 and frame.max() < video["frames"], (root.name, camera, ep["episode_index"])
            frames.append(np.stack((np.full_like(frame, video_offset + video_id[path]), frame), axis=-1))
        anchors.append(chosen)
        image_indices.append(np.stack(frames, axis=1))
    anchors = np.concatenate(anchors)

    rows = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    prompts = {row["task_index"]: next(v for k, v in row.items() if k != "task_index") for row in rows}
    return {
        "summary": {
            "task": root.name,
            "info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
            "episodes": len(episodes),
            "rows": len(states),
            "chunks": len(anchors),
        },
        "states": states,
        "anchors": anchors + row_offset,
        "image_indices": np.concatenate(image_indices),
        "prompts": [prompts[i] for i in task_ids[anchors]],
        "videos": videos,
    }


def norm_stats(values, rotations):
    flat = values.reshape(-1, values.shape[-1])
    q01, q99 = np.quantile(flat, [0.01, 0.99], axis=0)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    q01[rotations], q99[rotations], mean[rotations], std[rotations] = -1, 1, 0, 1
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def main(tasks: tuple[str, ...] = (), horizon: int = 50, overwrite: bool = False):
    """
    Args:
        tasks: Dataset directory names under data/. Default: every LeRobot dataset found there.
        horizon: Action chunk length; must match the model configuration.
        overwrite: Replace an existing preparation.
    """
    output = rebot.DATA_DIR / f"post_train_h{horizon}"
    names = sorted(tasks or (p.parent.parent.name for p in rebot.DATA_DIR.glob("*/meta/info.json")))
    if not names:
        raise FileNotFoundError(f"No LeRobot datasets under {rebot.DATA_DIR}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    staging = output.with_name(output.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    parts, rows, videos = [], 0, 0
    for name in names:
        part = read_task(rebot.DATA_DIR / name, horizon, rows, videos)
        rows, videos = rows + len(part["states"]), videos + len(part["videos"])
        parts.append(part)
        print(json.dumps(part["summary"]), flush=True)
    states = np.concatenate([p["states"] for p in parts])
    anchors = np.concatenate([p["anchors"] for p in parts])
    prompts = sorted({prompt for p in parts for prompt in p["prompts"]})
    prompt_indices = np.array([prompts.index(prompt) for p in parts for prompt in p["prompts"]], dtype=np.int32)

    chunks = np.lib.format.open_memmap(
        staging / "actions.npy", mode="w+", dtype=np.float32, shape=(len(anchors), horizon, _actions.ACTION_DIM)
    )
    for start in range(0, len(anchors), 4096):
        a = anchors[start : start + 4096]
        chunks[start : start + len(a)] = _actions.relative_actions(
            states[a], states[a[:, None] + np.arange(1, horizon + 1)]
        )
    chunks.flush()
    np.save(staging / "states.npy", states)
    np.save(staging / "anchors.npy", anchors)
    np.save(staging / "image_indices.npy", np.concatenate([p["image_indices"] for p in parts]))
    np.save(staging / "prompt_indices.npy", prompt_indices)
    # Statistics cover every state and every future target; rotations keep identity normalization.
    normalize.save(
        staging / "rebot",
        {
            "state": norm_stats(states, _actions.STATE_ROTATION),
            "actions": norm_stats(np.asarray(chunks), _actions.ACTION_ROTATION),
        },
    )
    manifest = {
        "version": _dataset.MANIFEST_VERSION,
        "horizon": horizon,
        "fps": _actions.FPS,
        "chunks": len(anchors),
        "sources": [p["summary"] for p in parts],
        "prompts": prompts,
        "videos": [video for p in parts for video in p["videos"]],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    del chunks
    shutil.rmtree(output, ignore_errors=True)
    staging.rename(output)
    print(f"Prepared {len(anchors)} windows from {len(names)} task(s): {output}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
