"""Prepare the downloaded Rebot v3 dataset without modifying the source files."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.shared import normalize
from openpi.training.rebot_dataset import CAMERAS, relative_actions


def video_info(source):
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(source),
            ]
        )
    )["streams"][0]
    count = int(info["nb_frames"])
    numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    return count, fps


def decode_video(task):
    source, target = task
    count, fps = video_info(source)
    if target.exists():
        shape = np.load(target, mmap_mode="r").shape
        if shape == (count, 224, 224, 3):
            return count, fps
        raise ValueError(f"Unexpected existing cache shape {target}: {shape}")
    tmp = target.with_suffix(".partial.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(count, 224, 224, 3))
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2:black",
        "-filter_threads",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        start = 0
        while block := process.stdout.read(224 * 224 * 3 * 32):
            frames = np.frombuffer(block, dtype=np.uint8).reshape(-1, 224, 224, 3)
            cache[start : start + len(frames)] = frames
            start += len(frames)
        if process.wait() != 0 or start != count:
            raise RuntimeError(f"Decode count mismatch {source}: {start} != {count}")
    cache.flush()
    del cache
    tmp.replace(target)
    print(f"Decoded {source.name} -> {target.name}: {count} frames", flush=True)
    return count, fps


def stats(values, rotations):
    flat = values.reshape(-1, values.shape[-1])
    q01, q99 = np.quantile(flat, [0.01, 0.99], axis=0)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    # UMI uses the continuous rotation representation without range rescaling.
    q01[rotations], q99[rotations], mean[rotations], std[rotations] = -1, 1, 0, 1
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def main(
    source: str,
    output: str,
    assets: str,
    horizon: int = 50,
    workers: int = 8,
    seed: int = 42,
    cache_images: bool = False,
):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared manifest already exists; use a new output path for a different preparation")
    info = json.loads((source / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == 30
    expected = ["x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "R02", "R12", "R22", "gripper_openness"]
    assert info["features"]["observation.state"]["names"] == [
        f"{arm}_{name}" for arm in ("left", "right") for name in expected
    ]
    table = pq.read_table(source / "data")
    episodes = pq.read_table(source / "meta/episodes").to_pylist()
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    labels = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episode_ids = np.asarray(table["episode_index"])
    timestamps = np.asarray(table["timestamp"])
    task_ids = np.asarray(table["task_index"])
    rows = np.asarray(table["index"])
    assert np.array_equal(rows, np.arange(len(states))), "Unexpected row ordering"
    assert np.isfinite(states).all() and np.isfinite(labels).all()
    valid = episode_ids[:-1] == episode_ids[1:]
    assert np.array_equal(labels[:-1][valid], states[1:][valid]), "Action is not exact next-state"
    for offset in (0, 13):
        rotation = states[:, offset + 3 : offset + 12].reshape(-1, 3, 3).swapaxes(-1, -2)
        assert np.max(abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3))) < 1e-4
        assert np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(episode_ids)))
    val_ids = np.sort(rng.permutation(ids)[: max(1, int(np.ceil(len(ids) * 0.1)))])
    train_ids = np.setdiff1d(ids, val_ids)
    video_paths = sorted(str(p.relative_to(source)) for p in (source / "videos").rglob("*.mp4"))
    videos = [{"source": p, "cache": f"videos/{i:03d}.npy"} for i, p in enumerate(video_paths)]
    video_id = {p: i for i, p in enumerate(video_paths)}
    jobs = [(source / v["source"], output / v["cache"]) for v in videos]
    print(f"Inspecting {len(jobs)} videos; image caching={cache_images}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(decode_video, jobs) if cache_images else pool.map(video_info, [j[0] for j in jobs])
        for video, (count, fps) in zip(videos, results, strict=True):
            video.update(frames=count, fps=fps)
    anchors, image_indices = [], []
    for ep in sorted(episodes, key=lambda e: e["episode_index"]):
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        assert end - start == ep["length"] and np.all(episode_ids[start:end] == ep["episode_index"])
        chosen = np.arange(start, end - horizon, dtype=np.int64)
        if not len(chosen):
            continue
        mapping = []
        for camera in CAMERAS:
            path = info["video_path"].format(
                video_key=camera,
                chunk_index=ep[f"videos/{camera}/chunk_index"],
                file_index=ep[f"videos/{camera}/file_index"],
            )
            vi = video_id[path]
            video_times = timestamps[chosen] + ep[f"videos/{camera}/from_timestamp"]
            frame = np.rint(video_times * videos[vi]["fps"]).astype(np.int64)
            assert frame.min() >= 0 and frame.max() < videos[vi]["frames"], (camera, ep["episode_index"])
            mapping.append(np.stack((np.full_like(frame, vi), frame), axis=-1))
        anchors.append(chosen)
        image_indices.append(np.stack(mapping, axis=1))
    anchors = np.concatenate(anchors)
    image_indices = np.concatenate(image_indices)
    chunks = np.lib.format.open_memmap(
        output / "actions.npy", mode="w+", dtype=np.float32, shape=(len(anchors), horizon, 20)
    )
    for start in range(0, len(anchors), 512):
        a = anchors[start : start + 512]
        future = states[a[:, None] + np.arange(1, horizon + 1)]
        chunks[start : start + len(a)] = relative_actions(states[a], future)
    chunks.flush()
    train_indices = np.flatnonzero(np.isin(episode_ids[anchors], train_ids))
    val_indices = np.flatnonzero(np.isin(episode_ids[anchors], val_ids))
    for name, value in {
        "states": states,
        "anchors": anchors,
        "image_indices": image_indices,
        "task_indices": task_ids[anchors],
        "train_indices": train_indices,
        "val_indices": val_indices,
    }.items():
        np.save(output / f"{name}.npy", value)
    print(f"Computing train-only normalization on {len(train_indices)} chunks", flush=True)
    norm_stats = {
        "state": stats(states[np.isin(episode_ids, train_ids)], np.r_[3:12, 16:25]),
        "actions": stats(np.asarray(chunks[train_indices]), np.r_[3:9, 13:19]),
    }
    normalize.save(Path(assets), norm_stats)
    task_rows = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    tasks = {str(r["task_index"]): next(v for k, v in r.items() if k != "task_index") for r in task_rows}
    manifest = {
        "version": 1,
        "source": str(source),
        "source_info_sha256": hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
        "horizon": horizon,
        "fps": info["fps"],
        "seed": seed,
        "rows": len(states),
        "chunks": len(anchors),
        "train_episodes": train_ids.tolist(),
        "val_episodes": val_ids.tolist(),
        "train_chunks": len(train_indices),
        "val_chunks": len(val_indices),
        "videos": videos,
        "tasks": tasks,
        "action_representation": "inv(T_observed) @ T_future; column rot6d; absolute gripper",
        "state_representation": "absolute xyz + column-major R9 + absolute gripper, left then right",
        "normalization": "train-only q01/q99; rotation identity normalization",
        "assets": str(assets),
        "cache_images": cache_images,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "videos"}, indent=2), flush=True)
    print("REBOT PREPARATION PASSED", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
