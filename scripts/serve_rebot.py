#!/usr/bin/env python3
"""Serve the post-trained Rebot checkpoint from the openpi repository."""

import argparse
import dataclasses
import logging
import math
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT.parent


def warmup_observation():
    import numpy as np

    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    state = np.zeros(26, np.float32)
    for offset in (0, 13):
        state[offset + 3:offset + 12] = np.eye(3).ravel(order="F")
        state[offset + 12] = .5
    return {"state": state,
            "image": {key: np.zeros((224, 224, 3), np.uint8) for key in keys},
            "image_mask": {key: np.bool_(True) for key in keys}, "prompt": "warmup"}


class HoldPolicy:
    """Hold the received state for transport-only simulation tests."""

    def __init__(self, latency):
        self.latency = latency

    def infer(self, observation):
        import numpy as np

        time.sleep(self.latency)
        return {"actions": np.repeat(np.asarray(observation["state"])[None], 50, axis=0)}


def create_policy(args):
    if args.mock_policy:
        return HoldPolicy(args.mock_latency)
    from openpi.policies import policy_config
    from openpi.training import config

    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/rebot/norm_stats.json").is_file():
        raise ValueError("Checkpoint must contain params/ and assets/rebot/norm_stats.json.")
    cfg = config.get_config("pi05_rebot_umi")
    cfg = dataclasses.replace(
        cfg, assets_base_dir=str(REPO_ROOT / "assets"),
        data=dataclasses.replace(cfg.data, assets=dataclasses.replace(
            cfg.data.assets, assets_dir=str(REPO_ROOT / "assets/pi05_rebot_umi"))),
    )
    return policy_config.create_trained_policy(
        cfg, checkpoint, sample_kwargs={"num_steps": args.num_steps})


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--host", default="127.0.0.1", help="Listen address (default: loopback)")
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--checkpoint", type=Path,
                     default=DEPLOY_ROOT / "checkpoints/pi05_rebot_umi_20000")
    cli.add_argument("--num-steps", type=int, default=10, help="Flow-matching sampling steps")
    cli.add_argument("--mock-policy", action="store_true", help="Serve a hold policy for simulation tests")
    cli.add_argument("--mock-latency", type=float, default=.08)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (not 1 <= args.port <= 65535 or args.num_steps <= 0
            or not math.isfinite(args.mock_latency) or args.mock_latency < 0):
        cli.error("invalid port, sampling steps, or mock latency")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(DEPLOY_ROOT / "cache/openpi"))
    sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages/openpi-client/src")]
    try:
        import numpy as np
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        print("Loading policy and compiling a warmup inference before listening...", flush=True)
        policy = create_policy(args)
        actions = np.asarray(policy.infer(warmup_observation())["actions"])
        if actions.shape != (50, 26) or not np.all(np.isfinite(actions)):
            raise ValueError("Policy must return finite absolute EEF states with shape (50, 26).")
        metadata = {"action_format": "absolute_eef26", "rotation_order": "column_major",
                    "control_frequency": 30, "action_horizon": 50,
                    "mock_policy": args.mock_policy, "num_steps": args.num_steps}
        server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata)
        print(f"Policy ready: ws://{args.host}:{args.port}; absolute EEF 50x26; "
              f"mock={args.mock_policy}.", flush=True)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Rebot policy server failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
