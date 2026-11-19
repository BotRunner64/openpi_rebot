#!/usr/bin/env python3
"""Serve the post-trained Rebot checkpoint from the openpi repository."""

import argparse
import logging
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

RTC_CHECKPOINT = REPO_ROOT / "models/pi05_rebot_unscrew_rtc_20000"


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


def create_policy(args):
    from openpi.policies import policy_config
    from openpi.training import config

    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/rebot/norm_stats.json").is_file():
        raise ValueError("Checkpoint must contain params/ and assets/rebot/norm_stats.json.")
    # Normalization statistics are read from the checkpoint, not from prepared training data.
    return policy_config.create_trained_policy(
        config.get_config("pi05_rebot_post_train"), checkpoint, sample_kwargs={"num_steps": args.num_steps})


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--host", default="127.0.0.1", help="Listen address (default: loopback)")
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--checkpoint", type=Path,
                     default=RTC_CHECKPOINT)
    cli.add_argument("--rtc-prefix-steps", type=int, choices=range(7), default=None,
                     help="Committed control steps (default: 3 for the bundled RTC checkpoint, 0 otherwise)")
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
    os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / "models"))
    sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages/openpi-client/src")]
    prefix_steps = args.rtc_prefix_steps
    if prefix_steps is None:
        using_rtc_checkpoint = args.checkpoint.expanduser().resolve() == RTC_CHECKPOINT.resolve()
        prefix_steps = 3 if using_rtc_checkpoint and not args.mock_policy else 0
    if args.mock_policy and prefix_steps:
        cli.error("mock policy does not support RTC prefixes")
    try:
        import rebot_policy_link as link

        if args.mock_policy:
            infer = link.HoldPolicy(args.mock_latency)
            cameras = link.REBOT_RGB
        else:
            import numpy as np

            from openpi.rebot import link as rebot_link

            print("Loading policy and compiling a warmup inference before listening...", flush=True)
            policy = create_policy(args)
            obs = warmup_observation()
            actions = np.asarray(policy.infer(obs)["actions"])
            if actions.shape != (50, 26) or not np.all(np.isfinite(actions)):
                raise ValueError("Policy must return finite absolute EEF states with shape (50, 26).")
            if prefix_steps:
                from openpi.rebot.actions import relative_actions

                # Compile the second JAX input shape before the controller connects.
                relative = relative_actions(obs["state"], actions[:prefix_steps])
                conditioned = policy.infer(obs, action_prefix=relative)["actions"]
                if conditioned.shape != actions.shape or not np.isfinite(conditioned).all():
                    raise ValueError("Invalid RTC warmup output.")
            infer = rebot_link.RebotLinkPolicy(policy, prefix_steps)
            cameras = rebot_link.CAMERAS
        # Extra fields are ignored by the controller but end up in its recorded run description.
        metadata = link.metadata(link.REBOT_STATE, cameras, mock=args.mock_policy, num_steps=args.num_steps,
                                 rtc_prefix_steps=prefix_steps,
                                 checkpoint=None if args.mock_policy else str(args.checkpoint.expanduser().resolve()))
        print(f"Policy ready: ws://{args.host}:{args.port}; rebot-policy-link; cameras "
              f"{metadata['rgb']}; mock={args.mock_policy}; RTC prefix={prefix_steps} steps.", flush=True)
        link.serve(infer, metadata, args.host, args.port)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Rebot policy server failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
