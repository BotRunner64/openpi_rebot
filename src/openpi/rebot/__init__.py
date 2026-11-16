"""Rebot bimanual arm: pi05 post-training and serving.

Everything Rebot-specific lives in this package and in ``scripts/*rebot*``. Datasets,
model weights and logs are git-ignored directories (or symlinks) of the repository.
"""

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"  # data/<task>: LeRobot v3 datasets
MODELS_DIR = REPO_ROOT / "models"  # base weights, tokenizer and training outputs
LOGS_DIR = REPO_ROOT / "logs"
