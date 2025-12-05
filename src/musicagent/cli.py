"""Shared CLI for MusicAgent.

Centralized construction of CLI parsers used across
preprocessing, training, and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_train_parser(description: str, default_save_dir: Path) -> argparse.ArgumentParser:
    """Create a training CLI parser with common arguments."""

    parser = argparse.ArgumentParser(description=description)

    # General training options
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=default_save_dir,
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="musicagent",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    # Data hyperparameters
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--storage-len", type=int)
    parser.add_argument("--max-transpose", type=int)

    # Model / training hyperparameters
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument( "--label-smoothing", type=float)
    parser.add_argument("--grad-clip",type=float)
    parser.add_argument("--weight-decay",type=float)
    parser.add_argument("--device", type=str)

    return parser


def build_train_offline_parser() -> argparse.ArgumentParser:
    """CLI parser for offline training."""

    return _build_train_parser("Train Offline Model", Path("checkpoints/offline"))


def build_train_online_parser() -> argparse.ArgumentParser:
    """CLI parser for online training."""

    return _build_train_parser(
        "Train Online Model (MLE Pretraining)",
        Path("checkpoints/online"),
    )


def _build_eval_parser(description: str, default_checkpoint: Path) -> argparse.ArgumentParser:
    """Create an evaluation CLI parser with common arguments."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    return parser


def build_eval_offline_parser() -> argparse.ArgumentParser:
    """CLI parser for offline evaluation."""

    return _build_eval_parser(
        "Evaluate Offline Model",
        Path("checkpoints/offline/best_model.pt"),
    )


def build_eval_online_parser() -> argparse.ArgumentParser:
    """CLI parser for online evaluation."""

    return _build_eval_parser(
        "Evaluate Online Model",
        Path("checkpoints/online/best_model.pt"),
    )


def build_preprocess_parser() -> argparse.ArgumentParser:
    """CLI parser for the preprocessing script."""

    parser = argparse.ArgumentParser(
        description="Preprocess HookTheory data to numpy arrays.",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-unknown-tokens",
        action="store_true",
    )
    parser.add_argument(
        "--unknown-token-warn-limit",
        type=int,
        default=20,
    )
    return parser
