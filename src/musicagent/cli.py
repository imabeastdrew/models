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
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=default_save_dir,
        help="Directory to store checkpoints and configs.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="musicagent",
        help="Weights & Biases project name.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom wandb run name.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a model checkpoint (.pt) to warm-start from.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (may be slower).",
    )

    # Data hyperparameters
    parser.add_argument("--max-len", type=int, help="Maximum input length.")
    parser.add_argument("--storage-len", type=int, help="Sequence length on disk.")
    parser.add_argument("--max-transpose", type=int, help="Max semitone shift.")

    # Model / training hyperparameters
    parser.add_argument("--d-model", type=int, help="Model dimension.")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, help="Number of layers.")
    parser.add_argument("--dropout", type=float, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps.")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu).")

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
        help="Path to model checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config device if set).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from distribution instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --sample).",
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
    parser.add_argument("--input", type=Path, help="Input JSON path.")
    parser.add_argument("--output", type=Path, help="Output directory.")
    return parser
