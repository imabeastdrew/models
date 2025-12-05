"""Evaluation modules and CLI for MusicAgent."""

import argparse
import sys

from musicagent.eval.metrics import (
    chord_length_entropy,
    chord_lengths,
    decode_tokens,
    note_in_chord_at_beat,
    note_in_chord_counts,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
)
from musicagent.eval.offline import OfflineEvalResult, evaluate_offline
from musicagent.eval.offline import main as offline_main
from musicagent.eval.online import (
    OnlineEvalResult,
    evaluate_online,
)
from musicagent.eval.online import (
    main as online_main,
)

__all__ = [
    # Metrics
    "chord_length_entropy",
    "chord_lengths",
    "decode_tokens",
    "note_in_chord_at_beat",
    "note_in_chord_counts",
    "note_in_chord_ratio",
    "onset_interval_emd",
    "onset_intervals",
    # Evaluation functions
    "evaluate_offline",
    "evaluate_online",
    # Result containers
    "OfflineEvalResult",
    "OnlineEvalResult",
    # CLI
    "main",
]


def main() -> None:
    """Entry point for eval

    Ex:

        musicagent-eval --mode offline --checkpoint checkpoints/offline/best_model.pt
        musicagent-eval --mode online --checkpoint checkpoints/online/best_model.pt
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Evaluation mode: 'online' or 'offline' (default: offline)",
    )
    args, remaining = parser.parse_known_args()

    # Put remaining args back for the sub-command
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "offline":
        offline_main()
    elif args.mode == "online":
        online_main()
    else:
        raise ValueError("Invalid mode: 'eval' must be either 'online' or 'offline'")
