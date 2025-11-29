"""Training modules for MusicAgent."""

import argparse
import sys

from musicagent.training.offline import main as offline_main
from musicagent.training.offline import train_offline
from musicagent.training.online import main as online_main
from musicagent.training.online import train_online

__all__ = [
    "train_offline",
    "train_online",
]


def main() -> None:
    """Unified entry point for training (online or offline mode)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Training mode: 'online' or 'offline' (default: offline)",
    )
    args, remaining = parser.parse_known_args()

    # Put remaining args back for the sub-command
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "online":
        online_main()
    else:
        offline_main()

