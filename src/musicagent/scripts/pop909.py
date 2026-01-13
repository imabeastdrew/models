from __future__ import annotations

from pathlib import Path

from musicagent.config import DataConfig
from musicagent.data.sources import extract_pop909_songs
from musicagent.scripts.preprocess import process_songs


def main() -> None:
    cfg = DataConfig()
    pop_root = Path("POP909-Dataset/POP909")
    if not pop_root.exists():
        raise FileNotFoundError(
            "POP909 root directory not found. "
            "Place the dataset under 'POP909-Dataset/POP909' or "
            "update `pop_root` in musicagent.scripts.pop909."
        )

    songs = extract_pop909_songs(pop_root)
    cfg.data_processed = Path("realchords_data_pop909")
    process_songs(cfg, songs)


if __name__ == "__main__":
    main()



