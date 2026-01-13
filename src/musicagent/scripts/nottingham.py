from __future__ import annotations

from pathlib import Path

from musicagent.config import DataConfig
from musicagent.data.sources import extract_nottingham_songs
from musicagent.scripts.preprocess import process_songs


def main() -> None:
    cfg = DataConfig()
    raw_dir = Path("nottingham-dataset/ABC_cleaned")
    if not raw_dir.exists():
        raise FileNotFoundError(
            "Nottingham ABC_cleaned directory not found. "
            "Place the dataset under 'nottingham-dataset/ABC_cleaned' or "
            "update `raw_dir` in musicagent.scripts.nottingham."
        )

    songs = extract_nottingham_songs(raw_dir)
    cfg.data_processed = Path("realchords_data_nottingham")
    process_songs(cfg, songs)


if __name__ == "__main__":
    main()



