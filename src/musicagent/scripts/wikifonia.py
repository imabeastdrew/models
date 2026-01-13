from __future__ import annotations

from pathlib import Path

from musicagent.config import DataConfig
from musicagent.data.sources import extract_wikifonia_songs
from musicagent.scripts.preprocess import process_songs


def main() -> None:
    cfg = DataConfig()
    xml_root = Path("wikifonia")
    if not xml_root.exists():
        raise FileNotFoundError(
            "Wikifonia directory not found. "
            "Place the MusicXML files under 'wikifonia/' or "
            "update `xml_root` in musicagent.scripts.wikifonia."
        )

    songs = extract_wikifonia_songs(xml_root)
    cfg.data_processed = Path("realchords_data_wikifonia")
    process_songs(cfg, songs)


if __name__ == "__main__":
    main()



