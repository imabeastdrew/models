"""Dataset-specific melody/chord extractors.

Each module in this package converts a particular raw dataset into a
common in-memory annotation schema that mirrors the HookTheory format:

    {
        "tags": [...],
        "split": "TRAIN" | "VALID" | "TEST",
        "annotations": {
            "num_beats": int,
            "meters": [...],
            "keys": [...],
            "melody": [
                {"onset": float, "offset": float, "pitch_class": int, "octave": int},
                ...
            ],
            "harmony": [
                {
                    "onset": float,
                    "offset": float,
                    "root_pitch_class": int,
                    "root_position_intervals": list[int],
                    "inversion": int,
                },
                ...
            ],
        },
    }

These annotations can then be fed into the frame-based preprocessing
pipeline defined in :mod:`musicagent.scripts.preprocess`.
"""

from .nottingham import extract_nottingham_songs  # noqa: F401
from .wikifonia import extract_wikifonia_songs  # noqa: F401
from .pop909 import extract_pop909_songs  # noqa: F401


