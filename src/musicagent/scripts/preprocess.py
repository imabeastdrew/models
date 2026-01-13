"""HookTheory JSON to arrays (preprocessing entry point).

This module contains the preprocessing pipeline that converts the raw
HookTheory dataset into numpy arrays and vocabularies used by the
offline and online models, and exposes a CLI entry point.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import ijson
import numpy as np
from tqdm import tqdm

from musicagent.cli import build_preprocess_parser
from musicagent.config import DataConfig

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(
        self,
        name: str,
        config: DataConfig,
        *,
        strict_unknown_tokens: bool = False,
        unknown_token_warn_limit: int = 20,
    ):
        """Simple token→ID vocabulary used during preprocessing.

        Args:
            name:
                Human‑readable name for logging.
            config:
                DataConfig containing special token IDs.
            strict_unknown_tokens:
                If True, :meth:`get_id` will raise a ``KeyError`` when asked to
                resolve an unknown token instead of silently mapping it to REST.
            unknown_token_warn_limit:
                Maximum number of warnings to emit for unknown tokens before
                suppressing further messages (when not in strict mode).
        """
        self.name = name
        self.config = config
        self.strict_unknown_tokens = strict_unknown_tokens
        self.unknown_token_warn_limit = max(0, int(unknown_token_warn_limit))
        self._unknown_token_count = 0

        self.token_to_id: dict[str, int] = {
            config.pad_token: config.pad_id,
            config.sos_token: config.sos_id,
            config.eos_token: config.eos_id,
            config.rest_token: config.rest_id,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.next_id = max(self.token_to_id.values()) + 1
        self.counts: Counter[str] = Counter()

    def add(self, token: str) -> None:
        self.counts[token] += 1
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

    def get_id(self, token: str) -> int:
        """Return the ID for ``token``, handling unknowns according to policy.

        By default, unknown tokens are mapped to REST but we also log a warning
        for the first few occurrences to surface potential schema drift. When
        ``strict_unknown_tokens`` is enabled, an unknown token raises instead.
        """
        # Fast path for in‑vocab tokens.
        token_id = self.token_to_id.get(token)
        if token_id is not None:
            return token_id

        # Strict mode: fail fast so mapping bugs don't go unnoticed.
        if self.strict_unknown_tokens:
            raise KeyError(f"Vocabulary '{self.name}': unknown token {token!r}")

        # Non‑strict mode: map to REST but emit a limited number of warnings.
        if self._unknown_token_count < self.unknown_token_warn_limit:
            self._unknown_token_count += 1
            suffix = ""
            if self._unknown_token_count == self.unknown_token_warn_limit:
                suffix = " (further unknown-token warnings will be suppressed)"

            logger.warning(
                "Vocabulary '%s': unknown token %r not in vocab; mapping to REST%s",
                self.name,
                token,
                suffix,
            )

        return self.config.rest_id

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "token_to_id": self.token_to_id,
                    "counts": dict[Any, int](self.counts.most_common()),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved {self.name} vocabulary to {path} (Size: {len(self.token_to_id)})")


class ChordMapper:
    def __init__(self):
        self.root_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    def get_token(self, root_pc: int, intervals: list[int], inversion: int) -> str:
        quality = "-".join(map(str, intervals))
        root = self.root_names[root_pc % 12]
        return f"{root}:{quality}/{inversion}"


def get_transposed_pitch(pitch: int, semitones: int) -> int:
    new_pitch = pitch + semitones
    return max(0, min(127, new_pitch))


def _augment_range(config: DataConfig) -> range:
    lo, hi = config.augment_range
    return range(lo, hi + 1)


def _augment_song(data: dict[str, Any], semitones: int, config: DataConfig) -> dict[str, Any]:
    """Deep-copy a song dict and transpose melody/chord roots by ``semitones``."""
    if semitones == 0:
        return copy.deepcopy(data)

    augmented = copy.deepcopy(data)
    annot = augmented.get("annotations", {})

    # Melody: shift pitch_class/octave while clamping MIDI to [0, 127].
    for note in annot.get("melody", []) or []:
        midi = note.get("pitch_class", 0) + note.get("octave", 0) * 12 + config.center_midi
        midi_shift = max(0, min(127, midi + semitones))
        offset = midi_shift - config.center_midi
        note["pitch_class"] = offset % 12
        note["octave"] = offset // 12

    # Chords: shift root pitch class.
    for chord in annot.get("harmony", []) or []:
        root_pc = chord.get("root_pitch_class", 0)
        chord["root_pitch_class"] = (root_pc + semitones) % 12

    augmented["augment_semitones"] = semitones
    return augmented


def _expand_augmented_songs(songs: list[dict[str, Any]], config: DataConfig) -> list[dict[str, Any]]:
    """Expand TRAIN split songs across the configured augmentation range."""
    expanded: list[dict[str, Any]] = []
    aug_range = _augment_range(config)
    for data in songs:
        split = data.get("split", "TRAIN").lower()
        if split == "train":
            for k in aug_range:
                expanded.append(_augment_song(data, k, config))
        else:
            expanded.append(copy.deepcopy(data))
    return expanded


def _iter_augmented_records(raw_path: Path, config: DataConfig) -> Iterable[dict[str, Any]]:
    """Yield records from raw JSON with train split expanded over augment_range."""
    with raw_path.open("rb") as f:
        for _, data in ijson.kvitems(f, ""):
            split = data.get("split", "TRAIN").lower()
            if split == "train":
                for k in _augment_range(config):
                    yield _augment_song(data, k, config)
            else:
                yield copy.deepcopy(data)


def process_songs(
    config: DataConfig,
    songs: list[dict[str, Any]],
    *,
    strict_unknown_tokens: bool = False,
    unknown_token_warn_limit: int = 20,
) -> None:
    """Preprocess an in-memory collection of annotated songs using a single vocab."""

    if not songs:
        raise ValueError("No songs provided for preprocessing")

    # Augment TRAIN split songs; include k=0 to retain originals.
    songs = _expand_augmented_songs(songs, config)

    config.data_processed.mkdir(exist_ok=True, parents=True)

    vocab = Vocabulary(
        "vocab",
        config,
        strict_unknown_tokens=strict_unknown_tokens,
        unknown_token_warn_limit=unknown_token_warn_limit,
    )
    chord_mapper = ChordMapper()

    # ------------------------------------------------------------------
    # Pass 1: Build Vocabulary
    # ------------------------------------------------------------------
    logger.info("Building vocab from %d songs (train augmented).", len(songs))
    for data in songs:
        annot = data.get("annotations", {})

        if annot.get("melody"):
            for n in annot["melody"]:
                pitch = n["pitch_class"] + (n["octave"] * 12) + config.center_midi
                tp_pitch = get_transposed_pitch(pitch, 0)
                vocab.add(f"pitch_{tp_pitch}_on")
                vocab.add(f"pitch_{tp_pitch}_hold")

        if annot.get("harmony"):
            for c in annot["harmony"]:
                intervals = c.get("root_position_intervals") or []
                if not intervals:
                    continue
                tp_root = (c["root_pitch_class"] + 0) % 12
                token = chord_mapper.get_token(tp_root, intervals, c["inversion"])
                vocab.add(f"{token}_on")
                vocab.add(f"{token}_hold")

    vocab.save(config.vocab)

    # ------------------------------------------------------------------
    # Tokenizing & Numpy
    # ------------------------------------------------------------------
    logger.info("Tokenizing songs to numpy (train augmented).")

    def _buffer_factory() -> dict[str, list[np.ndarray]]:
        return {"src": [], "tgt": []}

    buffers: defaultdict[str, dict[str, list[np.ndarray]]] = defaultdict(_buffer_factory)

    for data in songs:
        split = data.get("split", "TRAIN").lower()
        annot = data.get("annotations", {})
        num_beats = annot.get("num_beats", 0)
        if num_beats <= 0:
            continue

        total_frames = int(num_beats * config.frame_rate)
        transpose = 0

        arr_len = config.storage_len
        m_row = np.full(arr_len, config.rest_id, dtype=np.int32)
        c_row = np.full(arr_len, config.rest_id, dtype=np.int32)
        m_row[0] = config.sos_id
        c_row[0] = config.sos_id

        # Melody
        if annot.get("melody"):
            for n in annot["melody"]:
                start = int(round(n["onset"] * config.frame_rate))
                end = int(round(n["offset"] * config.frame_rate))

                pitch = n["pitch_class"] + (n["octave"] * 12) + config.center_midi
                tp_pitch = get_transposed_pitch(pitch, transpose)
                token_on = f"pitch_{tp_pitch}_on"
                token_hold = f"pitch_{tp_pitch}_hold"

                id_on = vocab.get_id(token_on)
                id_hold = vocab.get_id(token_hold)

                for idx in range(start, min(end, total_frames)):
                    arr_idx = idx + 1
                    if arr_idx >= arr_len - 1:
                        break
                    m_row[arr_idx] = id_on if idx == start else id_hold

        # Harmony
        if annot.get("harmony"):
            for c in annot["harmony"]:
                intervals = c.get("root_position_intervals") or []
                if not intervals:
                    continue

                start = int(round(c["onset"] * config.frame_rate))
                end = int(round(c["offset"] * config.frame_rate))

                tp_root = (c["root_pitch_class"] + transpose) % 12
                base_token = chord_mapper.get_token(tp_root, intervals, c["inversion"])
                token_on = f"{base_token}_on"
                token_hold = f"{base_token}_hold"

                id_on = vocab.get_id(token_on)
                id_hold = vocab.get_id(token_hold)

                for idx in range(start, min(end, total_frames)):
                    arr_idx = idx + 1
                    if arr_idx >= arr_len - 1:
                        break
                    c_row[arr_idx] = id_on if idx == start else id_hold

        eos_idx = min(total_frames + 1, arr_len - 1)
        m_row[eos_idx] = config.eos_id
        c_row[eos_idx] = config.eos_id
        if eos_idx + 1 < arr_len:
            m_row[eos_idx + 1 :] = config.pad_id
            c_row[eos_idx + 1 :] = config.pad_id

        buffers[split]["src"].append(m_row)
        buffers[split]["tgt"].append(c_row)

    for split, data in buffers.items():
        if not data["src"]:
            continue
        src_arr = np.stack(data["src"])
        tgt_arr = np.stack(data["tgt"])
        out_src = config.data_processed / f"{split}_src.npy"
        out_tgt = config.data_processed / f"{split}_tgt.npy"
        np.save(out_src, src_arr)
        np.save(out_tgt, tgt_arr)

    logger.info("Done.")


def process_dataset(
    config: DataConfig,
    *,
    strict_unknown_tokens: bool = False,
    unknown_token_warn_limit: int = 20,
) -> None:
    """Run preprocessing, given a `DataConfig`.

    Args:
        config:
            Data configuration controlling paths and special token IDs.
        strict_unknown_tokens:
            If True, raise an error when an unknown melody/chord token is
            encountered instead of silently mapping it to REST.
        unknown_token_warn_limit:
            Maximum number of warnings to emit for unknown tokens before
            suppressing further messages (only used when not in strict mode).
    """

    if not config.data_raw.exists():
        raise FileNotFoundError(f"{config.data_raw} does not exist")

    config.data_processed.mkdir(exist_ok=True, parents=True)
    logger.info(f"Processing {config.data_raw}...")

    vocab = Vocabulary(
        "vocab",
        config,
        strict_unknown_tokens=strict_unknown_tokens,
        unknown_token_warn_limit=unknown_token_warn_limit,
    )
    chord_mapper = ChordMapper()

    # ------------------------------------------------------------------
    # Pass 1: Build Vocabulary
    # ------------------------------------------------------------------
    logger.info("Building (train split augmented)...")
    for data in tqdm(_iter_augmented_records(config.data_raw, config), desc="Building Vocab"):
        annot = data.get("annotations", {})

        if annot.get("melody"):
            for n in annot["melody"]:
                pitch = n["pitch_class"] + (n["octave"] * 12) + config.center_midi
                tp_pitch = get_transposed_pitch(pitch, 0)
                vocab.add(f"pitch_{tp_pitch}_on")
                vocab.add(f"pitch_{tp_pitch}_hold")

        if annot.get("harmony"):
            for c in annot["harmony"]:
                # Some annotations have empty or missing interval lists.
                # These produce empty chord qualities, which
                # cannot be parsed by the evaluation metrics.
                intervals = c.get("root_position_intervals") or []
                if not intervals:
                    # Treat these as no chord and skip.
                    continue

                tp_root = (c["root_pitch_class"] + 0) % 12
                token = chord_mapper.get_token(tp_root, intervals, c["inversion"])
                vocab.add(f"{token}_on")
                vocab.add(f"{token}_hold")

    # Save vocabulary. From this point on, all sequences on disk
    # are stored directly in this unified ID space.
    vocab.save(config.vocab)

    # ------------------------------------------------------------------
    # Tokenizing & Numpy
    # ------------------------------------------------------------------
    logger.info("Tokenizing..")

    def _buffer_factory() -> dict[str, list[np.ndarray]]:
        return {"src": [], "tgt": []}

    buffers: defaultdict[str, dict[str, list[np.ndarray]]] = defaultdict(_buffer_factory)

    for data in tqdm(_iter_augmented_records(config.data_raw, config), desc="Tokenizing"):
        split = data.get("split", "TRAIN").lower()
        annot = data.get("annotations", {})
        num_beats = annot.get("num_beats", 0)
        if num_beats <= 0:
            continue

        total_frames = int(num_beats * config.frame_rate)

        transpose = 0

        arr_len = config.storage_len
        # Default to silence for all frames; padding is applied after EOS.
        m_row = np.full(arr_len, config.rest_id, dtype=np.int32)
        c_row = np.full(arr_len, config.rest_id, dtype=np.int32)

        m_row[0] = config.sos_id
        c_row[0] = config.sos_id

        # Melody
        if annot.get("melody"):
            for n in annot["melody"]:
                start = int(round(n["onset"] * config.frame_rate))
                end = int(round(n["offset"] * config.frame_rate))

                pitch = n["pitch_class"] + (n["octave"] * 12) + config.center_midi
                tp_pitch = get_transposed_pitch(pitch, transpose)
                token_on = f"pitch_{tp_pitch}_on"
                token_hold = f"pitch_{tp_pitch}_hold"

                # Look up melody token IDs; fall back to REST if missing.
                id_on = vocab.get_id(token_on)
                id_hold = vocab.get_id(token_hold)

                for idx in range(start, min(end, total_frames)):
                    arr_idx = idx + 1
                    if arr_idx >= arr_len - 1:
                        break
                    m_row[arr_idx] = id_on if idx == start else id_hold

        # Harmony
        if annot.get("harmony"):
            for c in annot["harmony"]:
                # As in the vocab pass, skip chords with empty / missing interval
                # lists so we don't create unparseable chord qualities.
                intervals = c.get("root_position_intervals") or []
                if not intervals:
                    continue

                start = int(round(c["onset"] * config.frame_rate))
                end = int(round(c["offset"] * config.frame_rate))

                tp_root = (c["root_pitch_class"] + transpose) % 12
                base_token = chord_mapper.get_token(tp_root, intervals, c["inversion"])
                token_on = f"{base_token}_on"
                token_hold = f"{base_token}_hold"

                # Look up chord token IDs; fall back to REST if missing.
                id_on = vocab.get_id(token_on)
                id_hold = vocab.get_id(token_hold)

                for idx in range(start, min(end, total_frames)):
                    arr_idx = idx + 1
                    if arr_idx >= arr_len - 1:
                        break
                    c_row[arr_idx] = id_on if idx == start else id_hold

        # EOS
        eos_idx = min(total_frames + 1, arr_len - 1)
        m_row[eos_idx] = config.eos_id
        c_row[eos_idx] = config.eos_id
        # Pad remaining frames.
        if eos_idx + 1 < arr_len:
            m_row[eos_idx + 1 :] = config.pad_id
            c_row[eos_idx + 1 :] = config.pad_id

        buffers[split]["src"].append(m_row)
        buffers[split]["tgt"].append(c_row)

    # Save buffers
    for split, data in buffers.items():
        if not data["src"]:
            continue

        src_arr = np.stack(data["src"])
        tgt_arr = np.stack(data["tgt"])

        out_src = config.data_processed / f"{split}_src.npy"
        out_tgt = config.data_processed / f"{split}_tgt.npy"

        np.save(out_src, src_arr)
        np.save(out_tgt, tgt_arr)

    logger.info("Done.")


def main() -> None:
    """CLI."""

    parser = build_preprocess_parser()
    args = parser.parse_args()

    cfg = DataConfig()
    if args.input:
        cfg.data_raw = args.input
    if args.output:
        cfg.data_processed = args.output

    process_dataset(
        cfg,
        strict_unknown_tokens=args.strict_unknown_tokens,
        unknown_token_warn_limit=args.unknown_token_warn_limit,
    )


if __name__ == "__main__":
    main()
