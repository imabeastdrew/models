"""HookTheory JSON to arrays (preprocessing entry point).

This module contains the preprocessing pipeline that converts the raw
HookTheory dataset into numpy arrays and vocabularies used by the
offline and online models, and exposes a CLI entry point.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
from tqdm import tqdm

from musicagent.cli import build_preprocess_parser
from musicagent.config import DataConfig

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, name: str, config: DataConfig):
        self.name = name
        self.config = config
        self.token_to_id: dict[str, int] = {
            config.pad_token: config.pad_id,
            config.sos_token: config.sos_id,
            config.eos_token: config.eos_id,
            config.rest_token: config.rest_id,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.next_id = max(self.token_to_id.values()) + 1
        self.counts: Counter = Counter()

    def add(self, token: str) -> None:
        self.counts[token] += 1
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.config.rest_id)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {"token_to_id": self.token_to_id, "counts": dict(self.counts.most_common())},
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


def process_dataset(config: DataConfig) -> None:
    """Run the full preprocessing pipeline given a :class:`DataConfig`."""

    if not config.data_raw.exists():
        raise FileNotFoundError(f"{config.data_raw} does not exist")

    config.data_processed.mkdir(exist_ok=True, parents=True)
    logger.info(f"Processing {config.data_raw}...")

    melody_vocab = Vocabulary("melody", config)
    chord_vocab = Vocabulary("chord", config)
    chord_mapper = ChordMapper()

    # ------------------------------------------------------------------
    # Pass 1: Build Vocabulary
    # ------------------------------------------------------------------
    logger.info("Pass 1: Building Vocabulary...")
    with open(config.data_raw, "rb") as f:
        for _, data in tqdm(ijson.kvitems(f, ""), desc="Building Vocab"):
            annot = data.get("annotations", {})

            # Augment vocab by all transpositions in [-max_transpose, max_transpose].
            for transpose in range(-config.max_transpose, config.max_transpose + 1):
                if annot.get("melody"):
                    for n in annot["melody"]:
                        pitch = n["pitch_class"] + (n["octave"] * 12) + config.center_midi
                        tp_pitch = get_transposed_pitch(pitch, transpose)
                        melody_vocab.add(f"pitch_{tp_pitch}_on")
                        melody_vocab.add(f"pitch_{tp_pitch}_hold")

                if annot.get("harmony"):
                    for c in annot["harmony"]:
                        # Some annotations may have empty or missing interval lists.
                        # These would produce an empty chord "quality" string, which
                        # later cannot be parsed by the evaluation metrics.
                        intervals = c.get("root_position_intervals") or []
                        if not intervals:
                            # Treat these as "no chord" and skip them in the vocab.
                            continue

                        tp_root = (c["root_pitch_class"] + transpose) % 12
                        token = chord_mapper.get_token(tp_root, intervals, c["inversion"])
                        chord_vocab.add(f"{token}_on")
                        chord_vocab.add(f"{token}_hold")

    # ------------------------------------------------------------------
    # Unified vocabulary
    # ------------------------------------------------------------------
    #
    # Build a single unified vocabulary that combines melody and chord
    # tokens in one ID space. This mirrors the layout used by the paper:
    #
    #   - Special tokens (pad/sos/eos/rest) share the same IDs across
    #     melody/chord/unified spaces.
    #   - Melody tokens keep their existing IDs from `melody_vocab`.
    #   - Chord tokens (excluding specials) are offset to live above the
    #     melody range so that a single embedding table can cover both
    #     tracks.
    #
    # The unified mapping is saved once at preprocessing time and is the
    # canonical source of truth for all model/vocab operations.
    logger.info("Building unified vocabulary...")

    unified_token_to_id: dict[str, int] = {}

    # 1) Copy melody tokens directly.
    unified_token_to_id.update(melody_vocab.token_to_id)
    melody_size = len(melody_vocab.token_to_id)

    special_ids = {
        config.pad_id,
        config.sos_id,
        config.eos_id,
        config.rest_id,
    }

    # 2) Add chord tokens, sharing special IDs and offsetting the rest.
    for token, cid in chord_vocab.token_to_id.items():
        if cid in special_ids:
            # Special tokens already exist in the melody vocab; ensure the
            # mapping is consistent.
            if token in unified_token_to_id:
                if unified_token_to_id[token] != cid:
                    logger.warning(
                        "Unified vocab conflict for token %s: melody=%d, chord=%d",
                        token,
                        unified_token_to_id[token],
                        cid,
                    )
            else:
                unified_token_to_id[token] = cid
        else:
            unified_token_to_id[token] = melody_size + cid

    unified_path = config.vocab_unified
    with unified_path.open("w") as vocab_file:
        json.dump(
            {
                "token_to_id": unified_token_to_id,
                "melody_size": melody_size,
                "chord_size": len(chord_vocab.token_to_id),
            },
            vocab_file,
            indent=2,
        )
    logger.info(
        "Saved unified vocabulary to %s (Size: %d)",
        unified_path,
        len(unified_token_to_id),
    )

    # Save separate melody and chord vocabularies for models that use
    # separate embedding tables.
    melody_vocab.save(config.vocab_melody)
    chord_vocab.save(config.vocab_chord)

    # ------------------------------------------------------------------
    # Pass 2: Tokenize and save to NPY (unified token IDs)
    # ------------------------------------------------------------------
    #
    # From this point on, all sequences on disk use the unified ID space.
    # The offline dataset reads `*_src.npy` / `*_tgt.npy` as melody/chord
    # tracks in unified IDs, and the online dataset interleaves them into
    # a single unified sequence. On‑the‑fly transposition in the Dataset
    # uses the same unified mapping.
    logger.info("Pass 2: Tokenizing to Numpy (unified IDs)...")

    def _buffer_factory() -> dict[str, list[np.ndarray]]:
        return {"src": [], "tgt": []}

    buffers: defaultdict[str, dict[str, list[np.ndarray]]] = defaultdict(_buffer_factory)

    with open(config.data_raw, "rb") as f:
        for _, data in tqdm(ijson.kvitems(f, ""), desc="Tokenizing"):
            split = data.get("split", "TRAIN").lower()
            annot = data.get("annotations", {})
            num_beats = annot.get("num_beats", 0)
            if num_beats <= 0:
                continue

            total_frames = int(num_beats * config.frame_rate)

            # Random transposition in [-max_transpose, max_transpose] is applied
            # on-the-fly in the Dataset for the train split.
            transpose = 0

            arr_len = config.storage_len
            # Default to silence (rest) for all frames; padding is applied after EOS.
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

                    # Look up unified IDs; fall back to REST if missing.
                    id_on = unified_token_to_id.get(token_on, config.rest_id)
                    id_hold = unified_token_to_id.get(token_hold, config.rest_id)

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

                    id_on = unified_token_to_id.get(token_on, config.rest_id)
                    id_hold = unified_token_to_id.get(token_hold, config.rest_id)

                    for idx in range(start, min(end, total_frames)):
                        arr_idx = idx + 1
                        if arr_idx >= arr_len - 1:
                            break
                        c_row[arr_idx] = id_on if idx == start else id_hold

            # EOS
            eos_idx = min(total_frames + 1, arr_len - 1)
            m_row[eos_idx] = config.eos_id
            c_row[eos_idx] = config.eos_id
            # Everything after EOS is padding, not silence.
            if eos_idx + 1 < arr_len:
                m_row[eos_idx + 1 :] = config.pad_id
                c_row[eos_idx + 1 :] = config.pad_id

            buffers[split]["src"].append(m_row)
            buffers[split]["tgt"].append(c_row)

    # Save buffers to disk
    for split, data in buffers.items():
        if not data["src"]:
            continue

        src_arr = np.stack(data["src"])
        tgt_arr = np.stack(data["tgt"])

        out_src = config.data_processed / f"{split}_src.npy"
        out_tgt = config.data_processed / f"{split}_tgt.npy"

        logger.info(f"Saving {split} src: {src_arr.shape} to {out_src}")
        np.save(out_src, src_arr)
        logger.info(f"Saving {split} tgt: {tgt_arr.shape} to {out_tgt}")
        np.save(out_tgt, tgt_arr)

    logger.info("Preprocessing complete.")


def main() -> None:
    """Run preprocessing from the command line."""

    parser = build_preprocess_parser()
    args = parser.parse_args()

    cfg = DataConfig()
    if args.input:
        cfg.data_raw = args.input
    if args.output:
        cfg.data_processed = args.output

    process_dataset(cfg)


if __name__ == "__main__":
    # Allow running as a module for local development:
    #   python -m musicagent.scripts.preprocess ...
    main()
