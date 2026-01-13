from __future__ import annotations

import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

from music21 import converter, harmony, key, meter, note
from note_seq import chord_symbols_lib

from musicagent.config import DataConfig

# Keep this consistent with DataConfig.center_midi (C4).
ZERO_OCTAVE_MIDI = DataConfig.center_midi


def _relative_pitch(midi_pitch: int) -> Tuple[int, int]:
    """Map absolute MIDI pitch to (octave, pitch_class) around C4=0."""
    rel = midi_pitch - ZERO_OCTAVE_MIDI
    return rel // 12, rel % 12


def _snap_to_grid(value: float, step: float = 0.25) -> float:
    """Round a time value to the nearest grid point."""
    return round(value / step) * step


def _quantize_timed_items(items: List[Dict], step: float = 0.25) -> List[Dict]:
    """Snap onsets/offsets to a regular grid and ensure positive durations."""
    out: List[Dict] = []
    for item in items:
        onset = _snap_to_grid(float(item["onset"]), step)
        offset = _snap_to_grid(float(item["offset"]), step)
        if offset <= onset:
            offset = onset + step
        new_item = dict(item)
        new_item["onset"] = onset
        new_item["offset"] = offset
        out.append(new_item)
    return out


def _resolve_melody_overlaps(items: List[Dict], min_q_len: float = 0.25) -> List[Dict]:
    """Make a monophonic melody by truncating overlapping notes."""
    if not items:
        return items
    notes = sorted(items, key=lambda x: (x["onset"], x["offset"]))
    resolved: List[Dict] = []
    i = 0
    while i < len(notes):
        current = deepcopy(notes[i])
        j = i + 1
        # Group notes that start together.
        same_onset: List[Dict] = [current]
        while j < len(notes) and notes[j]["onset"] == current["onset"]:
            same_onset.append(deepcopy(notes[j]))
            j += 1

        if len(same_onset) > 1:
            # Keep the longest note among those starting at the same time.
            chosen = max(
                same_onset, key=lambda n: (n["offset"] - n["onset"])
            )
        else:
            chosen = current

        # Truncate chosen note at the onset of the next note if they overlap.
        for k in range(j, len(notes)):
            nxt = notes[k]
            if nxt["onset"] >= chosen["offset"]:
                break
            chosen["offset"] = nxt["onset"]

        if chosen["offset"] <= chosen["onset"]:
            chosen["offset"] = chosen["onset"] + min_q_len
        resolved.append(chosen)
        i = j
    return resolved


def _drop_tiny_chords(chords: List[Dict], min_q_len: float = 1e-3) -> List[Dict]:
    return [c for c in chords if c["offset"] - c["onset"] > min_q_len]


def _normalize_wikifonia_label(symbol: str) -> str:
    """Normalize Wikifonia chord labels into something `note_seq` can parse."""
    if not symbol:
        return ""
    chord = symbol.strip()
    # Replace trailing '-' with flat, e.g. 'E-' -> 'Eb'.
    chord = re.sub(r"([A-Ga-g])-", r"\1b", chord)
    # Collapse whitespace.
    chord = re.sub(r"\s+", "", chord)
    # Standardize some common textual variants.
    chord = chord.replace("add9", "add9")
    chord = chord.replace("dim7", "dim7")
    return chord


def _simplify_unparsable(symbol: str) -> str:
    """Fallback simplification for chords that `note_seq` cannot interpret."""
    chord = symbol.strip()
    m = re.match(r"^([A-Ga-g][#b]?)(.*)$", chord)
    if not m:
        return "C"
    root = m.group(1)
    tail = m.group(2).lower()

    # If it looks minor and not explicit major.
    if "m" in tail and "maj" not in tail:
        if "7" in tail:
            return f"{root}m7"
        return f"{root}m"

    # If it has seventh/extended qualities, reduce to simple 7th.
    if any(t in tail for t in ("9", "11", "13", "7")):
        return f"{root}7"

    # Suspensions → basic sus4.
    if "sus" in tail:
        return f"{root}sus4"

    # Catch-all: major triad.
    return root


def _chord_symbol_to_intervals(symbol: str) -> Tuple[int, List[int], int]:
    """Convert a chord symbol into (root_pitch_class, intervals, inversion)."""
    label = _normalize_wikifonia_label(symbol)
    if not label:
        return 0, [], 0

    try:
        pitches = chord_symbols_lib.chord_symbol_pitches(label)
    except Exception:
        pitches = []

    if not pitches:
        # Try a simplified version.
        simple = _simplify_unparsable(label)
        try:
            pitches = chord_symbols_lib.chord_symbol_pitches(simple)
        except Exception:
            pitches = []

    if not pitches:
        return 0, [], 0

    root_pc = pitches[0] % 12
    steps: List[int] = []
    for i in range(1, len(pitches)):
        steps.append((pitches[i] - pitches[i - 1]) % 12)

    inversion = 0
    # Handle simple slash chords for inversion.
    if "/" in label:
        base, bass = label.split("/", 1)
        try:
            base_pitches = chord_symbols_lib.chord_symbol_pitches(base)
            bass_pitch = chord_symbols_lib.chord_symbol_pitches(bass)[0] % 12
            for idx, p in enumerate(base_pitches):
                if p % 12 == bass_pitch:
                    inversion = idx
                    break
        except Exception:
            inversion = 0

    return root_pc, steps, inversion


def _parse_single_musicxml(xml_path: Path) -> Dict | None:
    """Parse one MusicXML lead sheet into our annotation format."""
    try:
        score = converter.parse(str(xml_path))
    except Exception:
        return None
    if score is None:
        return None

    if score.parts:
        melody_notes_raw = score.parts[0].flat.notes
    else:
        melody_notes_raw = score.flat.notes

    melody: List[Dict] = []
    for elem in melody_notes_raw:
        if not isinstance(elem, note.Note):
            continue
        onset = float(elem.offset)
        q_len = float(elem.quarterLength)
        offset = onset + q_len
        octave, pc = _relative_pitch(int(elem.pitch.midi))
        melody.append(
            {
                "onset": onset,
                "offset": offset,
                "pitch_class": pc,
                "octave": octave,
            }
        )

    chord_syms = score.flat.getElementsByClass(harmony.ChordSymbol)
    chords: List[Dict] = []
    if chord_syms:
        try:
            harmony.realizeChordSymbolDurations(score)
        except Exception:
            pass
        for cs in chord_syms:
            try:
                onset = float(cs.offset)
                q_len = float(cs.quarterLength)
                offset = onset + q_len
                text = str(getattr(cs, "figure", cs))
                root_pc, intervals, inversion = _chord_symbol_to_intervals(text)
                if not intervals:
                    continue
                chords.append(
                    {
                        "onset": onset,
                        "offset": offset,
                        "root_pitch_class": root_pc,
                        "root_position_intervals": intervals,
                        "inversion": inversion,
                    }
                )
            except Exception:
                continue

    if not melody or not chords:
        return None

    # Quantize and clean timing.
    melody = _quantize_timed_items(melody, step=0.25)
    chords = _quantize_timed_items(chords, step=0.25)
    melody = _resolve_melody_overlaps(melody)
    chords = _drop_tiny_chords(chords)
    if not melody or not chords:
        return None

    # Determine approximate tempo structure.
    max_offset = max(
        max(n["offset"] for n in melody),
        max(c["offset"] for c in chords),
    )
    num_beats = int(max_offset) if max_offset > 0 else 32

    # Basic meter and key metadata.
    ts = score.flat.getElementsByClass(meter.TimeSignature)
    ks = score.flat.getElementsByClass(key.KeySignature)

    meters_meta: List[Dict] = [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}]
    if ts:
        try:
            beats_per_bar = int(ts[0].numerator)
            beat_unit = int(ts[0].denominator)
            meters_meta = [{"beat": 0, "beats_per_bar": beats_per_bar, "beat_unit": beat_unit}]
        except Exception:
            pass

    keys_meta: List[Dict] = [
        {
            "beat": 0,
            "tonic_pitch_class": 0,
            "scale_degree_intervals": [2, 2, 1, 2, 2, 2],
        }
    ]
    if ks:
        # We keep the pitch-class origin fixed for simplicity; only the scale
        # could be adapted from the key signature if desired.
        pass

    title = None
    composer = None
    if score.metadata:
        title = score.metadata.title
        composer = score.metadata.composer

    song_id = xml_path.stem
    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",
        "wikifonia": {
            "id": song_id,
            "title": title or song_id,
            "composer": composer,
            "file": xml_path.name,
        },
        "annotations": {
            "num_beats": num_beats,
            "meters": meters_meta,
            "keys": keys_meta,
            "melody": melody,
            "harmony": chords,
        },
    }


def _assign_splits(
    songs: List[Dict], train_ratio: float = 0.8, valid_ratio: float = 0.1, seed: int = 42
) -> None:
    """In-place split assignment."""
    if not songs:
        return
    rng = random.Random(seed)
    rng.shuffle(songs)

    n = len(songs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    for idx, s in enumerate(songs):
        if idx < n_train:
            s["split"] = "TRAIN"
        elif idx < n_train + n_valid:
            s["split"] = "VALID"
        else:
            s["split"] = "TEST"


def extract_wikifonia_songs(root_dir: Path | str) -> List[Dict]:
    """Load Wikifonia MusicXML files and return HookTheory-style annotations.

    Args:
        root_dir: Directory that contains `.mxl` files (searched recursively).
    """
    root_path = Path(root_dir)
    xml_files = sorted(root_path.rglob("*.mxl"))

    songs: List[Dict] = []
    for xml_path in xml_files:
        song = _parse_single_musicxml(xml_path)
        if song is not None:
            songs.append(song)

    _assign_splits(songs)
    return songs



