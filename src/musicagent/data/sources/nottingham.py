from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from music21 import converter, harmony, meter, note, stream

# HookTheory and ReaLchords treat MIDI 60 (C4) as the origin for (octave, pitch_class)
ZERO_OCTAVE_MIDI = 60


def _split_abc_multi_tune(content: str) -> List[str]:
    """Split an ABC file that may contain multiple tunes into sections."""
    # Tunes are usually separated by X:<index> headers at the start of a line.
    sections = re.split(r"\n(?=X:\s*\d+)", content)
    return [s for s in sections if s.strip()]


def _estimate_pickup_measures(s: stream.Stream) -> stream.Stream:
    """Return a copy of the stream with obvious pickup measures removed.

    The goal is to drop very short anacrusis bars that would otherwise skew
    onset timings. The heuristic here mirrors typical ABC practice but is
    intentionally simple and conservative.
    """
    if not s:
        return s

    try:
        if not s.hasMeasures():
            s = s.makeMeasures()
    except Exception:
        return s

    keep_measures: List[stream.Measure] = []
    for m in s.getElementsByClass(stream.Measure):
        try:
            bar_len = m.barDuration.quarterLength
            filled = sum(elem.quarterLength for elem in m.notesAndRests)
            padding_left = getattr(m, "paddingLeft", 0.0)
        except Exception:
            keep_measures.append(m)
            continue

        # Treat strongly under-filled bars or bars with explicit left padding
        # as pickups. Otherwise keep them.
        is_pickup = padding_left > 0 or filled < bar_len * 0.5
        if not is_pickup:
            keep_measures.append(m)

    if not keep_measures:
        return s

    # Rebuild a simple stream from kept measures.
    new_stream = stream.Stream()
    for m in keep_measures:
        new_stream.append(m)
    return new_stream


def _midi_to_relative(midi_pitch: int) -> Tuple[int, int]:
    """Convert absolute MIDI pitch to (octave, pitch_class) around C4=0."""
    rel = midi_pitch - ZERO_OCTAVE_MIDI
    octave = rel // 12
    pc = rel % 12
    return octave, pc


def _resolve_melody_overlaps(notes: List[Dict], min_q_len: float = 0.25) -> List[Dict]:
    """Truncate notes so the melody is strictly monophonic in time."""
    if len(notes) <= 1:
        return notes

    sorted_notes = sorted(notes, key=lambda x: (x["onset"], x["offset"]))
    result: List[Dict] = []
    for i, n in enumerate(sorted_notes):
        current = dict(n)
        for j in range(i + 1, len(sorted_notes)):
            nxt = sorted_notes[j]
            if nxt["onset"] >= current["offset"]:
                break
            # Next note starts before current ends → truncate current.
            current["offset"] = nxt["onset"]
        if current["offset"] <= current["onset"]:
            current["offset"] = current["onset"] + min_q_len
        result.append(current)
    return result


def _filter_short_chords(chords: List[Dict], min_q_len: float = 1e-3) -> List[Dict]:
    """Drop harmony entries with negligible duration."""
    return [c for c in chords if c["offset"] - c["onset"] > min_q_len]


def _parse_simple_chord_symbol(symbol: str) -> Tuple[int, List[int], int]:
    """Map a basic chord symbol to (root_pitch_class, intervals, inversion).

    This is intentionally limited to the common patterns that appear in
    Nottingham: major, minor, dominant 7th, minor 7th, and major 7th.
    Intervals are given as successive semitone steps from one chord tone to
    the next, in root position.
    """
    symbol = symbol.strip()
    if not symbol:
        return 0, [], 0

    # Basic root name parsing with optional accidental.
    root_match = re.match(r"^([A-Ga-g][#b]?)(.*)$", symbol)
    if not root_match:
        return 0, [], 0
    root_name = root_match.group(1).upper()
    suffix = root_match.group(2)

    name_to_pc = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    root_pc = name_to_pc.get(root_name, 0)

    # Decide basic quality from suffix.
    suffix_lower = suffix.lower()
    if suffix_lower.endswith("maj7"):
        steps = [4, 3, 4]  # maj3, min3, maj3
    elif suffix_lower.endswith("m7"):
        steps = [3, 4, 3]  # min3, maj3, min3
    elif suffix_lower.endswith("7"):
        steps = [4, 3, 3]  # maj3, min3, min3
    elif suffix_lower.endswith("m"):
        steps = [3, 4]  # min3, maj3
    else:
        steps = [4, 3]  # default: major triad

    return root_pc, steps, 0


def _extract_melody_and_harmony_from_stream(s: stream.Stream) -> Tuple[List[Dict], List[Dict]]:
    """Extract monophonic melody and chord annotations from a music21 stream."""
    # Ensure we are working on measures with pickups reasonably handled.
    s = _estimate_pickup_measures(s)

    melody: List[Dict] = []
    chords: List[Dict] = []

    # Melody: walk notes/rests in time order, tracking running time.
    current_time = 0.0
    for element in s.flat.notesAndRests:
        q_len = float(element.quarterLength)
        if isinstance(element, note.Note):
            onset = current_time
            offset = onset + q_len
            octave, pc = _midi_to_relative(int(element.pitch.midi))
            melody.append(
                {
                    "onset": onset,
                    "offset": offset,
                    "pitch_class": pc,
                    "octave": octave,
                }
            )
        current_time += q_len

    # Harmony: prefer music21 chord symbols with realized durations.
    try:
        chord_symbols = s.flat.getElementsByClass(harmony.ChordSymbol)
    except Exception:
        chord_symbols = []

    if chord_symbols:
        try:
            harmony.realizeChordSymbolDurations(s)
        except Exception:
            pass

        for cs in chord_symbols:
            try:
                onset = float(cs.offset)
                q_len = float(cs.quarterLength)
                offset = onset + q_len
                text = str(getattr(cs, "figure", cs))
                root_pc, intervals, inversion = _parse_simple_chord_symbol(text)
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

    # Fallback: crude extraction from quoted chord symbols in the ABC body.
    return melody, chords


def _fallback_chords_from_abc_text(section: str) -> List[Dict]:
    """Very simple chord reconstruction from quoted symbols in ABC text."""
    # Strip header (until K: line) and gather remaining music lines.
    lines = section.splitlines()
    in_body = False
    body_tokens: List[str] = []
    for line in lines:
        line = line.strip()
        if not in_body and line.startswith("K:"):
            in_body = True
            continue
        if in_body and line and not line.startswith("%"):
            body_tokens.append(line)
    content = " ".join(body_tokens)

    result: List[Dict] = []
    pos = 0.0
    default_len = 2.0  # two beats per chord when we have no timing info
    for match in re.finditer(r'"([^"]+)"', content):
        symbol = match.group(1)
        root_pc, intervals, inversion = _parse_simple_chord_symbol(symbol)
        if not intervals:
            continue
        onset = pos
        offset = onset + default_len
        result.append(
            {
                "onset": onset,
                "offset": offset,
                "root_pitch_class": root_pc,
                "root_position_intervals": intervals,
                "inversion": inversion,
            }
        )
        pos = offset
    return result


def _build_tune_from_section(section: str, file_stem: str, tune_index: int) -> Dict | None:
    """Parse a single ABC tune section into our annotation schema."""
    try:
        s = converter.parse(section, format="abc")
    except Exception:
        return None
    if s is None:
        return None

    melody, chords = _extract_melody_and_harmony_from_stream(s)
    if not chords:
        # Try crude fallback from ABC text if no chord symbols were found.
        chords = _fallback_chords_from_abc_text(section)

    if not melody or not chords:
        return None

    melody = _resolve_melody_overlaps(melody)
    chords = _filter_short_chords(chords)
    if not melody or not chords:
        return None

    max_offset = 0.0
    if melody:
        max_offset = max(max_offset, max(n["offset"] for n in melody))
    if chords:
        max_offset = max(max_offset, max(c["offset"] for c in chords))
    num_beats = int(max_offset) if max_offset > 0 else 32

    tune_id = f"nottingham_{file_stem}_{tune_index}"
    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",  # will be reassigned by the splitter
        "nottingham": {
            "id": tune_id,
            "file": file_stem,
            "index": str(tune_index),
        },
        "annotations": {
            "num_beats": num_beats,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "keys": [
                {
                    "beat": 0,
                    "tonic_pitch_class": 0,
                    "scale_degree_intervals": [2, 2, 1, 2, 2, 2],
                }
            ],
            "melody": melody,
            "harmony": chords,
        },
    }


def _split_songs_randomly(
    songs: List[Dict], train_ratio: float = 0.8, valid_ratio: float = 0.1, seed: int = 42
) -> None:
    """Assign TRAIN/VALID/TEST splits to songs in-place."""
    if not songs:
        return
    rng = random.Random(seed)
    rng.shuffle(songs)

    n = len(songs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train_end = n_train
    valid_end = n_train + n_valid

    for i, s in enumerate(songs):
        if i < train_end:
            s["split"] = "TRAIN"
        elif i < valid_end:
            s["split"] = "VALID"
        else:
            s["split"] = "TEST"


def extract_nottingham_songs(root_dir: Path | str) -> List[Dict]:
    """Load Nottingham ABC files and return HookTheory-style song annotations.

    Args:
        root_dir: Directory containing the cleaned ABC files (e.g. ABC_cleaned).
    """
    root_path = Path(root_dir)
    abc_files = sorted(root_path.glob("*.abc"))

    songs: List[Dict] = []
    for abc_path in abc_files:
        try:
            content = abc_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        sections = _split_abc_multi_tune(content)
        for idx, section in enumerate(sections):
            tune = _build_tune_from_section(section, abc_path.stem, idx)
            if tune is not None:
                songs.append(tune)

    _split_songs_randomly(songs)
    return songs



