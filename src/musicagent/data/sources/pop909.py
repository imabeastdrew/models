from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi
from note_seq import chord_symbols_lib

from musicagent.config import DataConfig


MIDDLE_C = DataConfig.center_midi


def _midi_to_relative(pitch: int) -> Tuple[int, int]:
    """Convert absolute MIDI pitch to (octave, pitch_class) around MIDDLE_C."""
    rel = pitch - MIDDLE_C
    return rel // 12, rel % 12


def _pc_to_name(pc: int) -> str:
    """Map pitch class to a canonical sharp-based note name."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[pc % 12]


def _root_and_suffix_from_pop909(label: str) -> Tuple[str | None, str | None]:
    """Split a POP909 chord label into root and quality+optional bass."""
    text = label.strip()
    if not text or text in {"N", "X"}:
        return None, None
    if ":" not in text:
        return text, ""
    root, tail = text.split(":", 1)
    return root.strip(), tail.strip()


def _degree_to_semitones(degree: str) -> int | None:
    """Convert a scale degree label used for POP909 slash bass into semitones."""
    mapping = {
        "3": 4,
        "b3": 3,
        "5": 7,
        "7": 11,
        "b7": 10,
    }
    return mapping.get(degree)


def _pop909_quality_to_suffix(quality: str) -> str | None:
    """Map POP909 quality strings to common chord suffixes for note_seq."""
    q = quality
    if q in ("", "maj"):
        return ""
    if q == "min":
        return "m"
    if q == "7":
        return "7"
    if q == "min7":
        return "m7"
    if q == "maj7":
        return "maj7"
    if q == "dim":
        return "dim"
    if q == "dim7":
        return "dim7"
    if q == "aug":
        return "aug"
    if q == "sus4":
        return "sus4"
    if q == "sus2":
        return "sus2"
    if q == "maj6":
        return "6"
    if q == "min6":
        return "m6"
    if q == "hdim7":
        return "m7b5"
    if q == "minmaj7":
        return "mM7"
    if q == "sus4(b7)":
        return "7sus"
    return None


def _label_to_noteseq_symbol(raw_label: str) -> Tuple[str | None, str | None]:
    """Convert a POP909 chord label to a note_seq-compatible symbol and base symbol.

    Returns:
        (full_symbol, base_symbol) where full_symbol may contain a slash
        bass (for inversion) and base_symbol is the unslashed chord name.
    """
    root, tail = _root_and_suffix_from_pop909(raw_label)
    if root is None:
        return None, None

    bass_name: str | None = None
    qual = tail
    if "/" in tail:
        qual, degree = tail.split("/", 1)
        qual = qual.strip()
        degree = degree.strip()
        shift = _degree_to_semitones(degree)
        if shift is not None:
            # Compute bass note name by applying the degree shift from the root.
            # Simple text-to-pc mapping using sharp-based names.
            pc_map = {
                "C": 0,
                "C#": 1,
                "Db": 1,
                "D": 2,
                "D#": 3,
                "Eb": 3,
                "E": 4,
                "F": 5,
                "F#": 6,
                "Gb": 6,
                "G": 7,
                "G#": 8,
                "Ab": 8,
                "A": 9,
                "A#": 10,
                "Bb": 10,
                "B": 11,
            }
            base_pc = pc_map.get(root, 0)
            bass_pc = (base_pc + shift) % 12
            bass_name = _pc_to_name(bass_pc)

    mapped_suffix = _pop909_quality_to_suffix(qual)
    if mapped_suffix is None:
        return None, None

    base_symbol = f"{root}{mapped_suffix}"
    full_symbol = base_symbol
    if bass_name is not None and bass_name != root:
        full_symbol = f"{base_symbol}/{bass_name}"
    return full_symbol, base_symbol


def _decode_chord_intervals(label: str) -> Tuple[int, List[int], int]:
    """Turn a POP909 chord label into (root_pitch_class, intervals, inversion)."""
    full_symbol, base_symbol = _label_to_noteseq_symbol(label)
    if base_symbol is None:
        return 0, [], 0

    try:
        pitches = chord_symbols_lib.chord_symbol_pitches(base_symbol)
    except Exception:
        pitches = []

    if not pitches:
        # Fallback: C major.
        return 0, [4, 3], 0

    root_pc = pitches[0] % 12
    steps: List[int] = []
    for i in range(1, len(pitches)):
        steps.append((pitches[i] - pitches[i - 1]) % 12)

    inversion = 0
    if full_symbol and "/" in full_symbol:
        base, bass = full_symbol.split("/", 1)
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


def _read_chord_file(path: Path) -> List[Dict]:
    """Parse POP909 chord_midi.txt into harmony annotations."""
    chords: List[Dict] = []
    if not path.exists():
        return chords

    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        start_s = float(parts[0])
        end_s = float(parts[1])
        label = parts[2]
        if label in {"N", "X"}:
            continue
        if end_s <= start_s:
            continue
        root_pc, intervals, inversion = _decode_chord_intervals(label)
        if not intervals:
            continue
        chords.append(
            {
                "onset": start_s,
                "offset": end_s,
                "root_pitch_class": root_pc,
                "root_position_intervals": intervals,
                "inversion": inversion,
            }
        )
    return chords


def _read_beats(path: Path) -> Tuple[List[float], List[bool]]:
    """Read beat_midi.txt into (beat_times, bar_start_flags)."""
    times: List[float] = []
    bar_flags: List[bool] = []
    if not path.exists():
        return times, bar_flags

    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        t = float(parts[0])
        # second column is downbeat flag (unused here)
        is_bar_start = float(parts[2]) > 0.5
        times.append(t)
        bar_flags.append(is_bar_start)
    return times, bar_flags


def _find_trim_time(
    melody: List[Dict], beat_times: List[float], bar_flags: List[bool], threshold_beats: int = 4
) -> float:
    """Return a time offset (seconds) to trim leading silence if long enough."""
    if not melody or not beat_times:
        return 0.0
    first_onset = min(n["onset"] for n in melody)
    # Closest beat to the first onset.
    closest_idx = min(
        range(len(beat_times)), key=lambda i: abs(beat_times[i] - first_onset)
    )
    if closest_idx < threshold_beats:
        return 0.0

    # Walk backwards to the nearest bar start.
    for i in range(closest_idx, -1, -1):
        if i < len(bar_flags) and bar_flags[i]:
            return beat_times[i]
    return 0.0


def _shift_times(items: List[Dict], offset: float) -> List[Dict]:
    """Subtract a constant time offset (seconds) and drop non-positive durations."""
    if offset == 0.0:
        return items
    out: List[Dict] = []
    for item in items:
        raw_onset = item["onset"] - offset
        raw_offset = item["offset"] - offset

        # Clamp onset to the new zero point but keep offset in the same frame of
        # reference so that a final duration check reflects the clamped values.
        new_onset = max(0.0, raw_onset)
        new_offset = raw_offset

        # Drop items that do not have strictly positive duration after trimming.
        if new_offset <= new_onset:
            continue

        new_item = dict(item)
        new_item["onset"] = new_onset
        new_item["offset"] = new_offset
        out.append(new_item)
    return out


def _extract_melody_from_midi(path: Path) -> List[Dict]:
    """Read MELODY track from POP909 MIDI and return note dicts in seconds."""
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return []

    target = None
    for inst in midi.instruments:
        if inst.name == "MELODY":
            target = inst
            break
    if target is None:
        return []

    notes: List[Dict] = []
    for n in target.notes:
        octave, pc = _midi_to_relative(n.pitch)
        notes.append(
            {
                "onset": float(n.start),
                "offset": float(n.end),
                "pitch_class": pc,
                "octave": octave,
            }
        )
    return notes


def _quantize_events_to_16th(events: List[Dict], beat_times: List[float]) -> List[Dict]:
    """Convert times in seconds to a quarter-note grid at 16th-note resolution."""
    if not events or len(beat_times) < 2:
        return events

    quantized: List[Dict] = []
    steps_per_beat = 4  # 16th notes

    for item in events:
        new_item = dict(item)
        for key in ("onset", "offset"):
            t = float(item[key])
            # Find enclosing beat interval.
            idx = 0
            for i in range(len(beat_times) - 1):
                if beat_times[i] <= t < beat_times[i + 1]:
                    idx = i
                    break
            else:
                idx = len(beat_times) - 1

            if idx < len(beat_times) - 1:
                start_t = beat_times[idx]
                end_t = beat_times[idx + 1]
                span = end_t - start_t if end_t > start_t else 1.0
                frac = (t - start_t) / span
            else:
                frac = 0.0

            sub = round(frac * steps_per_beat)
            sub = max(0, min(steps_per_beat, sub))
            beat_pos = idx + sub / steps_per_beat
            new_item[key] = beat_pos
        if new_item["offset"] <= new_item["onset"]:
            new_item["offset"] = new_item["onset"] + 1.0 / steps_per_beat
        quantized.append(new_item)
    return quantized


def _make_monophonic(melody: List[Dict]) -> List[Dict]:
    """Resolve overlapping melody notes by truncating earlier ones."""
    if not melody:
        return melody
    notes = sorted(melody, key=lambda x: (x["onset"], x["offset"]))
    resolved: List[Dict] = []
    i = 0
    while i < len(notes):
        current = deepcopy(notes[i])
        j = i + 1
        same_start: List[Dict] = [current]
        while j < len(notes) and notes[j]["onset"] == current["onset"]:
            same_start.append(deepcopy(notes[j]))
            j += 1
        if len(same_start) > 1:
            best = max(same_start, key=lambda n: (n["offset"] - n["onset"]))
        else:
            best = current
        for k in range(j, len(notes)):
            nxt = notes[k]
            if nxt["onset"] >= best["offset"]:
                break
            best["offset"] = nxt["onset"]
        if best["offset"] <= best["onset"]:
            best["offset"] = best["onset"] + 1e-3
        resolved.append(best)
        i = j
    return resolved


def _build_pop909_song(song_dir: Path) -> Dict | None:
    """Convert one POP909 index directory into our annotations schema."""
    song_id = song_dir.name
    midi_path = song_dir / f"{song_id}.mid"
    chord_path = song_dir / "chord_midi.txt"
    beat_path = song_dir / "beat_midi.txt"

    if not (midi_path.exists() and chord_path.exists() and beat_path.exists()):
        return None

    melody = _extract_melody_from_midi(midi_path)
    harmony = _read_chord_file(chord_path)
    beats, bar_flags = _read_beats(beat_path)

    if not melody or not harmony:
        return None

    trim_seconds = _find_trim_time(melody, beats, bar_flags)
    if trim_seconds > 0.0:
        melody = _shift_times(melody, trim_seconds)
        harmony = _shift_times(harmony, trim_seconds)
        beats = [t - trim_seconds for t in beats if t >= trim_seconds]

    if not beats or len(beats) < 2:
        return None

    melody_q = _quantize_events_to_16th(melody, beats)
    harmony_q = _quantize_events_to_16th(harmony, beats)
    melody_q = _make_monophonic(melody_q)

    if not melody_q or not harmony_q:
        return None

    max_offset = max(
        max(n["offset"] for n in melody_q),
        max(c["offset"] for c in harmony_q),
    )
    num_beats = int(max_offset) if max_offset > 0 else 32

    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",
        "pop909": {
            "id": song_id,
            "file": midi_path.name,
            "source": "POP909 Dataset",
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
            "melody": melody_q,
            "harmony": harmony_q,
        },
    }


def _assign_splits(
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
    for idx, s in enumerate(songs):
        if idx < n_train:
            s["split"] = "TRAIN"
        elif idx < n_train + n_valid:
            s["split"] = "VALID"
        else:
            s["split"] = "TEST"


def extract_pop909_songs(root_dir: Path | str) -> List[Dict]:
    """Load POP909 index folders and return HookTheory-style annotations."""
    root = Path(root_dir)
    song_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    songs: List[Dict] = []
    for d in song_dirs:
        song = _build_pop909_song(d)
        if song is not None:
            songs.append(song)
    _assign_splits(songs)
    return songs


