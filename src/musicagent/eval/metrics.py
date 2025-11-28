"""Evaluation metrics for generation."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitch / chord utilities
# ---------------------------------------------------------------------------

# 12 pitch classes
PITCH_CLASSES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']



def parse_chord_token(token: str) -> tuple[str | None, str | None]:
    """Parse chord token like 'C:maj/0_on' -> (root, quality).

    Returns (None, None) for special tokens (pad/sos/eos/rest) or unparseable.
    """
    if token.startswith('<') or token == 'rest':
        return None, None

    # Strip _on / _hold suffix
    if token.endswith('_on'):
        base = token[:-3]
    elif token.endswith('_hold'):
        base = token[:-5]
    else:
        return None, None

    # Split root:quality/inversion
    try:
        root, rest = base.split(':', 1)
        quality = rest.split('/')[0]  # ignore inversion for pitch-class membership
        return root, quality
    except ValueError:
        return None, None


def chord_pitch_classes(root: str, quality: str) -> set[int]:
    """Return set of pitch classes (0-11) for a chord.

    ``quality`` is a numeric interval string produced by
    :class:`scripts.preprocess.ChordMapper`, based on Hooktheory's
    ``root_position_intervals``.

    In our processed data this is typically a list of **successive intervals**
    between chord tones (e.g. ``"4-3"`` for a major triad), but our tests also
    exercise **absolute encodings** like ``"0-4-7"``. We therefore support
    both encodings:

    - Successive encoding (no explicit 0): e.g. ``"4-3"`` →
      cumulative sums → offsets {0, 4, 7}.
    - Absolute encoding (contains 0): e.g. ``"0-4-7"`` →
      interpreted directly as offsets from the root.
    """
    try:
        root_pc = PITCH_CLASSES.index(root)
    except ValueError:
        return set()

    try:
        raw_intervals = [int(x) for x in quality.split("-") if x != ""]
    except ValueError:
        # Log and treat as "no valid chord" so metric frames are skipped.
        logger.warning("Unparseable chord quality %r for root %r", quality, root)
        return set()

    if not raw_intervals:
        return set()

    # If an explicit 0 appears, interpret as absolute offsets (e.g. "0-4-7").
    if any(i == 0 for i in raw_intervals):
        rel_intervals = {i % 12 for i in raw_intervals}
        rel_intervals.add(0)  # ensure root is included even if not listed
    else:
        # Successive-interval encoding (e.g. "4-3" for a major triad):
        # take cumulative sums starting from 0.
        rel_intervals = {0}
        acc = 0
        for step in raw_intervals:
            acc = (acc + step) % 12
            rel_intervals.add(acc)

    return {(root_pc + interval) % 12 for interval in rel_intervals}


def parse_melody_token(token: str) -> int | None:
    """Parse melody token like 'pitch_60_on' -> MIDI pitch (60).

    Returns None for special tokens.
    """
    if not token.startswith('pitch_'):
        return None
    try:
        _, pitch_str, _ = token.split('_', 2)
        return int(pitch_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Metric 1: Note-in-Chord Ratio
# ---------------------------------------------------------------------------

def note_in_chord_ratio(
    melody_tokens: Sequence[str],
    chord_tokens: Sequence[str],
) -> float:
    """Compute fraction of frames where melody pitch class is in chord.

    Skips frames where either melody or chord is silent/special.
    """
    matches = 0
    total = 0

    for mel_tok, chd_tok in zip(melody_tokens, chord_tokens):
        midi_pitch = parse_melody_token(mel_tok)
        root, quality = parse_chord_token(chd_tok)

        # Skip silent / special frames where either melody or chord is not a real pitch.
        if midi_pitch is None or root is None or quality is None:
            continue

        melody_pc = midi_pitch % 12
        chord_pcs = chord_pitch_classes(root, quality)

        if melody_pc in chord_pcs:
            matches += 1
        total += 1

    return matches / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Chord-to-Note Onset Interval (EMD)
# ---------------------------------------------------------------------------

def onset_intervals(
    melody_tokens: Sequence[str],
    chord_tokens: Sequence[str],
) -> list[int]:
    """Compute chord-to-preceding-melody-note onset intervals (in frames)."""
    intervals: list[int] = []
    last_melody_onset: int | None = None

    for t, (mel_tok, chd_tok) in enumerate(zip(melody_tokens, chord_tokens)):
        # Track melody onsets
        if mel_tok.endswith('_on') and parse_melody_token(mel_tok) is not None:
            last_melody_onset = t

        # Track chord onsets
        if chd_tok.endswith('_on'):
            root, _ = parse_chord_token(chd_tok)
            if root is not None and last_melody_onset is not None:
                intervals.append(t - last_melody_onset)

    return intervals


def histogram(values: Sequence[int], max_bin: int = 16) -> np.ndarray:
    """Build histogram with bins [0, 1, ..., max_bin, >max_bin]."""
    counts = np.zeros(max_bin + 2, dtype=np.float64)
    for v in values:
        if v <= max_bin:
            counts[v] += 1
        else:
            counts[max_bin + 1] += 1
    # Normalize
    total = counts.sum()
    return counts / total if total > 0 else counts


def earth_movers_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """1D Earth Mover's Distance between two histograms."""
    cdf_a = np.cumsum(hist_a)
    cdf_b = np.cumsum(hist_b)
    return float(np.sum(np.abs(cdf_a - cdf_b)))


def onset_interval_emd(
    pred_intervals: Sequence[int],
    ref_intervals: Sequence[int],
    max_bin: int = 16,
) -> float:
    """EMD between predicted and reference onset-interval histograms."""
    hist_pred = histogram(pred_intervals, max_bin)
    hist_ref = histogram(ref_intervals, max_bin)
    return earth_movers_distance(hist_pred, hist_ref)


# ---------------------------------------------------------------------------
# Metric 3: Chord Length Entropy
# ---------------------------------------------------------------------------

def chord_lengths(chord_tokens: Sequence[str]) -> list[int]:
    """Compute list of chord durations (in frames)."""
    lengths: list[int] = []
    current_len = 0
    in_chord = False

    for tok in chord_tokens:
        root, _ = parse_chord_token(tok)
        if root is None:
            # Not a real chord (silence/special)
            if in_chord:
                lengths.append(current_len)
                in_chord = False
                current_len = 0
            continue

        if tok.endswith('_on'):
            if in_chord:
                lengths.append(current_len)
            current_len = 1
            in_chord = True
        elif tok.endswith('_hold'):
            current_len += 1

    if in_chord:
        lengths.append(current_len)

    return lengths


def chord_length_entropy(lengths: Sequence[int], max_bin: int = 32) -> float:
    """Entropy of chord-length distribution (nats)."""
    counts = Counter(min(length, max_bin + 1) for length in lengths)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


