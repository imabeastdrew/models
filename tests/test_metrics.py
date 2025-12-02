import logging
import math

import numpy as np

from musicagent.eval.metrics import (
    PITCH_CLASSES,
    chord_length_entropy,
    chord_lengths,
    chord_pitch_classes,
    earth_movers_distance,
    histogram,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
    parse_chord_token,
    parse_melody_token,
)


def test_parse_chord_token_roundtrip() -> None:
    """Chord tokens from preprocessing should round-trip to (root, quality)."""
    # Matches the format produced by scripts.preprocess.ChordMapper.get_token
    tok_on = "C:4-3/0_on"
    tok_hold = "C:4-3/0_hold"

    assert parse_chord_token(tok_on) == ("C", "4-3")
    assert parse_chord_token(tok_hold) == ("C", "4-3")


def test_parse_chord_token_special_and_invalid() -> None:
    """Special tokens and malformed tokens should yield (None, None)."""
    assert parse_chord_token("<pad>") == (None, None)
    assert parse_chord_token("rest") == (None, None)
    # Missing _on/_hold suffix
    assert parse_chord_token("C:4-3/0") == (None, None)


def test_parse_melody_token_valid_and_invalid() -> None:
    """Melody tokens in 'pitch_{midi}_{kind}' format should parse, others not."""
    assert parse_melody_token("pitch_60_on") == 60
    assert parse_melody_token("pitch_72_hold") == 72
    assert parse_melody_token("rest") is None
    assert parse_melody_token("<pad>") is None
    # Malformed pitch token
    assert parse_melody_token("pitch_not_an_int_on") is None


def test_chord_pitch_classes_numeric_quality() -> None:
    """chord_pitch_classes should interpret numeric interval strings correctly."""
    # Simple triad quality "4-3": successive intervals {4,3} → pcs {0,4,7}
    pcs_c = chord_pitch_classes("C", "4-3")
    assert pcs_c == {0, 4, 7}

    # Root shift should move all pitch classes by the root index.
    d_index = PITCH_CLASSES.index("D")
    pcs_d = chord_pitch_classes("D", "4-3")
    assert pcs_d == {(d_index + i) % 12 for i in {0, 4, 7}}


def test_chord_pitch_classes_logs_and_returns_empty_on_bad_quality(caplog) -> None:
    """Unparseable quality strings should log a warning and return empty set."""
    with caplog.at_level(logging.WARNING):
        pcs = chord_pitch_classes("C", "not-a-number")

    assert pcs == set()
    assert "Unparseable chord quality" in caplog.text


def test_note_in_chord_ratio_basic() -> None:
    """note_in_chord_ratio should count matches only when pitch class is in chord."""
    # Melody: E (pc 4), F (pc 5), Eb (pc 3)
    melody_tokens = ["pitch_64_on", "pitch_65_on", "pitch_63_on"]
    # Chord: C with quality "4-3" (pcs {0,4,7}) at every frame
    chord_tokens = ["C:4-3/0_on"] * 3

    ratio = note_in_chord_ratio(melody_tokens, chord_tokens)
    # E (4) and Eb (3) -> only E is in {0,4,7}, so 1 match out of 3
    assert ratio == 1 / 3


def test_note_in_chord_ratio_skips_special_frames() -> None:
    """Frames where melody or chord is special should be skipped."""
    melody_tokens = ["<pad>", "pitch_64_on", "rest"]
    chord_tokens = ["C:0-4-7/0_on", "C:0-4-7/0_on", "C:0-4-7/0_on"]

    ratio = note_in_chord_ratio(melody_tokens, chord_tokens)
    # Only middle frame is a valid (melody, chord) pair, and it's a match.
    assert ratio == 1.0


def test_note_in_chord_ratio_counts_bad_quality_chords() -> None:
    """Frames with bad-quality chords count toward total but never match."""
    # Melody: three valid notes.
    melody_tokens = ["pitch_60_on", "pitch_62_on", "pitch_64_on"]
    # Chords: all with unparseable quality → chord_pitch_classes returns empty set.
    chord_tokens = ["C:not-a-number/0_on"] * 3

    ratio = note_in_chord_ratio(melody_tokens, chord_tokens)
    # 0 matches out of 3 valid frames.
    assert ratio == 0.0


def test_onset_intervals_and_histogram_and_emd() -> None:
    """End-to-end check for onset_intervals → histogram → EMD."""
    # Simple melody/chord pattern with clear onset intervals.
    melody_tokens = [
        "pitch_60_on",  # t=0
        "pitch_60_hold",
        "pitch_60_hold",
        "pitch_60_on",  # t=3
    ]
    chord_tokens = [
        "C:0-4-7/0_on",  # t=0, interval 0 from preceding melody onset
        "C:0-4-7/0_hold",
        "C:0-4-7/0_on",  # t=2, interval 2 from preceding melody onset at t=0
        "C:0-4-7/0_hold",
    ]

    intervals = onset_intervals(melody_tokens, chord_tokens)
    assert intervals == [0, 2]

    hist = histogram(intervals, max_bin=4)
    # Two events, both within [0,4] → total probability 1
    assert np.isclose(hist.sum(), 1.0)
    # Bins 0 and 2 should be non-zero.
    assert hist[0] > 0.0
    assert hist[2] > 0.0

    # EMD should be zero when comparing histogram to itself.
    emd = earth_movers_distance(hist, hist)
    assert emd == 0.0

    # Non-zero EMD for a shifted histogram.
    hist_shifted = histogram([v + 1 for v in intervals], max_bin=4)
    emd_shifted = earth_movers_distance(hist, hist_shifted)
    assert emd_shifted > 0.0

    # onset_interval_emd should agree with direct EMD on the same inputs.
    emd_wrapper = onset_interval_emd(intervals, [v + 1 for v in intervals], max_bin=4)
    assert np.isclose(emd_shifted, emd_wrapper)


def test_onset_intervals_ignores_chords_before_first_melody_onset() -> None:
    """Chord onsets before any melody onset should produce no intervals."""
    melody_tokens = [
        "rest",  # no melody onset yet
        "pitch_60_on",  # first melody onset
    ]
    chord_tokens = [
        "C:0-4-7/0_on",  # chord onset before any melody onset
        "C:0-4-7/0_hold",  # no new chord onset
    ]

    intervals = onset_intervals(melody_tokens, chord_tokens)
    assert intervals == []


def test_chord_lengths_and_entropy() -> None:
    """Chord length utilities should produce sensible lengths and entropy."""
    chord_tokens = [
        "C:0-4-7/0_on",  # chord 1 starts
        "C:0-4-7/0_hold",  # chord 1 continues
        "rest",  # chord 1 ends (length 2)
        "C:0-4-7/0_on",  # chord 2 starts
        "C:0-4-7/0_hold",  # chord 2 continues
        "C:0-4-7/0_hold",  # chord 2 continues
        "rest",  # chord 2 ends (length 3)
    ]

    lengths = chord_lengths(chord_tokens)
    assert lengths == [2, 3]

    entropy = chord_length_entropy(lengths)
    # Two distinct lengths with equal probability → entropy = ln(2)
    assert np.isclose(entropy, math.log(2.0))
