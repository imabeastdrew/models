"""Evaluation modules for MusicAgent."""

from musicagent.eval.metrics import (
    chord_length_entropy,
    chord_lengths,
    decode_tokens,
    note_in_chord_at_beat,
    note_in_chord_counts,
    note_in_chord_ratio,
    onset_interval_emd,
    onset_intervals,
)
from musicagent.eval.offline import OfflineEvalResult, evaluate_offline
from musicagent.eval.online import (
    OnlineEvalResult,
    evaluate_online,
    extract_melody_and_chords,
)

__all__ = [
    # Metrics
    "chord_length_entropy",
    "chord_lengths",
    "decode_tokens",
    "note_in_chord_at_beat",
    "note_in_chord_counts",
    "note_in_chord_ratio",
    "onset_interval_emd",
    "onset_intervals",
    # Evaluation functions
    "evaluate_offline",
    "evaluate_online",
    "extract_melody_and_chords",
    # Result containers
    "OfflineEvalResult",
    "OnlineEvalResult",
]
