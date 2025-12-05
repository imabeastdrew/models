"""Analysis utilities for model evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from musicagent.eval.metrics import (
    chord_length_entropy,
    chord_lengths,
    histogram,
    melody_chord_root_intervals,
    note_in_chord_at_beat,
    onset_intervals,
)


@dataclass
class AdaptationDynamicsResult:
    """Result of per-beat NiC analysis."""

    valid_beats: list[int]
    """Beat indices that had enough samples."""
    beat_means: list[float]
    """Mean NiC ratio per valid beat."""
    beat_stds: list[float]
    """Standard deviation of NiC ratio per valid beat."""
    samples_per_beat: dict[int, int]
    """Number of samples for each beat (for all beats, not just valid ones)."""


@dataclass
class HistogramStats:
    """Histogram statistics for a single system or dataset."""

    # Raw values (all frames across all sequences)
    onset_intervals: list[int]
    chord_lengths: list[int]
    harmonic_intervals: list[int]

    # Normalized histograms
    onset_hist: np.ndarray
    chord_length_hist: np.ndarray
    harmonic_interval_hist: np.ndarray

    # Scalar summary
    chord_length_entropy: float


def compute_adaptation_dynamics(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    *,
    max_beats: int = 64,
    frame_rate: int = 4,
    min_samples: int = 10,
) -> AdaptationDynamicsResult:
    """Compute per-beat NiC statistics for adaptation dynamics analysis.

    Per paper Section K: excludes beats that are entirely silent.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        max_beats: Maximum number of beats to analyze
        frame_rate: Frames per beat (default 4 for 16th notes)
        min_samples: Minimum samples required per beat to include in results

    Returns:
        AdaptationDynamicsResult with per-beat statistics
    """
    beat_nic_all: dict[int, list[float]] = {b: [] for b in range(max_beats)}

    for mel_tokens, pred_tokens, _ in cached_predictions.values():
        # note_in_chord_at_beat returns None for silent beats
        beat_nic = note_in_chord_at_beat(mel_tokens, pred_tokens, frame_rate=frame_rate)

        for beat, nic in beat_nic.items():
            # Only include non-silent beats (nic is None for silent beats per paper)
            if beat < max_beats and nic is not None:
                beat_nic_all[beat].append(nic)

    # Compute mean and std per beat
    valid_beats: list[int] = []
    beat_means: list[float] = []
    beat_stds: list[float] = []

    for beat in range(max_beats):
        if len(beat_nic_all[beat]) >= min_samples:
            valid_beats.append(beat)
            beat_means.append(float(np.mean(beat_nic_all[beat])))
            beat_stds.append(float(np.std(beat_nic_all[beat])))

    samples_per_beat = {beat: len(beat_nic_all[beat]) for beat in range(max_beats)}

    return AdaptationDynamicsResult(
        valid_beats=valid_beats,
        beat_means=beat_means,
        beat_stds=beat_stds,
        samples_per_beat=samples_per_beat,
    )


def _build_histogram_stats(
    onset_values: Sequence[int],
    chord_length_values: Sequence[int],
    harmonic_values: Sequence[int],
    *,
    max_onset_bin: int = 16,
    max_length_bin: int = 32,
) -> HistogramStats:
    """Helper to build HistogramStats from raw values."""
    onset_values = list(onset_values)
    chord_length_values = list(chord_length_values)
    harmonic_values = list(harmonic_values)

    onset_hist = histogram(onset_values, max_bin=max_onset_bin)
    chord_length_hist = histogram(chord_length_values, max_bin=max_length_bin)
    # Harmonic intervals are pitch-class distances in [0, 11]
    harmonic_interval_hist = histogram(harmonic_values, max_bin=11)

    length_entropy = chord_length_entropy(chord_length_values)

    return HistogramStats(
        onset_intervals=onset_values,
        chord_lengths=chord_length_values,
        harmonic_intervals=harmonic_values,
        onset_hist=onset_hist,
        chord_length_hist=chord_length_hist,
        harmonic_interval_hist=harmonic_interval_hist,
        chord_length_entropy=length_entropy,
    )


def compute_histograms_from_cached_predictions(
    cached_predictions: dict[int, tuple[list[str], list[str], list[str]]],
    *,
    use_predicted_chords: bool = True,
    max_onset_bin: int = 16,
    max_length_bin: int = 32,
) -> HistogramStats:
    """Compute histograms for a model using its cached predictions.

    Args:
        cached_predictions: Dict mapping index -> (melody_tokens, pred_tokens, ref_tokens)
        use_predicted_chords: If True, use model predictions; otherwise use references.
        max_onset_bin: Maximum bin for onset-interval histogram (frames).
        max_length_bin: Maximum bin for chord-length histogram (frames).

    Returns:
        HistogramStats with raw values and normalized histograms.
    """
    all_onset: list[int] = []
    all_lengths: list[int] = []
    all_harmonic: list[int] = []

    for mel_tokens, pred_tokens, ref_tokens in cached_predictions.values():
        chord_tokens = pred_tokens if use_predicted_chords else ref_tokens

        all_onset.extend(onset_intervals(mel_tokens, chord_tokens))
        all_lengths.extend(chord_lengths(chord_tokens))
        all_harmonic.extend(melody_chord_root_intervals(mel_tokens, chord_tokens))

    return _build_histogram_stats(
        all_onset,
        all_lengths,
        all_harmonic,
        max_onset_bin=max_onset_bin,
        max_length_bin=max_length_bin,
    )


def compute_test_set_histograms(
    test_loader,
    id_to_token: dict[int, str],
    *,
    max_onset_bin: int = 16,
    max_length_bin: int = 32,
) -> HistogramStats:
    """Compute histograms for the ground-truth test set.

    This mirrors the logic in ``notebooks/dataset.ipynb`` but also computes
    melody–chord root harmonic intervals.

    Args:
        test_loader: DataLoader yielding ``(src, tgt)`` batches where ``src`` is
            a sequence of melody token IDs and ``tgt`` is a sequence of chord
            token IDs (in whatever ID spaces your loader uses).
        id_to_token: Mapping from token ID to string token. This can be a
            unified or per-modality mapping, as long as it is consistent with
            the IDs contained in ``src`` and ``tgt``.
        max_onset_bin: Maximum bin for onset-interval histogram (frames).
        max_length_bin: Maximum bin for chord-length histogram (frames).

    Returns:
        HistogramStats with raw values and normalized histograms.
    """

    def _decode(ids: Sequence[int]) -> list[str]:
        return [id_to_token.get(int(i), "<unk>") for i in ids]

    all_onset: list[int] = []
    all_lengths: list[int] = []
    all_harmonic: list[int] = []

    for src, tgt in test_loader:
        batch_size = src.size(0)
        for i in range(batch_size):
            mel_ids = src[i].cpu().tolist()
            chord_ids = tgt[i].cpu().tolist()

            mel_tokens = _decode(mel_ids)
            chord_tokens = _decode(chord_ids)

            all_onset.extend(onset_intervals(mel_tokens, chord_tokens))
            all_lengths.extend(chord_lengths(chord_tokens))
            all_harmonic.extend(melody_chord_root_intervals(mel_tokens, chord_tokens))

    return _build_histogram_stats(
        all_onset,
        all_lengths,
        all_harmonic,
        max_onset_bin=max_onset_bin,
        max_length_bin=max_length_bin,
    )
