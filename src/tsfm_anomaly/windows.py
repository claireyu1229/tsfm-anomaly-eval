"""Tail-covering windows with edge padding and point-wise score aggregation."""

from __future__ import annotations

import numpy as np


def tail_covering_starts(length: int, seq_len: int, stride: int) -> list[int]:
    """Regular window starts plus one final window ending at the last sample.

    Sequences no longer than ``seq_len`` produce a single window at 0, which
    callers pad with :func:`pad_window`.
    """
    if length <= 0:
        raise ValueError("Series must contain at least one point.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    if length <= seq_len:
        return [0]

    starts = list(range(0, length - seq_len + 1, stride))
    last_start = length - seq_len
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def pad_window(window: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Edge-pad a ``[time, channels]`` window to ``seq_len``.

    Returns the padded window and a ``[seq_len]`` mask with 1 for real
    observations and 0 for padding.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("window must have shape [time, channels].")

    real_length = len(window)
    if real_length <= 0:
        raise ValueError("Cannot pad an empty window.")
    if real_length > seq_len:
        raise ValueError(f"Window length {real_length} exceeds seq_len {seq_len}.")

    mask = np.zeros(seq_len, dtype=np.float32)
    mask[:real_length] = 1.0
    if real_length == seq_len:
        return window.copy(), mask

    padded = np.pad(window, ((0, seq_len - real_length), (0, 0)), mode="edge")
    return padded.astype(np.float32), mask


def tail_windows(
    values: np.ndarray,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack padded tail-covering windows, their masks, and their starts."""
    values = np.asarray(values, dtype=np.float32)
    starts = tail_covering_starts(len(values), seq_len, stride)
    windows, masks = [], []
    for start in starts:
        window, mask = pad_window(values[start : start + seq_len], seq_len)
        windows.append(window)
        masks.append(mask)
    return np.stack(windows), np.stack(masks), starts


def aggregate_window_scores(
    scores: np.ndarray,
    masks: np.ndarray,
    starts: list[int],
    length: int,
) -> np.ndarray:
    """Average overlapping per-window values back onto the original timeline."""
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for score, mask, start in zip(scores, masks, starts, strict=True):
        real = min(int(mask.sum()), length - start)
        total[start : start + real] += score[:real]
        count[start : start + real] += 1
    if np.any(count == 0):
        raise RuntimeError("Window aggregation left uncovered points")
    return total / count
