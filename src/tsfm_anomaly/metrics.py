"""Anomaly-detection metrics shared by every model in the benchmark.

Metrics are reported side by side, from lenient to strict:

- Adjusted (point-adjusted) best F1: one detected point counts the whole
  event as detected (Xu et al., WWW 2018). Known to overestimate.
- PA%K best F1: an event is adjusted only when at least K% of it is detected
  (Kim et al., AAAI 2022).
- Composite best F1: point-wise precision with event-wise recall
  (Garg et al., IEEE TNNLS 2022).
- Point-wise best F1 and AUPRC (scikit-learn): no adjustment at all.
- VUS-ROC / VUS-PR: range-aware areas under the curve (Paparrizos et al.,
  VLDB 2022), computed with the TSB-UAD library.

All best-F1 searches sweep thresholds in ascending order with predictions
``scores >= threshold`` and keep the first (lowest) threshold that reaches the
maximum F1. The searches are vectorized; for each threshold they compute the
same integer confusion counts and the same floating-point formulas as the
per-threshold loop in :func:`point_adjust_predicts` / :func:`pa_k_adjust`,
so results are identical to the reference loop implementation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-12


def binary_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous anomalous segments as half-open ``(start, end)`` pairs."""
    labels = np.asarray(labels).astype(bool).reshape(-1)
    changes = np.diff(np.pad(labels.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def point_confusion(labels: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    labels = np.asarray(labels).astype(bool).reshape(-1)
    pred = np.asarray(pred).astype(bool).reshape(-1)
    tp = int(np.sum(labels & pred))
    fp = int(np.sum(~labels & pred))
    fn = int(np.sum(labels & ~pred))
    precision, recall, f1 = _prf(tp, fp, fn)
    return float(precision), float(recall), float(f1)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return precision, recall, 2 * precision * recall / (precision + recall + EPS)


def _count_at_or_above(sorted_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Number of ``sorted_values >= t`` for every threshold ``t``."""
    return len(sorted_values) - np.searchsorted(sorted_values, thresholds, side="left")


def _prepare(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length.")
    return labels, scores


def _best_index(f1: np.ndarray) -> int:
    # np.argmax returns the first maximum, i.e. the lowest threshold.
    return int(np.argmax(f1))


def point_adjust_predicts(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Reference point adjustment: one detected point marks its whole event."""
    actual = np.asarray(labels).astype(bool).reshape(-1)
    adjusted = np.asarray(pred).astype(bool).reshape(-1).copy()
    for start, end in binary_events(actual):
        if adjusted[start:end].any():
            adjusted[start:end] = True
    return adjusted


def pa_k_adjust(labels: np.ndarray, pred: np.ndarray, k_percent: int) -> np.ndarray:
    """Reference PA%K: adjust an event only when at least K% of it is detected.

    K=0 keeps the traditional requirement of at least one detected point.
    """
    labels = np.asarray(labels).astype(bool).reshape(-1)
    adjusted = np.asarray(pred).astype(bool).reshape(-1).copy()
    for start, end in binary_events(labels):
        if int(adjusted[start:end].sum()) >= _pa_k_required(end - start, k_percent):
            adjusted[start:end] = True
    return adjusted


def _pa_k_required(event_length: int, k_percent: int) -> int:
    return 1 if k_percent == 0 else math.ceil(event_length * k_percent / 100.0)


def get_thresholds(scores: np.ndarray, mode: str = "unique", n_thresholds: int = 100) -> np.ndarray:
    scores = np.asarray(scores).astype(float).reshape(-1)
    if scores.size == 0:
        return np.array([], dtype=float)
    if mode == "unique":
        return np.unique(scores)
    if mode == "linear":
        return np.linspace(float(scores.min()), float(scores.max()), int(n_thresholds))
    if mode == "quantile":
        return np.quantile(scores, np.linspace(0.0, 1.0, int(n_thresholds)))
    raise ValueError(
        f"Unsupported threshold_mode: {mode}. Supported modes: unique, linear, quantile."
    )


def point_best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = _prepare(labels, scores)
    thresholds = np.unique(scores)
    tp = _count_at_or_above(np.sort(scores[labels]), thresholds)
    fp = _count_at_or_above(np.sort(scores[~labels]), thresholds)
    precision, recall, f1 = _prf(tp, fp, int(labels.sum()) - tp)
    best = _best_index(f1)
    return {
        "point_best_f1": float(f1[best]),
        "point_best_precision": float(precision[best]),
        "point_best_recall": float(recall[best]),
        "point_best_threshold": float(thresholds[best]),
    }


def adjusted_best_f1(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_mode: str = "unique",
    n_thresholds: int = 100,
) -> dict[str, float]:
    """Best F1 after point adjustment (PA)."""
    labels, scores = _prepare(labels, scores)
    thresholds = get_thresholds(scores, threshold_mode, n_thresholds)
    if thresholds.size == 0:
        return {
            "adjusted_best_f1": -1.0,
            "adjusted_best_precision": 0.0,
            "adjusted_best_recall": 0.0,
            "adjusted_best_threshold": 0.0,
        }

    tp = np.zeros(len(thresholds), dtype=np.int64)
    for start, end in binary_events(labels):
        tp += (end - start) * (scores[start:end].max() >= thresholds)
    fp = _count_at_or_above(np.sort(scores[~labels]), thresholds)
    precision, recall, f1 = _prf(tp, fp, int(labels.sum()) - tp)
    best = _best_index(f1)
    return {
        "adjusted_best_f1": float(f1[best]),
        "adjusted_best_precision": float(precision[best]),
        "adjusted_best_recall": float(recall[best]),
        "adjusted_best_threshold": float(thresholds[best]),
    }


def event_recall(labels: np.ndarray, pred: np.ndarray) -> float:
    events = binary_events(labels)
    if not events:
        return 0.0
    pred = np.asarray(pred).astype(bool).reshape(-1)
    return float(sum(bool(np.any(pred[start:end])) for start, end in events) / len(events))


def composite_best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Best F1 of point-wise precision and event-wise recall."""
    labels, scores = _prepare(labels, scores)
    thresholds = np.unique(scores)
    events = binary_events(labels)

    tp = _count_at_or_above(np.sort(scores[labels]), thresholds)
    fp = _count_at_or_above(np.sort(scores[~labels]), thresholds)
    precision = tp / (tp + fp + EPS)
    detected = np.zeros(len(thresholds), dtype=np.int64)
    for start, end in events:
        detected += scores[start:end].max() >= thresholds
    recall_event = detected / len(events) if events else np.zeros(len(thresholds))
    composite = 2 * precision * recall_event / (precision + recall_event + EPS)

    best = _best_index(composite)
    return {
        "composite_best_f1": float(composite[best]),
        "composite_point_precision": float(precision[best]),
        "composite_event_recall": float(recall_event[best]),
        "composite_best_threshold": float(thresholds[best]),
    }


def pa_k_best_f1(labels: np.ndarray, scores: np.ndarray, k_percent: int) -> dict[str, float]:
    labels, scores = _prepare(labels, scores)
    thresholds = np.unique(scores)

    tp = np.zeros(len(thresholds), dtype=np.int64)
    for start, end in binary_events(labels):
        event_scores = np.sort(scores[start:end])
        detected_points = _count_at_or_above(event_scores, thresholds)
        required = _pa_k_required(end - start, k_percent)
        # The event is adjusted when its `required`-th largest score passes.
        adjusted = event_scores[-required] >= thresholds
        tp += np.where(adjusted, end - start, detected_points)
    fp = _count_at_or_above(np.sort(scores[~labels]), thresholds)
    _, _, f1 = _prf(tp, fp, int(labels.sum()) - tp)

    best = _best_index(f1)
    return {
        f"pa_best_f1_k{k_percent}": float(f1[best]),
        f"pa_best_threshold_k{k_percent}": float(thresholds[best]),
    }


def vus_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold_count: int = 250
) -> dict[str, float]:
    # Imported here because TSB-UAD pulls in heavy dependencies (TensorFlow);
    # every other metric works without it.
    try:
        from TSB_UAD.vus.metrics import generate_curve
    except ImportError as exc:
        raise ImportError(
            "VUS metrics need TSB-UAD. Install with: pip install -e '.[vus]', "
            "or pass include_vus=False."
        ) from exc

    labels = np.asarray(labels).astype(np.int8).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    events = binary_events(labels)
    if not events:
        return {"vus_roc": float("nan"), "vus_pr": float("nan"), "vus_sliding_window": 0}

    sliding_window = max(1, 2 * max(end - start for start, end in events))
    *_, vus_roc, vus_pr = generate_curve(labels, scores, sliding_window, thre=int(threshold_count))
    return {
        "vus_roc": float(vus_roc),
        "vus_pr": float(vus_pr),
        "vus_sliding_window": int(sliding_window),
    }


def evaluate_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_mode: str = "unique",
    adjusted_threshold_count: int = 100,
    vus_threshold_count: int = 250,
    pa_k_values: Sequence[int] = (0, 10, 20, 50),
    include_vus: bool = True,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(np.int8).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Both normal and anomalous labels are required.")

    result: dict[str, float] = {}
    result.update(adjusted_best_f1(labels, scores, threshold_mode, adjusted_threshold_count))
    result.update(point_best_f1(labels, scores))
    result.update(composite_best_f1(labels, scores))
    result["auprc"] = float(average_precision_score(labels, scores))
    if include_vus:
        result.update(vus_metrics(labels, scores, vus_threshold_count))
    for k_percent in pa_k_values:
        result.update(pa_k_best_f1(labels, scores, k_percent))
    return result
