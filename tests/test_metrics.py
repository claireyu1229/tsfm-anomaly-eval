"""Metric correctness: vectorized searches vs. a reference loop, and hand-checked cases."""

import numpy as np
import pytest

from tsfm_anomaly import metrics as M


def _loop_best(labels, scores, thresholds, adjust):
    best = (-1.0, 0.0, 0.0, 0.0)
    for threshold in thresholds:
        pred = adjust(scores >= threshold)
        precision, recall, f1 = M.point_confusion(labels, pred)
        if f1 > best[0]:
            best = (f1, precision, recall, float(threshold))
    return best


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 300))
    labels = np.zeros(n, dtype=np.int8)
    for _ in range(int(rng.integers(1, 4))):
        start = int(rng.integers(0, n - 5))
        labels[start : start + int(rng.integers(1, 15))] = 1
    # Integer scores exercise ties; continuous scores exercise ordering.
    if seed % 2:
        scores = rng.integers(0, int(rng.integers(3, 40)), n).astype(float)
    else:
        scores = rng.random(n)
    scores[labels == 1] += rng.random() * rng.integers(0, 3)
    return labels, scores


@pytest.mark.parametrize("seed", range(60))
def test_vectorized_searches_match_reference_loop(seed):
    labels, scores = _random_case(seed)
    unique = np.unique(scores)

    expected = _loop_best(labels, scores, unique, lambda pred: pred)
    assert tuple(M.point_best_f1(labels, scores).values()) == expected

    for mode in ("unique", "linear", "quantile"):
        thresholds = M.get_thresholds(scores, mode, 37)
        expected = _loop_best(
            labels, scores, thresholds, lambda pred: M.point_adjust_predicts(labels, pred)
        )
        assert tuple(M.adjusted_best_f1(labels, scores, mode, 37).values()) == expected

    for k in (0, 10, 20, 50):
        f1, _, _, threshold = _loop_best(
            labels, scores, unique, lambda pred, k=k: M.pa_k_adjust(labels, pred, k)
        )
        assert M.pa_k_best_f1(labels, scores, k) == {
            f"pa_best_f1_k{k}": f1,
            f"pa_best_threshold_k{k}": threshold,
        }

    best = (-1.0, 0.0, 0.0, 0.0)
    for threshold in unique:
        pred = scores >= threshold
        precision, _, _ = M.point_confusion(labels, pred)
        recall = M.event_recall(labels, pred)
        f1 = 2 * precision * recall / (precision + recall + M.EPS)
        if f1 > best[0]:
            best = (f1, precision, recall, float(threshold))
    assert tuple(M.composite_best_f1(labels, scores).values()) == best


def test_point_adjustment_marks_whole_detected_event():
    labels = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert M.point_adjust_predicts(labels, pred).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert M.pa_k_adjust(labels, pred, 50).tolist() == [1, 0, 1, 0, 0, 0, 0, 0]
    assert M.pa_k_adjust(labels, pred, 30).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


def test_binary_events_are_half_open_intervals():
    assert M.binary_events(np.array([1, 1, 0, 0, 1, 0, 1])) == [(0, 2), (4, 5), (6, 7)]


def _one_hit_case():
    # A 3-point event where only one point gets a high score.
    labels = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.1)
    scores[2] = 0.9
    return labels, scores


def test_lenient_and_strict_metrics_disagree_on_a_partial_detection():
    labels, scores = _one_hit_case()
    # Threshold 0.9 flags 1 of 3 anomalous points and nothing else:
    # precision 1, recall 1/3 -> F1 0.5. Flagging everything gives F1 6/13.
    assert M.point_best_f1(labels, scores)["point_best_f1"] == pytest.approx(0.5)
    # Point adjustment counts the whole event as found -> F1 1.
    assert M.adjusted_best_f1(labels, scores)["adjusted_best_f1"] == pytest.approx(1.0)
    # PA%K with K=50 needs 2 of 3 points, so no adjustment happens.
    assert M.pa_k_best_f1(labels, scores, 50)["pa_best_f1_k50"] == pytest.approx(0.5)
    # Composite: point precision 1, event recall 1.
    assert M.composite_best_f1(labels, scores)["composite_best_f1"] == pytest.approx(1.0)


def test_auprc_extremes():
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    perfect = labels * 1.0 + 0.01
    result = M.evaluate_all_metrics(labels, perfect, include_vus=False)
    assert result["auprc"] == pytest.approx(1.0)
    # Constant scores carry no information: AUPRC equals the anomaly ratio.
    constant = M.evaluate_all_metrics(labels, np.ones(8), include_vus=False)
    assert constant["auprc"] == pytest.approx(labels.mean())


def test_evaluate_all_metrics_validates_input():
    labels, scores = _one_hit_case()
    result = M.evaluate_all_metrics(labels, scores, include_vus=False)
    assert "vus_pr" not in result and "pa_best_f1_k50" in result
    with pytest.raises(ValueError):
        M.evaluate_all_metrics(labels, np.where(scores > 0.5, np.nan, scores), include_vus=False)
    with pytest.raises(ValueError):
        M.evaluate_all_metrics(np.zeros(10), scores, include_vus=False)
    with pytest.raises(ValueError):
        M.evaluate_all_metrics(labels[:-1], scores, include_vus=False)


def test_vus_metrics_with_tsb_uad():
    pytest.importorskip("TSB_UAD")
    labels = np.zeros(400, dtype=int)
    labels[200:220] = 1
    scores = np.random.default_rng(0).random(400) * 0.1 + labels
    result = M.vus_metrics(labels, scores)
    assert 0.0 <= result["vus_pr"] <= 1.0 and 0.0 <= result["vus_roc"] <= 1.0
    assert result["vus_sliding_window"] == 40
