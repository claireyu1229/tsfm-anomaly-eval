"""No future information leaks into training, and windows cover every point."""

import numpy as np
import pytest

from tsfm_anomaly import preprocess as P
from tsfm_anomaly import windows as W

# id 7, train_end 600, anomaly 800..819 (inclusive)
FILENAME = "007_UCR_Anomaly_toy_600_800_819.txt"


@pytest.fixture
def ucr_file(tmp_path):
    rng = np.random.default_rng(0)
    series = np.sin(np.arange(1000) / 10.0) + 0.01 * rng.standard_normal(1000)
    series[800:820] += 5.0
    path = tmp_path / FILENAME
    np.savetxt(path, series)
    return path


def test_filename_metadata():
    meta = P.parse_ucr_filename(FILENAME)
    assert (meta.dataset_id, meta.dataset_name) == (7, "toy")
    assert (meta.train_end, meta.anomaly_start, meta.anomaly_end) == (600, 800, 819)


def test_anomaly_inside_training_segment_is_rejected():
    with pytest.raises(ValueError):
        P.parse_ucr_filename("008_UCR_Anomaly_toy_600_500_550.txt")


def test_split_keeps_anomaly_out_of_training(ucr_file):
    series = P.read_ucr_series(ucr_file)
    meta = P.parse_ucr_filename(ucr_file)
    train, test, labels = P.build_raw_ucr_split(series, meta)
    assert len(train) == 600 and len(train) + len(test) == len(series)
    assert np.array_equal(np.concatenate([train, test]), series)
    # Inclusive end: points 800..819 -> 20 test positions starting at 200.
    assert np.flatnonzero(labels).tolist() == list(range(200, 220))
    assert train.max() < series[800:820].min()


def test_pooled_labels_keep_any_anomalous_point(ucr_file):
    data = P.load_ucr(ucr_file, downsample_factor=10)
    assert data["train"].shape == (60, 1) and data["test"].shape == (40, 1)
    assert np.flatnonzero(data["labels"]).tolist() == [20, 21]
    assert data["labels"].sum() >= 1


def test_scaler_uses_training_statistics_only():
    train = np.arange(10, dtype=np.float32)[:, None]
    test = np.full((5, 1), 1000.0, dtype=np.float32)
    scaler = P.TrainScaler("standard").fit(train)
    before = (scaler.offset.copy(), scaler.scale.copy())
    scaled_test = scaler.transform(test)
    assert np.allclose(scaler.offset, train.mean()) and np.allclose(scaler.scale, train.std())
    assert np.array_equal(before[0], scaler.offset) and np.array_equal(before[1], scaler.scale)
    # Test values far outside the training range stay far outside after scaling.
    assert scaled_test.min() > 100


def test_validation_split_is_chronological():
    x = np.arange(2000, dtype=np.float32)[:, None]
    train, val = P.split_normal_train_validation(x, validation_ratio=0.2, window_size=512)
    assert train[-1, 0] < val[0, 0]  # validation comes strictly after training
    assert np.array_equal(np.concatenate([train, val]), x)
    assert len(train) >= 512 and len(val) >= 512


def test_short_series_reuses_the_whole_segment():
    x = np.arange(600, dtype=np.float32)
    train, val = P.split_normal_train_validation(x, validation_ratio=0.2, window_size=512)
    assert np.array_equal(train, val) and len(train) == 600


@pytest.mark.parametrize("length", [1, 100, 512, 513, 1500, 2048])
def test_tail_covering_windows_cover_every_point(length):
    starts = W.tail_covering_starts(length, seq_len=512, stride=512)
    covered = np.zeros(length, dtype=int)
    for start in starts:
        covered[start : start + 512] += 1
    assert covered.min() >= 1
    assert starts[-1] == max(0, length - 512)  # the last window ends at the last point


def test_padding_mask_marks_real_points_only():
    window, mask = W.pad_window(np.ones((100, 2), dtype=np.float32), seq_len=512)
    assert window.shape == (512, 2) and mask.sum() == 100 and mask[:100].all()


def test_window_scores_average_back_to_the_timeline():
    values = np.arange(700, dtype=np.float32)[:, None]
    windows, masks, starts = W.tail_windows(values, seq_len=512, stride=512)
    restored = W.aggregate_window_scores(windows[:, :, 0], masks, starts, len(values))
    assert np.allclose(restored, values[:, 0])
