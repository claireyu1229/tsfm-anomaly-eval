"""Shapes, labels, and seed reproducibility of the simulated data."""

import numpy as np
import pytest

from tsfm_anomaly import simulation as S


@pytest.fixture(scope="module")
def dataset():
    return S.generate_dataset(train_length=4000, test_length=5000, n_channels=10, seed=13)


def test_shapes_and_dtypes(dataset):
    x_train, cases = dataset
    assert x_train.shape == (4000, 10) and x_train.dtype == np.float32
    assert list(cases) == list(S.ANOMALY_TYPES)
    for x, labels in cases.values():
        assert x.shape == (5000, 10) and x.dtype == np.float32
        assert labels.shape == (5000,) and set(np.unique(labels)) == {0, 1}


def test_labels_mark_the_configured_interval(dataset):
    _, cases = dataset
    start = int(5000 * 0.35)
    expected_lengths = {"spike": 80}
    for name, (_, labels) in cases.items():
        anomalous = np.flatnonzero(labels)
        length = expected_lengths.get(name, 3000)
        assert anomalous[0] == start and anomalous[-1] == start + length - 1
        assert labels.sum() == length  # one contiguous interval


def test_same_seed_reproduces_and_other_seed_differs():
    a_train, a_cases = S.generate_dataset(2000, 3000, 4, seed=1)
    b_train, b_cases = S.generate_dataset(2000, 3000, 4, seed=1)
    c_train, _ = S.generate_dataset(2000, 3000, 4, seed=2)
    assert np.array_equal(a_train, b_train)
    for name in a_cases:
        assert np.array_equal(a_cases[name][0], b_cases[name][0])
    assert not np.array_equal(a_train, c_train)


def test_train_and_test_use_different_noise():
    x_train, cases = S.generate_dataset(3000, 3000, 2, seed=5)
    x_test, labels = cases["spike"]
    normal = labels == 0
    assert not np.array_equal(x_train[normal], x_test[normal])


def test_injection_changes_only_the_interval_and_channels():
    normal = S.generate_normal_series(3000, 4, seed=0)
    x, labels = S.inject_level_shift(normal, start=1000, length=500, channels=(0, 1), shift=2.5)
    inside = labels.astype(bool)
    assert np.allclose(x[inside][:, :2] - normal[inside][:, :2], 2.5)
    assert np.array_equal(x[~inside], normal[~inside])
    assert np.array_equal(x[:, 2:], normal[:, 2:])
    # The input is never modified in place.
    assert np.array_equal(normal, S.generate_normal_series(3000, 4, seed=0))


def test_overrides_set_position_length_and_strength():
    normal = S.generate_normal_series(4000, 2, seed=0)
    cases = S.make_simulated_cases(
        normal, seed=0, overrides={"spike": {"start": 100, "length": 10, "amplitude": 9.0}}
    )
    x, labels = cases["spike"]
    assert np.flatnonzero(labels).tolist() == list(range(100, 110))
    assert np.allclose(x[100:110, 0] - normal[100:110, 0], 9.0)


def test_unknown_anomaly_type_is_rejected():
    with pytest.raises(ValueError):
        S.make_simulated_cases(np.zeros((100, 2), dtype=np.float32), 0, overrides={"spikes": {}})


def test_cross_channel_relationship_holds_in_normal_data():
    x = S.generate_normal_series(5000, 4, seed=0)
    assert np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.9
    assert np.corrcoef(x[:, 2], x[:, 3])[0, 1] < 0.5
