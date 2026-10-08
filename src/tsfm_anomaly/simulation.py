"""Multivariate simulated series with five injected anomaly types.

Normal data: each channel has its own period, phase, and amplitude, plus two
shared periodic components, a slow linear trend, Gaussian noise, and two stable
cross-channel relationships (channel 1 follows channel 0, channel 3 is
anti-correlated with channel 2).

Anomalies: spike, level shift, variance change, frequency change, and a
correlation break. Every injector returns ``(x, labels)`` and leaves the input
unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

ANOMALY_TYPES = (
    "spike",
    "level_shift",
    "variance_change",
    "frequency_change",
    "correlation_anomaly",
)


def generate_normal_series(
    length: int,
    n_channels: int,
    seed: int,
    noise_std: float = 0.08,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)

    base_1 = np.sin(2 * np.pi * t / 80.0)
    base_2 = np.sin(2 * np.pi * t / 200.0 + 0.8)
    slow_trend = 0.00003 * t

    x = np.zeros((length, n_channels), dtype=np.float32)

    for channel in range(n_channels):
        period = 50 + 7 * channel
        phase = 0.3 * channel
        amplitude = 1.0 + 0.05 * channel

        seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)
        mixed = 0.35 * base_1 + 0.25 * base_2
        noise = rng.normal(0.0, noise_std, size=length).astype(np.float32)

        x[:, channel] = seasonal + mixed + slow_trend + noise

    # Add stable cross-channel relationships.
    if n_channels >= 2:
        x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]

    if n_channels >= 4:
        x[:, 3] = -0.5 * x[:, 2] + 0.5 * x[:, 3]

    return x.astype(np.float32)


def make_labels(length: int, start: int, end: int) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int64)
    labels[start:end] = 1
    return labels


def inject_spike(
    x: np.ndarray,
    start: int,
    length: int = 80,
    channels: Sequence[int] = (0,),
    amplitude: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    end = min(start + length, len(x))
    x[start:end, list(channels)] += amplitude
    return x.astype(np.float32), make_labels(len(x), start, end)


def inject_level_shift(
    x: np.ndarray,
    start: int,
    length: int = 3000,
    channels: Sequence[int] = (0, 1),
    shift: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    end = min(start + length, len(x))
    x[start:end, list(channels)] += shift
    return x.astype(np.float32), make_labels(len(x), start, end)


def inject_variance_change(
    x: np.ndarray,
    start: int,
    length: int = 3000,
    channels: Sequence[int] = (0, 1),
    noise_std: float = 1.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    end = min(start + length, len(x))
    rng = np.random.default_rng(seed)

    extra_noise = rng.normal(0.0, noise_std, size=(end - start, len(channels))).astype(np.float32)

    x[start:end, list(channels)] += extra_noise
    return x.astype(np.float32), make_labels(len(x), start, end)


def inject_frequency_change(
    x: np.ndarray,
    start: int,
    length: int = 3000,
    channels: Sequence[int] = (0, 1),
    new_period: float = 20.0,
    amplitude: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    end = min(start + length, len(x))
    local_t = np.arange(end - start, dtype=np.float32)

    new_wave = amplitude * np.sin(2 * np.pi * local_t / new_period)

    for channel in channels:
        x[start:end, channel] = new_wave + 0.05 * channel

    return x.astype(np.float32), make_labels(len(x), start, end)


def inject_correlation_anomaly(
    x: np.ndarray,
    start: int,
    length: int = 3000,
    target_channel: int = 1,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    end = min(start + length, len(x))
    rng = np.random.default_rng(seed)
    local_t = np.arange(end - start, dtype=np.float32)

    independent_signal = 1.2 * np.sin(2 * np.pi * local_t / 35.0 + 1.2) + rng.normal(
        0.0, 0.1, size=end - start
    )

    x[start:end, target_channel] = independent_signal.astype(np.float32)

    return x.astype(np.float32), make_labels(len(x), start, end)


def default_anomaly_params(seed: int) -> dict[str, dict[str, Any]]:
    """Original experiment settings; the random seeds derive from ``seed``."""
    return {
        "spike": {"length": 80, "channels": (0,), "amplitude": 5.0},
        "level_shift": {"length": 3000, "channels": (0, 1), "shift": 2.5},
        "variance_change": {
            "length": 3000,
            "channels": (0, 1),
            "noise_std": 1.0,
            "seed": seed + 10,
        },
        "frequency_change": {
            "length": 3000,
            "channels": (0, 1),
            "new_period": 20.0,
            "amplitude": 1.5,
        },
        "correlation_anomaly": {"length": 3000, "target_channel": 1, "seed": seed + 20},
    }


INJECTORS = {
    "spike": inject_spike,
    "level_shift": inject_level_shift,
    "variance_change": inject_variance_change,
    "frequency_change": inject_frequency_change,
    "correlation_anomaly": inject_correlation_anomaly,
}


def make_simulated_cases(
    normal_test: np.ndarray,
    seed: int,
    start_ratio: float = 0.35,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Inject each anomaly type into its own copy of ``normal_test``.

    By default every anomaly starts at ``start_ratio`` of the series. Any
    injector argument, including ``start``, can be overridden per type.
    """
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(INJECTORS))
    if unknown:
        raise ValueError(f"Unknown anomaly types: {unknown}")

    start = int(len(normal_test) * start_ratio)
    cases = {}
    for name, params in default_anomaly_params(seed).items():
        kwargs = {"start": start, **params, **overrides.get(name, {})}
        cases[name] = INJECTORS[name](normal_test, **kwargs)
    return cases


def generate_dataset(
    train_length: int = 20_000,
    test_length: int = 20_000,
    n_channels: int = 10,
    seed: int = 13,
    noise_std: float = 0.08,
    start_ratio: float = 0.35,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Normal training series and one labeled test case per anomaly type.

    Training and test series use different seeds, so the test noise is never
    seen during training.
    """
    x_train = generate_normal_series(train_length, n_channels, seed, noise_std)
    x_test_normal = generate_normal_series(test_length, n_channels, seed + 1, noise_std)
    cases = make_simulated_cases(x_test_normal, seed, start_ratio, overrides)
    return x_train, cases
