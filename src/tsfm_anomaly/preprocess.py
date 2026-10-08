"""UCR filename parsing, loading, max pooling, labels, scaling, and splits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

UCR_FILENAME_PATTERN = re.compile(
    r"^(?P<dataset_id>\d+)_UCR_Anomaly_"
    r"(?P<dataset_name>.+)_"
    r"(?P<train_end>\d+)_"
    r"(?P<anomaly_start>\d+)_"
    r"(?P<anomaly_end>\d+)\.txt$"
)


@dataclass(frozen=True)
class UCRMetadata:
    dataset_id: int
    dataset_name: str
    train_end: int
    anomaly_start: int
    anomaly_end: int
    filename: str


def parse_ucr_filename(path: str | Path) -> UCRMetadata:
    path = Path(path)
    match = UCR_FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Filename does not match the UCR anomaly format: {path.name}")

    values = match.groupdict()
    metadata = UCRMetadata(
        dataset_id=int(values["dataset_id"]),
        dataset_name=values["dataset_name"],
        train_end=int(values["train_end"]),
        anomaly_start=int(values["anomaly_start"]),
        anomaly_end=int(values["anomaly_end"]),
        filename=path.name,
    )

    if metadata.train_end <= 0:
        raise ValueError(f"Invalid train_end in {path.name}.")
    if metadata.anomaly_start < metadata.train_end:
        raise ValueError(f"Anomaly starts inside the training segment in {path.name}.")
    if metadata.anomaly_end < metadata.anomaly_start:
        raise ValueError(f"Invalid anomaly interval in {path.name}.")
    return metadata


def infer_ucr_category(
    dataset_name: str,
    category_groups: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Return an explicit override or the documented UCR name-prefix category."""
    for category, names in (category_groups or {}).items():
        if dataset_name in names:
            return str(category).upper()
    upper_name = dataset_name.upper()
    if upper_name.startswith("DISTORTED"):
        return "DISTORTED"
    if upper_name.startswith("NOISE"):
        return "NOISE"
    return "ORIGINAL"


def infer_ucr_base_family(dataset_name: str) -> str:
    """Group ORIGINAL/NOISE/DISTORTED variants without guessing further lineage."""
    upper_name = dataset_name.upper()
    for prefix in ("DISTORTED", "NOISE"):
        if upper_name.startswith(prefix):
            return dataset_name[len(prefix) :]
    return dataset_name


def discover_ucr_files(
    root: str | Path,
    file_glob: str = "*.txt",
    max_datasets: int | None = None,
) -> list[Path]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"UCR directory does not exist: {root.resolve()}")

    files = sorted(root.glob(file_glob), key=lambda p: parse_ucr_filename(p).dataset_id)
    if not files:
        raise FileNotFoundError(f"No files matching {file_glob!r} under {root.resolve()}")
    return files if max_datasets is None else files[:max_datasets]


def read_ucr_series(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        values = np.loadtxt(path, dtype=np.float32)
    except Exception:
        # Fallback for comma-separated files.
        values = np.loadtxt(path, dtype=np.float32, delimiter=",")

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Empty UCR file: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path.name}.")
    return values


def build_raw_ucr_split(
    series: np.ndarray,
    metadata: UCRMetadata,
    anomaly_end_inclusive: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metadata.train_end >= len(series):
        raise ValueError(
            f"train_end={metadata.train_end} is outside a series of length {len(series)}."
        )
    if metadata.anomaly_end >= len(series):
        raise ValueError(
            f"anomaly_end={metadata.anomaly_end} is outside a series of length {len(series)}."
        )

    train = series[: metadata.train_end].copy()
    test = series[metadata.train_end :].copy()
    labels = np.zeros(len(test), dtype=np.int64)

    relative_start = max(metadata.anomaly_start - metadata.train_end, 0)
    relative_end = metadata.anomaly_end - metadata.train_end
    stop = min(relative_end + 1 if anomaly_end_inclusive else relative_end, len(labels))
    if relative_start >= stop:
        raise ValueError(f"Empty anomaly interval after splitting {metadata.filename}.")

    labels[relative_start:stop] = 1
    return train, test, labels


def non_overlapping_pool(
    values: np.ndarray,
    factor: int,
    mode: str = "max",
) -> tuple[np.ndarray, int]:
    values = np.asarray(values)
    if factor <= 1:
        return values.copy(), 0

    usable = (len(values) // factor) * factor
    dropped = len(values) - usable
    if usable == 0:
        raise ValueError(f"Length {len(values)} is shorter than pooling factor {factor}.")

    reshaped = values[:usable].reshape(-1, factor)
    if mode == "max":
        pooled = reshaped.max(axis=1)
    elif mode == "mean":
        pooled = reshaped.mean(axis=1)
    else:
        raise ValueError("pooling mode must be 'max' or 'mean'.")
    return pooled.astype(values.dtype, copy=False), dropped


def pool_binary_labels(labels: np.ndarray, factor: int) -> tuple[np.ndarray, int]:
    pooled, dropped = non_overlapping_pool(np.asarray(labels, dtype=np.int64), factor, "max")
    return pooled.astype(np.int64), dropped


def load_ucr(
    path: str | Path,
    downsample_factor: int = 10,
    pooling_mode: str = "max",
    anomaly_end_inclusive: bool = True,
) -> dict[str, object]:
    """Load one UCR file as pooled ``[time, 1]`` train/test arrays and test labels.

    Values use ``pooling_mode``; labels are always max-pooled so that any
    anomalous raw point marks its pooled point as anomalous.
    """
    path = Path(path)
    metadata = parse_ucr_filename(path)
    series = read_ucr_series(path)
    train_raw, test_raw, labels_raw = build_raw_ucr_split(series, metadata, anomaly_end_inclusive)

    train, train_pool_tail = non_overlapping_pool(train_raw, downsample_factor, pooling_mode)
    test, test_pool_tail = non_overlapping_pool(test_raw, downsample_factor, pooling_mode)
    labels, label_pool_tail = pool_binary_labels(labels_raw, downsample_factor)

    common_length = min(len(test), len(labels))
    return {
        "metadata": metadata,
        "series_length_raw": int(len(series)),
        "train_length_raw": int(len(train_raw)),
        "test_length_raw": int(len(test_raw)),
        "train": train.reshape(-1, 1).astype(np.float32),
        "test": test[:common_length].reshape(-1, 1).astype(np.float32),
        "labels": labels[:common_length],
        "train_pool_tail": int(train_pool_tail),
        "test_pool_tail": int(test_pool_tail),
        "label_pool_tail": int(label_pool_tail),
    }


class TrainScaler:
    """Scaler fitted on normal training data only; never on validation or test."""

    def __init__(self, mode: str = "standard", eps: float = 1e-6):
        if mode not in {"standard", "minmax"}:
            raise ValueError("scaler mode must be 'standard' or 'minmax'.")
        self.mode = mode
        self.eps = eps
        self.offset: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TrainScaler":
        if self.mode == "standard":
            offset = x.mean(axis=0, keepdims=True)
            scale = x.std(axis=0, keepdims=True)
        else:
            offset = x.min(axis=0, keepdims=True)
            scale = x.max(axis=0, keepdims=True) - offset
        self.offset = offset.astype(np.float32)
        self.scale = np.maximum(scale.astype(np.float32), self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.offset is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted.")
        return ((x - self.offset) / self.scale).astype(np.float32)


def split_normal_train_validation(
    x: np.ndarray,
    validation_ratio: float,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the normal UCR training segment chronologically.

    Long sequences (>= 2 * window_size) get a non-overlapping split that keeps
    at least one full window on each side. Shorter sequences reuse the complete
    normal segment for both training and validation, so validation loss is a
    checkpoint-selection signal rather than an independent estimate.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must have shape [time, channels].")
    if len(x) < 2:
        raise ValueError("Normal training segment must contain at least two points.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if len(x) >= 2 * window_size:
        split = int(round(len(x) * (1.0 - validation_ratio)))
        split = min(max(window_size, split), len(x) - window_size)
        return x[:split].copy(), x[split:].copy()
    return x.copy(), x.copy()
