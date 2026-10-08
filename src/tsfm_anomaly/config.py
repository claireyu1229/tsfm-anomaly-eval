"""Typed experiment settings loaded from the YAML files in ``configs/``.

Each YAML section maps to one dataclass. Unknown keys raise an error so that a
typo cannot silently fall back to a default value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UCRConfig:
    ucr_root: str = "data/UCR_Anomaly_FullData"
    file_glob: str = "*.txt"
    anomaly_end_inclusive: bool = True
    downsample_factor: int = 10
    pooling_mode: str = "max"
    standardize: bool = True
    validation_ratio: float = 0.20
    max_datasets: int | None = None
    seed: int = 13
    device: str = "auto"
    resume: bool = True
    fail_fast: bool = False
    output_dir: str = "outputs/ucr"


@dataclass
class EvaluationConfig:
    threshold_mode: str = "unique"
    adjusted_threshold_count: int = 100
    vus_threshold_count: int = 250
    pa_k_values: tuple[int, ...] = (0, 10, 20, 50)
    include_vus: bool = True


@dataclass
class MomentConfig:
    model_id: str = "AutonLab/MOMENT-1-large"
    seq_len: int = 512
    patch_len: int = 8
    patch_stride: int = 8
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.05
    min_learning_rate: float = 1e-5
    warmup_learning_rate: float = 1e-5
    official_warmup_steps: int = 1000
    scaled_warmup_ratio: float = 0.10
    gradient_clip_norm: float = 5.0
    use_bfloat16_on_cuda: bool = True
    max_epochs: int = 20
    batch_size: int = 16
    train_stride: int = 512
    eval_stride: int = 512
    num_workers: int = 0
    run_zero_shot_baseline: bool = True


@dataclass
class TranADConfig:
    window: int = 10
    epochs: int = 5
    learning_rate: float = 0.006
    batch_size: int = 128


@dataclass
class SimulationConfig:
    seed: int = 13
    train_length: int = 20_000
    test_length: int = 20_000
    n_channels: int = 10
    noise_std: float = 0.08
    anomaly_start_ratio: float = 0.35
    validation_ratio: float = 0.20
    output_dir: str = "outputs/simulation"
    # Per-anomaly overrides of position, length, channels, and strength.
    anomalies: dict[str, dict[str, Any]] = field(default_factory=dict)


def _build(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    if "pa_k_values" in values:
        values["pa_k_values"] = tuple(values["pa_k_values"])
    return cls(**values)


SECTIONS = {
    "ucr": UCRConfig,
    "evaluation": EvaluationConfig,
    "moment": MomentConfig,
    "tranad": TranADConfig,
    "simulation": SimulationConfig,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Return ``{section_name: dataclass}`` for every section in the YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {unknown}")
    return {name: _build(SECTIONS[name], values) for name, values in raw.items()}
