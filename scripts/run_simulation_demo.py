#!/usr/bin/env python3
"""Quick demo on simulated data.

By default this runs on CPU in seconds and needs no model: it generates the
normal training series and the five anomaly cases, standardizes them with
statistics from the training series only, prints an overview, and saves a
figure of every anomaly type.

With ``--moment`` it also scores each case with MOMENT0 (zero-shot) and
MOMENTLP (head trained on the normal series) and prints the metrics. This
needs ``pip install -e '.[moment]'``; a GPU is recommended.

Usage:
    python scripts/run_simulation_demo.py
    python scripts/run_simulation_demo.py --moment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsfm_anomaly.config import EvaluationConfig, MomentConfig, load_config  # noqa: E402
from tsfm_anomaly.preprocess import TrainScaler, split_normal_train_validation  # noqa: E402
from tsfm_anomaly.simulation import generate_dataset  # noqa: E402

TITLES = {
    "spike": "Spike",
    "level_shift": "Level shift",
    "variance_change": "Variance change",
    "frequency_change": "Frequency change",
    "correlation_anomaly": "Correlation break (ch 1 decouples from ch 0)",
}
CHANNEL_COLORS = ("#2a78d6", "#eb6834")
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--figure", type=Path, help="figure path (default: <output_dir>/...)")
    parser.add_argument("--moment", action="store_true", help="also run MOMENT0 / MOMENTLP")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def overview(cases: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for name, (_, labels) in cases.items():
        anomalous = np.flatnonzero(labels)
        rows.append(
            {
                "anomaly_type": name,
                "start": int(anomalous[0]),
                "end": int(anomalous[-1]) + 1,
                "anomaly_points": int(labels.sum()),
                "anomaly_ratio": float(labels.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_cases(cases, figure_path: Path, margin: int = 600) -> None:
    """One panel per anomaly type: channels 0 and 1 around the anomaly onset."""
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 2.0 * len(cases)), sharex=False)
    fig.patch.set_facecolor("white")
    for ax, (name, (x, labels)) in zip(axes, cases.items(), strict=True):
        start = int(np.flatnonzero(labels)[0])
        lo, hi = max(0, start - margin), min(len(labels), start + margin)
        t = np.arange(lo, hi)
        in_window = labels[lo:hi].astype(bool)
        ax.fill_between(
            t,
            0,
            1,
            where=in_window,
            transform=ax.get_xaxis_transform(),
            color=BAND,
            linewidth=0,
            label="anomaly interval",
        )
        for channel, color in enumerate(CHANNEL_COLORS):
            ax.plot(t, x[lo:hi, channel], color=color, linewidth=1.0, label=f"channel {channel}")
        ax.set_title(TITLES[name], loc="left", fontsize=10, color=INK)
        ax.set_xlim(lo, hi - 1)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[-1].set_xlabel("time step", color=MUTED, fontsize=9)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Simulated anomalies (standardized, 2 of 10 channels)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def run_moment(x_train, cases, sim, moment_cfg, eval_cfg, device_name, out: Path) -> pd.DataFrame:
    from tsfm_anomaly.models.moment import (
        evaluate_method,
        load_moment,
        make_loaders,
        train_momentlp,
    )
    from tsfm_anomaly.utils import get_device, set_seed

    set_seed(sim.seed)
    device = get_device(device_name)
    model = load_moment(moment_cfg, device)
    rows = []
    for name, (x, labels) in cases.items():
        rows.append(
            {
                "anomaly_type": name,
                **evaluate_method("MOMENT0", model, x, labels, moment_cfg, eval_cfg, device),
            }
        )

    x_fit, x_val = split_normal_train_validation(x_train, sim.validation_ratio, moment_cfg.seq_len)
    train_loader, val_loader = make_loaders(x_fit, x_val, moment_cfg, device)
    model, _, _ = train_momentlp(
        model, train_loader, val_loader, moment_cfg, device, out / "checkpoints"
    )
    for name, (x, labels) in cases.items():
        rows.append(
            {
                "anomaly_type": name,
                **evaluate_method("MOMENTLP", model, x, labels, moment_cfg, eval_cfg, device),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sim = cfg["simulation"]
    out = Path(sim.output_dir)

    x_train_raw, cases_raw = generate_dataset(
        train_length=sim.train_length,
        test_length=sim.test_length,
        n_channels=sim.n_channels,
        seed=sim.seed,
        noise_std=sim.noise_std,
        start_ratio=sim.anomaly_start_ratio,
        overrides=sim.anomalies,
    )
    # Statistics come from the normal training series only.
    scaler = TrainScaler("standard").fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    cases = {name: (scaler.transform(x), labels) for name, (x, labels) in cases_raw.items()}

    print(f"train: {x_train.shape}, test cases: {len(cases)} x {cases['spike'][0].shape}")
    print(overview(cases).to_string(index=False))

    figure = args.figure or out / "simulation_anomalies.png"
    plot_cases(cases, figure)
    print(f"figure: {figure}")

    if args.moment:
        moment_cfg = cfg.get("moment", MomentConfig())
        eval_cfg = cfg.get("evaluation", EvaluationConfig())
        results = run_moment(x_train, cases, sim, moment_cfg, eval_cfg, args.device, out)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "simulation_metrics.csv", index=False)
        columns = ["anomaly_type", "method", "adjusted_best_f1", "pa_best_f1_k50", "auprc"]
        print(results[columns].to_string(index=False))
        print(f"metrics: {out / 'simulation_metrics.csv'}")


if __name__ == "__main__":
    main()
