"""Summaries of the per-dataset metric table.

Means alone hide where a model fails, so every summary is also broken down by
the UCR category (ORIGINAL / NOISE / DISTORTED), and two methods evaluated on
the same datasets are compared dataset by dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_COLUMNS = [
    "adjusted_best_f1",
    "pa_best_f1_k0",
    "pa_best_f1_k10",
    "pa_best_f1_k20",
    "pa_best_f1_k50",
    "composite_best_f1",
    "point_best_f1",
    "auprc",
    "vus_roc",
    "vus_pr",
]


def _metrics(frame: pd.DataFrame) -> list[str]:
    return [column for column in METRIC_COLUMNS if column in frame.columns]


def overall_summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("method")[_metrics(results)].agg(["mean", "median", "std"])


def category_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["method", "category"])
    summary = grouped[_metrics(results)].agg(["mean", "median"])
    summary[("dataset_count", "")] = grouped.size()
    return summary


def paired_comparison(results: pd.DataFrame, base: str, other: str) -> pd.DataFrame:
    """Per-dataset ``other - base`` differences: wins, ties, losses, mean, median."""
    left = results[results["method"] == base].set_index("dataset_id")
    right = results[results["method"] == other].set_index("dataset_id")
    common = left.index.intersection(right.index)
    rows = []
    for metric in _metrics(results):
        diff = right.loc[common, metric] - left.loc[common, metric]
        rows.append(
            {
                "metric": metric,
                "comparison": f"{other} - {base}",
                "datasets": len(common),
                "wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()),
                "mean_delta": float(diff.mean()),
                "median_delta": float(diff.median()),
            }
        )
    return pd.DataFrame(rows)


def write_summaries(results_csv: Path, output_dir: Path) -> None:
    if not results_csv.exists():
        return
    results = pd.read_csv(results_csv).drop_duplicates()
    if results.empty:
        return
    overall_summary(results).to_csv(output_dir / "summary_overall.csv")
    category_summary(results).to_csv(output_dir / "summary_by_category.csv")
    methods = set(results["method"])
    if {"MOMENT0", "MOMENTLP"} <= methods:
        paired_comparison(results, "MOMENT0", "MOMENTLP").to_csv(
            output_dir / "comparison_momentlp_vs_moment0.csv", index=False
        )
