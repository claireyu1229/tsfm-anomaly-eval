#!/usr/bin/env python3
"""Run MOMENT0/MOMENTLP or TranAD on every UCR series under one protocol.

Every model shares the same file parsing, train/test boundary, max pooling,
inclusive labels, chronological validation split, and metrics. Only the
model itself and its native scaling (z-score for MOMENT, min-max for TranAD,
both fitted on the normal training segment only) differ.

Each finished dataset is appended to ``metrics.csv`` immediately, so an
interrupted run resumes where it stopped.

Usage:
    python scripts/run_ucr_benchmark.py --config configs/ucr_moment.yaml
    python scripts/run_ucr_benchmark.py --config configs/ucr_tranad.yaml --max-datasets 3
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from tsfm_anomaly.config import EvaluationConfig, load_config
from tsfm_anomaly.metrics import evaluate_all_metrics
from tsfm_anomaly.preprocess import (
    TrainScaler,
    discover_ucr_files,
    infer_ucr_base_family,
    infer_ucr_category,
    load_ucr,
    parse_ucr_filename,
    split_normal_train_validation,
)
from tsfm_anomaly.summary import write_summaries
from tsfm_anomaly.utils import get_device, set_seed

# A dataset counts as done once this method's row is saved.
FINAL_METHOD = {"moment": "MOMENTLP", "tranad": "TranAD"}
# Model-native scaling, always fitted on the normal training segment only.
SCALER_MODE = {"moment": "standard", "tranad": "minmax"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--max-datasets", type=int, help="override ucr.max_datasets")
    parser.add_argument("--output-dir", help="override ucr.output_dir")
    parser.add_argument("--device", help="override ucr.device (auto/cpu/cuda/mps)")
    return parser.parse_args()


def append_rows_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        header = list(pd.read_csv(csv_path, nrows=0).columns)
        extra = [column for column in frame.columns if column not in header]
        if extra:
            raise RuntimeError(
                f"Refusing to append columns {extra} to the existing schema in {csv_path}. "
                "Use a new output directory."
            )
        frame.reindex(columns=header).to_csv(csv_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(csv_path, index=False)


def load_completed_dataset_ids(csv_path: Path, final_method: str, seed: int) -> set[int]:
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path)
    done = frame[(frame["method"] == final_method) & (frame["random_seed"] == seed)]
    return set(done["dataset_id"].astype(int))


def run_moment_dataset(model, pretrained_head_state, prepared, x_normal, x_test, labels, cfg, ctx):
    from tsfm_anomaly.models.moment import evaluate_method, make_loaders, train_momentlp

    moment_cfg, eval_cfg, device, out = cfg["moment"], ctx["evaluation"], ctx["device"], ctx["out"]
    metadata = prepared["metadata"]
    stem = f"{metadata.dataset_id:03d}_{metadata.dataset_name}"

    x_train, x_val = split_normal_train_validation(
        x_normal, validation_ratio=cfg["ucr"].validation_ratio, window_size=moment_cfg.seq_len
    )
    train_loader, val_loader = make_loaders(x_train, x_val, moment_cfg, device)
    shared = {
        "short_sequence_mode": bool(len(x_normal) < 2 * moment_cfg.seq_len),
        "training_padding_used": bool(len(x_train) < moment_cfg.seq_len),
        "validation_padding_used": bool(len(x_val) < moment_cfg.seq_len),
        "test_padding_used": bool(len(x_test) < moment_cfg.seq_len),
        "learning_rate": moment_cfg.learning_rate,
        "moment_version": ctx["moment_version"],
    }

    # Reset the pristine pretrained head before each dataset.
    model.head.load_state_dict(pretrained_head_state)
    for parameter in model.parameters():
        parameter.requires_grad = False

    rows = []
    if moment_cfg.run_zero_shot_baseline:
        row = evaluate_method(
            "MOMENT0",
            model,
            x_test,
            labels,
            moment_cfg,
            eval_cfg,
            device,
            artifact_path=out / "artifacts" / f"{stem}__moment0.npz",
        )
        rows.append({**row, **shared})

    # Reset again before LP training.
    model.head.load_state_dict(pretrained_head_state)
    model, history, training = train_momentlp(
        model, train_loader, val_loader, moment_cfg, device, out / "checkpoints" / stem
    )
    history_path = out / "training_histories" / f"{stem}_MOMENTLP.csv"
    history.to_csv(history_path, index=False)
    row = evaluate_method(
        "MOMENTLP",
        model,
        x_test,
        labels,
        moment_cfg,
        eval_cfg,
        device,
        artifact_path=out / "artifacts" / f"{stem}__momentlp.npz",
    )
    rows.append(
        {
            **row,
            **shared,
            "best_epoch": training["best_epoch"],
            "best_validation_mse": training["best_validation_mse"],
            "training_history_path": str(history_path),
        }
    )
    return rows


def run_tranad_dataset(prepared, x_normal, x_test, labels, cfg, ctx):
    from tsfm_anomaly.models.tranad import train_tranad

    tranad_cfg, eval_cfg, out = cfg["tranad"], ctx["evaluation"], ctx["out"]
    metadata = prepared["metadata"]
    stem = f"{metadata.dataset_id:03d}_{metadata.dataset_name}"

    x_train, x_val = split_normal_train_validation(
        x_normal, cfg["ucr"].validation_ratio, tranad_cfg.window
    )
    scores, reconstruction, history, training = train_tranad(
        x_train, x_val, x_test, tranad_cfg, ctx["device"], out / "checkpoints" / f"{stem}_TranAD.pt"
    )
    metrics = evaluate_all_metrics(
        labels,
        scores,
        threshold_mode=eval_cfg.threshold_mode,
        adjusted_threshold_count=eval_cfg.adjusted_threshold_count,
        vus_threshold_count=eval_cfg.vus_threshold_count,
        pa_k_values=eval_cfg.pa_k_values,
        include_vus=eval_cfg.include_vus,
    )
    np.savez_compressed(
        out / "artifacts" / f"{stem}__tranad.npz",
        labels=labels,
        anomaly_scores=scores,
        reconstruction=reconstruction,
    )
    history_path = out / "training_histories" / f"{stem}_TranAD.csv"
    history.to_csv(history_path, index=False)
    return [
        {
            "method": "TranAD",
            "anomaly_points": int(labels.sum()),
            "anomaly_ratio": float(labels.mean()),
            "learning_rate": tranad_cfg.learning_rate,
            "training_history_path": str(history_path),
            **training,
            **metrics,
        }
    ]


def save_provenance(cfg: dict, model_kind: str, device: torch.device, out: Path) -> None:
    provenance = {
        "model": model_kind,
        "config": {name: asdict(section) for name, section in cfg.items()},
        "scaler": f"{SCALER_MODE[model_kind]}, fitted on the normal training segment only",
        "validation": "chronological split of the normal training segment",
        "test_labels_used_only_by": "threshold-sweeping evaluation metrics",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    kinds = [kind for kind in FINAL_METHOD if kind in cfg]
    if len(kinds) != 1:
        raise ValueError("The config needs exactly one model section: 'moment' or 'tranad'.")
    model_kind = kinds[0]

    overrides = {
        key: value
        for key, value in {
            "max_datasets": args.max_datasets,
            "output_dir": args.output_dir,
            "device": args.device,
        }.items()
        if value is not None
    }
    ucr = replace(cfg["ucr"], **overrides)
    cfg["ucr"] = ucr
    evaluation = cfg.get("evaluation", EvaluationConfig())

    set_seed(ucr.seed)
    device = get_device(ucr.device)
    out = Path(ucr.output_dir)
    for sub in ("artifacts", "checkpoints", "training_histories"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    results_csv = out / "metrics.csv"
    errors_csv = out / "errors.csv"
    if results_csv.exists() and results_csv.stat().st_size > 0 and not ucr.resume:
        raise FileExistsError(
            f"Refusing to append a fresh run to existing results: {results_csv}. "
            "Set resume: true or choose a new output_dir."
        )

    files = discover_ucr_files(ucr.ucr_root, ucr.file_glob, ucr.max_datasets)
    save_provenance(cfg, model_kind, device, out)
    completed = (
        load_completed_dataset_ids(results_csv, FINAL_METHOD[model_kind], ucr.seed)
        if ucr.resume
        else set()
    )
    print(f"model={model_kind} device={device} datasets={len(files)} completed={len(completed)}")

    ctx = {"evaluation": evaluation, "device": device, "out": out}
    model = pretrained_head_state = None
    if model_kind == "moment":
        from tsfm_anomaly.models.moment import get_moment_version, load_moment

        model = load_moment(cfg["moment"], device)
        # Only the reconstruction head changes during LP, so preserving just the
        # pristine head is enough to reset the model between UCR datasets.
        pretrained_head_state = copy.deepcopy(model.head.state_dict())
        ctx["moment_version"] = get_moment_version()

    for path in tqdm(files, desc="UCR datasets"):
        metadata = parse_ucr_filename(path)
        if metadata.dataset_id in completed:
            continue
        try:
            prepared = load_ucr(
                path,
                downsample_factor=ucr.downsample_factor,
                pooling_mode=ucr.pooling_mode,
                anomaly_end_inclusive=ucr.anomaly_end_inclusive,
            )
            x_normal, x_test, labels = prepared["train"], prepared["test"], prepared["labels"]
            if len(x_normal) < 2 or len(x_test) < 2:
                raise ValueError("Downsampled train or test segment has fewer than two points.")
            if int(np.sum(labels)) == 0:
                raise ValueError("No anomaly labels remain after preprocessing.")

            scaler = TrainScaler(SCALER_MODE[model_kind]).fit(x_normal)
            x_normal, x_test = scaler.transform(x_normal), scaler.transform(x_test)

            if model_kind == "moment":
                rows = run_moment_dataset(
                    model, pretrained_head_state, prepared, x_normal, x_test, labels, cfg, ctx
                )
            else:
                rows = run_tranad_dataset(prepared, x_normal, x_test, labels, cfg, ctx)

            dataset_info = {
                "dataset_id": metadata.dataset_id,
                "dataset_name": metadata.dataset_name,
                "filename": metadata.filename,
                "category": infer_ucr_category(metadata.dataset_name),
                "base_family": infer_ucr_base_family(metadata.dataset_name),
                "random_seed": ucr.seed,
                "device": str(device),
                "downsample_factor": ucr.downsample_factor,
                "scaler_mode": SCALER_MODE[model_kind],
                "series_length_raw": prepared["series_length_raw"],
                "train_length_raw": prepared["train_length_raw"],
                "test_length_raw": prepared["test_length_raw"],
                "train_length_processed": len(x_normal),
                "test_length_processed": len(x_test),
                "train_pool_tail": prepared["train_pool_tail"],
                "test_pool_tail": prepared["test_pool_tail"],
            }
            append_rows_csv([{**dataset_info, **row} for row in rows], results_csv)
            completed.add(metadata.dataset_id)
            print(f"Completed {metadata.dataset_id:03d}: {metadata.dataset_name}")

        except Exception as exc:
            append_rows_csv(
                [
                    {
                        "dataset_id": metadata.dataset_id,
                        "dataset_name": metadata.dataset_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
                errors_csv,
            )
            print(f"FAILED {metadata.dataset_id:03d} {metadata.dataset_name}: {exc}")
            if ucr.fail_fast:
                raise

        finally:
            if model is not None:
                # Return the model to the pretrained head before the next file.
                model.head.load_state_dict(pretrained_head_state)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_summaries(results_csv, out)
    print(f"results={results_csv}")


if __name__ == "__main__":
    main()
