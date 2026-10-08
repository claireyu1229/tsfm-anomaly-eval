"""MOMENT zero-shot (MOMENT0) and linear probing (MOMENTLP) for anomaly detection.

Both methods score a point by its reconstruction error:

- MOMENT0 uses the pretrained reconstruction head as is.
- MOMENTLP freezes the whole backbone and trains only the reconstruction head
  on normal data, selecting the epoch with the lowest validation MSE.

MOMENT itself applies RevIN (instance normalization) to every input window;
this module only feeds windows of already train-standardized data.
"""

from __future__ import annotations

import copy
import importlib.metadata
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..config import EvaluationConfig, MomentConfig
from ..metrics import evaluate_all_metrics
from ..windows import pad_window, tail_covering_starts
from .scheduler import LinearWarmupCosineScheduler


def get_moment_version() -> str:
    try:
        return importlib.metadata.version("momentfm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_moment(cfg: MomentConfig, device: torch.device) -> nn.Module:
    """Load pretrained MOMENT with the reconstruction head."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        cfg.model_id,
        model_kwargs={
            "task_name": "reconstruction",
            "seq_len": cfg.seq_len,
            "patch_len": cfg.patch_len,
            "patch_stride_len": cfg.patch_stride,
        },
    )
    model.init()
    model = model.to(device)
    if not hasattr(model, "head"):
        raise AttributeError("Loaded MOMENT model has no `head` module.")
    return model


class WindowDataset(Dataset):
    """
    Tail-covering MOMENT windows.

    Sequences shorter than seq_len are edge-padded to one full window.
    input_mask marks only the original, non-padded observations.
    """

    def __init__(self, x: np.ndarray, seq_len: int, stride: int):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError("x must have shape [time, channels].")
        if len(x) == 0:
            raise ValueError("x must contain at least one point.")

        self.x = x
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.starts = tail_covering_starts(len(self.x), self.seq_len, self.stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        end = min(start + self.seq_len, len(self.x))
        real_window = self.x[start:end]

        padded_window, input_mask = pad_window(real_window, seq_len=self.seq_len)

        # MOMENT input: [channels, time]
        return (
            torch.from_numpy(padded_window.T.copy()),
            torch.from_numpy(input_mask),
        )


def make_loaders(
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: MomentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = WindowDataset(x_train, seq_len=cfg.seq_len, stride=cfg.train_stride)
    val_dataset = WindowDataset(x_val, seq_len=cfg.seq_len, stride=cfg.eval_stride)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg.batch_size, len(val_dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def masked_reconstruction_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared reconstruction error over real observations only.

    reconstruction/target: [batch, channels, time]
    input_mask:           [batch, time], 1=real and 0=padding
    """
    if reconstruction.shape != target.shape:
        raise ValueError(
            f"Reconstruction and target shapes differ: {reconstruction.shape} vs {target.shape}"
        )

    mask = input_mask.to(dtype=target.dtype).unsqueeze(1)
    mask = mask.expand_as(target)

    squared_error = (reconstruction - target).pow(2) * mask
    denominator = mask.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


@torch.no_grad()
def validation_reconstruction_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    model.eval()
    total_squared_error = 0.0
    total_real_elements = 0.0

    for x_batch, input_mask in loader:
        x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
        input_mask = input_mask.to(device, non_blocking=(device.type == "cuda"))

        with autocast_context(device, use_bfloat16):
            output = model(x_enc=x_batch, input_mask=input_mask)
            if output.reconstruction is None:
                raise RuntimeError("Model returned no reconstruction.")

            mask = input_mask.to(dtype=x_batch.dtype).unsqueeze(1)
            mask = mask.expand_as(x_batch)
            squared_error = (output.reconstruction - x_batch).pow(2) * mask

        total_squared_error += float(squared_error.detach().float().sum().cpu())
        total_real_elements += float(mask.detach().float().sum().cpu())

    return total_squared_error / max(total_real_elements, 1.0)


def train_momentlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: MomentConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, pd.DataFrame, dict[str, float | int]]:
    # Official MOMENTLP rule: freeze everything except reconstruction head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("head.")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected}")

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.max_epochs * max(len(train_loader), 1)
    scaled_warmup_steps = min(
        cfg.official_warmup_steps,
        max(1, int(round(total_steps * cfg.scaled_warmup_ratio))),
    )

    scheduler = LinearWarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=scaled_warmup_steps,
        initial_lr=cfg.learning_rate,
        warmup_lr=cfg.warmup_learning_rate,
        min_lr=cfg.min_learning_rate,
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_head_state = None
    global_step = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.max_epochs + 1):
        # Keep the frozen backbone in eval mode and only the head in train mode.
        model.eval()
        model.head.train()

        running_squared_error = 0.0
        running_real_elements = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", leave=False)

        gradient_norm = torch.tensor(0.0)

        for x_batch, input_mask in progress:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            input_mask = input_mask.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, cfg.use_bfloat16_on_cuda):
                output = model(x_enc=x_batch, input_mask=input_mask)
                if output.reconstruction is None:
                    raise RuntimeError("Model returned no reconstruction.")
                loss = masked_reconstruction_mse(
                    reconstruction=output.reconstruction,
                    target=x_batch,
                    input_mask=input_mask,
                )

            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.head.parameters(),
                max_norm=cfg.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            real_elements = float(input_mask.sum().detach().cpu()) * x_batch.shape[1]
            running_squared_error += float(loss.detach().float().cpu()) * real_elements
            running_real_elements += real_elements

            progress.set_postfix(
                loss=float(loss.detach().float().cpu()),
                lr=optimizer.param_groups[0]["lr"],
            )

        train_loss = running_squared_error / max(running_real_elements, 1.0)
        val_loss = validation_reconstruction_loss(
            model=model,
            loader=val_loader,
            device=device,
            use_bfloat16=cfg.use_bfloat16_on_cuda,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_head_state = copy.deepcopy(model.head.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_id": cfg.model_id,
                    "config": asdict(cfg),
                    "head_state_dict": best_head_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                },
                output_dir / "best_checkpoint.pt",
            )

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_reconstruction_mse": train_loss,
                "validation_reconstruction_mse": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_last_batch": float(gradient_norm),
                "best_epoch_so_far": best_epoch,
                "best_validation_mse_so_far": best_val_loss,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_loss:.8f} | "
            f"val={val_loss:.8f} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"best={best_epoch}"
        )

    if best_head_state is None:
        raise RuntimeError("No best checkpoint was created.")

    model.head.load_state_dict(best_head_state)
    model.eval()

    metadata = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_loss,
        "total_optimizer_steps": total_steps,
        "scaled_warmup_steps": scaled_warmup_steps,
        "official_warmup_steps_reference": cfg.official_warmup_steps,
    }
    return model, pd.DataFrame(history), metadata


@torch.no_grad()
def reconstruct_series(
    model: nn.Module,
    x: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, np.ndarray | int]:
    """
    Reconstruct the complete test series.

    Short test sequences are edge-padded internally to seq_len. Padded
    positions are discarded before anomaly scores and metrics are returned.
    """
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must have shape [time, channels].")
    if len(x) == 0:
        raise ValueError("Test series is empty.")
    if len(labels) != len(x):
        raise ValueError(f"Label length {len(labels)} != test length {len(x)}.")

    original_length = len(x)
    starts = tail_covering_starts(original_length, seq_len, stride)

    score_sum = np.zeros((original_length, x.shape[1]), dtype=np.float64)
    reconstruction_sum = np.zeros_like(score_sum)
    coverage = np.zeros(original_length, dtype=np.float64)

    model.eval()

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = []
        masks = []
        real_lengths = []

        for start in batch_starts:
            end = min(start + seq_len, original_length)
            real_window = x[start:end]
            padded_window, input_mask = pad_window(real_window, seq_len=seq_len)
            windows.append(padded_window.T)
            masks.append(input_mask)
            real_lengths.append(len(real_window))

        batch = np.stack(windows, axis=0).astype(np.float32)
        mask_batch = np.stack(masks, axis=0).astype(np.float32)

        x_batch = torch.from_numpy(batch).to(device)
        input_mask = torch.from_numpy(mask_batch).to(device)

        with autocast_context(device, use_bfloat16):
            output = model(x_enc=x_batch, input_mask=input_mask)

        if output.reconstruction is None:
            raise RuntimeError("Model returned no reconstruction.")

        reconstruction = output.reconstruction.detach().float().cpu().numpy()

        for local_index, start in enumerate(batch_starts):
            real_length = real_lengths[local_index]
            end = start + real_length

            true_window = batch[local_index, :, :real_length].T
            reconstructed_window = reconstruction[local_index, :, :real_length].T
            squared_error = (true_window - reconstructed_window) ** 2

            score_sum[start:end] += squared_error
            reconstruction_sum[start:end] += reconstructed_window
            coverage[start:end] += 1.0

    if np.any(coverage == 0):
        missing = int(np.sum(coverage == 0))
        raise RuntimeError(f"{missing} points received no window coverage.")

    channel_mse = score_sum / coverage[:, None]
    reconstruction = reconstruction_sum / coverage[:, None]
    anomaly_scores = channel_mse.mean(axis=1)

    return {
        "labels": labels,
        "anomaly_scores": anomaly_scores.astype(np.float64),
        "channel_mse": channel_mse.astype(np.float64),
        "reconstruction": reconstruction.astype(np.float32),
        "coverage": coverage.astype(np.int64),
        "evaluated_length": original_length,
        "dropped_tail_length": 0,
    }


def evaluate_method(
    method_name: str,
    model: nn.Module,
    x_test: np.ndarray,
    labels: np.ndarray,
    moment_cfg: MomentConfig,
    eval_cfg: EvaluationConfig,
    device: torch.device,
    artifact_path: Path | None = None,
) -> dict[str, object]:
    """Score ``x_test`` by reconstruction MSE and compute every metric."""
    reconstructed = reconstruct_series(
        model=model,
        x=x_test,
        labels=labels,
        seq_len=moment_cfg.seq_len,
        stride=moment_cfg.eval_stride,
        batch_size=moment_cfg.batch_size,
        device=device,
        use_bfloat16=moment_cfg.use_bfloat16_on_cuda,
    )

    metric_values = evaluate_all_metrics(
        labels=reconstructed["labels"],
        scores=reconstructed["anomaly_scores"],
        threshold_mode=eval_cfg.threshold_mode,
        adjusted_threshold_count=eval_cfg.adjusted_threshold_count,
        vus_threshold_count=eval_cfg.vus_threshold_count,
        pa_k_values=eval_cfg.pa_k_values,
        include_vus=eval_cfg.include_vus,
    )

    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            method=np.array(method_name),
            labels=reconstructed["labels"],
            anomaly_scores=reconstructed["anomaly_scores"],
            channel_mse=reconstructed["channel_mse"],
            reconstruction=reconstructed["reconstruction"],
            coverage=reconstructed["coverage"],
        )

    return {
        "method": method_name,
        "evaluated_length": reconstructed["evaluated_length"],
        "anomaly_points": int(np.sum(reconstructed["labels"])),
        "anomaly_ratio": float(np.mean(reconstructed["labels"])),
        **metric_values,
    }
