"""TranAD baseline (Tuli et al., VLDB 2022) under the shared UCR protocol.

The architecture is a device-independent port of the official implementation
(imperial-qore/TranAD@7ffb98d0c18189cc3d9ab732b4cb0278200a0af0, BSD 3-Clause,
Copyright (c) 2022, Shreshth Tuli; see ``licenses/TranAD-BSD-3-Clause.txt``).
Compatibility changes are limited to making dimensions explicit. The official
``nn.LeakyReLU(True)`` is kept: its negative slope of 1 makes it an identity.

Training follows the official two-phase loss and optimizer. Unlike the
official script, the epoch is selected on a chronological normal validation
segment, and scaling is fitted on normal training data only.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import TranADConfig


class PositionalEncoding(nn.Module):
    """TranAD positional encoding from imperial-qore/TranAD."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(d_model, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class TranADEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)


class TranADDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory, need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)


class TranAD(nn.Module):
    """Official two-phase TranAD architecture."""

    def __init__(self, features: int = 1, window_size: int = 10):
        super().__init__()
        self.features = features
        self.window_size = window_size
        d_model = 2 * features
        self.pos_encoder = PositionalEncoding(d_model, 0.1, window_size)
        self.transformer_encoder = nn.TransformerEncoder(
            TranADEncoderLayer(d_model, features, 16, 0.1), 1
        )
        self.transformer_decoder1 = nn.TransformerDecoder(
            TranADDecoderLayer(d_model, features, 16, 0.1), 1
        )
        self.transformer_decoder2 = nn.TransformerDecoder(
            TranADDecoderLayer(d_model, features, 16, 0.1), 1
        )
        self.fcn = nn.Sequential(nn.Linear(d_model, features), nn.Sigmoid())

    def _encode(self, src: torch.Tensor, condition: torch.Tensor, target: torch.Tensor):
        source = torch.cat((src, condition), dim=2)
        source = self.pos_encoder(source * math.sqrt(self.features))
        memory = self.transformer_encoder(source)
        return target.repeat(1, 1, 2), memory

    def forward(self, src: torch.Tensor, target: torch.Tensor):
        condition = torch.zeros_like(src)
        phase1 = self.fcn(self.transformer_decoder1(*self._encode(src, condition, target)))
        condition = (phase1 - src) ** 2
        phase2 = self.fcn(self.transformer_decoder2(*self._encode(src, condition, target)))
        return phase1, phase2


def causal_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """TranAD's official left-padded causal windows, one score per point."""
    values = np.asarray(values, dtype=np.float32)
    windows = []
    for index in range(len(values)):
        if index >= window_size:
            window = values[index - window_size : index]
        else:
            window = np.concatenate(
                [np.repeat(values[0:1], window_size - index, axis=0), values[:index]], axis=0
            )
        windows.append(window)
    return np.stack(windows)


def train_tranad(train, validation, test, cfg: TranADConfig, device, checkpoint_path: Path):
    # The pinned reference implementation uses float64 on CPU. Apple MPS does
    # not implement float64, so MPS runs use explicitly recorded float32.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    train_windows = torch.from_numpy(causal_windows(train, cfg.window)).to(dtype)
    val_windows = torch.from_numpy(causal_windows(validation, cfg.window)).to(dtype)
    test_windows = torch.from_numpy(causal_windows(test, cfg.window)).to(dtype)
    loader = DataLoader(TensorDataset(train_windows), batch_size=cfg.batch_size, shuffle=False)
    model = TranAD(features=train.shape[1], window_size=cfg.window).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    history = []
    best_state, best_val, best_epoch = None, float("inf"), 0
    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            source = batch.permute(1, 0, 2)
            target = source[-1:].contiguous()
            phase1, phase2 = model(source, target)
            loss = torch.mean((phase1 - target) ** 2) / epoch
            loss = loss + (1.0 - 1.0 / epoch) * torch.mean((phase2 - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_scores, _ = score_tranad(model, val_windows, cfg.batch_size, device)
        val_loss = float(np.mean(val_scores))
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mse": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    scores, reconstruction = score_tranad(model, test_windows, cfg.batch_size, device)
    return (
        scores,
        reconstruction,
        pd.DataFrame(history),
        {
            "best_epoch": best_epoch,
            "best_validation_mse": best_val,
            "training_seconds": time.time() - start_time,
            "numeric_precision": str(dtype).replace("torch.", ""),
        },
    )


@torch.no_grad()
def score_tranad(model, windows, batch_size, device):
    model.eval()
    all_scores, all_reconstruction = [], []
    loader = DataLoader(TensorDataset(windows), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        batch = batch.to(device)
        source = batch.permute(1, 0, 2)
        target = source[-1:].contiguous()
        _, reconstruction = model(source, target)
        all_scores.append(torch.mean((reconstruction - target) ** 2, dim=-1)[0].cpu())
        all_reconstruction.append(reconstruction[0].cpu())
    return torch.cat(all_scores).numpy(), torch.cat(all_reconstruction).numpy()
