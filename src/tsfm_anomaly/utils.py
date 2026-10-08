"""Reproducibility and device helpers shared by the PyTorch runners."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Return the requested device, or CUDA > MPS > CPU when ``"auto"``."""
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
