"""Evaluating time-series foundation models for anomaly detection.

Modules that need only NumPy/pandas (``simulation``, ``preprocess``,
``windows``, ``metrics``) are importable without PyTorch. Model code lives in
``tsfm_anomaly.models`` and requires the ``torch`` or ``moment`` extra.
"""

__version__ = "0.1.0"
