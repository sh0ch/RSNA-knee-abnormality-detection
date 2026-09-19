"""Training utilities."""

from __future__ import annotations

import inspect
from typing import Any

from rsna_knee.training.loss import compute_pos_weight, masked_bce_with_logits
from rsna_knee.training.metrics import macro_roc_auc

__all__ = [
    "compute_pos_weight",
    "macro_roc_auc",
    "masked_bce_with_logits",
    "predict_test_ensemble",
    "prevalence_baseline_predictions",
    "run_kfold_training",
    "run_phase2_training",
]


def _call_kfold(*args: Any, **kwargs: Any) -> Any:
    """Forward to ``loop.run_kfold_training``, dropping kwargs a stale copy rejects."""
    from rsna_knee.training.loop import run_kfold_training as _fn

    try:
        params = inspect.signature(_fn).parameters
    except (TypeError, ValueError):
        return _fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return _fn(*args, **kwargs)
    dropped = [key for key in kwargs if key not in params]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"Dropped stale-loop kwargs: {dropped}", flush=True)
    return _fn(*args, **kwargs)


def run_kfold_training(*args: Any, **kwargs: Any) -> Any:
    return _call_kfold(*args, **kwargs)


def run_phase2_training(*args: Any, **kwargs: Any) -> Any:
    return _call_kfold(*args, **kwargs)


def predict_test_ensemble(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import predict_test_ensemble as _fn

    return _fn(*args, **kwargs)


def prevalence_baseline_predictions(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import prevalence_baseline_predictions as _fn

    return _fn(*args, **kwargs)
