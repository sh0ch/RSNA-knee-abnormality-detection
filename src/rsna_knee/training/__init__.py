"""Training utilities."""

from __future__ import annotations

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


def run_kfold_training(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import run_kfold_training as _fn

    return _fn(*args, **kwargs)


def run_phase2_training(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import run_kfold_training as _kfold
    from rsna_knee.training import loop as _loop

    fn = getattr(_loop, "run_phase2_training", _kfold)
    return fn(*args, **kwargs)


def predict_test_ensemble(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import predict_test_ensemble as _fn

    return _fn(*args, **kwargs)


def prevalence_baseline_predictions(*args: Any, **kwargs: Any) -> Any:
    from rsna_knee.training.loop import prevalence_baseline_predictions as _fn

    return _fn(*args, **kwargs)
