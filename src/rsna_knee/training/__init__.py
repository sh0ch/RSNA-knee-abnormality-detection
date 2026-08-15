"""Training utilities."""

from rsna_knee.training.loss import compute_pos_weight, masked_bce_with_logits
from rsna_knee.training.metrics import macro_roc_auc

__all__ = [
    "compute_pos_weight",
    "macro_roc_auc",
    "masked_bce_with_logits",
    "predict_test_ensemble",
    "prevalence_baseline_predictions",
    "run_kfold_training",
]


def __getattr__(name: str):
    """Lazy-import torch-dependent training loop helpers."""
    if name in {
        "predict_test_ensemble",
        "prevalence_baseline_predictions",
        "run_kfold_training",
    }:
        from rsna_knee.training import loop as _loop

        return getattr(_loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
