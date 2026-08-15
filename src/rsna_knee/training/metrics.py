"""Training metrics aligned with competition evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import TARGET_LABELS


def macro_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> float:
    """
    Macro-averaged ROC-AUC across all 12 abnormalities.

    Matches Kaggle evaluation. Skips labels with only one class present in y_true.
    """
    labels = labels or TARGET_LABELS
    scores: list[float] = []
    for i, _name in enumerate(labels):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, y_pred[:, i]))
    if not scores:
        raise ValueError("No label had both classes present for ROC-AUC.")
    return float(np.mean(scores))
