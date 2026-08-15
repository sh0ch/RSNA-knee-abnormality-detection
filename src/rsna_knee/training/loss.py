"""Training losses."""

from __future__ import annotations

from typing import Any

import numpy as np


def masked_bce_with_logits(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    pos_weight: Any | None = None,
) -> Any:
    """
    BCE-with-logits that ignores labels where ``mask == 0`` (NaN / missing).

    Args:
        logits: ``[B, C]``
        targets: ``[B, C]`` float in {0,1} (NaNs already zeroed)
        mask: ``[B, C]`` float 0/1
        pos_weight: optional ``[C]`` positive-class weights
    """
    import torch.nn.functional as F

    if mask.sum() <= 0:
        return logits.sum() * 0.0

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def compute_pos_weight(labels: np.ndarray, masks: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Per-label pos_weight = neg/pos from labeled entries only.

    ``labels`` / ``masks`` shaped ``[N, C]``. Adds ``eps`` to avoid division by zero.
    """
    labeled = masks > 0.5
    pos = (labels * labeled).sum(axis=0)
    neg = ((1.0 - labels) * labeled).sum(axis=0)
    return ((neg + eps) / (pos + eps)).astype(np.float32)
