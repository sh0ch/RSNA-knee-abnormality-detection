"""Lightweight MRI augmentations using torchvision (Kaggle image has it)."""

from __future__ import annotations

import torch
from torchvision.transforms import functional as TF


def augment_study_batch(
    images: torch.Tensor,
    *,
    hflip_prob: float = 0.5,
    rotate_deg: float = 10.0,
    contrast_jitter: float = 0.15,
) -> torch.Tensor:
    """
    Apply shared-per-study augmentations to ``[B, S, 3, H, W]`` tensors.

    Uses the same random draw for all slices in a study so anatomy stays coherent.
    """
    if images.ndim != 5:
        raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")

    out = images.clone()
    batch = out.shape[0]
    for b in range(batch):
        study = out[b]  # [S, 3, H, W]
        if torch.rand(1).item() < hflip_prob:
            study = torch.flip(study, dims=[-1])

        angle = (torch.rand(1).item() * 2 - 1) * rotate_deg
        if abs(angle) > 0.5:
            # Rotate each slice the same amount.
            rotated = []
            for s in range(study.shape[0]):
                rotated.append(TF.rotate(study[s], angle=angle, fill=0.0))
            study = torch.stack(rotated, dim=0)

        if contrast_jitter > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast_jitter
            mean = study.mean(dim=(-2, -1), keepdim=True)
            study = (study - mean) * factor + mean
            study = study.clamp(0.0, 1.0)

        out[b] = study
    return out
