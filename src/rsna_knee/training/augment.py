"""Lightweight MRI augmentations using torchvision (Kaggle image has it)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rotate_nchw(images: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate ``[N, C, H, W]`` around the center with one grid_sample."""
    n = images.size(0)
    ang = images.new_tensor(math.radians(angle_deg))
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    theta = images.new_zeros(n, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(
        images,
        grid,
        align_corners=False,
        padding_mode="zeros",
        mode="bilinear",
    )


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
        study = out[b]
        if torch.rand(1, device=images.device).item() < hflip_prob:
            study = torch.flip(study, dims=[-1])

        angle = (torch.rand(1, device=images.device).item() * 2 - 1) * rotate_deg
        if abs(angle) > 0.5:
            study = _rotate_nchw(study, angle)

        if contrast_jitter > 0:
            factor = 1.0 + (torch.rand(1, device=images.device).item() * 2 - 1) * contrast_jitter
            mean = study.mean(dim=(-2, -1), keepdim=True)
            study = (study - mean) * factor + mean
            study = study.clamp(0.0, 1.0)

        out[b] = study
    return out
