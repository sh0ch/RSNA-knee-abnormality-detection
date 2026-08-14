"""Image/volume transforms (extend with albumentations/MONAI on Kaggle)."""

from __future__ import annotations

import numpy as np


def center_crop_or_pad(
    volume: np.ndarray,
    target_depth: int,
    target_height: int,
    target_width: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Center-crop or zero-pad a [D, H, W] volume to a fixed shape."""
    d, h, w = volume.shape
    out = np.full((target_depth, target_height, target_width), fill_value, dtype=volume.dtype)

    # Depth
    d0 = max(0, (d - target_depth) // 2)
    od0 = max(0, (target_depth - d) // 2)
    d_take = min(d, target_depth)

    # Height
    h0 = max(0, (h - target_height) // 2)
    oh0 = max(0, (target_height - h) // 2)
    h_take = min(h, target_height)

    # Width
    w0 = max(0, (w - target_width) // 2)
    ow0 = max(0, (target_width - w) // 2)
    w_take = min(w, target_width)

    out[
        od0 : od0 + d_take,
        oh0 : oh0 + h_take,
        ow0 : ow0 + w_take,
    ] = volume[d0 : d0 + d_take, h0 : h0 + h_take, w0 : w0 + w_take]
    return out
