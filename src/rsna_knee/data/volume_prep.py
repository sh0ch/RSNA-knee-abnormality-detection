"""Volume resampling helpers for 2.5D slice stacks."""

from __future__ import annotations

import numpy as np


def sample_depth_indices(num_slices: int, target_depth: int) -> np.ndarray:
    """Evenly sample ``target_depth`` indices from ``[0, num_slices)``."""
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if target_depth <= 0:
        raise ValueError("target_depth must be positive")
    if num_slices == target_depth:
        return np.arange(num_slices, dtype=np.int64)
    if num_slices == 1:
        return np.zeros(target_depth, dtype=np.int64)
    return np.linspace(0, num_slices - 1, target_depth).round().astype(np.int64)


def resize_slice(slice_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a 2D float array with bilinear interpolation (no torch required)."""
    src_h, src_w = slice_2d.shape
    if src_h == height and src_w == width:
        return slice_2d.astype(np.float32, copy=False)

    y = np.linspace(0, src_h - 1, height)
    x = np.linspace(0, src_w - 1, width)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    top = slice_2d[y0][:, x0] * (1 - wx) + slice_2d[y0][:, x1] * wx
    bot = slice_2d[y1][:, x0] * (1 - wx) + slice_2d[y1][:, x1] * wx
    out = top * (1 - wy)[:, None] + bot * wy[:, None]
    return out.astype(np.float32)


def resize_volume(volume: np.ndarray, depth: int, height: int, width: int) -> np.ndarray:
    """Sample depth then resize each slice to ``(height, width)``."""
    indices = sample_depth_indices(volume.shape[0], depth)
    sampled = volume[indices]
    out = np.empty((depth, height, width), dtype=np.float32)
    for i in range(depth):
        out[i] = resize_slice(sampled[i], height, width)
    return out


def stack_adjacent_as_rgb(volume: np.ndarray) -> np.ndarray:
    """
    Build 2.5D 3-channel slices: ``[s-1, s, s+1]`` for each depth index.

    Returns array shaped ``[D, 3, H, W]``.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected [D,H,W], got shape {volume.shape}")
    depth = volume.shape[0]
    prev = np.concatenate([volume[:1], volume[:-1]], axis=0)
    nxt = np.concatenate([volume[1:], volume[-1:]], axis=0)
    stacked = np.stack([prev, volume, nxt], axis=1)
    assert stacked.shape == (depth, 3, volume.shape[1], volume.shape[2])
    return stacked.astype(np.float32, copy=False)


def prepare_series_tensor(
    volume: np.ndarray,
    *,
    depth: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Normalize-ready volume → ``[D, 3, H, W]`` float32 tensor array."""
    resized = resize_volume(volume, depth, height, width)
    return stack_adjacent_as_rgb(resized)
