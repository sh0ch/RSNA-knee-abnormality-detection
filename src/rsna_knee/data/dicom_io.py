"""DICOM reading and volume assembly for knee MRI series."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


def list_dicom_files(series_path: Path) -> list[Path]:
    """Return sorted paths to .dcm files in a series directory."""
    if not series_path.is_dir():
        raise FileNotFoundError(f"Series directory not found: {series_path}")
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")
    return files


def read_dicom_slice(path: Path) -> tuple[Dataset, np.ndarray]:
    """Read a single DICOM slice; returns metadata and pixel array."""
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    return ds, pixels


def _sort_key(ds: Dataset) -> float:
    """Best-effort slice ordering key (InstanceNumber, then ImagePositionPatient)."""
    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return 0.0


def _read_header(path: Path) -> Dataset:
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _apply_rescale(ds: Dataset, pixels: np.ndarray) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


def load_series_volume(
    series_path: Path,
    *,
    apply_rescale: bool = True,
    depth: int | None = None,
) -> tuple[np.ndarray, list[Dataset]]:
    """
    Load a series into a 3D volume [D, H, W].

    If ``depth`` is set, only those evenly sampled slices are decoded (headers
    of every file are still read for InstanceNumber / ImagePositionPatient sort).
    Applies RescaleSlope/Intercept when present.
    """
    paths = list_dicom_files(series_path)

    if depth is None:
        datasets: list[Dataset] = []
        slices: list[np.ndarray] = []
        for path in paths:
            ds, pixels = read_dicom_slice(path)
            if apply_rescale:
                pixels = _apply_rescale(ds, pixels)
            datasets.append(ds)
            slices.append(pixels)
        order = np.argsort([_sort_key(ds) for ds in datasets])
        datasets = [datasets[int(i)] for i in order]
        volume = np.stack([slices[int(i)] for i in order], axis=0)
        return volume, datasets

    from rsna_knee.data.volume_prep import sample_depth_indices

    headers = [_read_header(path) for path in paths]
    order = np.argsort([_sort_key(ds) for ds in headers])
    ordered_paths = [paths[int(i)] for i in order]
    indices = sample_depth_indices(len(ordered_paths), depth)

    pixel_cache: dict[int, tuple[Dataset, np.ndarray]] = {}
    datasets = []
    slices = []
    for raw_i in indices:
        i = int(raw_i)
        if i not in pixel_cache:
            ds, pixels = read_dicom_slice(ordered_paths[i])
            if apply_rescale:
                pixels = _apply_rescale(ds, pixels)
            pixel_cache[i] = (ds, pixels)
        ds, pixels = pixel_cache[i]
        datasets.append(ds)
        slices.append(pixels)

    return np.stack(slices, axis=0), datasets


def series_metadata_summary(datasets: list[Dataset]) -> dict[str, object]:
    """Extract commonly useful metadata from the first slice of a series."""
    ds = datasets[0]
    return {
        "modality": getattr(ds, "Modality", None),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "num_slices": len(datasets),
        "pixel_spacing": list(getattr(ds, "PixelSpacing", [])),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0) or 0.0),
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "series_uid": getattr(ds, "SeriesInstanceUID", None),
    }


def normalize_volume(
    volume: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Clip and scale a volume to [0, 1] using robust percentiles."""
    lo = np.percentile(volume, percentile_low)
    hi = np.percentile(volume, percentile_high)
    clipped = np.clip(volume, lo, hi)
    return (clipped - lo) / (hi - lo + eps)
