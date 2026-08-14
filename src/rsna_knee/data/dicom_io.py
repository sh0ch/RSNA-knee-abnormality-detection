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


def load_series_volume(
    series_path: Path,
    *,
    apply_rescale: bool = True,
) -> tuple[np.ndarray, list[Dataset]]:
    """
    Load all slices in a series into a 3D volume [D, H, W].

    Handles mixed transfer syntaxes via pydicom. Applies RescaleSlope/Intercept
    when present so intensities are in a consistent float space.
    """
    paths = list_dicom_files(series_path)
    datasets: list[Dataset] = []
    slices: list[np.ndarray] = []

    for path in paths:
        ds, pixels = read_dicom_slice(path)
        if apply_rescale:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            pixels = pixels * slope + intercept
        datasets.append(ds)
        slices.append(pixels)

    order = np.argsort([_sort_key(ds) for ds in datasets])
    datasets = [datasets[i] for i in order]
    volume = np.stack([slices[i] for i in order], axis=0)
    return volume, datasets


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
