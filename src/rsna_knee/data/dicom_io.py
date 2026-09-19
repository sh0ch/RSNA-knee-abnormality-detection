"""DICOM reading and volume assembly for knee MRI series."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

# Header-only tags for InstanceNumber sort (skip the rest of the DICOM dataset).
_SORT_TAGS = ["InstanceNumber", "ImagePositionPatient"]

# Process-local: ordered .dcm paths per series. Shared across fold threads so
# epoch 2+ and overlapping folds skip glob + header re-reads.
_ORDERED_PATHS: dict[str, list[Path]] = {}
_ORDERED_LOCK = threading.Lock()


def list_dicom_files(series_path: Path) -> list[Path]:
    """Return sorted paths to .dcm files in a series directory."""
    if not series_path.is_dir():
        raise FileNotFoundError(f"Series directory not found: {series_path}")
    files = [
        Path(entry.path)
        for entry in os.scandir(series_path)
        if entry.is_file() and entry.name.lower().endswith(".dcm")
    ]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")
    files.sort()
    return files


_UNCOMPRESSED_TS = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.2",
}
_BYTE_MISMATCH = re.compile(
    r"less than expected \((\d+) vs (\d+) bytes\)",
    re.IGNORECASE,
)


def _transfer_syntax_uid(ds: Dataset) -> str:
    meta = getattr(ds, "file_meta", None)
    if meta is not None and getattr(meta, "TransferSyntaxUID", None):
        return str(meta.TransferSyntaxUID)
    return str(getattr(ds, "TransferSyntaxUID", "") or "")


def _reshape_uint8_plane(
    raw: bytes, rows: int, cols: int, samples: int
) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8, count=rows * cols * samples)
    if samples <= 1:
        return arr.reshape(rows, cols)
    return arr.reshape(rows, cols, samples)


def _pixel_array_tolerant(ds: Dataset) -> np.ndarray:
    """``pixel_array`` with a BitsAllocated 16-vs-8 workaround used by some RSNA files."""
    try:
        return np.asarray(ds.pixel_array)
    except ValueError as exc:
        msg = str(exc)
        match = _BYTE_MISMATCH.search(msg)
        if match is None:
            raise
        actual, _expected = int(match.group(1)), int(match.group(2))
        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
        samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
        ts = _transfer_syntax_uid(ds)
        uncompressed = not ts or ts in _UNCOMPRESSED_TS
        plane = rows * cols * samples
        # Header claims 16-bit but PixelData is one byte per sample (actual == rows*cols).
        if not (uncompressed and plane > 0 and actual == plane):
            raise
        ds.BitsAllocated = 8
        stored = int(getattr(ds, "BitsStored", 8) or 8)
        ds.BitsStored = min(max(stored, 1), 8)
        ds.HighBit = int(ds.BitsStored) - 1
        try:
            return np.asarray(ds.pixel_array)
        except ValueError:
            raw = bytes(ds.PixelData)
            if len(raw) < plane:
                raise exc from None
            return _reshape_uint8_plane(raw, rows, cols, samples)


def read_dicom_slice(path: Path) -> tuple[Dataset, np.ndarray]:
    """Read a single DICOM slice; returns metadata and pixel array."""
    ds = pydicom.dcmread(str(path), force=True)
    pixels = _pixel_array_tolerant(ds).astype(np.float32)
    if pixels.ndim > 2:
        pixels = pixels[0]
    return ds, pixels


def _read_slice_or_none(path: Path) -> tuple[Dataset, np.ndarray] | None:
    try:
        return read_dicom_slice(path)
    except Exception as exc:  # noqa: BLE001 — one bad file must not kill the series
        logger.warning("Skipping unreadable DICOM %s: %s", path.name, exc)
        return None


def _sort_key(ds: Dataset) -> float:
    """Best-effort slice ordering key (InstanceNumber, then ImagePositionPatient)."""
    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return 0.0


def _read_header(path: Path) -> Dataset:
    return pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        specific_tags=_SORT_TAGS,
        force=True,
    )


def ordered_dicom_paths(series_path: Path) -> list[Path]:
    """DICOM paths sorted by InstanceNumber, cached for the process lifetime."""
    key = str(series_path)
    with _ORDERED_LOCK:
        cached = _ORDERED_PATHS.get(key)
    if cached is not None:
        return cached
    paths = list_dicom_files(series_path)
    headers = [_read_header(path) for path in paths]
    order = np.argsort([_sort_key(ds) for ds in headers])
    ordered = [paths[int(i)] for i in order]
    with _ORDERED_LOCK:
        _ORDERED_PATHS[key] = ordered
    return ordered


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

    If ``depth`` is set, only those evenly sampled slices are decoded. Slice
    order is cached after the first header pass. Applies RescaleSlope/Intercept
    when present.
    """
    ordered_paths = ordered_dicom_paths(series_path)

    if depth is None:
        datasets: list[Dataset] = []
        slices: list[np.ndarray] = []
        for path in ordered_paths:
            got = _read_slice_or_none(path)
            if got is None:
                continue
            ds, pixels = got
            if apply_rescale:
                pixels = _apply_rescale(ds, pixels)
            datasets.append(ds)
            slices.append(pixels)
        if not slices:
            raise ValueError(f"No readable slices in {series_path}")
        volume = np.stack(slices, axis=0)
        return volume, datasets

    from rsna_knee.data.volume_prep import sample_depth_indices

    indices = sample_depth_indices(len(ordered_paths), depth)
    n_paths = len(ordered_paths)
    pixel_cache: dict[int, tuple[Dataset, np.ndarray]] = {}
    unreadable: set[int] = set()
    datasets = []
    slices = []
    for raw_i in indices:
        start = int(raw_i)
        found: tuple[Dataset, np.ndarray] | None = None
        for delta in range(n_paths):
            candidates = [start] if delta == 0 else [start + delta, start - delta]
            for idx in candidates:
                if idx < 0 or idx >= n_paths or idx in unreadable:
                    continue
                if idx not in pixel_cache:
                    got = _read_slice_or_none(ordered_paths[idx])
                    if got is None:
                        unreadable.add(idx)
                        continue
                    ds, pixels = got
                    if apply_rescale:
                        pixels = _apply_rescale(ds, pixels)
                    pixel_cache[idx] = (ds, pixels)
                found = pixel_cache[idx]
                break
            if found is not None:
                break
        if found is None:
            continue
        datasets.append(found[0])
        slices.append(found[1])

    if not slices:
        raise ValueError(f"No readable slices in {series_path}")
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
