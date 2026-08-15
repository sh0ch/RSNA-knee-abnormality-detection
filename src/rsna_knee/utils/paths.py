"""Environment-aware path resolution for local dev vs Kaggle kernels."""

from __future__ import annotations

import os
from pathlib import Path

from rsna_knee.constants import (
    COMPETITION_SLUG,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TEST_SERIES_CSV,
    TEST_SERIES_DIR,
    TRAIN_CSV,
    TRAIN_SERIES_CSV,
    TRAIN_SERIES_DIR,
)


def is_kaggle_kernel() -> bool:
    """Return True when running inside a Kaggle notebook/kernel."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
        return True
    if "KAGGLE_URL_BASE" in os.environ:
        return True
    # Filesystem fallback when env vars are unavailable (e.g. some batch runs).
    return Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()


def is_kaggle_notebook() -> bool:
    """Return True when any Kaggle notebook environment is detected."""
    return is_kaggle_kernel()


def project_root() -> Path:
    """Repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def _as_path(data_root: Path | str | None) -> Path:
    """Coerce an optional data root to Path (handles str overrides from notebooks)."""
    if data_root is None:
        return default_data_root()
    return Path(data_root)


def _kaggle_data_candidates() -> list[Path]:
    """Known mount locations for the competition dataset on Kaggle."""
    slug = COMPETITION_SLUG
    return [
        Path("/kaggle/input/competitions") / slug,
        Path("/kaggle/input") / slug,
    ]


def default_data_root() -> Path:
    """Resolve the competition data directory."""
    if is_kaggle_kernel():
        for candidate in _kaggle_data_candidates():
            if (candidate / TRAIN_CSV).is_file():
                return candidate
        # Prefer current Kaggle competition layout when probing fails (e.g. during import)
        return Path("/kaggle/input/competitions") / COMPETITION_SLUG

    env_override = os.environ.get("RSNA_DATA_ROOT")
    if env_override:
        return Path(env_override)

    # Local default: tiny synthetic sample, not the full competition download
    return project_root() / "data" / "sample"


def series_dir(data_root: Path | str | None = None, split: str = "train") -> Path:
    """Path to DICOM series root for train or test split."""
    root = _as_path(data_root)
    subdir = TRAIN_SERIES_DIR if split == "train" else TEST_SERIES_DIR
    return root / subdir


def csv_path(name: str, data_root: Path | str | None = None) -> Path:
    """Resolve a competition CSV filename under the data root."""
    root = _as_path(data_root)
    return root / name


def train_csv(data_root: Path | str | None = None) -> Path:
    return csv_path(TRAIN_CSV, data_root)


def train_series_csv(data_root: Path | str | None = None) -> Path:
    return csv_path(TRAIN_SERIES_CSV, data_root)


def test_csv(data_root: Path | str | None = None) -> Path:
    return csv_path(TEST_CSV, data_root)


def test_series_csv(data_root: Path | str | None = None) -> Path:
    return csv_path(TEST_SERIES_CSV, data_root)


def sample_submission_csv(data_root: Path | str | None = None) -> Path:
    return csv_path(SAMPLE_SUBMISSION_CSV, data_root)
