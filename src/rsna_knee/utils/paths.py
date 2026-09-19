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


def _discover_kaggle_data_root() -> Path | None:
    """Find the directory that contains train.csv under /kaggle/input."""
    for candidate in _kaggle_data_candidates():
        if (candidate / TRAIN_CSV).is_file():
            return candidate

    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    skip = {TRAIN_SERIES_DIR, TEST_SERIES_DIR}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if TRAIN_CSV in filenames:
            return Path(dirpath)
        dirnames[:] = [name for name in dirnames if name not in skip]
        if len(rel.parts) >= 4:
            dirnames.clear()
    return None


def default_data_root() -> Path:
    """Resolve the competition data directory."""
    if is_kaggle_kernel():
        found = _discover_kaggle_data_root()
        if found is not None:
            return found
        listing = "(missing)"
        inp = Path("/kaggle/input")
        if inp.is_dir():
            listing = ", ".join(p.name for p in sorted(inp.iterdir())) or "(empty)"
        raise FileNotFoundError(
            "Competition data is not mounted (no train.csv under /kaggle/input).\n"
            f"/kaggle/input: {listing}\n"
            "On Kaggle, open kernel simonhochwebde/rsna-knee-phase1-image, "
            "Add Input → competition rsna-knee-abnormality-detection, "
            "then restart Jupyter Server. Package sync cannot provide DICOMs."
        )

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


def default_volume_cache_dir() -> Path | None:
    """Ephemeral uint8 volume cache (Phase 2 streaming). ``None`` means disabled."""
    env = os.environ.get("RSNA_VOLUME_CACHE")
    if env:
        return Path(env)
    if not is_kaggle_kernel():
        return None
    for candidate in (
        Path("/tmp/rsna_volume_cache"),
        Path("/kaggle/tmp/rsna_volume_cache"),
        Path("/kaggle/working/volume_cache"),
    ):
        if candidate.parent.is_dir():
            return candidate
    return None
