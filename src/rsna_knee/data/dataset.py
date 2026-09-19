"""PyTorch datasets and data loading."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna_knee.constants import (
    ANATOMICAL_PLANE_COL,
    FLUID_COL,
    REPORT_COL,
    SERIES_ID_COL,
    STUDY_ID_COL,
    TARGET_LABELS,
)
from rsna_knee.data.dicom_io import load_series_volume, normalize_volume
from rsna_knee.data.schema import (
    labels_present_mask,
    load_test_series_table,
    load_test_table,
    load_train_series_table,
    load_train_table,
)
from rsna_knee.data.volume_prep import resize_volume, stack_adjacent_as_rgb
from rsna_knee.reports.hybrid import CONF_SUFFIX, load_pseudo_labels
from rsna_knee.utils.paths import default_data_root, series_dir

logger = logging.getLogger(__name__)

# Preferred anatomical plane order when selecting up to N series per study.
_PLANE_PRIORITY: dict[str, int] = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


def uint8_cache_nbytes(
    n_studies: int,
    volume_shape: tuple[int, int, int],
    max_series: int,
) -> int:
    """Bytes for packed grayscale volumes ``[S, D, H, W]`` uint8."""
    depth, height, width = volume_shape
    return int(n_studies) * int(max_series) * depth * height * width


def disk_cache_fits(
    n_studies: int,
    volume_shape: tuple[int, int, int],
    max_series: int,
    cache_dir: Path,
    *,
    reserve_bytes: int = 2 * 1024**3,
) -> bool:
    """True if ``cache_dir`` has room for uint8 volumes plus a free-space reserve."""
    needed = uint8_cache_nbytes(n_studies, volume_shape, max_series)
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    return free > int(needed * 1.15) + reserve_bytes


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp.npy"
    try:
        np.save(tmp, array)
        os.replace(tmp, path)
    finally:
        if tmp.is_file():
            with contextlib.suppress(OSError):
                tmp.unlink()


def _cache_file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def warm_volume_cache(dataset: KneeStudyDataset, *, max_workers: int = 8) -> int:
    """Write missing disk-cache files. Already-cached studies are not loaded into RAM."""
    if dataset.cache_dir is None:
        return 0
    n = len(dataset)
    todo = [
        uid for uid in dataset.study_ids if dataset._find_cached_npy(uid) is None
    ]
    n_have = n - len(todo)
    free_gib = shutil.disk_usage(dataset.cache_dir).free / (1024**3)
    print(
        f"Volume cache: {n_have}/{n} on disk, {len(todo)} to decode  "
        f"({free_gib:.1f} GiB free under {dataset.cache_dir})",
        flush=True,
    )
    if not todo:
        return n_have

    workers = max(1, min(int(max_workers), len(todo)))
    errors: list[str] = []

    def _one(study_uid: str) -> str | None:
        try:
            dataset._ensure_disk_cache(study_uid)
            return None
        except Exception as exc:  # noqa: BLE001 — keep warming the rest
            return f"{study_uid}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, uid) for uid in todo]
        for fut in tqdm(
            as_completed(futures),
            total=len(todo),
            desc="cache volumes (disk)",
            file=sys.stdout,
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            err = fut.result()
            if err:
                errors.append(err)
                print(err, flush=True)
    if errors:
        print(f"Volume cache: {len(errors)} studies failed (see above)", flush=True)
    return n - len(errors)


def prepare_disk_volume_cache(
    data_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    *,
    labeled_only: bool = False,
    pseudo_labels_path: Path | str | None = None,
    min_confidence: float = 0.7,
    volume_shape: tuple[int, int, int] = (16, 256, 256),
    max_series: int = 3,
    max_workers: int = 8,
) -> KneeStudyDataset:
    """Decode packed uint8 volumes into ``cache_dir`` (Kaggle Dataset staging)."""
    from rsna_knee.data.volume_cache import discover_volume_cache_roots
    from rsna_knee.utils.paths import is_kaggle_kernel

    if cache_dir is None:
        cache_dir = (
            Path("/kaggle/working/volume_cache")
            if is_kaggle_kernel()
            else Path("volume_cache")
        )
    cache_dir = Path(cache_dir)
    ds = KneeStudyDataset(
        data_root,
        split="train",
        labeled_only=labeled_only,
        pseudo_labels_path=pseudo_labels_path,
        min_confidence=min_confidence,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=False,
        cache_dir=cache_dir,
    )
    extra = discover_volume_cache_roots()
    tmp = Path("/tmp/rsna_volume_cache")
    if tmp.is_dir():
        extra = [tmp, tmp / ds._shape_subdir(), *extra]
    ds.enable_disk_cache(cache_dir, extra_read_dirs=extra)
    warm_volume_cache(ds, max_workers=max_workers)
    return ds


def materialize_writable_volume_cache(dataset: KneeStudyDataset) -> int:
    """Copy npy files that exist only on read mounts into the writable cache dir."""
    if dataset.cache_dir is None:
        return 0
    n_copied = 0
    for uid in dataset.study_ids:
        dest = dataset._cache_path(uid)
        if _cache_file_ready(dest):
            continue
        src = dataset._find_cached_npy(uid)
        if src is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n_copied += 1
    if n_copied:
        print(f"Copied {n_copied} npy files into {dataset.cache_dir}", flush=True)
    return n_copied


class StudyIndex:
    """Lightweight index over train studies, series, and labels."""

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or default_data_root()
        self.studies = load_train_table(self.data_root)
        self.series = load_train_series_table(self.data_root)
        self._validate()

    def _validate(self) -> None:
        missing = set(TARGET_LABELS) - set(self.studies.columns)
        if missing:
            raise ValueError(f"train.csv missing label columns: {missing}")

    def get_study(self, study_uid: str) -> pd.Series:
        row = self.studies[self.studies[STUDY_ID_COL] == study_uid]
        if row.empty:
            raise KeyError(f"Unknown study: {study_uid}")
        return row.iloc[0]

    def get_series_for_study(self, study_uid: str) -> pd.DataFrame:
        return self.series[self.series[STUDY_ID_COL] == study_uid].copy()

    def labels_for_study(self, study_uid: str) -> np.ndarray:
        row = self.get_study(study_uid)
        return row[TARGET_LABELS].to_numpy(dtype=np.float32)

    def report_for_study(self, study_uid: str) -> str:
        row = self.get_study(study_uid)
        return str(row.get(REPORT_COL, ""))

    def load_series_volume(
        self,
        study_uid: str,
        series_uid: str,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        path = series_dir(self.data_root, split="train") / study_uid / series_uid
        volume, _ = load_series_volume(path)
        if normalize:
            volume = normalize_volume(volume)
        return volume

    def iter_studies(self) -> list[str]:
        return self.studies[STUDY_ID_COL].tolist()

    def labeled_study_ids(self) -> list[str]:
        """Study UIDs with at least one non-null explicit label."""
        mask = labels_present_mask(self.studies)
        return self.studies.loc[mask, STUDY_ID_COL].tolist()


def fluid_sensitive_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to fluid-sensitive (PD/STIR) series when the column is present."""
    if FLUID_COL not in series_df.columns:
        return series_df
    fluid = series_df[series_df[FLUID_COL] == 1]
    return fluid if not fluid.empty else series_df


def select_series_for_study(
    series_df: pd.DataFrame,
    *,
    max_series: int = 3,
) -> pd.DataFrame:
    """
    Prefer fluid-sensitive series, then diversify by anatomical plane.

    Returns up to ``max_series`` rows ordered by plane priority (sag → cor → ax).
    """
    preferred = fluid_sensitive_series(series_df)
    if preferred.empty:
        preferred = series_df

    if ANATOMICAL_PLANE_COL in preferred.columns:
        plane = preferred[ANATOMICAL_PLANE_COL].fillna("").astype(str).str.strip().str.lower()
        preferred = preferred.assign(_plane_rank=plane.map(_PLANE_PRIORITY).fillna(99))
        # One series per distinct plane when possible, then fill remaining slots.
        selected_rows: list[pd.Series] = []
        used_uids: set[str] = set()
        for _, group in preferred.sort_values("_plane_rank").groupby("_plane_rank", sort=True):
            row = group.iloc[0]
            uid = str(row[SERIES_ID_COL])
            if uid not in used_uids:
                selected_rows.append(row)
                used_uids.add(uid)
            if len(selected_rows) >= max_series:
                break
        if len(selected_rows) < max_series:
            for _, row in preferred.sort_values("_plane_rank").iterrows():
                uid = str(row[SERIES_ID_COL])
                if uid in used_uids:
                    continue
                selected_rows.append(row)
                used_uids.add(uid)
                if len(selected_rows) >= max_series:
                    break
        out = pd.DataFrame(selected_rows).drop(columns=["_plane_rank"], errors="ignore")
        return out.reset_index(drop=True)

    return preferred.head(max_series).reset_index(drop=True)


def _labels_and_mask(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels = row[TARGET_LABELS].to_numpy(dtype=np.float32)
    mask = (~np.isnan(labels)).astype(np.float32)
    labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)
    return labels, mask


class KneeStudyDataset:
    """
    Study-level dataset for Phase 1/2 image baseline.

    Each item is one study: stacked 2.5D slices from up to ``max_series``
    fluid-sensitive series, plus multilabel targets and a NaN mask.

    Phase 2: optional ``pseudo_labels_path`` supplies report-derived labels
    for report-only studies. Explicit ground-truth labels always take priority.

    ``cache_dir`` is the writable packed uint8 store. Optional read-only
    Kaggle Dataset mounts are searched first so a published cache survives
    Jupyter restarts.

    Returns dict with:
      - ``study_uid``: str
      - ``image``: float32 array ``[S, 3, H, W]`` (S = max_series * depth)
      - ``labels``: float32 ``[12]``
      - ``mask``: float32 ``[12]`` (1 = supervised label present)
      - ``confidence``: float32 ``[12]`` (1.0 for ground truth; pseudo-label conf)
      - ``report``: str (train split only, empty for test)
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        split: str = "train",
        study_ids: list[str] | None = None,
        labeled_only: bool = True,
        pseudo_labels_path: Path | str | None = None,
        min_confidence: float = 0.7,
        volume_shape: tuple[int, int, int] = (16, 256, 256),
        max_series: int = 3,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        require_dicom: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        self.split = split
        self.volume_shape = volume_shape
        self.max_series = max_series
        self.cache = cache
        self.cache_dir: Path | None = None
        self._cache_read_dirs: list[Path] = []
        self.require_dicom = require_dicom
        self.min_confidence = float(min_confidence)
        self._cache: dict[str, np.ndarray] = {}
        self._pack_locks: dict[str, threading.Lock] = {}
        self._pack_locks_guard = threading.Lock()
        self._pseudo: pd.DataFrame | None = None
        if pseudo_labels_path is not None:
            self._pseudo = load_pseudo_labels(pseudo_labels_path).set_index(STUDY_ID_COL)

        if split == "train":
            self.studies = load_train_table(self.data_root)
            self.series = load_train_series_table(self.data_root)
            if study_ids is None:
                if labeled_only:
                    study_ids = self.studies.loc[
                        labels_present_mask(self.studies), STUDY_ID_COL
                    ].tolist()
                else:
                    study_ids = self.studies[STUDY_ID_COL].tolist()
        else:
            self.studies = load_test_table(self.data_root)
            self.series = load_test_series_table(self.data_root)
            if study_ids is None:
                study_ids = self.studies[STUDY_ID_COL].tolist()

        self.study_ids = [str(uid) for uid in study_ids]
        self._study_pos = {
            str(uid): i for i, uid in enumerate(self.studies[STUDY_ID_COL].astype(str))
        }
        self._series_by_study: dict[str, pd.DataFrame] = {
            str(uid): grp.reset_index(drop=True)
            for uid, grp in self.series.groupby(
                self.series[STUDY_ID_COL].astype(str), sort=False
            )
        }
        self._empty_series = self.series.iloc[0:0]
        if cache_dir is not None:
            self.enable_disk_cache(cache_dir)

    def _shape_subdir(self) -> str:
        from rsna_knee.data.volume_cache import volume_cache_shape_name

        return volume_cache_shape_name(self.volume_shape, self.max_series)

    def enable_disk_cache(
        self,
        cache_dir: Path | str,
        *,
        extra_read_dirs: list[Path | str] | None = None,
    ) -> None:
        """Writable cache under ``cache_dir / d{D}_h{H}_w{W}_s{S}/`` plus read mounts."""
        self.cache_dir = Path(cache_dir) / self._shape_subdir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_read_dirs = []
        seen = {str(self.cache_dir)}
        for raw in extra_read_dirs or []:
            root = Path(raw)
            for cand in (root / self._shape_subdir(), root):
                key = str(cand)
                if key in seen or not cand.is_dir():
                    continue
                seen.add(key)
                self._cache_read_dirs.append(cand)

    def _find_cached_npy(self, study_uid: str) -> Path | None:
        name = f"{study_uid}.npy"
        if self.cache_dir is not None:
            path = self.cache_dir / name
            if _cache_file_ready(path):
                return path
        for folder in self._cache_read_dirs:
            path = folder / name
            if _cache_file_ready(path):
                return path
        return None

    def __len__(self) -> int:
        return len(self.study_ids)

    def _study_row(self, study_uid: str) -> pd.Series:
        pos = self._study_pos.get(str(study_uid))
        if pos is None:
            raise KeyError(f"Unknown study: {study_uid}")
        return self.studies.iloc[pos]

    def stacked_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Labels / masks / confidence for every study — no DICOM reads."""
        n = len(self.study_ids)
        k = len(TARGET_LABELS)
        labels = np.zeros((n, k), dtype=np.float32)
        masks = np.zeros((n, k), dtype=np.float32)
        confidence = np.zeros((n, k), dtype=np.float32)
        if self.split != "train":
            return labels, masks, confidence
        for i, uid in enumerate(self.study_ids):
            labels[i], masks[i], confidence[i] = self._resolve_labels(
                uid, self._study_row(uid)
            )
        return labels, masks, confidence

    def __getitem__(self, index: int) -> dict[str, Any]:
        study_uid = self.study_ids[index]
        try:
            image = self._load_image(study_uid)
        except Exception as exc:  # noqa: BLE001 — one study must not kill a fold
            logger.warning("Zero image for %s: %s", study_uid, exc)
            depth, height, width = self.volume_shape
            image = self._gray_to_rgb(
                np.zeros((self.max_series, depth, height, width), dtype=np.float32)
            )

        report = ""
        if self.split == "train":
            row = self._study_row(study_uid)
            labels, mask, confidence = self._resolve_labels(study_uid, row)
            report = str(row.get(REPORT_COL, ""))
        else:
            labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            mask = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            confidence = np.zeros(len(TARGET_LABELS), dtype=np.float32)

        return {
            "study_uid": study_uid,
            "image": image,
            "labels": labels,
            "mask": mask,
            "confidence": confidence,
            "report": report,
        }

    def _resolve_labels(self, study_uid: str, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ground truth for labeled studies; pseudo-labels otherwise."""
        explicit_mask = bool(row[TARGET_LABELS].notna().any())
        if explicit_mask:
            labels, mask = _labels_and_mask(row)
            confidence = mask.copy()
            return labels, mask, confidence

        labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        mask = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        confidence = np.zeros(len(TARGET_LABELS), dtype=np.float32)

        if self._pseudo is not None and study_uid in self._pseudo.index:
            pseudo_row = self._pseudo.loc[study_uid]
            for i, name in enumerate(TARGET_LABELS):
                conf_col = f"{name}{CONF_SUFFIX}"
                conf = float(pseudo_row.get(conf_col, 0.0))
                if conf >= self.min_confidence:
                    labels[i] = float(pseudo_row.get(name, 0.0))
                    mask[i] = 1.0
                    confidence[i] = conf
        return labels, mask, confidence

    def _cache_path(self, study_uid: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{study_uid}.npy"

    def _study_lock(self, study_uid: str) -> threading.Lock:
        with self._pack_locks_guard:
            lock = self._pack_locks.get(study_uid)
            if lock is None:
                lock = threading.Lock()
                self._pack_locks[study_uid] = lock
            return lock

    def _packed_to_rgb(self, packed: np.ndarray) -> np.ndarray:
        return self._gray_to_rgb(packed.astype(np.float32) * (1.0 / 255.0))

    def _gray_to_rgb(self, gray: np.ndarray) -> np.ndarray:
        parts = [stack_adjacent_as_rgb(gray[s]) for s in range(self.max_series)]
        return np.concatenate(parts, axis=0)

    def _ensure_disk_cache(self, study_uid: str) -> None:
        """Write the uint8 cache file if missing. Does not keep the array in RAM."""
        if self._find_cached_npy(study_uid) is not None:
            return
        if self.cache_dir is None:
            return
        path = self._cache_path(study_uid)
        with self._study_lock(study_uid):
            if self._find_cached_npy(study_uid) is not None:
                return
            try:
                packed = self._decode_packed(study_uid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Placeholder cache for %s: %s", study_uid, exc)
                depth, height, width = self.volume_shape
                packed = np.zeros(
                    (self.max_series, depth, height, width), dtype=np.uint8
                )
            _atomic_save_npy(path, packed)

    def _load_packed(self, study_uid: str) -> np.ndarray:
        """Grayscale uint8 ``[max_series, D, H, W]`` from disk cache or DICOMs."""
        found = self._find_cached_npy(study_uid)
        if found is not None:
            return np.load(found)
        try:
            packed = self._decode_packed(study_uid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Zero volume for %s: %s", study_uid, exc)
            depth, height, width = self.volume_shape
            packed = np.zeros((self.max_series, depth, height, width), dtype=np.uint8)
        if self.cache_dir is not None:
            with self._study_lock(study_uid):
                found = self._find_cached_npy(study_uid)
                if found is not None:
                    return np.load(found)
                _atomic_save_npy(self._cache_path(study_uid), packed)
        return packed

    def _decode_grayscale(self, study_uid: str) -> np.ndarray:
        series_rows = select_series_for_study(
            self._series_by_study.get(study_uid, self._empty_series),
            max_series=self.max_series,
        )
        depth, height, width = self.volume_shape
        gray = np.zeros((self.max_series, depth, height, width), dtype=np.float32)
        slot = 0
        for _, row in series_rows.iterrows():
            if slot >= self.max_series:
                break
            series_uid = str(row[SERIES_ID_COL])
            path = series_dir(self.data_root, split=self.split) / study_uid / series_uid
            if not path.is_dir():
                if self.require_dicom:
                    raise FileNotFoundError(f"Missing series directory: {path}")
                continue
            try:
                volume, _ = load_series_volume(path, depth=depth)
                volume = normalize_volume(volume)
                gray[slot] = resize_volume(
                    volume, depth=depth, height=height, width=width
                )
                slot += 1
            except Exception as exc:  # noqa: BLE001 — keep other series for this study
                logger.warning("Skipping series %s in %s: %s", series_uid, study_uid, exc)
                continue
        return gray

    def _decode_packed(self, study_uid: str) -> np.ndarray:
        gray = self._decode_grayscale(study_uid)
        return np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)

    def _load_image(self, study_uid: str) -> np.ndarray:
        if self.cache and study_uid in self._cache:
            return self._cache[study_uid]
        if self.cache_dir is not None:
            image = self._packed_to_rgb(self._load_packed(study_uid))
        else:
            image = self._gray_to_rgb(self._decode_grayscale(study_uid))
        if self.cache:
            self._cache[study_uid] = image
        return image


def collate_studies(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate KneeStudyDataset items into batched numpy / list fields."""
    return {
        "study_uid": [item["study_uid"] for item in batch],
        "image": np.stack([item["image"] for item in batch], axis=0),
        "labels": np.stack([item["labels"] for item in batch], axis=0),
        "mask": np.stack([item["mask"] for item in batch], axis=0),
        "confidence": np.stack([item["confidence"] for item in batch], axis=0),
        "report": [item.get("report", "") for item in batch],
    }
