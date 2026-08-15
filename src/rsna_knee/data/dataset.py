"""PyTorch datasets and data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from rsna_knee.data.volume_prep import prepare_series_tensor
from rsna_knee.utils.paths import default_data_root, series_dir

# Preferred anatomical plane order when selecting up to N series per study.
_PLANE_PRIORITY: dict[str, int] = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


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
    Study-level dataset for Phase 1 image baseline.

    Each item is one study: stacked 2.5D slices from up to ``max_series``
    fluid-sensitive series, plus multilabel targets and a NaN mask.

    Returns dict with:
      - ``study_uid``: str
      - ``image``: float32 array ``[S, 3, H, W]`` (S = max_series * depth)
      - ``labels``: float32 ``[12]``
      - ``mask``: float32 ``[12]`` (1 = supervised label present)
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        split: str = "train",
        study_ids: list[str] | None = None,
        labeled_only: bool = True,
        volume_shape: tuple[int, int, int] = (16, 256, 256),
        max_series: int = 3,
        cache: bool = True,
        require_dicom: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        self.split = split
        self.volume_shape = volume_shape
        self.max_series = max_series
        self.cache = cache
        self.require_dicom = require_dicom
        self._cache: dict[str, np.ndarray] = {}

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

        self.study_ids = list(study_ids)

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        study_uid = self.study_ids[index]
        image = self._load_image(study_uid)

        if self.split == "train":
            row = self.studies[self.studies[STUDY_ID_COL] == study_uid].iloc[0]
            labels, mask = _labels_and_mask(row)
        else:
            labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            mask = np.zeros(len(TARGET_LABELS), dtype=np.float32)

        return {
            "study_uid": study_uid,
            "image": image,
            "labels": labels,
            "mask": mask,
        }

    def _load_image(self, study_uid: str) -> np.ndarray:
        if self.cache and study_uid in self._cache:
            return self._cache[study_uid]

        series_rows = select_series_for_study(
            self.series[self.series[STUDY_ID_COL] == study_uid],
            max_series=self.max_series,
        )
        depth, height, width = self.volume_shape
        tensors: list[np.ndarray] = []

        for _, row in series_rows.iterrows():
            series_uid = str(row[SERIES_ID_COL])
            path = series_dir(self.data_root, split=self.split) / study_uid / series_uid
            if not path.is_dir():
                if self.require_dicom:
                    raise FileNotFoundError(f"Missing series directory: {path}")
                continue
            volume, _ = load_series_volume(path)
            volume = normalize_volume(volume)
            tensors.append(prepare_series_tensor(volume, depth=depth, height=height, width=width))

        if not tensors:
            # Placeholder when DICOMs are absent (CSV-only local roots).
            image = np.zeros((self.max_series * depth, 3, height, width), dtype=np.float32)
        else:
            image = np.concatenate(tensors, axis=0)
            # Pad to a fixed number of series slots for stable batching.
            target_slices = self.max_series * depth
            if image.shape[0] < target_slices:
                pad = np.zeros(
                    (target_slices - image.shape[0], 3, height, width),
                    dtype=np.float32,
                )
                image = np.concatenate([image, pad], axis=0)
            elif image.shape[0] > target_slices:
                image = image[:target_slices]

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
    }
