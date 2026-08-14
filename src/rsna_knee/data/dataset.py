"""PyTorch datasets and data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import (
    FLUID_COL,
    REPORT_COL,
    SERIES_ID_COL,
    STUDY_ID_COL,
    TARGET_LABELS,
)
from rsna_knee.data.dicom_io import load_series_volume, normalize_volume
from rsna_knee.utils.paths import default_data_root, series_dir, train_csv, train_series_csv


class StudyIndex:
    """Lightweight index over train studies, series, and labels."""

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or default_data_root()
        self.studies = pd.read_csv(train_csv(self.data_root))
        self.series = pd.read_csv(train_series_csv(self.data_root))
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


def fluid_sensitive_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to fluid-sensitive (PD/STIR) series when the column is present."""
    if FLUID_COL not in series_df.columns:
        return series_df
    fluid = series_df[series_df[FLUID_COL] == 1]
    return fluid if not fluid.empty else series_df
