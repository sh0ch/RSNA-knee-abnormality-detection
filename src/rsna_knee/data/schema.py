"""Normalize competition CSV schemas to canonical column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.constants import (
    ANATOMICAL_PLANE_COL,
    FAT_SUPPRESSION_COL,
    FLUID_COL,
    KAGGLE_LABEL_COLUMNS,
    PATIENT_SEX_COL,
    REPORT_COL,
    SAMPLE_SUBMISSION_CSV,
    SERIES_ID_COL,
    STUDY_ID_COL,
    SUBMISSION_LABEL_COLUMNS,
    TARGET_LABELS,
    TEST_CSV,
    TEST_SERIES_CSV,
    TRAIN_CSV,
    TRAIN_SERIES_CSV,
)
from rsna_knee.utils.paths import (
    sample_submission_csv,
    test_csv,
    test_series_csv,
    train_csv,
    train_series_csv,
)

# Known header variants -> canonical column name
_LABEL_ALIASES: dict[str, str] = {}
for canonical, kaggle in zip(TARGET_LABELS, KAGGLE_LABEL_COLUMNS, strict=True):
    _LABEL_ALIASES[canonical] = canonical
    _LABEL_ALIASES[kaggle] = canonical

_SERIES_ALIASES: dict[str, str] = {
    "FluidSensitiveSeries": FLUID_COL,
    "Fluid_Sensitive": FLUID_COL,
    FLUID_COL: FLUID_COL,
    "Fat_Suppression": FAT_SUPPRESSION_COL,
    FAT_SUPPRESSION_COL: FAT_SUPPRESSION_COL,
    "Anatomical_Plane": ANATOMICAL_PLANE_COL,
    ANATOMICAL_PLANE_COL: ANATOMICAL_PLANE_COL,
    STUDY_ID_COL: STUDY_ID_COL,
    SERIES_ID_COL: SERIES_ID_COL,
}


def _rename_columns(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    rename = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=rename)


def normalize_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle train.csv headers to canonical TARGET_LABELS columns."""
    out = _rename_columns(df, _LABEL_ALIASES)
    missing = [label for label in TARGET_LABELS if label not in out.columns]
    if missing:
        raise ValueError(
            f"train.csv missing label columns after normalization: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    for label in TARGET_LABELS:
        out[label] = pd.to_numeric(out[label], errors="coerce")
    return out


def normalize_series_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle train_series.csv headers to canonical names."""
    return _rename_columns(df, _SERIES_ALIASES)


def submission_columns() -> list[str]:
    """Submission CSV label headers in competition order."""
    return KAGGLE_LABEL_COLUMNS.copy()


def predictions_to_submission(
    study_ids: list[str],
    predictions: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """
    Build a submission DataFrame with Kaggle column names.

    ``predictions`` must use canonical TARGET_LABELS as columns (or be a 2D array
    with columns in TARGET_LABELS order).
    """
    if isinstance(predictions, pd.Series):
        preds = predictions.to_frame().T
    else:
        preds = predictions.copy()

    missing = [label for label in TARGET_LABELS if label not in preds.columns]
    if missing:
        raise ValueError(f"Predictions missing canonical label columns: {missing}")

    out = pd.DataFrame({STUDY_ID_COL: study_ids})
    for canonical, kaggle_col in SUBMISSION_LABEL_COLUMNS.items():
        out[kaggle_col] = preds[canonical].to_numpy()
    return out


def load_train_table(data_root: Path | str | None = None) -> pd.DataFrame:
    return normalize_train_df(pd.read_csv(train_csv(data_root)))


def load_train_series_table(data_root: Path | str | None = None) -> pd.DataFrame:
    return normalize_series_df(pd.read_csv(train_series_csv(data_root)))


def load_test_table(data_root: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(test_csv(data_root))


def load_test_series_table(data_root: Path | str | None = None) -> pd.DataFrame:
    return normalize_series_df(pd.read_csv(test_series_csv(data_root)))


def load_sample_submission(data_root: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(sample_submission_csv(data_root))
