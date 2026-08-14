"""Tests for competition CSV schema normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.constants import (
    FLUID_COL,
    KAGGLE_LABEL_COLUMNS,
    TARGET_LABELS,
)
from rsna_knee.data.schema import (
    label_coverage,
    labels_present_mask,
    normalize_series_df,
    normalize_train_df,
    predictions_to_submission,
)


def _kaggle_train_row() -> pd.DataFrame:
    labels = dict(zip(KAGGLE_LABEL_COLUMNS, [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], strict=True))
    return pd.DataFrame(
        {
            "StudyInstanceUID": ["1.2.3"],
            "Report": ["Sample report"],
            **labels,
        }
    )


def test_normalize_train_df_kaggle_columns() -> None:
    raw = _kaggle_train_row()
    out = normalize_train_df(raw)
    assert list(out[TARGET_LABELS].iloc[0]) == [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]


def test_normalize_train_df_snake_case_columns() -> None:
    raw = pd.DataFrame(
        {
            "StudyInstanceUID": ["1.2.3"],
            "Report": ["Sample report"],
            **{label: 0 for label in TARGET_LABELS},
        }
    )
    raw["acl_tear"] = 1
    out = normalize_train_df(raw)
    assert out["acl_tear"].iloc[0] == 1


def test_normalize_series_df_kaggle_columns() -> None:
    raw = pd.DataFrame(
        {
            "StudyInstanceUID": ["1.2.3"],
            "SeriesInstanceUID": ["4.5.6"],
            "Fluid_Sensitive": [1],
            "Fat_Suppression": [0],
            "Anatomical_Plane": ["Sagittal"],
        }
    )
    out = normalize_series_df(raw)
    assert out[FLUID_COL].iloc[0] == 1
    assert out["fat_suppression"].iloc[0] == 0
    assert out["anatomical_plane"].iloc[0] == "Sagittal"


def test_label_coverage_partial_labels() -> None:
    raw = _kaggle_train_row()
    unlabeled = raw.assign(**dict.fromkeys(KAGGLE_LABEL_COLUMNS, float("nan")))
    out = normalize_train_df(pd.concat([raw, unlabeled], ignore_index=True))
    cov = label_coverage(out)
    assert cov.loc[cov["label"] == "acl_tear", "labeled_studies"].iloc[0] == 1
    assert labels_present_mask(out).sum() == 1


def test_predictions_to_submission_uses_kaggle_headers() -> None:
    preds = pd.DataFrame({label: [0.1] for label in TARGET_LABELS})
    sub = predictions_to_submission(["study-1"], preds)
    assert list(sub.columns) == ["StudyInstanceUID", *KAGGLE_LABEL_COLUMNS]
    assert sub["ACL"].iloc[0] == pytest.approx(0.1)
