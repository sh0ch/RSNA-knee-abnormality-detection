"""Tests for DICOM I/O and study indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rsna_knee.data import StudyIndex, load_series_volume, normalize_volume
from rsna_knee.training import macro_roc_auc


def test_load_series_volume(sample_data_dir: Path) -> None:
    index = StudyIndex(sample_data_dir)
    study_uid = index.iter_studies()[0]
    series_df = index.get_series_for_study(study_uid)
    series_uid = series_df.iloc[0]["SeriesInstanceUID"]

    volume, datasets = load_series_volume(
        sample_data_dir / "train_series" / study_uid / series_uid
    )
    assert volume.ndim == 3
    assert volume.shape[0] == len(datasets)
    assert volume.dtype == np.float32


def test_normalize_volume() -> None:
    vol = np.random.randn(10, 32, 32).astype(np.float32) * 100
    normed = normalize_volume(vol)
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0 + 1e-6


def test_study_index_labels(sample_data_dir: Path) -> None:
    index = StudyIndex(sample_data_dir)
    study_uid = index.iter_studies()[0]
    labels = index.labels_for_study(study_uid)
    assert labels.shape == (12,)
    assert set(labels.tolist()).issubset({0.0, 1.0})


def test_macro_roc_auc_perfect() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]], dtype=float)
    score = macro_roc_auc(y_true, y_pred, labels=["a", "b"])
    assert score == 1.0
