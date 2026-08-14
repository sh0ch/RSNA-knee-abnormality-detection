"""Tests for environment-aware path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsna_knee.constants import TRAIN_CSV
from rsna_knee.utils import paths


def test_csv_path_accepts_string_root(tmp_path: Path) -> None:
    (tmp_path / TRAIN_CSV).write_text("StudyInstanceUID\n", encoding="utf-8")
    root_str = str(tmp_path)
    assert paths.train_csv(root_str) == tmp_path / TRAIN_CSV


def test_series_dir_accepts_string_root(tmp_path: Path) -> None:
    root_str = str(tmp_path)
    assert paths.series_dir(root_str, split="train") == tmp_path / "train_series"


def test_kaggle_candidates_include_competitions_subfolder() -> None:
    candidates = paths._kaggle_data_candidates()
    assert Path("/kaggle/input/competitions/rsna-knee-abnormality-detection") in candidates
    assert Path("/kaggle/input/rsna-knee-abnormality-detection") in candidates


def test_default_data_root_on_kaggle_uses_competitions_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    competitions_root = tmp_path / "input" / "competitions" / "rsna-knee-abnormality-detection"
    competitions_root.mkdir(parents=True)
    (competitions_root / TRAIN_CSV).write_text("StudyInstanceUID\n", encoding="utf-8")

    monkeypatch.setenv("KAGGLE_KERNEL_RUN", "True")
    monkeypatch.setattr(
        paths,
        "_kaggle_data_candidates",
        lambda: [competitions_root, tmp_path / "input" / "rsna-knee-abnormality-detection"],
    )

    assert paths.default_data_root() == competitions_root
