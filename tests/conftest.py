"""Pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_sample_data import create_sample_dataset


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("sample_data")
    create_sample_dataset(data_dir, num_train_studies=2, num_test_studies=1)
    return data_dir
