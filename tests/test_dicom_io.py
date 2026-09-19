"""Tests for DICOM I/O and study indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rsna_knee.data import StudyIndex, load_series_volume, normalize_volume
from rsna_knee.data.dicom_io import read_dicom_slice
from rsna_knee.training import macro_roc_auc


def _write_slice(
    path: Path,
    pixels: np.ndarray,
    *,
    bits_allocated: int,
    pixel_bytes: bytes,
) -> None:
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


def test_read_dicom_slice_8bit_buffer_16bit_header(tmp_path: Path) -> None:
    """Some RSNA files store 8-bit PixelData while BitsAllocated says 16."""
    rows, cols = 32, 40
    pixels = np.arange(rows * cols, dtype=np.uint8).reshape(rows, cols)
    path = tmp_path / "mismatch.dcm"
    _write_slice(
        path,
        pixels,
        bits_allocated=16,
        pixel_bytes=pixels.tobytes(),
    )
    _, got = read_dicom_slice(path)
    assert got.shape == (rows, cols)
    np.testing.assert_array_equal(got.astype(np.uint8), pixels)


def test_load_series_skips_unreadable_slice(tmp_path: Path) -> None:
    series = tmp_path / "series"
    series.mkdir()
    good = np.full((16, 16), 77, dtype=np.uint16)
    _write_slice(
        series / "ok.dcm",
        good,
        bits_allocated=16,
        pixel_bytes=good.tobytes(),
    )
    _write_slice(
        series / "bad.dcm",
        good,
        bits_allocated=16,
        pixel_bytes=b"\x00\x01",
    )
    volume, datasets = load_series_volume(series)
    assert volume.shape[0] == 1
    assert len(datasets) == 1


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


def test_load_series_volume_samples_depth(sample_data_dir: Path) -> None:
    from rsna_knee.data.volume_prep import sample_depth_indices

    index = StudyIndex(sample_data_dir)
    study_uid = index.iter_studies()[0]
    series_df = index.get_series_for_study(study_uid)
    series_uid = series_df.iloc[0]["SeriesInstanceUID"]
    path = sample_data_dir / "train_series" / study_uid / series_uid

    full, _ = load_series_volume(path)
    sampled, datasets = load_series_volume(path, depth=4)
    idx = sample_depth_indices(full.shape[0], 4)
    np.testing.assert_allclose(sampled, full[idx])
    assert len(datasets) == 4


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
