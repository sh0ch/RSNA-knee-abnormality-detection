#!/usr/bin/env python3
"""Generate a minimal synthetic dataset for local DICOM pipeline testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rsna_knee.constants import (
    FLUID_COL,
    PATIENT_SEX_COL,
    REPORT_COL,
    SAMPLE_SUBMISSION_CSV,
    SERIES_ID_COL,
    STUDY_ID_COL,
    TARGET_LABELS,
    TEST_CSV,
    TEST_SERIES_CSV,
    TEST_SERIES_DIR,
    TRAIN_CSV,
    TRAIN_SERIES_CSV,
    TRAIN_SERIES_DIR,
)
from rsna_knee.utils.paths import project_root


def _make_uid() -> str:
    return generate_uid()


def _write_dicom_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    pixel_array: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    ds.PixelData = pixel_array.astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)


def _write_series(
    root: Path,
    study_uid: str,
    series_uid: str,
    num_slices: int,
    size: int = 64,
) -> None:
    series_path = root / study_uid / series_uid
    rng = np.random.default_rng(hash(series_uid) % (2**32))
    base = rng.integers(200, 800, size=(size, size), dtype=np.uint16)
    for i in range(1, num_slices + 1):
        noise = rng.integers(0, 30, size=(size, size), dtype=np.uint16)
        pixels = np.clip(base.astype(np.int32) + noise, 0, 4095).astype(np.uint16)
        _write_dicom_slice(
            series_path / f"slice_{i:04d}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            instance_number=i,
            pixel_array=pixels,
        )


def create_sample_dataset(
    output_dir: Path,
    *,
    num_train_studies: int = 4,
    num_test_studies: int = 2,
    series_per_study: int = 2,
    slices_per_series: int = 8,
) -> None:
    """Write CSVs and synthetic DICOMs mirroring competition layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    train_rows: list[dict] = []
    train_series_rows: list[dict] = []
    test_rows: list[dict] = []
    test_series_rows: list[dict] = []

    train_root = output_dir / TRAIN_SERIES_DIR
    test_root = output_dir / TEST_SERIES_DIR

    for _ in range(num_train_studies):
        study_uid = _make_uid()
        labels = {label: int(rng.random() > 0.7) for label in TARGET_LABELS}
        train_rows.append(
            {
                STUDY_ID_COL: study_uid,
                PATIENT_SEX_COL: rng.choice(["M", "F"]),
                REPORT_COL: "Sample radiology report for local testing.",
                **labels,
            }
        )
        for s in range(series_per_study):
            series_uid = _make_uid()
            train_series_rows.append(
                {
                    STUDY_ID_COL: study_uid,
                    SERIES_ID_COL: series_uid,
                    FLUID_COL: 1 if s == 0 else 0,
                }
            )
            _write_series(train_root, study_uid, series_uid, slices_per_series)

    for _ in range(num_test_studies):
        study_uid = _make_uid()
        test_rows.append({STUDY_ID_COL: study_uid, PATIENT_SEX_COL: rng.choice(["M", "F"])})
        for s in range(series_per_study):
            series_uid = _make_uid()
            test_series_rows.append(
                {
                    STUDY_ID_COL: study_uid,
                    SERIES_ID_COL: series_uid,
                    FLUID_COL: 1 if s == 0 else 0,
                }
            )
            _write_series(test_root, study_uid, series_uid, slices_per_series)

    pd.DataFrame(train_rows).to_csv(output_dir / TRAIN_CSV, index=False)
    pd.DataFrame(train_series_rows).to_csv(output_dir / TRAIN_SERIES_CSV, index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / TEST_CSV, index=False)
    pd.DataFrame(test_series_rows).to_csv(output_dir / TEST_SERIES_CSV, index=False)

    submission = pd.DataFrame(
        {
            STUDY_ID_COL: [r[STUDY_ID_COL] for r in test_rows],
            **dict.fromkeys(
                [
                    "ACL",
                    "MCL",
                    "Medial Meniscus",
                    "Lateral Meniscus",
                    "Medial OA",
                    "Lateral OA",
                    "PF OA",
                    "Effusion",
                    "Synovitis",
                    "Baker's",
                    "Contusion",
                    "Fracture",
                ],
                0.5,
            ),
        }
    )
    submission.to_csv(output_dir / SAMPLE_SUBMISSION_CSV, index=False)

    print(f"Sample dataset written to {output_dir}")
    print(f"  Train studies: {num_train_studies}, Test studies: {num_test_studies}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "sample",
        help="Output directory for synthetic data",
    )
    parser.add_argument("--train-studies", type=int, default=4)
    parser.add_argument("--test-studies", type=int, default=2)
    args = parser.parse_args()
    create_sample_dataset(
        args.output,
        num_train_studies=args.train_studies,
        num_test_studies=args.test_studies,
    )


if __name__ == "__main__":
    main()
