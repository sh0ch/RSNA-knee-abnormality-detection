# Data layout

## Official dataset (Kaggle only)

Mount path in notebooks (auto-detected by `default_data_root()`):

```
/kaggle/input/competitions/rsna-knee-abnormality-detection/
```

Older notebooks may still see `/kaggle/input/rsna-knee-abnormality-detection/` when the dataset is attached directly. The package checks both locations.

| File / directory | Description |
|------------------|-------------|
| `train.csv` | One row per study: `StudyInstanceUID`, `PatientSex`, `Report`, 12 label columns |
| `train_series.csv` | One row per MRI series; `FluidSensitiveSeries` flags PD/STIR sequences |
| `train_series/` | DICOM slices: `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm` |
| `test.csv` | Study IDs for inference (no `Report` during scoring) |
| `test_series.csv` | Series metadata for test studies |
| `test_series/` | Test DICOMs (public example set; replaced during scoring) |
| `sample_submission.csv` | Submission template |

## DICOM notes

- Each `.dcm` file is a single slice; series typically have 20–45 slices (median 30).
- Transfer syntaxes vary (uncompressed, JPEG Lossless, JPEG 2000, etc.) — use `pydicom`.
- Metadata is stripped to an allowlisted set of 86 tags.
- Intensities, orientation, and resolution vary across sites.

## Local sample data

This repo does **not** include competition data. Generate a tiny synthetic copy for pipeline tests:

```bash
python scripts/create_sample_data.py
```

Output goes to `data/sample/` by default (~few MB). Set `RSNA_DATA_ROOT` to point elsewhere if needed.

## Recommended preprocessing (starting point)

1. Group series by `StudyInstanceUID`.
2. Prefer fluid-sensitive series (`FluidSensitiveSeries == 1`) for soft-tissue findings.
3. Stack slices into 3D volumes; sort by `InstanceNumber` or `ImagePositionPatient[2]`.
4. Apply `RescaleSlope` / `RescaleIntercept` before normalization.
5. Resize/crop to a fixed tensor shape for 3D CNNs or sample 2D slices for 2D backbones.
6. For multimodal models: encode `Report` with a text encoder during training only.

See `src/rsna_knee/data/dicom_io.py` for reference implementations.
