# Data layout

## Official dataset (Kaggle only)

Mount path in notebooks (auto-detected by `default_data_root()`):

```
/kaggle/input/competitions/rsna-knee-abnormality-detection/
```

Older notebooks may still see `/kaggle/input/rsna-knee-abnormality-detection/` when the dataset is attached directly. The package checks both locations.

| File / directory | Description |
|------------------|-------------|
| `train.csv` | One row per study: `StudyInstanceUID`, `Report`, 12 label columns. **Only ~1.3% of studies (58 / 4,407) have explicit 0/1 labels**; the rest are report-only (NaN). See [PROJECT_LOG.md](PROJECT_LOG.md). |
| `train_series.csv` | One row per MRI series; `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
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

## Local data (two options)

### `data/kaggle/` — official CSVs (recommended for tabular EDA)

Copy the five competition **CSV files** into `data/kaggle/` (no DICOM). See [data/kaggle/README.md](../data/kaggle/README.md).

| File | Rows (full competition) |
|------|-------------------------|
| `train.csv` | 4,407 |
| `train_series.csv` | 24,371 |
| `test.csv` | 3 (public sample) |
| `test_series.csv` | 15 |
| `sample_submission.csv` | 3 |

Use real reports, labels, and series metadata locally without a Kaggle session:

```bash
# PowerShell
$env:RSNA_DATA_ROOT = "data/kaggle"
```

Or in `.env`: `RSNA_DATA_ROOT=data/kaggle`

CSV contents are **gitignored** — do not commit them.

### `data/sample/` — synthetic CSV + DICOM (pipeline tests)

Generate a tiny fake dataset for unit tests and DICOM I/O:

```bash
python scripts/create_sample_data.py
```

Output goes to `data/sample/` by default (~few MB). Includes miniature `train_series/` DICOM folders.

### Choosing a local root

| Goal | Set `RSNA_DATA_ROOT` to |
|------|-------------------------|
| Explore real reports & CSV schema | `data/kaggle` |
| Test DICOM loading / transforms | `data/sample` (after `create_sample_data.py`) |
| Full MRI EDA | Kaggle kernel (DICOM too large for git) |

If unset, `default_data_root()` uses `data/sample/` when not on Kaggle.

## Recommended preprocessing (starting point)

1. Group series by `StudyInstanceUID`.
2. Prefer fluid-sensitive series (`Fluid_Sensitive == 1`, normalized to `fluid_sensitive` in code) for soft-tissue findings.
3. Stack slices into 3D volumes; sort by `InstanceNumber` or `ImagePositionPatient[2]`.
4. Apply `RescaleSlope` / `RescaleIntercept` before normalization.
5. Resize/crop to a fixed tensor shape for 3D CNNs or sample 2D slices for 2D backbones.
6. For multimodal models: encode `Report` with a text encoder during training only.

See `src/rsna_knee/data/schema.py` for CSV column normalization (`ACL` → `acl_tear`, etc.).
