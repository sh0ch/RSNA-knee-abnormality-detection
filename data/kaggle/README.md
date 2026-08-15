# Competition CSVs (local copy)

Official **tabular files only** from [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), kept locally for CSV/report EDA without running on Kaggle.

## Contents

| File | Description |
|------|-------------|
| `train.csv` | 4,407 studies — `Report` + 12 label columns |
| `train_series.csv` | 24,371 series — fluid/fat flags, anatomical plane |
| `test.csv` | Test study IDs (public sample) |
| `test_series.csv` | Test series metadata |
| `sample_submission.csv` | Submission template |

**Not included here:** `train_series/` and `test_series/` DICOM folders (~multi-TB). Use Kaggle kernels for full MRI data.

## How to obtain

1. Accept the competition rules on Kaggle.
2. Download from the [Data tab](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) — CSV files only, or copy them from a full download.
3. Place the five CSVs in this directory (flat layout as above).

Alternatively, with the [Kaggle API](https://github.com/Kaggle/kaggle-api) authenticated:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/kaggle
kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data/kaggle
# ... repeat for test.csv, test_series.csv, sample_submission.csv
```

## Use in this repo

Point the package at this folder:

```bash
# PowerShell (session)
$env:RSNA_DATA_ROOT = "data/kaggle"

# Or in .env (gitignored)
RSNA_DATA_ROOT=data/kaggle
```

Then in Python or `notebooks/02_eda_phase0.ipynb`:

```python
from rsna_knee.utils.paths import default_data_root
from rsna_knee.data.schema import read_competition_csv, load_train_table

print(default_data_root())  # .../data/kaggle
train = read_competition_csv(default_data_root() / "train.csv")
```

`load_train_table()` and friends work the same as on Kaggle. DICOM cells will fail unless you also have `train_series/` under the same root (use `data/sample/` or Kaggle for imaging).

## Git

CSV contents are **gitignored** (competition data must not be committed). Only this README is tracked.

See [docs/DATA.md](../../docs/DATA.md) for full schema and layout.
