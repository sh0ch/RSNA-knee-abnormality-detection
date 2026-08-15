# RSNA Knee Abnormality Detection

Multimodal deep learning workspace for the [RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition.

Predict 12 knee abnormalities from DICOM MRI series (and radiology reports during training). Develop locally on a tiny synthetic dataset; train and submit on Kaggle where the full data lives.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/create_sample_data.py
pytest
```

## Workflow

| Environment | Purpose | Data |
|-------------|---------|------|
| **Local (CSVs)** | Tabular EDA, reports, schema | `data/kaggle/` — official CSVs ([README](data/kaggle/README.md)) |
| **Local (DICOM)** | Pipeline tests, unit tests | `data/sample/` — synthetic (~few MB) |
| **Kaggle** | Full training, inference, submissions | `/kaggle/input/competitions/rsna-knee-abnormality-detection` |

```
Local dev ──push──▶ GitHub ──clone──▶ Kaggle Notebook ──submit──▶ Leaderboard
```

## Repository structure

```
├── src/rsna_knee/       # Core package (DICOM I/O, datasets, metrics)
├── configs/             # default.yaml (local) · kaggle.yaml (notebook)
├── scripts/             # create_sample_data.py · sync_kaggle_eda.py
├── kaggle/eda/          # Generated EDA kernel for Kaggle
├── notebooks/           # Local Jupyter experiments
├── tests/               # pytest suite
└── docs/                # Documentation + project log
```

## Documentation

| Doc | Contents |
|-----|----------|
| **[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md)** | **Living insights & decisions (start here after EDA)** |
| [docs/COMPETITION.md](docs/COMPETITION.md) | Task, labels, dates, rules summary |
| [docs/DATA.md](docs/DATA.md) | CSV schemas, DICOM layout, preprocessing |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, standards, project workflow |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Running notebooks and submissions |

## Key finding (Phase 0)

Only **58 / 4,407** training studies have explicit 0/1 labels; the rest are report-only. See [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

## Targets (12 multilabel)

ACL/MCL tear · medial/lateral meniscus injury · medial/lateral/patellofemoral osteoarthritis · joint effusion · synovitis · Baker's cyst · bone contusion · fracture

**Metric:** Macro ROC-AUC

## Example — load a DICOM series locally

```python
from rsna_knee.data import StudyIndex, load_series_volume

index = StudyIndex()  # uses data/sample/ by default
study_uid = index.iter_studies()[0]
series_uid = index.get_series_for_study(study_uid).iloc[0]["SeriesInstanceUID"]

volume, meta = load_series_volume(
    index.data_root / "train_series" / study_uid / series_uid
)
print(volume.shape)  # (slices, H, W)
```

## EDA notebook

- Source: `notebooks/02_eda_phase0.ipynb`
- Kaggle: `python scripts/sync_kaggle_eda.py --push` → see [docs/KAGGLE.md](docs/KAGGLE.md)

## License

MIT for this repository's code. Competition data is subject to [Kaggle/RSNA terms](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
