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
Edit notebooks/ in Cursor
  -> connect to Kaggle Jupyter Server (daily)
  -> publish Dataset rsna-knee-code when src/ changes
  -> push_kaggle_kernel.py + Save & Run All (submit)
```

See **[docs/KAGGLE.md](docs/KAGGLE.md)** for the Cursor ↔ Kaggle Jupyter Server steps.

## Repository structure

```
├── src/rsna_knee/       # Core package (DICOM I/O, datasets, metrics, models)
├── configs/             # default.yaml · kaggle.yaml · kernel settings
├── scripts/             # sample data · publish Dataset · push kernel · weights
├── kaggle/eda/          # EDA kernel-metadata.json (submit settings)
├── kaggle/train/        # Phase 1 kernel-metadata.json (internet OFF)
├── notebooks/           # Source notebooks (edit here; connect remote kernel)
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
| [docs/KAGGLE.md](docs/KAGGLE.md) | Jupyter Server daily loop + submit |

## Key finding (Phase 0)

Only **58 / 4,407** training studies have explicit 0/1 labels; the rest are report-only. See [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

## Phase 1 notebook

- Source: `notebooks/03_phase1_image_baseline.ipynb`
- Package on Kaggle: `python scripts/publish_code_dataset.py`
- Submit push: `python scripts/push_kaggle_kernel.py train`
- Trains **from scratch** (pipeline smoke; weak score expected on 58 labels)

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
- Daily: Cursor → Kaggle Jupyter Server — [docs/KAGGLE.md](docs/KAGGLE.md)
- Submit push: `python scripts/push_kaggle_kernel.py eda`

## License

MIT for this repository's code. Competition data is subject to [Kaggle/RSNA terms](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
