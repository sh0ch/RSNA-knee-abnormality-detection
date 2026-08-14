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
| **Local** | Prototyping, unit tests, DICOM pipeline | `data/sample/` (synthetic, ~few MB) |
| **Kaggle** | Full training, inference, submissions | `/kaggle/input/rsna-knee-abnormality-detection` |

```
Local dev ──push──▶ GitHub ──clone──▶ Kaggle Notebook ──submit──▶ Leaderboard
```

## Repository structure

```
├── src/rsna_knee/       # Core package (DICOM I/O, datasets, metrics)
├── configs/             # default.yaml (local) · kaggle.yaml (notebook)
├── scripts/             # create_sample_data.py
├── kaggle/kernels/      # Notebook template for Kaggle
├── notebooks/           # Local Jupyter experiments
├── tests/               # pytest suite
└── docs/                # Detailed documentation
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/COMPETITION.md](docs/COMPETITION.md) | Task, labels, dates, rules summary |
| [docs/DATA.md](docs/DATA.md) | CSV schemas, DICOM layout, preprocessing |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, standards, project workflow |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Running notebooks and submissions |

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

## Kaggle notebook

See [docs/KAGGLE.md](docs/KAGGLE.md) and `kaggle/kernels/train_template.ipynb`.

## License

MIT for this repository's code. Competition data is subject to [Kaggle/RSNA terms](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
