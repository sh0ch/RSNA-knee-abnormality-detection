# Development guide

## Philosophy

- **Develop locally** on synthetic/sample data — fast iteration, no multi-TB downloads.
- **Train and submit on Kaggle** — full dataset, GPU, and official scoring environment.
- **Keep logic in `src/rsna_knee/`** — notebooks stay thin; reusable code lives in the package.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install package + dev tools
pip install -e ".[dev]"

# Optional: training deps (PyTorch, MONAI, etc.)
pip install -e ".[train]"

# Generate local sample DICOM dataset
python scripts/create_sample_data.py
```

## Verify installation

```bash
pytest
ruff check src tests scripts
```

## Project layout

```
├── configs/           # YAML experiment configs (default, kaggle)
├── data/sample/       # Generated synthetic data (gitignored contents)
├── docs/              # Competition and workflow documentation
├── kaggle/kernels/    # Notebook templates for Kaggle
├── notebooks/         # Local Jupyter notebooks
├── scripts/           # CLI utilities
├── src/rsna_knee/     # Importable package
│   ├── constants.py   # Labels, column names, paths
│   ├── data/          # DICOM I/O, datasets, transforms
│   ├── models/        # Model architectures
│   ├── training/      # Metrics, training loops
│   └── utils/         # Config, path resolution
└── tests/             # Unit tests
```

## Coding standards

- Python 3.10+, type hints on public APIs
- `ruff` for linting and import sorting
- Config-driven experiments (`configs/*.yaml`)
- No hardcoded absolute paths — use `rsna_knee.utils.paths`
- No competition data in git

## Workflow

1. Prototype DICOM loading / transforms locally with `data/sample/`.
2. Move stable code into `src/rsna_knee/`.
3. Copy or sync `kaggle/kernels/train_template.ipynb` to Kaggle.
4. Add competition data source in notebook settings.
5. Install this repo in the Kaggle notebook (see `docs/KAGGLE.md`).
6. Submit notebook output to the leaderboard.

## Adding a model

1. Implement in `src/rsna_knee/models/`.
2. Wire config keys in `configs/default.yaml`.
3. Add a Kaggle training cell that imports from the installed package.
4. Log validation macro ROC-AUC with `rsna_knee.training.macro_roc_auc`.
