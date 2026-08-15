# Development guide

## Philosophy

- **Develop locally** on synthetic/sample data — fast iteration, no multi-TB downloads.
- **Train and submit on Kaggle** — full dataset, GPU, and official scoring environment.
- **Keep logic in `src/rsna_knee/`** — notebooks stay thin; reusable code lives in the package.
- **Record learnings in [PROJECT_LOG.md](PROJECT_LOG.md)** — confirmed numbers and decisions, not notebook outputs.

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
├── data/kaggle/       # Optional local copy of competition CSVs (gitignored contents)
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

1. Explore real CSVs/reports locally with `data/kaggle/` (`RSNA_DATA_ROOT=data/kaggle`), or prototype DICOM with `data/sample/`.
2. Move stable code into `src/rsna_knee/`.
3. For **submissions**, use the Phase 1 offline notebook (`scripts/sync_kaggle_train.py`) — internet must be OFF.
4. Add competition data in notebook settings; enable GPU.
5. Submit notebook output (`submission.csv`) to the leaderboard.

## Adding a model

1. Implement in `src/rsna_knee/models/`.
2. Wire config keys in `configs/default.yaml` / `configs/kaggle.yaml`.
3. Re-run `python scripts/sync_kaggle_train.py` so the vendored package on Kaggle updates.
4. Log validation macro ROC-AUC with `rsna_knee.training.macro_roc_auc`.
