# Running on Kaggle

## Competition submission (internet OFF)

Scoring runs **without internet**. Do not rely on `git clone`, `pip install`, or downloading weights inside the submit notebook.

### Phase 1 image baseline (recommended)

| File | Purpose |
|------|---------|
| `notebooks/03_phase1_image_baseline.ipynb` | Source of truth (edit here) |
| `kaggle/train/train.ipynb` | Generated Kaggle copy (vendors `src/rsna_knee`) |
| `scripts/sync_kaggle_train.py` | Regenerate + optional `kaggle kernels push` |
| `configs/kaggle_train.yaml` | Kernel id / optional pretrained dataset |

```bash
# After editing the notebook or src/
python scripts/sync_kaggle_train.py --push
```

Kernel settings (`kaggle/train/kernel-metadata.json`):

- `enable_internet: false`
- `enable_gpu: true`
- Competition data attached

The generated notebook attaches Dataset `simonhochwebde/rsna-knee-code` (`src/` + `configs/`) and puts it on `sys.path`. No source blobs are embedded in the notebook.

**Phase 1 init:** trains **from scratch** (pipeline validation on 58 labels — score expected to be weak). Optional ImageNet init later: `scripts/export_pretrained_weights.py` + attach Dataset + set `model.allow_random_init: false`.

### Output

- Write `/kaggle/working/submission.csv` with columns matching `sample_submission.csv`.
- **Save Version → Save & Run All**, then submit the notebook output.

## Data paths

```python
from rsna_knee.utils.paths import default_data_root, is_kaggle_kernel

print(is_kaggle_kernel())  # True
print(default_data_root())  # /kaggle/input/competitions/rsna-knee-abnormality-detection
```

## Load config

```python
from rsna_knee.utils.config import load_config

cfg = load_config("kaggle")
```

## EDA notebook (Phase 0) — internet OK for exploration

EDA may use git clone + pip (interactive / non-submit). **Do not use that pattern for leaderboard submissions.**

**Workflow doc:** [KAGGLE_NOTEBOOK_SYNC.md](KAGGLE_NOTEBOOK_SYNC.md)

| File | Purpose |
|------|---------|
| `notebooks/02_eda_phase0.ipynb` | Source of truth (edit here) |
| `kaggle/eda/eda_phase0.ipynb` | Generated Kaggle copy (do not edit) |
| `scripts/sync_kaggle_eda.py` | Regenerate, push, pull, logs, import-run |

```bash
python scripts/sync_kaggle_eda.py --push
python scripts/sync_kaggle_eda.py --import-run PATH_TO_DOWNLOAD.ipynb
```

GPU is not required for EDA (`enable_gpu: false` in metadata).

## Deprecated

`kaggle/kernels/train_template.ipynb` is a pointer only — use Phase 1 above.
