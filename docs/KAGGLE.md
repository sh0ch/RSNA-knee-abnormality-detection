# Running on Kaggle

## 1. Create a notebook

1. Go to [Competition Notebooks](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code).
2. Click **New Notebook**.
3. In **Input**, add dataset: `rsna-knee-abnormality-detection`.
4. Enable **GPU** (and Internet if installing from GitHub).

## 2. Install this repository

Add a cell at the top of your notebook:

```python
import os
import subprocess

REPO_URL = "https://github.com/YOUR_USERNAME/RSNA_knee_abnormality_detection.git"
BRANCH = "main"
WORK_DIR = "/kaggle/working/rsna_knee_repo"

if not os.path.exists(WORK_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, WORK_DIR], check=True)

subprocess.run(["pip", "install", "-q", "-e", f"{WORK_DIR}[train]"], check=True)
```

Replace `YOUR_USERNAME` after pushing to GitHub.

**Alternative:** Upload `src/rsna_knee` as a Kaggle Dataset and pip-install from `/kaggle/input/...`.

## 3. Data paths

```python
from rsna_knee.utils.paths import default_data_root, is_kaggle_kernel

print(is_kaggle_kernel())  # True
print(default_data_root())  # /kaggle/input/competitions/rsna-knee-abnormality-detection
```

## 4. Load config

```python
import sys
sys.path.insert(0, "/kaggle/working/rsna_knee_repo/src")

from rsna_knee.utils.config import load_config

cfg = load_config("kaggle")
```

## 5. Submission

- Output `submission.csv` with columns: `StudyInstanceUID` + 12 label columns.
- Use **Save Version → Save & Run All** before submitting to the leaderboard.
- Notebook must complete within Kaggle time limits.

## 6. Syncing local changes

After editing code locally:

1. Commit and push to GitHub.
2. Re-run the clone/install cell in Kaggle (or pin a release tag for stability).

Template notebook: `kaggle/kernels/train_template.ipynb`

## EDA notebook (Phase 0)

Pre-built Kaggle notebook with repo bootstrap included:

| File | Purpose |
|------|---------|
| `kaggle/eda/eda_phase0.ipynb` | Full EDA on competition data |
| `kaggle/eda/kernel-metadata.json` | Settings for `kaggle kernels push` |
| `scripts/build_eda_notebook.py` | Regenerate clean `notebooks/02_eda_phase0.ipynb` (no outputs) |
| `scripts/sync_kaggle_eda.py` | Regenerate + push EDA notebook to Kaggle |

### Sync notebook to Kaggle

**Source of truth:** `notebooks/02_eda_phase0.ipynb`  
**Generated copy:** `kaggle/eda/eda_phase0.ipynb` (do not edit by hand)

Settings: `configs/kaggle_eda.yaml` (kernel id, GitHub clone URL)

```bash
python scripts/sync_kaggle_eda.py           # regenerate kaggle/eda/eda_phase0.ipynb
python scripts/sync_kaggle_eda.py --push    # regenerate + kaggle kernels push
python scripts/sync_kaggle_eda.py --pull    # download from Kaggle (merge into source manually)
```

### Push from your machine (manual)

1. Push this repo to GitHub (the bootstrap cell clones it on Kaggle).
2. Edit `configs/kaggle_eda.yaml` if needed (`repo_url`, `kernel_id`).
3. Configure the [Kaggle API](https://github.com/Kaggle/kaggle-api).
4. Run `python scripts/sync_kaggle_eda.py --push`
5. Open the notebook on Kaggle → **Save & Run All** (Internet must be enabled).

GPU is not required for EDA (`enable_gpu: false` in metadata).

## Kernel metadata

`kaggle/kernel-metadata.json` documents suggested settings for the training template (`kaggle kernels push -p kaggle` after pointing metadata at the train kernel). For EDA, use `kaggle/eda/kernel-metadata.json` instead.
