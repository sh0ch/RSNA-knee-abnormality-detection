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
print(default_data_root())  # /kaggle/input/rsna-knee-abnormality-detection
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

## Kernel metadata

`kaggle/kernel-metadata.json` documents suggested settings for the Kaggle CLI (`kaggle kernels push`).
