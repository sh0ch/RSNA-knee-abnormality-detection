# Running on Kaggle

Two paths:

1. **Daily loop** — edit notebooks in Cursor, run cells on a **Kaggle Jupyter Server** (GPU / competition data).
2. **Submit** — push the notebook once, then **Save Version → Save & Run All** on Kaggle (required for scoring).

Package code (`src/rsna_knee/`) is **not** git-cloned. Publish it as Dataset `simonhochwebde/rsna-knee-code` and attach that Dataset to the kernel.

## Daily loop — Cursor ↔ Kaggle Jupyter Server

[Official docs](https://www.kaggle.com/docs/notebooks) (VS Code Compatible URL; Cursor uses the same kernel picker).

1. On Kaggle, open the EDA or Phase 1 kernel. Attach:
   - Competition: `rsna-knee-abnormality-detection`
   - Dataset: `simonhochwebde/rsna-knee-code`
   - GPU for training (prefer **T4**, not P100 — current Kaggle PyTorch needs sm_70+)
2. **Run → Kaggle Jupyter Server → Start**.
3. Under **Manually Connect**, copy the **VS Code Compatible URL**.
4. In Cursor, open:
   - EDA: `notebooks/02_eda_phase0.ipynb`
   - Train: `notebooks/03_phase1_image_baseline.ipynb`
5. Kernel picker → **Select Another Kernel** → **Existing Jupyter Server** → paste the URL → name it (e.g. `Kaggle GPU`).
6. Run cells. Check with `!nvidia-smi` if using GPU.

Reconnect with a fresh URL if the session times out. The local `.ipynb` is **not** synced to the Kaggle notebook file — only the execution backend is shared.

After changing `src/` or `configs/`, republish the Dataset before expecting new imports on the remote kernel:

```bash
python scripts/publish_code_dataset.py
```

Then **restart** the Kaggle Jupyter Server session so it mounts the new Dataset version.

## Submit path (leaderboard)

Interactive Cursor sessions do **not** create a scored kernel version.

```bash
# If src/ or configs/ changed:
python scripts/publish_code_dataset.py

# Copy source notebook into kaggle/{eda|train}/ and push:
python scripts/push_kaggle_kernel.py eda
python scripts/push_kaggle_kernel.py train
```

On Kaggle:

1. Open the kernel → confirm Input has competition data + `rsna-knee-code`.
2. Train: **Internet OFF**, **GPU ON**. EDA: internet may stay ON.
3. **Save Version → Save & Run All**.
4. Submit `/kaggle/working/submission.csv` from the Output tab (train).

## Kernels

| Role | Edit locally | Kernel id | Metadata |
|------|--------------|-----------|----------|
| EDA | `notebooks/02_eda_phase0.ipynb` | `simonhochwebde/rsna-knee-eda-phase-0` | `kaggle/eda/kernel-metadata.json` |
| Phase 1 | `notebooks/03_phase1_image_baseline.ipynb` | `simonhochwebde/rsna-knee-phase1-image` | `kaggle/train/kernel-metadata.json` |

Configs: `configs/kaggle_eda.yaml`, `configs/kaggle_train.yaml`.

## Data paths

```python
from rsna_knee.utils.paths import default_data_root, is_kaggle_kernel

print(is_kaggle_kernel())  # True on Kaggle
print(default_data_root())  # /kaggle/input/competitions/rsna-knee-abnormality-detection
```

## Load config

```python
from rsna_knee.utils.config import load_config

cfg = load_config("kaggle")  # or "default" locally
```

## Optional: import a downloaded run

After downloading an executed notebook from the Kaggle UI, copy it under `kaggle/eda/runs/` or `kaggle/train/runs/` (gitignored) for local review.
