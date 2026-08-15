# Phase 1 train kernel (Kaggle)

| Local source | Submit push |
|--------------|-------------|
| `notebooks/03_phase1_image_baseline.ipynb` | `python scripts/push_kaggle_kernel.py train` |

Daily loop: Cursor → Kaggle Jupyter Server (see [docs/KAGGLE.md](../../docs/KAGGLE.md)).

Package: `python scripts/publish_code_dataset.py` → Dataset `rsna-knee-code`.

Settings: `kernel-metadata.json` (internet OFF, GPU ON, competition + code Dataset).
