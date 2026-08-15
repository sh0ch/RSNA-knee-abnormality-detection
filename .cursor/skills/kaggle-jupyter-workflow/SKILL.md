---
name: kaggle-jupyter-workflow
description: >-
  RSNA Kaggle daily loop (Cursor ↔ Kaggle Jupyter Server), publish code Dataset,
  and thin kernel push for Save & Run All submit. Use when the user asks to
  connect to Kaggle, run on GPU, publish rsna-knee-code, push EDA/train kernel,
  or submit.
---

# Kaggle Jupyter Server workflow

No MCP. No dual-notebook generators. Source of truth is `notebooks/*.ipynb`.

## Step 0 — Ask which kernel (required)

If unclear, ask:

1. **EDA (Phase 0)** — `notebooks/02_eda_phase0.ipynb` → `simonhochwebde/rsna-knee-eda-phase-0`
2. **Phase 1 train** — `notebooks/03_phase1_image_baseline.ipynb` → `simonhochwebde/rsna-knee-phase1-image`

## Daily loop (interactive)

User does this in the UI (agent documents / reminds; cannot paste the VS Code URL for them):

1. Open the kernel on Kaggle. Attach competition data + Dataset `simonhochwebde/rsna-knee-code`. Prefer **T4** GPU for train (not P100).
2. **Run → Kaggle Jupyter Server → Start**.
3. Copy **VS Code Compatible URL** (Manually Connect).
4. In Cursor: open the source notebook → kernel picker → **Existing Jupyter Server** → paste URL.
5. Run cells. Remote imports use `/kaggle/input/rsna-knee-code/src` (setup cell in the notebook).

If `src/` or `configs/` changed since last publish:

```bash
python scripts/publish_code_dataset.py
```

Then restart the Kaggle Jupyter Server session.

Docs: [docs/KAGGLE.md](../../docs/KAGGLE.md).

## Submit path (Save & Run All)

Interactive sessions do **not** score. For a leaderboard version:

```bash
# If package changed:
python scripts/publish_code_dataset.py

python scripts/push_kaggle_kernel.py eda    # or: train
```

Then on Kaggle: **Save Version → Save & Run All**. Train: internet OFF, GPU ON. Submit `submission.csv` from Output.

## Review a downloaded run

If the user downloads an executed notebook from Kaggle, copy it under `kaggle/eda/runs/` or `kaggle/train/runs/` (gitignored) and summarize outputs. Do not invent paths in Downloads.

## Do not

- Reintroduce `sync_kaggle_*.py` generators or embed `_VENDORED` source blobs
- Git-clone the repo inside notebooks
- Edit a generated `kaggle/*/…ipynb` as source of truth (source is `notebooks/`)
- Assume MCP is required
