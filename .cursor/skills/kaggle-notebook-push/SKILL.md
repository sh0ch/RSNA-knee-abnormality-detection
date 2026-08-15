---
name: kaggle-notebook-push
description: >-
  Push the RSNA EDA notebook from local repo to Kaggle. Regenerates
  kaggle/eda/eda_phase0.ipynb and runs kaggle kernels push. Use when the user
  asks to push/sync/upload the notebook to Kaggle, publish EDA changes, or run
  sync_kaggle_eda --push.
---

# Kaggle notebook push (local → Kaggle)

Push analysis changes to the Kaggle kernel **simonhochwebde/rsna-knee-eda-phase-0**.

## Source of truth

| Edit | Do not edit |
|------|-------------|
| `notebooks/02_eda_phase0.ipynb` | `kaggle/eda/eda_phase0.ipynb` (auto-generated) |

Settings: `configs/kaggle_eda.yaml`  
Full workflow: [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md)

## Workflow

Copy this checklist and track progress:

```
Push progress:
- [ ] Step 1: Confirm source notebook is ready
- [ ] Step 2: GitHub sync if library code changed
- [ ] Step 3: Regenerate + push to Kaggle
- [ ] Step 4: Tell user what to do on Kaggle
```

### Step 1: Confirm source notebook

- Only `notebooks/02_eda_phase0.ipynb` should be edited for analysis cells.
- Bootstrap cells are injected by the sync script — do not add them manually.

### Step 2: GitHub sync (if needed)

If `src/rsna_knee/` changed since last push, Kaggle bootstrap clones stale code unless GitHub is updated:

1. Check: `git status` and `git diff src/rsna_knee/`
2. If changed: remind user to commit and push to `main` (or ask if they want a commit).
3. Do **not** push to Kaggle until GitHub has the library changes the notebook depends on.

### Step 3: Regenerate + push

Run from repo root:

```bash
python scripts/sync_kaggle_eda.py --push
```

If Kaggle API auth fails, tell user to run `kaggle auth login` once.

On success, report the kernel URL from CLI output.

### Step 4: User actions on Kaggle

Tell the user:

1. Open the kernel on Kaggle.
2. **Session → Restart session** if bootstrap or `src/rsna_knee/` changed.
3. **Run All** (interactive) or **Save version → Save & Run All** (batch).
4. Verify setup cell prints:
   ```
   Environment : Kaggle
   Data root   : /kaggle/input/competitions/rsna-knee-abnormality-detection
   ```
5. Download the executed notebook when done, then use **kaggle-notebook-pull** skill.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `kaggle auth login` required | User must authenticate once |
| Push succeeds but Kaggle shows old cells | User may need to refresh kernel page |
| `Environment : local` on Kaggle | Stale session — restart session; ensure GitHub has latest `paths.py` |

## Do not

- Edit `kaggle/eda/eda_phase0.ipynb` by hand
- Commit large notebook outputs from local runs into the source notebook before push (sync strips outputs)
