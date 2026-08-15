---
name: kaggle-notebook-push
description: >-
  Push an RSNA Kaggle notebook from local repo to Kaggle (EDA or Phase 1 train).
  Regenerates the generated kernel notebook and runs kaggle kernels push. Use when
  the user asks to push/sync/upload a notebook to Kaggle, publish EDA or train
  changes, or run sync_kaggle_eda / sync_kaggle_train --push.
---

# Kaggle notebook push (local → Kaggle)

Push local notebook + package changes to a Kaggle kernel.

## Step 0 — Ask which kernel (required)

**Before doing anything else**, if the user has not already said which notebook to push, ask with the AskQuestion tool (or a clear yes/no choice):

> Which notebook should I push to Kaggle?

Options:

1. **EDA (Phase 0)** — `notebooks/02_eda_phase0.ipynb` → kernel `simonhochwebde/rsna-knee-eda-phase-0`
2. **Phase 1 train** — `notebooks/03_phase1_image_baseline.ipynb` → kernel `simonhochwebde/rsna-knee-phase1-image`

Do **not** assume EDA. Do **not** push both unless the user asks for both.

If the user already named one (e.g. “push the train notebook”), skip AskQuestion and use that target.

## Targets

| Choice | Edit (source of truth) | Generated (do not edit) | Config | Sync command |
|--------|------------------------|-------------------------|--------|--------------|
| EDA | `notebooks/02_eda_phase0.ipynb` | `kaggle/eda/eda_phase0.ipynb` | `configs/kaggle_eda.yaml` | `python scripts/sync_kaggle_eda.py --push` |
| Phase 1 | `notebooks/03_phase1_image_baseline.ipynb` | `kaggle/train/train.ipynb` | `configs/kaggle_train.yaml` | `python scripts/sync_kaggle_train.py --push` |

Docs: [docs/KAGGLE.md](../../docs/KAGGLE.md), [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md) (EDA loop).

## Workflow

Copy this checklist and track progress:

```
Push progress:
- [ ] Step 0: User chose EDA or Phase 1
- [ ] Step 1: Confirm source notebook is ready
- [ ] Step 2: GitHub / vendor notes
- [ ] Step 3: Regenerate + push to Kaggle
- [ ] Step 4: Tell user what to do on Kaggle
```

### Step 1: Confirm source notebook

- Edit only the **source** notebook for that target (table above).
- Bootstrap / vendor cells are injected by the sync script — do not add them by hand to the source.

### Step 2: Code availability on Kaggle

**EDA** — bootstrap **git-clones** GitHub (`enable_internet: true` for exploration):

1. Check `git status` / `git diff src/rsna_knee/`
2. If library code changed: remind user to commit and push to `main` before Kaggle run (or ask to commit).
3. Do not push to Kaggle until GitHub has the library changes the EDA notebook depends on.

**Phase 1 train** — notebook **vendors** `src/rsna_knee` into the generated ipynb (`enable_internet: false`):

1. No GitHub push required for package code to appear on Kaggle.
2. Re-run sync after any `src/rsna_knee/` change so the vendor cell embeds the latest sources.
3. From-scratch training — no pretrained Dataset required.

### Step 3: Regenerate + push

From repo root, run the command for the chosen target (table above).

If Kaggle API auth fails, tell user to run `kaggle auth login` once.

On success, report the kernel URL from CLI output.

### Step 4: User actions on Kaggle

**EDA**

1. Open the EDA kernel.
2. **Session → Restart session** if bootstrap or `src/rsna_knee/` changed.
3. **Run All** or **Save version → Save & Run All**.
4. Verify setup prints `Environment : Kaggle` and the competition data root.
5. Download the executed notebook when done → **kaggle-notebook-pull**.

**Phase 1 train**

1. Open the Phase 1 kernel.
2. Confirm **Internet OFF**, **GPU ON**, competition data attached.
3. **Save version → Save & Run All** (needed for submission).
4. Confirm `/kaggle/working/submission.csv` is produced.
5. Download the executed notebook when done → **kaggle-notebook-pull** (choose Phase 1).

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `kaggle auth login` required | User must authenticate once |
| Push succeeds but Kaggle shows old cells | Refresh kernel page; restart session |
| EDA shows `Environment : local` | Stale session — restart; ensure GitHub has latest `paths.py` |
| Phase 1 missing package imports | Re-run `sync_kaggle_train.py --push` so vendor cell is refreshed |

## Do not

- Edit generated notebooks under `kaggle/eda/` or `kaggle/train/` by hand
- Commit large notebook outputs into the source notebook before push (sync strips outputs)
- Push the wrong kernel without asking (Step 0)
