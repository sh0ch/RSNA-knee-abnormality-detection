---
name: kaggle-notebook-pull
description: >-
  Import a downloaded Kaggle notebook run (EDA or Phase 1 train) into the repo,
  extract results, and summarize findings. Use when the user ran a notebook on
  Kaggle, downloaded an ipynb, asks to pull/import/sync results from Kaggle, or
  wants PROJECT_LOG updated from a Kaggle run.
---

# Kaggle notebook pull (Kaggle → local)

Import an **executed** notebook from Kaggle and process results for review.

## Step 0 — Ask which kernel (required)

**Before importing**, if the user has not already said which run this is, ask with the AskQuestion tool:

> Which Kaggle run should I import?

Options:

1. **EDA (Phase 0)** — stores under `kaggle/eda/runs/`
2. **Phase 1 train** — stores under `kaggle/train/runs/`

Do **not** assume EDA. If the user already named one (or the filename clearly indicates train vs eda), skip AskQuestion.

## What Kaggle provides

| Method | Has cell outputs? |
|--------|-------------------|
| Download notebook from Kaggle UI | **Yes** (preferred) |
| `sync_kaggle_*.py --pull` | No (source only; EDA only today) |
| `sync_kaggle_*.py --logs` | Text log from last Save & Run All only |

This skill uses **`--import-run`** for full results.

### Artifacts by target

**EDA**

| Path | Content |
|------|---------|
| `kaggle/eda/runs/latest.ipynb` | Full notebook with outputs (gitignored) |
| `kaggle/eda/runs/last_run_summary.txt` | Text extract |
| `kaggle/eda/runs/run_YYYYMMDD_HHMMSS.ipynb` | Timestamped archive |

**Phase 1 train**

| Path | Content |
|------|---------|
| `kaggle/train/runs/latest.ipynb` | Full notebook with outputs (gitignored) |
| `kaggle/train/runs/last_run_summary.txt` | Text extract |
| `kaggle/train/runs/run_YYYYMMDD_HHMMSS.ipynb` | Timestamped archive |

Docs: [docs/KAGGLE.md](../../docs/KAGGLE.md), [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md).

## Workflow

Copy this checklist and track progress:

```
Pull progress:
- [ ] Step 0: User chose EDA or Phase 1
- [ ] Step 1: Get notebook path from user
- [ ] Step 2: Import run
- [ ] Step 3: Read and summarize results
- [ ] Step 4: Offer PROJECT_LOG update (if user wants)
```

### Step 1: Get notebook path from user

**If the user did not provide a path** (no `@file`, no quoted path), ask:

> Which notebook should I import? Paste the path or `@`-mention the downloaded `.ipynb` (e.g. from Downloads).

Accept:

- Absolute path (Windows or Unix)
- `@`-attached file in chat
- Existing `kaggle/eda/runs/latest.ipynb` or `kaggle/train/runs/latest.ipynb` only if user says to re-process the last import for that target

Do **not** guess a path in Downloads.

### Step 2: Import run

From repo root (quote paths with spaces):

**EDA**

```bash
python scripts/sync_kaggle_eda.py --import-run "PATH_TO_NOTEBOOK.ipynb"
```

**Phase 1 train**

```bash
python scripts/sync_kaggle_train.py --import-run "PATH_TO_NOTEBOOK.ipynb"
```

If the file is missing, report clearly and return to Step 1.

Optional — commit run log only (no outputs):

```bash
# EDA
python scripts/sync_kaggle_eda.py --logs

# Phase 1
python scripts/sync_kaggle_train.py --logs
```

### Step 3: Read and summarize results

1. Read the matching `last_run_summary.txt` first (`kaggle/eda/runs/` or `kaggle/train/runs/`).
2. If summary is insufficient, read that target’s `latest.ipynb` outputs.

Summarize for the user:

- **Environment** — Kaggle vs local, data root path
- **Errors** — any failed cells or tracebacks
- **Key numbers** — EDA: row counts, label coverage, DICOM stats; Phase 1: OOF macro ROC-AUC, fold AUCs, submission shape/path
- **Notable findings** — anything surprising vs prior runs or PROJECT_LOG
- **Figures** — note if matplotlib outputs were captured (summary shows `[figure]`)

### Step 4: PROJECT_LOG (only if asked)

Update `docs/PROJECT_LOG.md` only when the user explicitly asks (e.g. "update PROJECT_LOG from this run").

When updating:

- Add confirmed numbers, not speculative ones
- Note date and that values were verified on Kaggle
- Do not paste raw notebook output wholesale

## Merge analysis cells back (rare)

If the user edited analysis cells **on Kaggle** and wants them locally:

1. Compare downloaded run vs the source notebook for that target
2. Copy analysis cells only — **skip bootstrap cells** (tags: `kaggle-bootstrap`)
3. Remind: next `--push` regenerates the Kaggle copy from source

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Empty summary / no outputs | User downloaded source, not executed version — re-download after Run All |
| `Environment : local` in summary | Run failed or wrong environment |
| `--logs` shows ERROR | Read `last_commit.log`; check data mount / vendor / GPU |
| Wrong runs folder | Re-import with the correct target (Step 0) |

## Typical user prompts

- "Import my Kaggle run" → ask EDA vs Phase 1 (Step 0), then ask for path
- "Pull the train notebook from Downloads/…" → Phase 1 `--import-run`
- "Summarize the last Kaggle run" → ask which target if unclear; read that `last_run_summary.txt`
