---
name: kaggle-notebook-pull
description: >-
  Import a downloaded Kaggle notebook run into the repo, extract results, and
  summarize findings. Use when the user ran a notebook on Kaggle, downloaded
  an ipynb, asks to pull/import/sync results from Kaggle, or wants PROJECT_LOG
  updated from a Kaggle run.
---

# Kaggle notebook pull (Kaggle → local)

Import an **executed** notebook from Kaggle and process results for review.

## What Kaggle provides

| Method | Has cell outputs? |
|--------|-------------------|
| Download notebook from Kaggle UI | **Yes** (preferred) |
| `sync_kaggle_eda.py --pull` | No (source only) |
| `sync_kaggle_eda.py --logs` | Text log from last Save & Run All only |

This skill uses **`--import-run`** for full results.

Artifacts written:

| Path | Content |
|------|---------|
| `kaggle/eda/runs/latest.ipynb` | Full notebook with outputs (gitignored) |
| `kaggle/eda/runs/last_run_summary.txt` | Text extract for quick review |
| `kaggle/eda/runs/run_YYYYMMDD_HHMMSS.ipynb` | Timestamped archive |

Full workflow: [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md)

## Workflow

Copy this checklist and track progress:

```
Pull progress:
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
- Existing `kaggle/eda/runs/latest.ipynb` only if user says to re-process the last import

Do **not** guess a path in Downloads.

### Step 2: Import run

Run from repo root (quote paths with spaces):

```bash
python scripts/sync_kaggle_eda.py --import-run "PATH_TO_NOTEBOOK.ipynb"
```

If the file is missing, report clearly and return to Step 1.

Optional — commit run log only (no outputs):

```bash
python scripts/sync_kaggle_eda.py --logs
```

Output: `kaggle/eda/runs/last_commit.log`

### Step 3: Read and summarize results

1. Read `kaggle/eda/runs/last_run_summary.txt` first.
2. If summary is insufficient, read `kaggle/eda/runs/latest.ipynb` outputs.

Summarize for the user:

- **Environment** — Kaggle vs local, data root path
- **Errors** — any failed cells or tracebacks
- **Key numbers** — row counts, label coverage, DICOM stats
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

1. Compare downloaded run vs `notebooks/02_eda_phase0.ipynb`
2. Copy analysis cells only — **skip bootstrap cells** (tags: `kaggle-bootstrap`)
3. Remind: next `--push` regenerates the Kaggle copy from source

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Empty summary / no outputs | User downloaded source, not executed version — re-download after Run All |
| `Environment : local` in summary | Run failed or wrong environment — see [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md) |
| `--logs` shows ERROR | Read `last_commit.log`; often stale clone or missing competition data |

## Typical user prompts

- "Import my Kaggle run" → ask for path, then run workflow
- "I downloaded the notebook to Downloads/…" → `--import-run` that path
- "Summarize the last Kaggle run" → read `last_run_summary.txt` if import already done; otherwise ask for path
