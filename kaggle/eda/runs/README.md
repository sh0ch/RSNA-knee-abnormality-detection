# Kaggle run artifacts (local only)

Populated by:

```bash
python scripts/sync_kaggle_eda.py --import-run PATH_TO_DOWNLOADED.ipynb
python scripts/sync_kaggle_eda.py --logs
```

| File | Purpose |
|------|---------|
| `latest.ipynb` | Last downloaded notebook **with outputs** (gitignored) |
| `last_run_summary.txt` | Text extract of stdout / key tables for Cursor |
| `last_commit.log` | Stdout/stderr from last **Save & Run All** (via `--logs`) |

See [docs/KAGGLE_NOTEBOOK_SYNC.md](../../docs/KAGGLE_NOTEBOOK_SYNC.md).
