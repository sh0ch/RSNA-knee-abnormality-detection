---
name: kaggle-jupyter-workflow
description: >-
  RSNA Kaggle setup and daily loop: sync code to Jupyter Server, export/publish
  pretrained weights, publish rsna-knee-code Dataset, push eda/train/phase2
  kernels, Save & Run All submit. Use when connecting to Kaggle, running on GPU,
  deploying code, publishing datasets, or submitting.
---

# Kaggle workflow (RSNA Knee)

No MCP. No dual-notebook generators. Source of truth is `notebooks/*.ipynb`.

Full reference: [docs/KAGGLE.md](../../docs/KAGGLE.md).

## Step 0 — Ask which kernel (if unclear)

| Phase | Notebook | Kernel id | Push target |
|-------|----------|-----------|-------------|
| EDA | `notebooks/02_eda_phase0.ipynb` | `simonhochwebde/rsna-knee-eda-phase-0` | `eda` |
| Phase 1 | `notebooks/03_phase1_image_baseline.ipynb` | `simonhochwebde/rsna-knee-phase1-image` | `train` |
| Phase 2 | `notebooks/04_phase2_pseudo_labels.ipynb` | `simonhochwebde/rsna-knee-phase2-pseudo` | `phase2` |

## Required Kaggle Inputs (train / phase2)

Attach on the kernel before starting Jupyter Server or Save & Run All:

- **Competition:** `rsna-knee-abnormality-detection`
- **Code Dataset:** `simonhochwebde/rsna-knee-code`
- **Pretrained (Phase 1+2 with ImageNet init):** `simonhochwebde/rsna-knee-pretrained`
- **GPU:** prefer **T4** (not P100 — PyTorch needs sm_70+)
- **Submit runs:** internet **OFF**, GPU **ON**

---

## Full setup (deploy + submit)

Run from repo root when preparing a scored kernel version or first-time Phase 2 deploy.

**Agent:** run these commands (user must complete Kaggle UI steps below).

```bash
python scripts/sync_code_to_jupyter.py          # or publish_code_dataset.py
python scripts/export_pretrained_weights.py     # once, for ImageNet weights
python scripts/publish_pretrained_weights.py
python scripts/publish_code_dataset.py
python scripts/push_kaggle_kernel.py phase2
```

### What each command does

| Command | When | Effect |
|---------|------|--------|
| `sync_code_to_jupyter.py` | **Daily interactive** editing in Cursor | Pushes `src/` + `configs/` to `/kaggle/working/` on a live Jupyter Server. No Dataset version bump. Requires VS Code Compatible URL in `.env` as `KAGGLE_JUPYTER_URL`. |
| `publish_code_dataset.py` | **Before Save & Run All** or when not using sync | Versions `simonhochwebde/rsna-knee-code` for offline kernels (internet OFF). Restart Jupyter session after publish if using Dataset mount. |
| `export_pretrained_weights.py` | **Once** (or when switching variant) | Downloads ImageNet ConvNeXt-Tiny to `data/pretrained/convnext_tiny_imagenet.pth`. Then `publish_pretrained_weights.py`. RadImageNet: `--variant radimagenet --source /path/to/rad.pth`. |
| `publish_pretrained_weights.py` | After export | Versions Dataset `simonhochwebde/rsna-knee-pretrained`. Restart Jupyter so the mount updates. |
| `push_kaggle_kernel.py <target>` | Before submit | Copies source notebook → `kaggle/{eda,train,phase2}/`, pushes to Kaggle API. Targets: `eda`, `train`, `phase2`. |

### sync vs publish

- **Editing cells in Cursor against a live Kaggle kernel?** → `sync_code_to_jupyter.py` (fast iteration).
- **Scoring / Save & Run All without an active Cursor session?** → `publish_code_dataset.py` then `push_kaggle_kernel.py <target>`.

Both may be needed: sync for dev, publish+push for the scored version.

### After push — user steps on Kaggle

1. Open the kernel (e.g. `simonhochwebde/rsna-knee-phase2-pseudo`).
2. Confirm Inputs: competition + `rsna-knee-code` (+ `rsna-knee-pretrained` for Phase 2).
3. **Save Version → Save & Run All** (internet OFF, GPU ON).
4. Submit `/kaggle/working/submission.csv` from Output.

---

## Daily loop (interactive)

User does UI steps; agent runs sync and reminds about attachments.

1. On Kaggle, open the kernel. Attach competition + `rsna-knee-code` (+ pretrained if needed).
2. **Run → Kaggle Jupyter Server → Start**.
3. Copy **VS Code Compatible URL** (Manually Connect).
4. In Cursor: open the source notebook → kernel picker → **Existing Jupyter Server** → paste URL.
5. Sync local code:

```bash
python scripts/sync_code_to_jupyter.py
```

Re-run the notebook setup cell. Imports use `/kaggle/working/src`.

If `rsna_knee` not found: URL is stale or sync was not run. If `/kaggle/input` is empty: competition not attached on Kaggle — DICOMs cannot sync from PC.

After `src/` or `configs/` changes intended for **offline** kernels:

```bash
python scripts/publish_code_dataset.py
```

Restart the Jupyter Server session.

---

## Phase-specific notes

### Phase 1 (`train`)

- Config: `configs/kaggle.yaml` / `kaggle_train.yaml`
- From-scratch default (`allow_random_init: true`) needs no pretrained Dataset.

### Phase 2 (`phase2`)

- Config: `configs/kaggle_phase2.yaml`
- Notebook: `notebooks/04_phase2_pseudo_labels.ipynb`
- Expects ImageNet weights (`allow_random_init: false`) → attach `rsna-knee-pretrained`.
- LLM confirmer (section 2b): attach Hugging Face model **Qwen/Qwen2.5-1.5B-Instruct** via Add Input → Models. Restart Jupyter after attaching. Loader uses `/kaggle/input/models/...` — do not download from huggingface.co.
- Pseudo-labels: `/kaggle/working/outputs/pseudo_labels.csv` this session. Pull locally with `python scripts/pull_pseudo_labels.py`, later upload with `python scripts/publish_pseudo_labels.py`. Set `FORCE_RELABEL = False` to skip regeneration.
- Outputs: `/kaggle/working/outputs/pseudo_labels.csv`, `/kaggle/working/submission.csv`
- Compare backbones: same pseudo-labels, swap `model.pretrained_weights` / `pretrained_variant` in config.

### Pretrained export (one-time)

```bash
python scripts/export_pretrained_weights.py
python scripts/publish_pretrained_weights.py
```

Document license in `docs/PROJECT_LOG.md`.

---

## Review a downloaded run

Copy executed notebook to `kaggle/{eda,train,phase2}/runs/` (gitignored). Summarize outputs; do not invent paths in Downloads.

---

## Do not

- Reintroduce `sync_kaggle_*.py` generators or `_VENDORED` source blobs
- Git-clone the repo inside notebooks
- Edit `kaggle/*/*.ipynb` as source of truth (source is `notebooks/`)
- Use reports at inference (competition rule)
- Commit pretrained `.pth` files or pseudo-label artifacts to git
