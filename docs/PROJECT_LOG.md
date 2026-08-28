# Project log

Living notes from EDA, experiments, and competition learning.  
**Update this file** when you confirm something on Kaggle or change modeling strategy — not the notebook outputs.

| Phase | Status | Doc / artifact |
|-------|--------|----------------|
| 0 — EDA | Done (2026-08-15) | `notebooks/02_eda_phase0.ipynb` |
| 1 — Image baseline | Done (2026-08-28) | `notebooks/03_phase1_image_baseline.ipynb` |
| 2 — Reports / semi-supervised | In progress | `notebooks/04_phase2_pseudo_labels.ipynb` |

---

## Phase 0 — Exploratory data analysis (2026-08-15)

Verified on Kaggle with full competition data (`/kaggle/input/competitions/rsna-knee-abnormality-detection`).

### Dataset scale

| Split | Studies | Series |
|-------|---------|--------|
| Train | 4,407 | 24,371 |
| Test | 3 | 15 |

~**5.5 series per study** on average (24,371 / 4,407). Test sample: **5 series per study** (15 / 3).

### Partial explicit labels (critical)

Only a **small subset** of training studies have ground-truth 0/1 labels. The rest are **report-only** (all 12 label columns NaN). This is [by design](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data): labels can be derived from multilingual radiology reports.

| Category | Count | Share |
|----------|------:|------:|
| Explicitly labeled (≥1 non-null label) | **58** | **1.3%** |
| Report-only (all labels NaN) | 4,349 | 98.7% |

- Label values where present: **0.0 / 1.0 only** (float64).
- All 58 labeled studies have **all 12 labels filled** (non-null count = 58 per column).
- **`train.head(3)` will show NaN labels** — those rows are report-only, not a parsing bug.

**Implication:** Pure supervised image training on explicit labels alone uses **58 studies**. Report NLP / pseudo-labeling / distillation is central, not optional.

### CSV schema (Kaggle → code)

`train.csv` headers (submission uses the same label names):

`StudyInstanceUID`, `Report`, `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

`train_series.csv`: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`

Normalized in code via `rsna_knee.data.schema` → snake_case (`acl_tear`, `fluid_sensitive`, …). See [DATA.md](DATA.md) and [COMPETITION.md](COMPETITION.md).

### Series metadata (spot-check)

- `Fluid_Sensitive` / `Fat_Suppression`: binary flags per series.
- `Anatomical_Plane`: Sagittal / Axial / Coronal (sometimes empty).
- Prefer **fluid-sensitive** series for soft-tissue findings (PD/STIR-like).

### Reports

- Present for all training studies; **not available at inference**.
- Multilingual (e.g. Spanish, German, English observed in sample rows).
- Reports can contain newlines; use `low_memory=False` when reading CSV.

### DICOM (spot-check, Kaggle 2026-08-15)

- Path: `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm`
- Example series: **22 slices**, **512×512**, `Modality=MR`, `PixelSpacing≈0.33 mm`, `SliceThickness≈3.4 mm`
- Raw `pixel_array`: `uint16` (example slice min=0, max=843, mean≈156); `RescaleSlope` / `RescaleIntercept` present but **not applied** in raw EDA cells
- Files named by **SOP Instance UID**, not slice index

#### Slice ordering (critical)

**Do not sort DICOM files by filename.** On Kaggle, alphabetical filename order scrambles anatomy (e.g. InstanceNumbers 5, 10, 3, 2, 9 for the first five files).

Sort slices before viewing or stacking:

1. **`InstanceNumber`** (primary)
2. **`ImagePositionPatient[2]`** (fallback)

After sorting, example series runs InstanceNumber **1 → 22** contiguously. Use `rsna_knee.data.dicom_io.load_series_volume` in pipelines (same sort logic).

Slice count, resolution, and orientation vary by site — full distribution still TBD (sample one series only).

### Phase 1 decisions (from Phase 0)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary supervised set | 58 labeled studies | Only ground truth available |
| Loss | BCE with **NaN mask** | Do not treat missing labels as 0 |
| Series selection | Fluid-sensitive first | Competition + EDA convention |
| Volume shape (start) | `[16, 256, 256]` per series | 2.5D stacks; see `configs/default.yaml` |
| DICOM slice order | Sort by `InstanceNumber` | Filename order is not slice order |
| Text at inference | **No** | Rules / test pipeline |
| CV | Study-level k-fold on labeled set | n=58 → high score variance expected |

### Open questions

- [ ] Label prevalence per class on the 58 labeled studies (run EDA §2 on Kaggle).
- [ ] Train vs test shift in series count, planes, sites.
- [ ] Report length / language distribution.
- [ ] Best strategy for 4,349 report-only studies (keyword rules vs clinical LLM vs weak supervision).

---

## Phase 1 — Image baseline (2026-08-15)

### What we built

- **2.5D ConvNeXt-Tiny** + gated attention MIL (`rsna_knee.models.mil_2p5d`)
- `KneeStudyDataset`: labeled-only, fluid-sensitive series (up to 3 planes), InstanceNumber sort, cached volumes
- Masked BCE + pos_weight, study-level 5-fold, mixup + MRI augs, fold ensemble + TTA
- **From-scratch init** for Phase 1 (pipeline / submit path); ImageNet optional later via `scripts/export_pretrained_weights.py`
- **Offline Kaggle path:** Dataset `rsna-knee-code` (`src/` + `configs/`) via `scripts/publish_code_dataset.py`; internet OFF. Daily runs: Cursor → Kaggle Jupyter Server ([KAGGLE.md](KAGGLE.md)).

### Artifacts

| Path | Role |
|------|------|
| `notebooks/03_phase1_image_baseline.ipynb` | Source notebook (edit + remote kernel) |
| `kaggle/train/kernel-metadata.json` | Submit kernel settings |
| `configs/kaggle.yaml` / `configs/kaggle_train.yaml` | Train + kernel ids |

### Pretrained weights

- **Not required for Phase 1** (`allow_random_init: true`). Goal is a working train → `submission.csv` loop.
- Optional later: torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` exported to `convnext_tiny_imagenet.pth`

### Next

- ~~Connect Cursor to Kaggle Jupyter Server (prefer T4), publish Dataset, run GPU train~~
- ~~Submit via `python scripts/push_kaggle_kernel.py train` + Save & Run All~~
- ~~Record OOF / LB macro ROC-AUC here after first successful Kaggle run~~
- Then Phase 2: report-derived labels for the 4,349 unlabeled studies; revisit ImageNet init

### Results (Kaggle interactive run, 2026-08-28)

| Metric | Value |
|--------|------:|
| OOF macro ROC-AUC | **0.5295** |
| Training | 5-fold × 8 epochs, dual T4, from-scratch ConvNeXt-Tiny MIL |
| Studies | 58 labeled only |

**Best per-label OOF AUC:** medial meniscus (0.71), joint effusion (0.69), synovitis (0.63).

**Worst per-label OOF AUC:** fracture (0.32), bone contusion (0.40).

`submission.csv` schema validated. Leaderboard score: TBD after offline Save & Run All submit.

---

## Phase 2 — Pseudo-labels from reports (2026-08-28)

### What we built

- **Hybrid labeler:** `rsna_knee.reports` — keyword rules (EN/DE/ES) + **Qwen2.5-1.5B-Instruct** confirmer (Apache-2.0; Kaggle HF model, not a Dataset upload) for ACL/MCL/OA/synovitis rule-positives. Attach via Add Input → Models; do not download from huggingface.co on the kernel.
- **Report EDA:** language heuristics, length stats, keyword hit rates on labeled studies
- **Pseudo-label training:** extended `KneeStudyDataset` + confidence-weighted masked BCE
- **OOF on 58 labeled** while training on all 4,407 studies (`eval_labeled_only`)
- **Pretrained comparison:** `resolve_pretrained_weights(variant=imagenet|radimagenet)` + export script
- **Multimodal (train-only):** `MultimodalMIL` — text fusion at train, image-only at inference
- **Notebook:** `notebooks/04_phase2_pseudo_labels.ipynb`
- **Kaggle kernel:** `simonhochwebde/rsna-knee-phase2-pseudo` via `push_kaggle_kernel.py phase2`

### Configs

| File | Use |
|------|-----|
| `configs/phase2.yaml` | Local smoke |
| `configs/kaggle_phase2.yaml` | Kaggle full run |

### Next

- Run report EDA + pseudo-label generation on Kaggle full data
- Train with ImageNet pretrained; compare OOF vs Phase 1 baseline (0.5295)
- Compare RadImageNet variant under same pseudo-labels (if rules permit)
- Enable multimodal experiment after pseudo-label image baseline lands
- Record Phase 2 OOF / LB scores here

---

## Template for future entries

```markdown
## Phase N — Title (YYYY-MM-DD)

### What we tried
- ...

### Results
- ...

### Decisions
- ...

### Next
- ...
```

---

## Changelog

| Date | Update |
|------|--------|
| 2026-08-15 | Phase 0: partial labels (58/4407), schema mapping, paths, DICOM spot-check |
| 2026-08-15 | Kaggle v7: test counts (3/15), 512×512 MR, **filename ≠ slice order** — sort by InstanceNumber |
| 2026-08-15 | Phase 1 scaffold: ConvNeXt-Tiny MIL notebook, offline vendor sync; **from-scratch** default |
| 2026-08-15 | Workflow: Cursor ↔ Kaggle Jupyter Server; Dataset publish + thin kernel push; removed dual-notebook sync clones |
| 2026-08-28 | Phase 1 done: OOF 0.5295 on 58 labeled; submission.csv validated on Kaggle |
| 2026-08-28 | Phase 2 scaffold: hybrid report labeler, pseudo-label training, multimodal model, notebook 04 |
