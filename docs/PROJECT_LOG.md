# Project log

Living notes from EDA, experiments, and competition learning.  
**Update this file** when you confirm something on Kaggle or change modeling strategy — not the notebook outputs.

| Phase | Status | Doc / artifact |
|-------|--------|----------------|
| 0 — EDA | Done (2026-08-15) | `notebooks/02_eda_phase0.ipynb` |
| 1 — Image baseline | Not started | — |
| 2 — Reports / semi-supervised | Not started | — |

---

## Phase 0 — Exploratory data analysis (2026-08-15)

Verified on Kaggle with full competition data (`/kaggle/input/competitions/rsna-knee-abnormality-detection`).

### Dataset scale

| Split | Studies | Series |
|-------|---------|--------|
| Train | 4,407 | 24,371 |
| Test | *(run EDA for current count)* | *(run EDA)* |

~**5.5 series per study** on average (24,371 / 4,407).

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

### DICOM (spot-check)

- Path: `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm`
- Example series: **22 slices**, files readable via `pydicom`.
- Slice count, resolution, and orientation vary by site (full distribution: run EDA §7 on Kaggle).

### Phase 1 decisions (from Phase 0)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary supervised set | 58 labeled studies | Only ground truth available |
| Loss | BCE with **NaN mask** | Do not treat missing labels as 0 |
| Series selection | Fluid-sensitive first | Competition + EDA convention |
| Volume shape (start) | `[32, 256, 256]` | See `configs/default.yaml` |
| Text at inference | **No** | Rules / test pipeline |
| CV | Study-level k-fold on labeled set | n=58 → high score variance expected |

### Open questions

- [ ] Label prevalence per class on the 58 labeled studies (run EDA §2 on Kaggle).
- [ ] Train vs test shift in series count, planes, sites.
- [ ] Report length / language distribution.
- [ ] Best strategy for 4,349 report-only studies (keyword rules vs clinical LLM vs weak supervision).

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
| 2026-08-15 | Phase 0 complete: partial labels (58/4407), schema mapping, paths, DICOM spot-check |
