# Competition reference

Official page: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## Task

Predict the per-study probability of **12 knee abnormalities** from multimodal inputs:

- DICOM MRI series (multiple sequences per study)
- Radiology reports (training only — **not available at inference**)

## Targets (12 labels)

Kaggle `train.csv` uses short headers (`ACL`, `MCL`, `Medial Meniscus`, …). This repo normalizes them to snake_case constants in `rsna_knee.constants.TARGET_LABELS` via `rsna_knee.data.schema`.

| Canonical (code) | Kaggle CSV | Finding |
|------------------|------------|---------|
| `acl_tear` | ACL | ACL Tear |
| `mcl_tear` | MCL | MCL Tear |
| `medial_meniscus_injury` | Medial Meniscus | Medial Meniscus Injury |
| `lateral_meniscus_injury` | Lateral Meniscus | Lateral Meniscus Injury |
| `medial_osteoarthritis` | Medial OA | Medial Osteoarthritis |
| `lateral_osteoarthritis` | Lateral OA | Lateral Osteoarthritis |
| `patellofemoral_osteoarthritis` | PF OA | Patellofemoral Osteoarthritis |
| `joint_effusion` | Effusion | Joint Effusion |
| `synovitis` | Synovitis | Synovitis |
| `bakers_cyst` | Baker's | Baker's Cyst |
| `bone_contusion` | Contusion | Bone Contusion |
| `fracture` | Fracture | Fracture |

## Evaluation

**Macro ROC-AUC** averaged across all 12 abnormalities.

## Training labels

Only a **subset** of training studies include explicit 0/1 labels; the majority are report-only (NaN labels). Verified counts and modeling implications: **[PROJECT_LOG.md](PROJECT_LOG.md)**.

## Key dates (2026)

| Milestone | Date |
|-----------|------|
| Competition opens | ~July 30, 2026 |
| Team merger deadline | October 15, 2026 |
| Final submission | October 22, 2026 |
| Winners requirements | November 5, 2026 |

## Competition rules (summary)

Always read the [official rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) on Kaggle. This repo encodes the following constraints in `.cursor/rules/`:

1. **No external data** unless explicitly allowed in the current rules version.
2. **No manual labeling** of the test set.
3. **Submissions via Kaggle** — code must run in Kaggle Notebooks for scoring.
4. **Reports are train-only** — models must not rely on report text at inference.
5. **Team limits** — respect Kaggle team size and merger deadlines.
6. **Winner obligations** — top teams must open-source solutions and provide reproducible code.
7. **No private sharing** of predictions or code outside your registered team.
8. **Efficiency track** — separate prize for compute-efficient models; document runtime and hardware.

If rules change on Kaggle, update `docs/COMPETITION.md` and `.cursor/rules/kaggle-competition.mdc`.
