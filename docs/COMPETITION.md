# Competition reference

Official page: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## Task

Predict the per-study probability of **12 knee abnormalities** from multimodal inputs:

- DICOM MRI series (multiple sequences per study)
- Radiology reports (training only — **not available at inference**)

## Targets (12 labels)

| Column | Finding |
|--------|---------|
| `acl_tear` | ACL Tear |
| `mcl_tear` | MCL Tear |
| `medial_meniscus_injury` | Medial Meniscus Injury |
| `lateral_meniscus_injury` | Lateral Meniscus Injury |
| `medial_osteoarthritis` | Medial Osteoarthritis |
| `lateral_osteoarthritis` | Lateral Osteoarthritis |
| `patellofemoral_osteoarthritis` | Patellofemoral Osteoarthritis |
| `joint_effusion` | Joint Effusion |
| `synovitis` | Synovitis |
| `bakers_cyst` | Baker's Cyst |
| `bone_contusion` | Bone Contusion |
| `fracture` | Fracture |

## Evaluation

**Macro ROC-AUC** averaged across all 12 abnormalities.

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
