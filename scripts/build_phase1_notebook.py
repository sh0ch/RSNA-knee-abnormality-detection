#!/usr/bin/env python3
"""Write the Phase 1 image baseline source notebook (no stored outputs)."""

from __future__ import annotations

import json

from rsna_knee.utils.paths import project_root


def _md(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def build_notebook() -> dict:
    cells = [
        _md([
            "# Phase 1 — Image baseline (offline-submittable)",
            "",
            "2.5D **ConvNeXt-Tiny** + gated attention MIL on the **58 labeled** studies.",
            "",
            "- Reports are **not** used at inference (competition rule).",
            "- Kaggle scoring: **internet OFF** — package is vendored; ImageNet weights come from an attached Dataset.",
            "- Sync: `python scripts/sync_kaggle_train.py --push`",
            "",
            "See [docs/PROJECT_LOG.md](../docs/PROJECT_LOG.md).",
        ]),
        _md(["## Setup"]),
        _code("""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Resolve repo root when running locally from notebooks/
for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "src" / "rsna_knee").is_dir():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path.cwd().parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.data import (
    KneeStudyDataset,
    load_sample_submission,
    load_test_table,
    load_train_table,
    predictions_to_submission,
)
from rsna_knee.data.schema import labels_present_mask
from rsna_knee.models import build_model, resolve_pretrained_weights
from rsna_knee.training import (
    macro_roc_auc,
    predict_test_ensemble,
    prevalence_baseline_predictions,
    run_kfold_training,
)
from rsna_knee.utils.config import load_config
from rsna_knee.utils.paths import default_data_root, is_kaggle_kernel

ON_KAGGLE = is_kaggle_kernel()
CFG_NAME = "kaggle" if ON_KAGGLE else "default"
cfg = load_config(CFG_NAME)
DATA_ROOT = default_data_root()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Environment : {'Kaggle' if ON_KAGGLE else 'local'}")
print(f"Data root   : {DATA_ROOT}")
print(f"Device      : {DEVICE}")
print(f"Config      : {CFG_NAME}")
print(f"Torch       : {torch.__version__}")
"""),
        _md([
            "## 1. Labeled-study sanity check",
            "",
            "Only explicitly labeled studies are used for supervised training.",
        ]),
        _code("""
train = load_train_table(DATA_ROOT)
has_labels = labels_present_mask(train)
n_labeled = int(has_labels.sum())
print(f"Train studies : {len(train):,}")
print(f"Labeled       : {n_labeled:,} ({100 * has_labels.mean():.1f}%)")

volume_shape = tuple(cfg["data"]["volume_shape"])
max_series = int(cfg["data"].get("max_series", 3))
print(f"Volume shape  : {volume_shape}  |  max_series={max_series}")
"""),
        _md([
            "## 2. Prevalence baseline (submission floor)",
            "",
            "Constant positive rates from labeled studies — validates submission format.",
        ]),
        _code("""
labeled = train.loc[has_labels]
label_mat = labeled[TARGET_LABELS].to_numpy(dtype=np.float32)
mask_mat = (~np.isnan(label_mat)).astype(np.float32)
label_mat = np.nan_to_num(label_mat, nan=0.0)

test = load_test_table(DATA_ROOT)
n_test = len(test)
prev_preds = prevalence_baseline_predictions(label_mat, mask_mat, n_test)
prev_df = predictions_to_submission(test["StudyInstanceUID"].tolist(), pd.DataFrame(prev_preds, columns=TARGET_LABELS))

rates = prev_preds[0]
print("Positive rates (labeled):")
for name, rate in zip(TARGET_LABELS, rates):
    print(f"  {name:30s} {rate:.3f}")

OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else (REPO_ROOT / "outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
prev_path = OUT_DIR / "submission_prevalence.csv"
prev_df.to_csv(prev_path, index=False)
print(f"Wrote {prev_path}")
display(prev_df.head())
"""),
        _md([
            "## 3. Pretrained weights + model smoke check",
            "",
            "On Kaggle, missing weights **fail fast** (no silent random init).",
        ]),
        _code("""
allow_random = not ON_KAGGLE  # local smoke may skip weights; Kaggle must have Dataset
try:
    weights_path = resolve_pretrained_weights(allow_missing=allow_random)
except FileNotFoundError:
    weights_path = None
    if ON_KAGGLE:
        raise
    print("WARNING: no pretrained weights — model will use random init (local only).")

print(f"Weights: {weights_path}")

smoke = KneeStudyDataset(
    DATA_ROOT,
    split="train",
    labeled_only=True,
    volume_shape=volume_shape,
    max_series=max_series,
    cache=False,
    require_dicom=True,
)
print(f"Labeled dataset size: {len(smoke)}")
if len(smoke) == 0:
    raise RuntimeError("No labeled studies — check data root / CSVs.")

sample = smoke[0]
print(f"Sample image shape: {sample['image'].shape}  (S, 3, H, W)")
print(f"Labels: {sample['labels']}")

model = build_model(
    cfg["model"]["name"],
    pretrained_path=weights_path,
    allow_random_init=allow_random,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

with torch.no_grad():
    x = torch.from_numpy(sample["image"]).unsqueeze(0)
    logits = model(x)
print(f"Logits shape: {tuple(logits.shape)}")
"""),
        _md([
            "## 4. K-fold training",
            "",
            "Study-level folds on labeled studies. Local default: 1 epoch smoke. Kaggle: full `configs/kaggle.yaml`.",
        ]),
        _code("""
CKPT_DIR = Path(cfg["paths"]["checkpoint_dir"])
if not CKPT_DIR.is_absolute():
    CKPT_DIR = REPO_ROOT / CKPT_DIR
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = int(cfg["training"]["max_epochs"])
# Keep local CPU runs short
if not ON_KAGGLE and DEVICE.type == "cpu":
    MAX_EPOCHS = min(MAX_EPOCHS, 1)
    print(f"Local CPU: capping max_epochs → {MAX_EPOCHS}")

train_result = run_kfold_training(
    DATA_ROOT,
    n_folds=int(cfg["training"]["n_folds"]),
    max_epochs=MAX_EPOCHS,
    batch_size=int(cfg["training"]["batch_size"]),
    learning_rate=float(cfg["training"]["learning_rate"]),
    volume_shape=volume_shape,
    max_series=max_series,
    checkpoint_dir=CKPT_DIR,
    seed=int(cfg["seed"]),
    pretrained_path=weights_path,
    allow_random_init=allow_random,
    tta=bool(cfg["inference"].get("tta", False)),
    num_workers=int(cfg["data"].get("num_workers", 0)),
    model_name=cfg["model"]["name"],
)

print(f"OOF macro ROC-AUC: {train_result['overall_auc']:.4f}")
for fr in train_result["fold_results"]:
    print(f"  fold {fr.fold}: {fr.val_auc:.4f}  {fr.checkpoint_path}")
"""),
        _md(["## 5. Out-of-fold metric detail"]),
        _code("""
oof_preds = train_result["oof_preds"]
oof_labels = train_result["oof_labels"]
print("OOF macro ROC-AUC:", macro_roc_auc(oof_labels, oof_preds))

per_label = []
for i, name in enumerate(TARGET_LABELS):
    yt = oof_labels[:, i]
    if len(np.unique(yt)) < 2:
        per_label.append({"label": name, "auc": float("nan")})
        continue
    from sklearn.metrics import roc_auc_score
    per_label.append({"label": name, "auc": float(roc_auc_score(yt, oof_preds[:, i]))})
display(pd.DataFrame(per_label).sort_values("auc", ascending=False))
"""),
        _md([
            "## 6. Test inference → submission.csv",
            "",
            "Ensemble fold checkpoints (+ TTA on Kaggle). Writes `/kaggle/working/submission.csv`.",
        ]),
        _code("""
ckpt_paths = [fr.checkpoint_path for fr in train_result["fold_results"]]
study_ids, test_preds = predict_test_ensemble(
    ckpt_paths,
    DATA_ROOT,
    volume_shape=volume_shape,
    max_series=max_series,
    batch_size=int(cfg["training"]["batch_size"]),
    tta=bool(cfg["inference"].get("tta", False)),
    allow_random_init=True,
    model_name=cfg["model"]["name"],
    num_workers=int(cfg["data"].get("num_workers", 0)),
)

sub = predictions_to_submission(
    study_ids,
    pd.DataFrame(test_preds, columns=TARGET_LABELS),
)
sub_path = OUT_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Wrote {sub_path}  shape={sub.shape}")
display(sub.head())

# Schema check vs sample submission
sample = load_sample_submission(DATA_ROOT)
assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
print("submission.csv columns match sample_submission.csv")
"""),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> None:
    root = project_root()
    out = root / "notebooks" / "03_phase1_image_baseline.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
