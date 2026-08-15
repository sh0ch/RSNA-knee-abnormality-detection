#!/usr/bin/env python3
"""Write a clean source EDA notebook (no stored outputs)."""

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
            "# Phase 0 — EDA",
            "",
            "Exploratory analysis for [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).",
            "",
            "**Run on Kaggle** for real data. Locally: `python scripts/create_sample_data.py` first.",
            "",
            "Confirmed findings live in **[docs/PROJECT_LOG.md](../docs/PROJECT_LOG.md)** — update that file after each phase.",
            "",
            "Sync to Kaggle: `python scripts/sync_kaggle_eda.py --push`",
        ]),
        _md(["## Setup"]),
        _code("""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve repo root when running locally from notebooks/
for candidate in (Path.cwd(), Path.cwd().parent):
    if (candidate / "src" / "rsna_knee").is_dir():
        REPO_ROOT = candidate
        break
else:
    REPO_ROOT = Path.cwd().parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import (
    FLUID_COL,
    PATIENT_SEX_COL,
    REPORT_COL,
    SERIES_ID_COL,
    STUDY_ID_COL,
    TARGET_LABELS,
)
from rsna_knee.data.dicom_io import load_series_volume, series_metadata_summary
from rsna_knee.data.schema import (
    label_coverage,
    labels_present_mask,
    load_test_series_table,
    load_test_table,
    load_train_series_table,
    load_train_table,
)
from rsna_knee.utils.paths import default_data_root, is_kaggle_kernel, series_dir

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

DATA_ROOT = default_data_root()
RNG = np.random.default_rng(42)
print(f"Environment : {'Kaggle' if is_kaggle_kernel() else 'local'}")
print(f"Data root   : {DATA_ROOT}")
"""),
        _code("""
train = load_train_table(DATA_ROOT)
train_series = load_train_series_table(DATA_ROOT)
test = load_test_table(DATA_ROOT)
test_series = load_test_series_table(DATA_ROOT)

has_labels = labels_present_mask(train)
print(f"Train studies : {len(train):,}  |  series : {len(train_series):,}")
print(f"Test studies  : {len(test):,}  |  series : {len(test_series):,}")
print(f"Labeled studies : {has_labels.sum():,} ({100 * has_labels.mean():.1f}%)")
print(f"Report-only     : {(~has_labels).sum():,}")
"""),
        _md([
            "## 1. Overview",
            "",
            "First rows are often **report-only** (NaN labels). Inspect labeled rows separately.",
        ]),
        _code("""
print("Report-only sample (labels NaN is expected):")
display(train.loc[~has_labels, [STUDY_ID_COL, REPORT_COL, *TARGET_LABELS]].head(3))

if has_labels.any():
    print("\\nLabeled sample (0/1 values):")
    display(train.loc[has_labels, [STUDY_ID_COL, *TARGET_LABELS]].head(3))

display(train_series.head(3))
"""),
        _md(["## 2. Label prevalence (labeled studies only)"]),
        _code("""
label_df = label_coverage(train).sort_values("positive_rate", ascending=False, na_position="last")
label_df
"""),
        _code("""
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = label_df.dropna(subset=["positive_rate"])
ax.barh(plot_df["display_name"], 100 * plot_df["positive_rate"], color="steelblue")
ax.set_xlabel("Positive rate among labeled studies (%)")
ax.set_title(f"Label prevalence (n={has_labels.sum()} labeled studies)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""),
        _md(["## 3. Label co-occurrence (labeled studies)"]),
        _code("""
labeled = train.loc[has_labels, TARGET_LABELS]
y = labeled.to_numpy(dtype=float)
cooccur = (y.T @ y) / len(labeled)
cooccur_df = pd.DataFrame(cooccur, index=TARGET_LABELS, columns=TARGET_LABELS)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cooccur_df, cmap="YlOrRd", vmin=0)
ax.set_xticks(range(len(TARGET_LABELS)), TARGET_LABELS, rotation=90, fontsize=8)
ax.set_yticks(range(len(TARGET_LABELS)), TARGET_LABELS, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
"""),
        _md(["## 4. Series per study"]),
        _code("""
series_per_study = train_series.groupby(STUDY_ID_COL).size()
fluid_per_study = train_series.groupby(STUDY_ID_COL)[FLUID_COL].sum()

print(series_per_study.describe())
print(f"Studies with fluid-sensitive series: {(fluid_per_study > 0).mean() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
series_per_study.value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_xlabel("Series per study")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
"""),
        _md(["## 5. Reports (train only)"]),
        _code("""
reports = train[REPORT_COL].fillna("").astype(str)
words = reports.str.split().str.len()
print(f"Words — mean: {words.mean():.0f}, median: {words.median():.0f}, p95: {words.quantile(0.95):.0f}")
print(f"Empty reports: {(reports.str.strip() == '').sum()}")

words.clip(upper=words.quantile(0.99)).hist(bins=40, figsize=(8, 4))
plt.xlabel("Words per report (clipped p99)")
plt.tight_layout()
plt.show()
"""),
        _md(["## 6. Train vs test metadata"]),
        _code("""
train_sps = train_series.groupby(STUDY_ID_COL).size()
test_sps = test_series.groupby(STUDY_ID_COL).size()
print(f"Mean series/study — train: {train_sps.mean():.2f}, test: {test_sps.mean():.2f}")

if PATIENT_SEX_COL in train.columns and PATIENT_SEX_COL in test.columns:
    print("\\nPatient sex %:")
    print(pd.DataFrame({
        "train": train[PATIENT_SEX_COL].value_counts(normalize=True) * 100,
        "test": test[PATIENT_SEX_COL].value_counts(normalize=True) * 100,
    }))
"""),
        _md(["## 7. DICOM geometry (sample)"]),
        _code("""
SAMPLE = 200 if is_kaggle_kernel() else min(8, len(train_series))
sample = train_series.sample(SAMPLE, random_state=42)
rows = []
for _, row in sample.iterrows():
    path = series_dir(DATA_ROOT, split="train") / row[STUDY_ID_COL] / row[SERIES_ID_COL]
    try:
        vol, dsets = load_series_volume(path)
        meta = series_metadata_summary(dsets)
        rows.append({"slices": vol.shape[0], "h": vol.shape[1], "w": vol.shape[2], **meta})
    except OSError:
        pass

dicom_df = pd.DataFrame(rows)
print(dicom_df[["slices", "h", "w"]].describe())
"""),
        _md([
            "## 8. Update project log",
            "",
            "Copy any new numbers or decisions into **`docs/PROJECT_LOG.md`**.",
        ]),
        _code("""
print("Update docs/PROJECT_LOG.md with:")
print(f"  - labeled studies: {has_labels.sum()} / {len(train)}")
print(f"  - mean series/study: {train_series.groupby(STUDY_ID_COL).size().mean():.2f}")
if len(dicom_df):
    print(f"  - median slices (sample): {dicom_df['slices'].median():.0f}")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out = project_root() / "notebooks" / "02_eda_phase0.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(nb['cells'])} cells, no outputs)")


if __name__ == "__main__":
    main()
