"""Report exploratory analysis helpers (Phase 2)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import REPORT_COL, STUDY_ID_COL, TARGET_LABELS
from rsna_knee.data.schema import labels_present_mask, load_train_table
from rsna_knee.reports.rules import RuleLabeler

# Lightweight language heuristics (no external langdetect dependency).
_LANG_MARKERS: dict[str, list[str]] = {
    "english": [
        r"\bthe\b",
        r"\band\b",
        r"\bwith\b",
        r"\bno\b",
        r"\bfindings\b",
        r"\bimpression\b",
    ],
    "german": [
        r"\bund\b",
        r"\bkein\b",
        r"\bkeine\b",
        r"\bohne\b",
        r"\bbefund\b",
        r"\bknie\b",
        r"\bmeniskus\b",
    ],
    "spanish": [
        r"\by\b",
        r"\bsin\b",
        r"\bcon\b",
        r"\bmenisco\b",
        r"\bhallazgos\b",
        r"\brodilla\b",
    ],
}


def _normalize_report(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def detect_report_language(text: str) -> str:
    """Heuristic language tag: english, german, spanish, or other."""
    norm = _normalize_report(text)
    if not norm:
        return "empty"
    scores = {
        lang: sum(1 for pat in patterns if re.search(pat, norm))
        for lang, patterns in _LANG_MARKERS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "other"
    return best_lang


def report_length_stats(reports: pd.Series) -> dict[str, float]:
    """Character and word counts for a report column."""
    lengths = reports.fillna("").astype(str).str.len()
    words = reports.fillna("").astype(str).str.split().str.len()
    return {
        "n_reports": float(len(reports)),
        "chars_mean": float(lengths.mean()),
        "chars_median": float(lengths.median()),
        "chars_p95": float(lengths.quantile(0.95)),
        "words_mean": float(words.mean()),
        "words_median": float(words.median()),
    }


def language_distribution(reports: pd.Series) -> pd.DataFrame:
    """Count reports per detected language."""
    langs = reports.fillna("").astype(str).map(detect_report_language)
    counts = langs.value_counts().rename_axis("language").reset_index(name="count")
    counts["fraction"] = counts["count"] / max(len(reports), 1)
    return counts


def keyword_hit_rates_on_labeled(
    train_df: pd.DataFrame,
    *,
    labeler: RuleLabeler | None = None,
) -> pd.DataFrame:
    """
    Per-label rule hit rate on explicitly labeled studies.

    A "hit" means the rule labeler assigned confidence >= 0.7 for that label
    (positive or negative), indicating the keyword rules found a signal.
    """
    labeler = labeler or RuleLabeler()
    labeled = train_df.loc[labels_present_mask(train_df)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in labeled.iterrows():
        result = labeler.label_report(str(row.get(REPORT_COL, "")))
        for i, name in enumerate(TARGET_LABELS):
            rows.append(
                {
                    "label": name,
                    "hit": float(result.confidence[i] >= 0.7),
                    "rule_label": float(result.labels[i]),
                    "ground_truth": float(row[name]),
                    "match": float(result.labels[i] == row[name]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("label", as_index=False)
        .agg(
            hit_rate=("hit", "mean"),
            accuracy=("match", "mean"),
            n=("hit", "count"),
        )
        .sort_values("hit_rate", ascending=False)
    )
    return summary


def report_eda_summary(data_root: Any = None) -> dict[str, Any]:
    """
    Full report EDA summary for logging in PROJECT_LOG or notebooks.

    Returns dict with length stats, language distribution, and labeled keyword
    hit rates. Safe to run locally on sample data or on Kaggle full data.
    """
    train = load_train_table(data_root)
    reports = train[REPORT_COL]
    labeled_mask = labels_present_mask(train)

    return {
        "n_studies": int(len(train)),
        "n_labeled": int(labeled_mask.sum()),
        "length_stats": report_length_stats(reports),
        "language_distribution": language_distribution(reports).to_dict(orient="records"),
        "keyword_hit_rates_labeled": keyword_hit_rates_on_labeled(train).to_dict(orient="records"),
    }
