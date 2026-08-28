"""Evaluate report labelers against explicit ground-truth labels."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from rsna_knee.constants import REPORT_COL, STUDY_ID_COL, TARGET_LABELS
from rsna_knee.data.schema import labels_present_mask, load_train_table
from rsna_knee.reports.rules import RuleLabelResult


class ReportLabeler(Protocol):
    def label_report(self, report: str) -> RuleLabelResult: ...


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Per-label precision, recall, F1, and AUC (when both classes present).

    ``y_true`` / ``y_pred``: ``[N, C]`` float arrays.
    """
    label_names = label_names or TARGET_LABELS
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yt_bin = (yt >= 0.5).astype(int)
        yp_bin = (yp >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt_bin, yp_bin, average="binary", zero_division=0
        )
        auc = float("nan")
        if len(np.unique(yt_bin)) > 1:
            try:
                auc = float(roc_auc_score(yt_bin, yp))
            except ValueError:
                auc = float("nan")
        rows.append(
            {
                "label": name,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "auc": auc,
                "support_pos": int(yt_bin.sum()),
                "support_neg": int((1 - yt_bin).sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_labeler(
    labeler: ReportLabeler,
    data_root: Any = None,
) -> tuple[pd.DataFrame, float]:
    """
    Evaluate a labeler on the 58 explicitly labeled studies.

    Returns per-label metrics DataFrame and macro-averaged F1.
    """
    train = load_train_table(data_root)
    labeled = train.loc[labels_present_mask(train)]

    y_true = labeled[TARGET_LABELS].to_numpy(dtype=np.float32)
    y_pred = np.zeros_like(y_true)
    for i, (_, row) in enumerate(labeled.iterrows()):
        result = labeler.label_report(str(row.get(REPORT_COL, "")))
        y_pred[i] = result.labels

    metrics = per_label_metrics(y_true, y_pred)
    macro_f1 = float(metrics["f1"].mean())
    return metrics, macro_f1
