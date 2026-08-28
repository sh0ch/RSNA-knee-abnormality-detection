"""Merge rule-based and LLM labelers into study-level pseudo-label artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import REPORT_COL, STUDY_ID_COL, TARGET_LABELS
from rsna_knee.data.schema import labels_present_mask, load_train_table
from rsna_knee.reports.llm import LLMLabeler
from rsna_knee.reports.rules import RuleLabeler, RuleLabelResult

CONF_SUFFIX = "_conf"
SOURCE_COL = "label_source"


@dataclass
class HybridLabelResult:
    study_uid: str
    labels: np.ndarray
    confidence: np.ndarray
    label_source: str


class HybridLabeler:
    """
    Hybrid labeler: rules first, LLM for ambiguous labels, ground truth for labeled.
    """

    def __init__(
        self,
        *,
        rule_labeler: RuleLabeler | None = None,
        llm_labeler: LLMLabeler | None = None,
        route_threshold: float = 0.7,
    ) -> None:
        self.rule_labeler = rule_labeler or RuleLabeler(route_threshold=route_threshold)
        self.llm_labeler = llm_labeler or LLMLabeler()
        self.route_threshold = route_threshold

    def label_report(
        self,
        report: str,
        *,
        ground_truth: np.ndarray | None = None,
        has_ground_truth: bool = False,
    ) -> HybridLabelResult:
        if has_ground_truth and ground_truth is not None:
            labels = np.nan_to_num(ground_truth, nan=0.0).astype(np.float32)
            confidence = np.ones(len(TARGET_LABELS), dtype=np.float32)
            return HybridLabelResult(
                study_uid="",
                labels=labels,
                confidence=confidence,
                label_source="ground_truth",
            )

        rules: RuleLabelResult = self.rule_labeler.label_report(report)
        labels = rules.labels.copy()
        confidence = rules.confidence.copy()
        sources = list(rules.sources)
        label_source = "rules"

        ambiguous_idx = np.where(rules.ambiguous_mask)[0].tolist()
        if ambiguous_idx and self.llm_labeler is not None:
            llm = self.llm_labeler.label_labels_only(report, ambiguous_idx)
            labels[ambiguous_idx] = llm.labels[ambiguous_idx]
            confidence[ambiguous_idx] = llm.confidence[ambiguous_idx]
            for i in ambiguous_idx:
                sources[i] = "llm"
            label_source = "hybrid"

        return HybridLabelResult(
            study_uid="",
            labels=labels,
            confidence=confidence,
            label_source=label_source,
        )

    def label_dataframe(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Label all studies in a normalized train table."""
        has_labels = labels_present_mask(train_df)
        rows: list[dict[str, Any]] = []

        for idx, row in train_df.iterrows():
            study_uid = str(row[STUDY_ID_COL])
            report = str(row.get(REPORT_COL, ""))
            gt = row[TARGET_LABELS].to_numpy(dtype=np.float32) if has_labels.loc[idx] else None
            result = self.label_report(
                report,
                ground_truth=gt,
                has_ground_truth=bool(has_labels.loc[idx]),
            )
            record: dict[str, Any] = {STUDY_ID_COL: study_uid, SOURCE_COL: result.label_source}
            for i, name in enumerate(TARGET_LABELS):
                record[name] = float(result.labels[i])
                record[f"{name}{CONF_SUFFIX}"] = float(result.confidence[i])
            rows.append(record)

        return pd.DataFrame(rows)


def generate_pseudo_labels(
    data_root: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    llm_model_path: Path | str | None = None,
    route_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Generate pseudo-label artifact for all training studies.

    Writes parquet or CSV to ``output_path`` when provided.
    """
    train = load_train_table(data_root)
    llm = LLMLabeler(model_path=llm_model_path) if llm_model_path else LLMLabeler()
    labeler = HybridLabeler(llm_labeler=llm, route_threshold=route_threshold)
    df = labeler.label_dataframe(train)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)

    return df


def load_pseudo_labels(path: Path | str) -> pd.DataFrame:
    """Load a pseudo-label artifact (parquet or CSV)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def confidence_columns() -> list[str]:
    return [f"{name}{CONF_SUFFIX}" for name in TARGET_LABELS]
