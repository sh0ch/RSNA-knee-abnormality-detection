"""Merge rule-based and LLM labelers into study-level pseudo-label artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rsna_knee.constants import REPORT_COL, STUDY_ID_COL, TARGET_LABELS
from rsna_knee.data.schema import labels_present_mask, load_train_table
from rsna_knee.reports.llm import (
    DEFAULT_CONFIRM_LABELS,
    LLMLabeler,
    LLMLabelResult,
    format_hybrid_match_debug,
)
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
    Rules first; LLM confirms low-precision rule-positives.

    Confirmed positives keep the LLM confidence. Vetoes become label=0 with
    confidence=0 (masked in training) so a wrong veto is not a hard negative.
    High-precision rule hits (fracture, Baker, effusion, …) are not sent to the LLM.
    """

    def __init__(
        self,
        *,
        rule_labeler: RuleLabeler | None = None,
        llm_labeler: LLMLabeler | None = None,
        route_threshold: float = 0.7,
        confirm_labels: Sequence[str] | None = None,
        fill_ambiguous: bool = False,
        verbose: bool = False,
        debug_max: int | None = None,
    ) -> None:
        self.rule_labeler = rule_labeler or RuleLabeler(route_threshold=route_threshold)
        self.llm_labeler = llm_labeler if llm_labeler is not None else LLMLabeler()
        self.route_threshold = route_threshold
        self.confirm_labels = list(
            confirm_labels if confirm_labels is not None else DEFAULT_CONFIRM_LABELS
        )
        self.fill_ambiguous = fill_ambiguous
        self.verbose = verbose
        self.debug_max = debug_max
        self._debug_calls = 0
        if verbose:
            setattr(self.llm_labeler, "verbose", True)
            if debug_max is not None:
                setattr(self.llm_labeler, "debug_max", debug_max)
        self._confirm_idx = [
            TARGET_LABELS.index(name)
            for name in self.confirm_labels
            if name in TARGET_LABELS
        ]

    def _llm_ready(self) -> bool:
        return bool(getattr(self.llm_labeler, "is_available", lambda: False)())

    def _route_indices(self, rules: RuleLabelResult) -> list[int]:
        routed: set[int] = set()
        for i in self._confirm_idx:
            if float(rules.labels[i]) >= 0.5:
                routed.add(i)
        if self.fill_ambiguous:
            routed.update(int(i) for i in np.where(rules.ambiguous_mask)[0])
        return sorted(routed)

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
        label_source = "rules"

        routed = self._route_indices(rules)
        if routed and self._llm_ready():
            llm: LLMLabelResult = self.llm_labeler.label_labels_only(report, routed)
            decisions: list[tuple[str, int, int, str]] = []
            for i in routed:
                name = TARGET_LABELS[i]
                rule_positive = float(rules.labels[i]) >= 0.5
                llm_positive = float(llm.labels[i]) >= 0.5
                if rule_positive and llm_positive:
                    labels[i] = 1.0
                    confidence[i] = float(llm.confidence[i])
                    action = "confirm (keep 1)"
                elif rule_positive and not llm_positive:
                    labels[i] = 0.0
                    confidence[i] = 0.0
                    action = "veto (label=0, conf=0 masked)"
                elif self.fill_ambiguous and not rule_positive:
                    labels[i] = float(llm.labels[i])
                    confidence[i] = float(llm.confidence[i]) if llm_positive else 0.0
                    action = "fill_ambiguous"
                else:
                    action = "unchanged"
                decisions.append((name, int(rule_positive), int(llm_positive), action))
            label_source = "hybrid"
            if self.verbose:
                self._debug_calls += 1
                if self.debug_max is None or self._debug_calls <= self.debug_max:
                    print(
                        format_hybrid_match_debug(
                            call_index=self._debug_calls, decisions=decisions
                        ),
                        flush=True,
                    )

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

        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="pseudo-labels"):
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
    llm_model_id: str | None = None,
    route_threshold: float = 0.7,
    confirm_labels: Sequence[str] | None = None,
    fill_ambiguous: bool = False,
    require_llm: bool = False,
    llm_labeler: LLMLabeler | None = None,
) -> pd.DataFrame:
    """
    Generate pseudo-label artifact for all training studies.

    Writes parquet or CSV to ``output_path`` when provided.
    """
    train = load_train_table(data_root)
    llm = llm_labeler
    if llm is None:
        llm = LLMLabeler(model_path=llm_model_path, model_id=llm_model_id)
    if require_llm:
        llm.warmup()
    labeler = HybridLabeler(
        llm_labeler=llm,
        route_threshold=route_threshold,
        confirm_labels=confirm_labels,
        fill_ambiguous=fill_ambiguous,
    )
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
