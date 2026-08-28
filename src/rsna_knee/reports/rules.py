"""Keyword-based report labeler with multilingual synonyms and negation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rsna_knee.constants import TARGET_LABELS

# Per-label positive keyword patterns (case-insensitive).
_POSITIVE_PATTERNS: dict[str, list[str]] = {
    "acl_tear": [
        r"\bacl\b",
        r"anterior cruciate",
        r"acl tear",
        r"acl rupture",
        r"acl injury",
        r"kreuzband",
        r"vorderes kreuzband",
        r"ligamento cruzado anterior",
        r"\blca\b",
    ],
    "mcl_tear": [
        r"\bmcl\b",
        r"medial collateral",
        r"mcl tear",
        r"mcl injury",
        r"mediales kollateral",
        r"innenseitenband",
        r"ligamento colateral medial",
        r"\blcm\b",
    ],
    "medial_meniscus_injury": [
        r"medial meniscus",
        r"medial meniscal",
        r"innenmeniskus",
        r"meniskus medial",
        r"menisco medial",
        r"meniscal tear.{0,30}medial",
        r"medial.{0,20}meniscal tear",
    ],
    "lateral_meniscus_injury": [
        r"lateral meniscus",
        r"lateral meniscal",
        r"außenmeniskus",
        r"aussenmeniskus",
        r"meniskus lateral",
        r"menisco lateral",
        r"meniscal tear.{0,30}lateral",
        r"lateral.{0,20}meniscal tear",
    ],
    "medial_osteoarthritis": [
        r"medial osteoarthritis",
        r"medial oa\b",
        r"medial compartment.{0,20}arthr",
        r"mediale arthrose",
        r"arthrose.{0,20}medial",
        r"osteoarthritis.{0,20}medial",
    ],
    "lateral_osteoarthritis": [
        r"lateral osteoarthritis",
        r"lateral oa\b",
        r"lateral compartment.{0,20}arthr",
        r"laterale arthrose",
        r"arthrose.{0,20}lateral",
        r"osteoarthritis.{0,20}lateral",
    ],
    "patellofemoral_osteoarthritis": [
        r"patellofemoral",
        r"pf oa\b",
        r"retropatellar",
        r"patello-femoral",
        r"patellofemoral osteoarthritis",
        r"chondromalacia",
        r"retropatellare arthrose",
        r"artrosis patelofemoral",
    ],
    "joint_effusion": [
        r"\beffusion\b",
        r"joint fluid",
        r"intraarticular fluid",
        r"\berguss\b",
        r"gelenkerguss",
        r"derrame",
        r"efusion articular",
        r"hydrops",
    ],
    "synovitis": [
        r"\bsynovitis\b",
        r"synovialitis",
        r"synovial thickening",
        r"synoviale hyperplasie",
        r"synovial hypertrophy",
        r"synovitis",
        r"synovitis activa",
    ],
    "bakers_cyst": [
        r"baker'?s cyst",
        r"popliteal cyst",
        r"popliteal cyste",
        r"popliteale zyste",
        r"quiste de baker",
        r"quiste popliteo",
    ],
    "bone_contusion": [
        r"\bcontusion\b",
        r"bone bruise",
        r"bone marrow edema",
        r"bone marrow oedema",
        r"knochenmarködem",
        r"knochenmarkodem",
        r"bone contusion",
        r"edema óseo",
        r"edema oseo",
    ],
    "fracture": [
        r"\bfracture\b",
        r"\bfractured\b",
        r"\bfraktur\b",
        r"\bfractura\b",
        r"bone break",
        r"avulsion fracture",
        r"stress fracture",
    ],
}

_NEGATION_PATTERNS: list[str] = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bkein\b",
    r"\bkeine\b",
    r"\bkeinen\b",
    r"\bohne\b",
    r"\bsin\b",
    r"\bausencia de\b",
    r"\bno evidence of\b",
    r"\bunremarkable\b",
]

_AMBIGUOUS_CONFIDENCE = 0.5
_CLEAR_CONFIDENCE = 0.9
_ABSENT_CONFIDENCE = 0.85


@dataclass(frozen=True)
class RuleLabelResult:
    """Rule labeler output for one report."""

    labels: np.ndarray  # float32 [12] in {0, 1}
    confidence: np.ndarray  # float32 [12] in [0, 1]
    ambiguous_mask: np.ndarray  # bool [12] — route to LLM when True
    sources: list[str]  # per-label source tag


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _window_has_negation(text: str, start: int, end: int, window: int = 40) -> bool:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return any(re.search(pat, snippet) for pat in _NEGATION_PATTERNS)


class RuleLabeler:
    """Multilingual keyword labeler with local negation handling."""

    def __init__(
        self,
        *,
        high_confidence: float = _CLEAR_CONFIDENCE,
        ambiguous_confidence: float = _AMBIGUOUS_CONFIDENCE,
        route_threshold: float = 0.7,
    ) -> None:
        self.high_confidence = high_confidence
        self.ambiguous_confidence = ambiguous_confidence
        self.route_threshold = route_threshold
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in _POSITIVE_PATTERNS.items()
        }

    def label_report(self, report: str) -> RuleLabelResult:
        """Label a single report text."""
        text = _normalize(report)
        labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        confidence = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        ambiguous = np.zeros(len(TARGET_LABELS), dtype=bool)
        sources: list[str] = []

        for i, label in enumerate(TARGET_LABELS):
            patterns = self._compiled[label]
            matches: list[tuple[bool, int, int]] = []
            for pat in patterns:
                for m in pat.finditer(text):
                    negated = _window_has_negation(text, m.start(), m.end())
                    matches.append((negated, m.start(), m.end()))

            if not matches:
                labels[i] = 0.0
                confidence[i] = 0.0
                ambiguous[i] = True
                sources.append("none")
                continue

            pos_hits = sum(1 for neg, _, _ in matches if not neg)
            neg_hits = sum(1 for neg, _, _ in matches if neg)

            if pos_hits > 0 and neg_hits > 0:
                labels[i] = 1.0 if pos_hits > neg_hits else 0.0
                confidence[i] = self.ambiguous_confidence
                ambiguous[i] = True
                sources.append("conflict")
            elif pos_hits > 0:
                labels[i] = 1.0
                confidence[i] = self.high_confidence
                ambiguous[i] = False
                sources.append("positive")
            else:
                labels[i] = 0.0
                confidence[i] = _ABSENT_CONFIDENCE
                ambiguous[i] = False
                sources.append("negated")

            if confidence[i] < self.route_threshold:
                ambiguous[i] = True

        return RuleLabelResult(
            labels=labels,
            confidence=confidence,
            ambiguous_mask=ambiguous,
            sources=sources,
        )
