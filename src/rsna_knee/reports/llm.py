"""LLM-based label extraction for ambiguous report cases (offline-capable)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.constants import TARGET_DISPLAY_NAMES, TARGET_LABELS

_LABEL_JSON_KEYS = list(TARGET_LABELS)

_SYSTEM_PROMPT = """You are a radiology coding assistant. Given a knee MRI report, extract binary labels (1=present, 0=absent) for each finding. Respond with JSON only, using these keys:
acl_tear, mcl_tear, medial_meniscus_injury, lateral_meniscus_injury, medial_osteoarthritis, lateral_osteoarthritis, patellofemoral_osteoarthritis, joint_effusion, synovitis, bakers_cyst, bone_contusion, fracture
Use 0 or 1 for each key. If unclear, use 0."""


@dataclass
class LLMLabelResult:
    labels: np.ndarray  # float32 [12]
    confidence: np.ndarray  # float32 [12]
    raw_response: str


def _parse_label_json(text: str) -> dict[str, int]:
    """Extract JSON object from model output."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: int(bool(data.get(k, 0))) for k in _LABEL_JSON_KEYS}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return {k: int(bool(data.get(k, 0))) for k in _LABEL_JSON_KEYS}
        except json.JSONDecodeError:
            pass

    return dict.fromkeys(_LABEL_JSON_KEYS, 0)


def _labels_from_dict(data: dict[str, int], *, confidence: float) -> LLMLabelResult:
    labels = np.array([float(data[k]) for k in TARGET_LABELS], dtype=np.float32)
    conf = np.full(len(TARGET_LABELS), confidence, dtype=np.float32)
    return LLMLabelResult(labels=labels, confidence=conf, raw_response=json.dumps(data))


class LLMLabeler:
    """
    Extract labels via a local HuggingFace causal/seq2seq model (offline on Kaggle).

    When ``model_path`` is None, falls back to a conservative all-zero extractor
    with low confidence (for tests / CI without transformers).
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        device: str = "cpu",
        max_new_tokens: int = 256,
        llm_confidence: float = 0.75,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.llm_confidence = llm_confidence
        self._pipe: Any = None

    def _ensure_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        if self.model_path is None or not self.model_path.exists():
            return None
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for LLM labeling. "
                "Install with: pip install -e '.[train]'"
            ) from exc

        self._pipe = pipeline(
            "text-generation",
            model=str(self.model_path),
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        return self._pipe

    def _build_prompt(self, report: str) -> str:
        findings = "\n".join(f"- {TARGET_DISPLAY_NAMES[k]}" for k in TARGET_LABELS)
        return (
            f"{_SYSTEM_PROMPT}\n\nFindings to code:\n{findings}\n\n"
            f"Report:\n{report}\n\nJSON:"
        )

    def label_report(self, report: str) -> LLMLabelResult:
        """Extract 12 binary labels from one report."""
        pipe = self._ensure_pipeline()
        if pipe is None:
            return _labels_from_dict(dict.fromkeys(_LABEL_JSON_KEYS, 0), confidence=0.3)

        prompt = self._build_prompt(report)
        out = pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = out[0]["generated_text"] if out else ""
        if prompt in text:
            text = text.split(prompt, 1)[-1]
        parsed = _parse_label_json(text)
        return _labels_from_dict(parsed, confidence=self.llm_confidence)

    def label_labels_only(self, report: str, label_indices: list[int]) -> LLMLabelResult:
        """Extract only selected label indices (for partial LLM routing)."""
        full = self.label_report(report)
        mask = np.zeros(len(TARGET_LABELS), dtype=bool)
        mask[label_indices] = True
        labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        confidence = np.zeros(len(TARGET_LABELS), dtype=np.float32)
        labels[mask] = full.labels[mask]
        confidence[mask] = full.confidence[mask]
        return LLMLabelResult(labels=labels, confidence=confidence, raw_response=full.raw_response)
