"""LLM confirmer for low-precision rule hits (offline-capable on Kaggle)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.constants import TARGET_DISPLAY_NAMES, TARGET_LABELS
from rsna_knee.utils.paths import is_kaggle_kernel

_LABEL_JSON_KEYS = list(TARGET_LABELS)

# Prefer Qwen2.5-1.5B-Instruct: also on Kaggle as an HF model, works with stock
# transformers. Qwen3.5-4B is attachable too but needs transformers from git main.
DEFAULT_LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Labels whose rule-positives are not precise enough to train on unaudited.
DEFAULT_CONFIRM_LABELS: tuple[str, ...] = (
    "acl_tear",
    "mcl_tear",
    "medial_osteoarthritis",
    "lateral_osteoarthritis",
    "patellofemoral_osteoarthritis",
    "synovitis",
)

_KAGGLE_LLM_MOUNTS: tuple[str, ...] = (
    "rsna-knee-llm",
    "qwen3.5-4b",
    "qwen-qwen3.5-4b",
    "qwen2.5-1.5b-instruct",
    "qwen-qwen2.5-1.5b-instruct",
    "qwen25-15b-instruct",
    "qwen2.5-1.5b",
)
# Kaggle Models layout: /kaggle/input/models/<owner>/<model>/<framework>/<variation>/<version>
_KAGGLE_LLM_RELATIVE: tuple[str, ...] = (
    "models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1",
    "models/qwen-lm/qwen2.5/transformers/1.5b-instruct",
)
_SKIP_KAGGLE_MOUNTS: tuple[str, ...] = (
    "rsna-knee-abnormality-detection",
    "competitions",
    "rsna-knee-code",
    "rsna-knee-pretrained",
)

_SYSTEM_PROMPT = (
    "You are a conservative radiology coder. Reply with JSON only. "
    "Use 1 only if the report clearly supports the finding. "
    "Use 0 if the finding is absent, negated, or unclear."
)


@dataclass
class LLMLabelResult:
    labels: np.ndarray  # float32 [12]
    confidence: np.ndarray  # float32 [12]
    raw_response: str


def resolve_llm_source(
    model_path: Path | str | None = None,
    model_id: str | None = None,
) -> tuple[str | None, str]:
    """
    Resolve a local directory of model files.

    Search order: explicit path → Kaggle Input / HF cache → local ``data/llm/``.
    ``model_id`` is only a folder-name hint. On Kaggle this never returns a
    HuggingFace hub id (that would make transformers call huggingface.co).
    """
    from rsna_knee.utils.paths import project_root

    if model_path is not None:
        path = Path(model_path)
        if path.is_file():
            path = path.parent
        found = _find_config_dir(path)
        if found is not None:
            return str(found), "path"

    if is_kaggle_kernel():
        found = _scan_kaggle_llm_mounts()
        if found is not None:
            return str(found), "kaggle"
        cached = _hf_cache_snapshot(model_id or DEFAULT_LLM_MODEL_ID)
        if cached is not None:
            return str(cached), "kaggle"
        return None, "none"

    llm_root = project_root() / "data" / "llm"
    for dirname in (
        "qwen2.5-1.5b-instruct",
        "qwen3.5-4b",
        "Qwen2.5-1.5B-Instruct",
        "Qwen3.5-4B",
    ):
        found = _find_config_dir(llm_root / dirname)
        if found is not None:
            return str(found), "path"

    # Local-only fallback. Callers on Kaggle never reach here.
    hub_id = (model_id or "").strip()
    if hub_id:
        return hub_id, "hub"
    return None, "none"


def _find_config_dir(root: Path, *, max_depth: int = 6) -> Path | None:
    """Return the directory that contains config.json, walking up to max_depth."""
    if not root.is_dir():
        return None
    if (root / "config.json").is_file():
        return root
    frontier: list[tuple[Path, int]] = [(root, 0)]
    seen = 0
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth or seen > 256:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            seen += 1
            if not child.is_dir():
                continue
            if (child / "config.json").is_file():
                return child
            frontier.append((child, depth + 1))
    return None


def _scan_kaggle_llm_mounts(root: Path | None = None) -> Path | None:
    root = root if root is not None else Path("/kaggle/input")
    if not root.is_dir():
        return None
    for rel in _KAGGLE_LLM_RELATIVE:
        found = _find_config_dir(root / rel)
        if found is not None:
            return found
    for name in _KAGGLE_LLM_MOUNTS:
        found = _find_config_dir(root / name)
        if found is not None:
            return found
    models_root = root / "models"
    if models_root.is_dir():
        for hint in ("qwen-lm", "qwen", "Qwen"):
            found = _find_config_dir(models_root / hint)
            if found is not None:
                return found
    try:
        children = sorted(root.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir() or child.name in _SKIP_KAGGLE_MOUNTS:
            continue
        if child.name == "models":
            continue
        found = _find_config_dir(child)
        if found is not None:
            return found
    return None


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            roots.append(Path(raw))
    roots.extend(
        (
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("/root/.cache/huggingface/hub"),
        )
    )
    return roots


def _hf_cache_snapshot(model_id: str) -> Path | None:
    slug = "models--" + model_id.replace("/", "--")
    for cache in _hf_cache_roots():
        found = _find_config_dir(cache / slug)
        if found is not None:
            return found
    return None


def _parse_label_json(text: str, keys: list[str] | None = None) -> dict[str, int]:
    """Extract JSON object from model output. Missing keys default to 0."""
    keys = keys or _LABEL_JSON_KEYS
    defaults = dict.fromkeys(keys, 0)
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    data = _load_json_object(text)
    if data is None:
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            data = _load_json_object(match.group())
    if not isinstance(data, dict):
        return defaults
    out = dict(defaults)
    for key in keys:
        if key in data:
            out[key] = int(bool(data[key]))
            continue
        # Models sometimes emit display names.
        for alt, canonical in TARGET_DISPLAY_NAMES.items():
            if alt == key and canonical.lower() in {str(k).lower() for k in data}:
                raw = data.get(canonical, data.get(canonical.lower()))
                out[key] = int(bool(raw))
    return out


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def format_llm_call_debug(
    *,
    call_index: int,
    keys: list[str],
    report: str,
    raw_reply: str,
    parsed: dict[str, int],
    report_chars: int = 1800,
) -> str:
    """Human-readable dump of one confirmer call (prompt context + parse)."""
    excerpt = report or ""
    clipped = excerpt[:report_chars]
    clip_note = "" if len(excerpt) <= report_chars else f"\n… [{len(excerpt)} chars total]"
    asked = {k: int(parsed.get(k, 0)) for k in keys}
    unmatched = [k for k in keys if k not in parsed]
    lines = [
        f"=== LLM call {call_index} ===",
        f"Asked keys : {keys}",
        f"Report ({len(excerpt)} chars):",
        clipped + clip_note,
        "Raw reply:",
        (raw_reply or "").strip() or "<empty>",
        f"Parsed JSON for asked keys: {asked}",
    ]
    if unmatched:
        lines.append(f"Missing keys (default 0): {unmatched}")
    return "\n".join(lines)


def format_hybrid_match_debug(
    *,
    call_index: int,
    decisions: list[tuple[str, int, int, str]],
) -> str:
    """How rule positives were confirmed or vetoed after the LLM reply."""
    lines = [f"=== Match {call_index} (rule AND llm → keep; veto → mask conf=0) ==="]
    for name, rule_bit, llm_bit, action in decisions:
        lines.append(f"  {name}: rule={rule_bit} llm={llm_bit} → {action}")
    if not decisions:
        lines.append("  (no confirm-labels were rule-positive; LLM not called)")
    return "\n".join(lines)


def _labels_from_dict(
    data: dict[str, int],
    *,
    confidence: float,
    raw_response: str | None = None,
) -> LLMLabelResult:
    labels = np.array([float(data.get(k, 0)) for k in TARGET_LABELS], dtype=np.float32)
    conf = np.full(len(TARGET_LABELS), confidence, dtype=np.float32)
    return LLMLabelResult(
        labels=labels,
        confidence=conf,
        raw_response=raw_response if raw_response is not None else json.dumps(data),
    )


class LLMLabeler:
    """
    Confirm selected findings with a local instruct model.

    When no weights are available, ``is_available()`` is False and callers should
    keep rule outputs (tests / CI). Do not treat the zero stub as a veto.
    """

    DEFAULT_CONFIRM_LABELS = DEFAULT_CONFIRM_LABELS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_id: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 160,
        llm_confidence: float = 0.8,
        max_report_chars: int = 6000,
        verbose: bool = False,
        debug_max: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.llm_confidence = llm_confidence
        self.max_report_chars = max_report_chars
        self.verbose = verbose
        self.debug_max = debug_max
        self._debug_calls = 0
        self.source, self.source_kind = resolve_llm_source(model_path, model_id)
        self._pipe: Any = None
        self._load_error: str | None = None

    def describe(self) -> str:
        if self.source is None:
            return "unavailable"
        return f"{self.source_kind}:{self.source}"

    def is_available(self) -> bool:
        if self._pipe is not None:
            return True
        if self._load_error is not None:
            return False
        if self.source is None:
            return False
        try:
            import torch  # noqa: F401
            from transformers import pipeline  # noqa: F401
        except ImportError:
            return False
        return True

    def warmup(self) -> None:
        """Load weights now so a missing model fails before labeling 58/4407 studies."""
        if self._ensure_pipeline() is None:
            extra = ""
            if is_kaggle_kernel() and Path("/kaggle/input").is_dir():
                names = sorted(p.name for p in Path("/kaggle/input").iterdir())
                extra = f" Mounted inputs: {names}."
            raise FileNotFoundError(
                (
                    self._load_error
                    or (
                        "No LLM weights on this kernel. Expected "
                        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1. "
                        "Add Input → Models → Qwen2.5 1.5B Instruct, restart Jupyter, "
                        "then re-run Setup after syncing src/."
                    )
                )
                + extra
            )

    def unload(self) -> None:
        """Drop GPU weights so Phase 2 training can use cuda:0."""
        pipe = self._pipe
        self._pipe = None
        if pipe is None:
            return
        del pipe
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        if self.source is None:
            self._load_error = "No LLM source resolved."
            return None

        source_path = Path(self.source)
        on_kaggle = is_kaggle_kernel()
        # Hub ids make transformers call huggingface.co. Never do that on Kaggle.
        if on_kaggle or self.source_kind != "hub":
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if on_kaggle and (self.source_kind == "hub" or not source_path.is_dir()):
            self._load_error = (
                f"Refusing HuggingFace hub id {self.source!r}. Expected local dir "
                "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1. "
                "Sync src/ and re-run Setup."
            )
            return None

        try:
            import torch
            from transformers import pipeline
        except ImportError:
            self._load_error = (
                "transformers is required for LLM labeling. "
                "Install with: pip install -e '.[train]'"
            )
            return None

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe_kwargs: dict[str, Any] = {
            "task": "text-generation",
            "model": self.source,
            "tokenizer": self.source,
            "torch_dtype": dtype,
            "trust_remote_code": False,
        }
        if device == "cpu":
            pipe_kwargs["device"] = "cpu"
        else:
            pipe_kwargs["device_map"] = "auto"
        try:
            self._pipe = pipeline(**pipe_kwargs)
        except Exception as exc:  # noqa: BLE001 — surface load failures to warmup()
            self._load_error = f"Failed to load LLM from {self.source}: {exc}"
            return None
        tokenizer = getattr(self._pipe, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return self._pipe

    def _build_prompt(self, report: str, label_indices: list[int] | None = None) -> str:
        indices = label_indices if label_indices is not None else list(range(len(TARGET_LABELS)))
        names = [TARGET_LABELS[i] for i in indices]
        keys = ", ".join(names)
        findings = "\n".join(f"- {n}: {TARGET_DISPLAY_NAMES[n]}" for n in names)
        clipped = (report or "")[: self.max_report_chars]
        user = (
            f"Return JSON only with keys: {keys}. Values must be 0 or 1.\n\n"
            f"Findings:\n{findings}\n\n"
            f"Report:\n{clipped}\n"
        )
        pipe = self._pipe
        tokenizer = getattr(pipe, "tokenizer", None) if pipe is not None else None
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            try:
                return str(
                    apply(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                try:
                    return str(
                        apply(messages, tokenize=False, add_generation_prompt=True)
                    )
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        return f"{_SYSTEM_PROMPT}\n\n{user}\nJSON:"

    def label_report(self, report: str) -> LLMLabelResult:
        """Extract all 12 labels (used when fill_ambiguous is on)."""
        return self.label_labels_only(report, list(range(len(TARGET_LABELS))))

    def label_labels_only(self, report: str, label_indices: list[int]) -> LLMLabelResult:
        """Confirm only the requested label indices."""
        keys = [TARGET_LABELS[i] for i in label_indices]
        empty = dict.fromkeys(_LABEL_JSON_KEYS, 0)
        pipe = self._ensure_pipeline()
        if pipe is None:
            return _labels_from_dict(empty, confidence=0.3)

        prompt = self._build_prompt(report, label_indices)
        out = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = ""
        if out:
            item = out[0]
            text = str(item.get("generated_text", item) if isinstance(item, dict) else item)
        if prompt and prompt in text:
            text = text.split(prompt, 1)[-1]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        parsed = _parse_label_json(text, keys=keys)
        if self.verbose:
            self._debug_calls += 1
            if self.debug_max is None or self._debug_calls <= self.debug_max:
                print(
                    format_llm_call_debug(
                        call_index=self._debug_calls,
                        keys=keys,
                        report=report,
                        raw_reply=text,
                        parsed=parsed,
                    ),
                    flush=True,
                )
        merged = dict(empty)
        merged.update(parsed)
        result = _labels_from_dict(
            merged, confidence=self.llm_confidence, raw_response=text
        )
        mask = np.zeros(len(TARGET_LABELS), dtype=bool)
        mask[list(label_indices)] = True
        result.labels[~mask] = 0.0
        result.confidence[~mask] = 0.0
        return result
