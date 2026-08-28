"""Phase 2 report labeling and pseudo-label tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import REPORT_COL, STUDY_ID_COL, TARGET_LABELS
from rsna_knee.reports.eda import detect_report_language, report_eda_summary
from rsna_knee.reports.evaluate import evaluate_labeler, per_label_metrics
from rsna_knee.reports.hybrid import HybridLabeler, generate_pseudo_labels
from rsna_knee.reports.llm import LLMLabelResult
from rsna_knee.reports.rules import RuleLabeler


def test_detect_report_language() -> None:
    assert detect_report_language("The knee shows joint effusion and synovitis.") == "english"
    assert detect_report_language("Knie ohne Erguss und mit Meniskus.") == "german"
    assert detect_report_language("Rodilla con derrame articular sin fractura.") == "spanish"


def test_rule_labeler_effusion_positive() -> None:
    labeler = RuleLabeler()
    result = labeler.label_report("Moderate joint effusion is present in the knee.")
    idx = TARGET_LABELS.index("joint_effusion")
    assert result.labels[idx] == 1.0
    assert result.confidence[idx] >= 0.7


def test_rule_labeler_negation() -> None:
    labeler = RuleLabeler()
    result = labeler.label_report("No joint effusion. No fracture identified.")
    eff_idx = TARGET_LABELS.index("joint_effusion")
    frac_idx = TARGET_LABELS.index("fracture")
    assert result.labels[eff_idx] == 0.0
    assert result.labels[frac_idx] == 0.0


def _label(result, name: str) -> float:
    return float(result.labels[TARGET_LABELS.index(name)])


def test_rule_labeler_acl_mcl_require_injury_language() -> None:
    labeler = RuleLabeler()
    mention = labeler.label_report("The ACL and MCL are visualized. Patellofemoral joint is normal.")
    assert _label(mention, "acl_tear") == 0.0
    assert _label(mention, "mcl_tear") == 0.0
    assert mention.sources[TARGET_LABELS.index("acl_tear")] == "none"
    assert mention.sources[TARGET_LABELS.index("mcl_tear")] == "none"

    torn = labeler.label_report("Complete ACL tear. High-grade MCL sprain.")
    assert _label(torn, "acl_tear") == 1.0
    assert _label(torn, "mcl_tear") == 1.0
    assert torn.confidence[TARGET_LABELS.index("acl_tear")] >= 0.7

    intact = labeler.label_report("ACL is intact. No MCL tear.")
    assert _label(intact, "acl_tear") == 0.0
    assert _label(intact, "mcl_tear") == 0.0


def test_rule_labeler_oa_uses_compartment_findings_not_anatomy() -> None:
    labeler = RuleLabeler()
    anatomy = labeler.label_report("Patellofemoral joint and retropatellar cartilage are visualized.")
    assert _label(anatomy, "patellofemoral_osteoarthritis") == 0.0
    assert anatomy.sources[TARGET_LABELS.index("patellofemoral_osteoarthritis")] == "none"

    medial = labeler.label_report("Medial compartment cartilage loss with degenerative chondral change.")
    assert _label(medial, "medial_osteoarthritis") == 1.0

    lateral = labeler.label_report("Lateral compartment chondrosis.")
    assert _label(lateral, "lateral_osteoarthritis") == 1.0

    pf = labeler.label_report("Chondromalacia patellae with trochlear cartilage thinning.")
    assert _label(pf, "patellofemoral_osteoarthritis") == 1.0


def test_rule_labeler_synovitis_hypertrophy() -> None:
    labeler = RuleLabeler()
    result = labeler.label_report("Synovial hypertrophy and Hoffa synovitis.")
    assert _label(result, "synovitis") == 1.0


def test_hybrid_without_llm_keeps_rule_positives() -> None:
    class UnavailableLLM:
        def is_available(self) -> bool:
            return False

    hybrid = HybridLabeler(
        llm_labeler=UnavailableLLM(),  # type: ignore[arg-type]
        confirm_labels=["acl_tear"],
    )
    result = hybrid.label_report("Complete ACL tear.")
    assert _label(result, "acl_tear") == 1.0
    assert result.label_source == "rules"


def test_parse_label_json_fenced() -> None:
    from rsna_knee.reports.llm import _parse_label_json

    parsed = _parse_label_json(
        '```json\n{"acl_tear": 1, "mcl_tear": 0}\n```',
        keys=["acl_tear", "mcl_tear", "fracture"],
    )
    assert parsed["acl_tear"] == 1
    assert parsed["mcl_tear"] == 0
    assert parsed["fracture"] == 0


def test_parse_label_json_strips_think_tags() -> None:
    from rsna_knee.reports.llm import _parse_label_json

    parsed = _parse_label_json(
        '<think>The ACL looks torn.</think>\n{"acl_tear": 1, "mcl_tear": 0}',
        keys=["acl_tear", "mcl_tear"],
    )
    assert parsed["acl_tear"] == 1
    assert parsed["mcl_tear"] == 0


def test_format_llm_call_debug_includes_parse() -> None:
    from rsna_knee.reports.llm import format_hybrid_match_debug, format_llm_call_debug

    text = format_llm_call_debug(
        call_index=1,
        keys=["acl_tear", "mcl_tear"],
        report="Complete ACL tear.",
        raw_reply='{"acl_tear": 1, "mcl_tear": 0}',
        parsed={"acl_tear": 1, "mcl_tear": 0},
    )
    assert "Asked keys" in text
    assert "acl_tear" in text
    assert '{"acl_tear": 1, "mcl_tear": 0}' in text
    match = format_hybrid_match_debug(
        call_index=1,
        decisions=[("mcl_tear", 1, 0, "veto (label=0, conf=0 masked)")],
    )
    assert "mcl_tear: rule=1 llm=0" in match


def test_scan_kaggle_models_qwen25_instruct(tmp_path: Path) -> None:
    from rsna_knee.reports.llm import _scan_kaggle_llm_mounts

    nested = (
        tmp_path
        / "models"
        / "qwen-lm"
        / "qwen2.5"
        / "transformers"
        / "1.5b-instruct"
        / "1"
    )
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")
    assert _scan_kaggle_llm_mounts(tmp_path) == nested


def test_resolve_kaggle_never_returns_hub_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from rsna_knee.reports import llm as llm_mod

    monkeypatch.setattr(llm_mod, "is_kaggle_kernel", lambda: True)
    monkeypatch.setattr(llm_mod, "_scan_kaggle_llm_mounts", lambda: None)
    monkeypatch.setattr(llm_mod, "_hf_cache_snapshot", lambda _mid: None)
    source, kind = llm_mod.resolve_llm_source(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    assert source is None
    assert kind == "none"


def test_hybrid_confirm_vetoes_unconfirmed_mcl() -> None:
    class ScriptedLLM:
        def is_available(self) -> bool:
            return True

        def label_labels_only(self, report: str, label_indices: list[int]) -> LLMLabelResult:
            text = report.lower()
            labels = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            conf = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            for i in label_indices:
                name = TARGET_LABELS[i]
                if name == "mcl_tear":
                    labels[i] = 1.0 if "mcl tear" in text else 0.0
                elif name == "acl_tear":
                    labels[i] = 1.0 if "acl tear" in text else 0.0
                conf[i] = 0.8
            return LLMLabelResult(labels=labels, confidence=conf, raw_response="scripted")

    hybrid = HybridLabeler(
        llm_labeler=ScriptedLLM(),  # type: ignore[arg-type]
        confirm_labels=["mcl_tear", "acl_tear"],
        fill_ambiguous=False,
    )
    confirmed = hybrid.label_report("Complete ACL tear. High-grade MCL sprain.")
    assert _label(confirmed, "acl_tear") == 1.0
    assert _label(confirmed, "mcl_tear") == 0.0
    assert confirmed.confidence[TARGET_LABELS.index("mcl_tear")] == 0.0
    assert confirmed.label_source == "hybrid"

    keep = hybrid.label_report("Complete ACL tear and MCL tear.")
    assert _label(keep, "acl_tear") == 1.0
    assert _label(keep, "mcl_tear") == 1.0


def test_hybrid_uses_ground_truth_for_labeled(sample_data_dir: Path) -> None:
    from rsna_knee.data.schema import load_train_table

    train = load_train_table(sample_data_dir)
    labeler = HybridLabeler()
    row = train.iloc[0]
    gt = row[TARGET_LABELS].to_numpy(dtype=np.float32)
    result = labeler.label_report(
        str(row[REPORT_COL]),
        ground_truth=gt,
        has_ground_truth=True,
    )
    np.testing.assert_array_equal(result.labels, gt)
    assert result.label_source == "ground_truth"


def test_resolve_pseudo_labels_path_prefers_existing(tmp_path: Path) -> None:
    from rsna_knee.reports.persist import resolve_pseudo_labels_path

    missing = tmp_path / "nope.csv"
    assert resolve_pseudo_labels_path(missing) is None
    found = tmp_path / "pseudo_labels.csv"
    found.write_text("StudyInstanceUID\n1\n", encoding="utf-8")
    assert resolve_pseudo_labels_path(found) == found


def test_generate_pseudo_labels(sample_data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "pseudo_labels.csv"
    df = generate_pseudo_labels(sample_data_dir, output_path=out)
    assert len(df) >= 1
    assert STUDY_ID_COL in df.columns
    assert out.is_file()
    for name in TARGET_LABELS:
        assert name in df.columns
        assert f"{name}_conf" in df.columns


def test_evaluate_labeler_runs(sample_data_dir: Path) -> None:
    metrics, macro_f1 = evaluate_labeler(RuleLabeler(), sample_data_dir)
    assert len(metrics) == len(TARGET_LABELS)
    assert 0.0 <= macro_f1 <= 1.0


def test_report_eda_summary(sample_data_dir: Path) -> None:
    summary = report_eda_summary(sample_data_dir)
    assert summary["n_studies"] >= 1
    assert "length_stats" in summary
    assert "language_distribution" in summary


def test_knee_study_dataset_pseudo_labels(sample_data_dir: Path, tmp_path: Path) -> None:
    from rsna_knee.data import KneeStudyDataset

    pseudo_path = tmp_path / "pseudo.csv"
    generate_pseudo_labels(sample_data_dir, output_path=pseudo_path)

    ds = KneeStudyDataset(
        sample_data_dir,
        split="train",
        labeled_only=False,
        pseudo_labels_path=pseudo_path,
        min_confidence=0.0,
        volume_shape=(8, 32, 32),
        max_series=2,
        cache=False,
        require_dicom=False,
    )
    assert len(ds) >= 2
    for i in range(len(ds)):
        item = ds[i]
        assert "confidence" in item
        assert item["confidence"].shape == (12,)
        assert item["labels"].shape == (12,)
        assert item["mask"].shape == (12,)
        assert item["mask"].sum() >= 0

    labels, masks, conf = ds.stacked_labels()
    assert labels.shape == (len(ds), 12)
    np.testing.assert_array_equal(labels[1], ds[1]["labels"])
    np.testing.assert_array_equal(masks[1], ds[1]["mask"])
    np.testing.assert_array_equal(conf[1], ds[1]["confidence"])


def test_confidence_weighted_loss() -> None:
    torch = pytest.importorskip("torch")
    from rsna_knee.training.loss import masked_bce_with_logits

    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.ones(2, 3)
    confidence = torch.tensor([[1.0, 0.5, 1.0], [1.0, 1.0, 0.5]])
    loss = masked_bce_with_logits(logits, targets, mask, confidence=confidence)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_multimodal_forward_image_only() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from rsna_knee.models.multimodal import build_multimodal_model

    model = build_multimodal_model(allow_random_init=True, slice_chunk=4)
    model.eval()
    x = torch.rand(1, 4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x, use_text=False)
    assert logits.shape == (1, 12)
