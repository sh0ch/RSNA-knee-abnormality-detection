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
    item = ds[0]
    assert "confidence" in item
    assert item["confidence"].shape == (12,)
    assert item["mask"].sum() >= 0


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
