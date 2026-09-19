"""Phase 1 image baseline tests (torch optional for CI without [train])."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.data.schema import predictions_to_submission
from rsna_knee.data.volume_prep import (
    prepare_series_tensor,
    sample_depth_indices,
    stack_adjacent_as_rgb,
)
from rsna_knee.training.loss import compute_pos_weight, masked_bce_with_logits


def test_fold_waves() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from rsna_knee.training.loop import _fold_waves

    assert _fold_waves(5, 2) == [[(0, 0), (1, 1)], [(2, 0), (3, 1)], [(4, 0)]]
    assert _fold_waves(3, 1) == [[(0, 0)], [(1, 0)], [(2, 0)]]
    assert _fold_waves(2, 2) == [[(0, 0), (1, 1)]]


def test_sample_depth_indices() -> None:
    idx = sample_depth_indices(22, 16)
    assert len(idx) == 16
    assert idx[0] == 0
    assert idx[-1] == 21


def test_stack_adjacent_as_rgb() -> None:
    vol = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    stacked = stack_adjacent_as_rgb(vol)
    assert stacked.shape == (5, 3, 4, 4)
    np.testing.assert_array_equal(stacked[0, 1], vol[0])
    np.testing.assert_array_equal(stacked[0, 0], vol[0])  # edge clamp
    np.testing.assert_array_equal(stacked[2, 0], vol[1])
    np.testing.assert_array_equal(stacked[2, 2], vol[3])


def test_prepare_series_tensor() -> None:
    vol = np.random.randn(22, 64, 48).astype(np.float32)
    out = prepare_series_tensor(vol, depth=16, height=32, width=32)
    assert out.shape == (16, 3, 32, 32)
    assert out.dtype == np.float32


def test_compute_pos_weight() -> None:
    labels = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    masks = np.ones_like(labels)
    pw = compute_pos_weight(labels, masks)
    assert pw.shape == (2,)
    assert pw[0] < pw[1]  # label0 more positive → lower pos_weight


def test_masked_bce_requires_torch() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    loss = masked_bce_with_logits(logits, targets, mask)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_predictions_to_submission_schema() -> None:
    preds = np.full((2, 12), 0.25, dtype=np.float32)
    import pandas as pd

    df = predictions_to_submission(
        ["a", "b"],
        pd.DataFrame(preds, columns=TARGET_LABELS),
    )
    assert df.shape == (2, 13)
    assert "StudyInstanceUID" in df.columns
    assert "ACL" in df.columns


def test_resolve_pretrained_weights_allow_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RSNA_PRETRAINED_WEIGHTS", raising=False)
    monkeypatch.setattr(
        "rsna_knee.models.weights.is_kaggle_kernel",
        lambda: False,
    )
    monkeypatch.setattr(
        "rsna_knee.models.weights.pretrained_weights_dir",
        lambda: tmp_path / "pretrained",
    )
    from rsna_knee.models.weights import resolve_pretrained_weights as resolve

    assert resolve(allow_missing=True) is None


def test_resolve_pretrained_weights_finds_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rsna_knee.models.weights.is_kaggle_kernel",
        lambda: False,
    )
    weight = tmp_path / "convnext_tiny_imagenet.pth"
    weight.write_bytes(b"fake")
    from rsna_knee.models.weights import resolve_pretrained_weights as resolve

    found = resolve(explicit=weight, allow_missing=False)
    assert found == weight


def test_knee_study_dataset_shapes(sample_data_dir: Path) -> None:
    from rsna_knee.data import KneeStudyDataset

    ds = KneeStudyDataset(
        sample_data_dir,
        split="train",
        labeled_only=True,
        volume_shape=(8, 32, 32),
        max_series=2,
        cache=True,
    )
    assert len(ds) >= 1
    item = ds[0]
    assert item["image"].shape == (2 * 8, 3, 32, 32)
    assert item["labels"].shape == (12,)
    assert item["mask"].shape == (12,)


def test_knee_study_dataset_disk_cache(sample_data_dir: Path, tmp_path: Path) -> None:
    from rsna_knee.data import KneeStudyDataset, warm_volume_cache

    cache_dir = tmp_path / "vol_cache"
    ds = KneeStudyDataset(
        sample_data_dir,
        split="train",
        labeled_only=True,
        volume_shape=(8, 32, 32),
        max_series=2,
        cache=False,
        cache_dir=cache_dir,
    )
    first = ds[0]["image"].copy()
    npy_files = list(ds.cache_dir.glob("*.npy"))
    assert npy_files
    packed = np.load(npy_files[0])
    assert packed.dtype == np.uint8
    assert packed.shape == (2, 8, 32, 32)

    warm_volume_cache(ds, max_workers=2)
    second = ds[0]["image"]
    np.testing.assert_allclose(first, second, atol=1 / 255)


def test_prepare_disk_volume_cache(sample_data_dir: Path, tmp_path: Path) -> None:
    from rsna_knee.data import prepare_disk_volume_cache

    ds = prepare_disk_volume_cache(
        sample_data_dir,
        tmp_path / "stage",
        labeled_only=True,
        volume_shape=(8, 32, 32),
        max_series=2,
        max_workers=2,
    )
    npy_files = list(ds.cache_dir.glob("*.npy"))
    assert len(npy_files) == len(ds)


def test_volume_cache_extra_read_dirs(sample_data_dir: Path, tmp_path: Path) -> None:
    from rsna_knee.data import KneeStudyDataset

    writer = KneeStudyDataset(
        sample_data_dir,
        split="train",
        labeled_only=True,
        volume_shape=(8, 32, 32),
        max_series=2,
        cache=False,
        cache_dir=tmp_path / "write",
    )
    uid = writer.study_ids[0]
    _ = writer[0]
    src = writer._find_cached_npy(uid)
    assert src is not None
    mount = tmp_path / "mount"
    mount.mkdir()
    (mount / src.name).write_bytes(src.read_bytes())

    reader = KneeStudyDataset(
        sample_data_dir,
        split="train",
        labeled_only=True,
        volume_shape=(8, 32, 32),
        max_series=2,
        cache=False,
        cache_dir=tmp_path / "empty_write",
    )
    reader.enable_disk_cache(tmp_path / "empty_write", extra_read_dirs=[mount])
    found = reader._find_cached_npy(uid)
    assert found is not None
    assert found.parent == mount


def test_discover_volume_cache_datasets_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.data import volume_cache as vc

    monkeypatch.setattr(vc, "is_kaggle_kernel", lambda: True)
    monkeypatch.setattr(vc, "_kaggle_input_root", lambda: tmp_path / "input")
    shaped = (
        tmp_path
        / "input"
        / "datasets"
        / "simonhochwebde"
        / "rsna-knee-volume-cache"
        / "d16_h256_w256_s3"
    )
    shaped.mkdir(parents=True)
    (shaped / "abc.npy").write_bytes(b"x")
    dirs = vc.discover_volume_cache_read_dirs("d16_h256_w256_s3")
    assert shaped in dirs
    assert "abc" in vc.cached_npy_stems(dirs)


def test_discover_volume_cache_part_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.data import volume_cache as vc

    monkeypatch.setattr(vc, "is_kaggle_kernel", lambda: True)
    monkeypatch.setattr(vc, "_kaggle_input_root", lambda: tmp_path / "input")
    shaped = (
        tmp_path
        / "input"
        / "datasets"
        / "simonhochwebde"
        / "rsna-knee-volume-cache-p2"
        / "d16_h256_w256_s3"
    )
    shaped.mkdir(parents=True)
    (shaped / "uid.npy").write_bytes(b"x")
    dirs = vc.discover_volume_cache_read_dirs("d16_h256_w256_s3")
    assert shaped in dirs
    assert "uid" in vc.cached_npy_stems(dirs)


def test_shard_npy_files_three_parts(tmp_path: Path) -> None:
    from rsna_knee.data import volume_cache as vc

    folder = tmp_path / "npy"
    folder.mkdir()
    for i in range(10):
        (folder / f"{i:02d}.npy").write_bytes(b"x")
    shards = vc.shard_npy_files(folder, 3)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert {p.name for part in shards for p in part} == {f"{i:02d}.npy" for i in range(10)}
    staged = vc.stage_volume_cache_part(
        folder, 2, n_parts=3, stage_root=tmp_path / "stage"
    )
    assert len(list(staged.glob("*.npy"))) == 3


def test_kaggle_argv_uses_cli_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna_knee.data import volume_cache as vc

    fake = tmp_path / "kaggle"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(vc.shutil, "which", lambda name: str(fake) if name == "kaggle" else None)
    assert vc.kaggle_argv() == [str(fake)]


def test_ordered_dicom_paths_cached(sample_data_dir: Path) -> None:
    from rsna_knee.data import StudyIndex
    from rsna_knee.data.dicom_io import ordered_dicom_paths

    index = StudyIndex(sample_data_dir)
    study_uid = index.iter_studies()[0]
    series_uid = index.get_series_for_study(study_uid).iloc[0]["SeriesInstanceUID"]
    path = sample_data_dir / "train_series" / study_uid / series_uid
    first = ordered_dicom_paths(path)
    second = ordered_dicom_paths(path)
    assert first is second
    assert first


def test_model_forward_random_init() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from rsna_knee.models import build_model

    model = build_model("convnext_tiny_mil", allow_random_init=True)
    model.eval()
    x = torch.rand(1, 4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 12)


def test_select_series_for_study(sample_data_dir: Path) -> None:
    from rsna_knee.data import StudyIndex, select_series_for_study

    index = StudyIndex(sample_data_dir)
    study = index.iter_studies()[0]
    series = index.get_series_for_study(study)
    selected = select_series_for_study(series, max_series=2)
    assert 1 <= len(selected) <= 2
