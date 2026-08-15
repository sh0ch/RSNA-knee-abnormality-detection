"""K-fold training and inference loops for Phase 1 image baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.data.dataset import KneeStudyDataset
from rsna_knee.models.mil_2p5d import (
    build_model,
    horizontal_flip_tta,
    mixup_batch,
)
from rsna_knee.training.augment import augment_study_batch
from rsna_knee.training.loss import compute_pos_weight, masked_bce_with_logits
from rsna_knee.training.metrics import macro_roc_auc


@dataclass
class FoldResult:
    fold: int
    checkpoint_path: Path
    val_auc: float
    oof_indices: np.ndarray
    oof_preds: np.ndarray
    oof_labels: np.ndarray
    oof_masks: np.ndarray


class _NumpyStudyDataset(Dataset):
    """Thin torch Dataset wrapper around KneeStudyDataset."""

    def __init__(self, base: KneeStudyDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        return {
            "study_uid": item["study_uid"],
            "image": torch.from_numpy(item["image"]),
            "labels": torch.from_numpy(item["labels"]),
            "mask": torch.from_numpy(item["mask"]),
        }


def _collate_torch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "study_uid": [item["study_uid"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
    }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: torch.Tensor | None,
    use_mixup: bool = True,
    use_aug: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        masks = batch["mask"].to(device)

        if use_aug:
            images = augment_study_batch(images)
        if use_mixup and images.size(0) > 1:
            images, labels, masks = mixup_batch(images, labels, masks)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = masked_bce_with_logits(logits, labels, masks, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    uids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        if tta:
            logits_flip = model(horizontal_flip_tta(images))
            probs = 0.5 * (probs + torch.sigmoid(logits_flip))
        preds.append(probs.cpu().numpy())
        labels.append(batch["labels"].numpy())
        masks.append(batch["mask"].numpy())
        uids.extend(batch["study_uid"])
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(masks, axis=0),
        uids,
    )


def run_kfold_training(
    data_root: Path | str | None = None,
    *,
    n_folds: int = 5,
    max_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    volume_shape: tuple[int, int, int] = (16, 256, 256),
    max_series: int = 3,
    checkpoint_dir: Path | str = "checkpoints",
    seed: int = 42,
    pretrained_path: Path | str | None = None,
    allow_random_init: bool = False,
    tta: bool = True,
    num_workers: int = 0,
    model_name: str = "convnext_tiny_mil",
) -> dict[str, Any]:
    """
    Train study-level k-fold on labeled studies only.

    Returns OOF predictions, per-fold metrics, and checkpoint paths.
    """
    device = _device()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base = KneeStudyDataset(
        data_root,
        split="train",
        labeled_only=True,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=True,
    )
    if len(base) == 0:
        raise RuntimeError("No labeled studies found for training.")

    ds = _NumpyStudyDataset(base)
    labels_all = np.stack([base[i]["labels"] for i in range(len(base))], axis=0)
    masks_all = np.stack([base[i]["mask"] for i in range(len(base))], axis=0)
    pos_w_np = compute_pos_weight(labels_all, masks_all)
    pos_weight = torch.tensor(pos_w_np, device=device)

    kf = KFold(n_splits=min(n_folds, len(base)), shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []
    oof_preds = np.zeros((len(base), len(TARGET_LABELS)), dtype=np.float32)
    oof_labels = labels_all.copy()
    oof_masks = masks_all.copy()
    oof_filled = np.zeros(len(base), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(base)))):
        train_loader = DataLoader(
            Subset(ds, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_torch,
        )
        val_loader = DataLoader(
            Subset(ds, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_torch,
        )

        model = build_model(
            model_name,
            pretrained_path=pretrained_path,
            allow_random_init=allow_random_init,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_auc = -1.0
        ckpt_path = ckpt_dir / f"fold{fold}.pt"
        best_state = None

        for _epoch in range(max_epochs):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight=pos_weight,
            )
            scheduler.step()
            val_preds, val_labels, val_masks, _ = predict_loader(
                model, val_loader, device, tta=False
            )
            # Only score labels with both classes in this fold when possible.
            try:
                auc = macro_roc_auc(val_labels, val_preds)
            except ValueError:
                auc = float("nan")
            if auc == auc and auc > best_auc:  # not NaN
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_auc = float("nan")

        torch.save({"model": best_state, "fold": fold, "val_auc": best_auc}, ckpt_path)
        model.load_state_dict(best_state)
        model.to(device)
        val_preds, val_labels, val_masks, _ = predict_loader(
            model, val_loader, device, tta=tta
        )
        oof_preds[val_idx] = val_preds
        oof_filled[val_idx] = True

        fold_results.append(
            FoldResult(
                fold=fold,
                checkpoint_path=ckpt_path,
                val_auc=best_auc,
                oof_indices=val_idx,
                oof_preds=val_preds,
                oof_labels=val_labels,
                oof_masks=val_masks,
            )
        )
        print(f"Fold {fold}: val macro ROC-AUC = {best_auc:.4f}  → {ckpt_path}")

    assert oof_filled.all(), "Not all studies received OOF predictions"
    try:
        overall_auc = macro_roc_auc(oof_labels, oof_preds)
    except ValueError:
        overall_auc = float("nan")

    return {
        "fold_results": fold_results,
        "oof_preds": oof_preds,
        "oof_labels": oof_labels,
        "oof_masks": oof_masks,
        "study_ids": list(base.study_ids),
        "overall_auc": overall_auc,
        "pos_weight": pos_w_np,
        "checkpoint_dir": ckpt_dir,
    }


@torch.no_grad()
def predict_test_ensemble(
    checkpoint_paths: list[Path | str],
    data_root: Path | str | None = None,
    *,
    volume_shape: tuple[int, int, int] = (16, 256, 256),
    max_series: int = 3,
    batch_size: int = 2,
    tta: bool = True,
    allow_random_init: bool = False,
    model_name: str = "convnext_tiny_mil",
    num_workers: int = 0,
) -> tuple[list[str], np.ndarray]:
    """
    Average fold checkpoints (+ optional TTA) on the test split.

    Streams studies without caching the full test set.
    """
    device = _device()
    base = KneeStudyDataset(
        data_root,
        split="test",
        labeled_only=False,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=False,
    )
    ds = _NumpyStudyDataset(base)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_torch,
    )

    accum: np.ndarray | None = None
    study_ids: list[str] | None = None
    n_models = 0

    for ckpt in checkpoint_paths:
        payload = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model = build_model(model_name, allow_random_init=True).to(device)
        model.load_state_dict(state)
        preds, _, _, uids = predict_loader(model, loader, device, tta=tta)
        if accum is None:
            accum = preds.astype(np.float64)
            study_ids = uids
        else:
            accum += preds
            if study_ids != uids:
                raise RuntimeError("Study ID order changed between checkpoint passes")
        n_models += 1
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accum is None or study_ids is None or n_models == 0:
        raise RuntimeError("No checkpoints provided for test inference")
    return study_ids, (accum / n_models).astype(np.float32)


def prevalence_baseline_predictions(
    train_labels: np.ndarray,
    train_masks: np.ndarray,
    n_test: int,
) -> np.ndarray:
    """Constant positive-rate predictions from labeled train studies."""
    rates = []
    for i in range(train_labels.shape[1]):
        m = train_masks[:, i] > 0.5
        if m.any():
            rates.append(float(train_labels[m, i].mean()))
        else:
            rates.append(0.5)
    rates_arr = np.asarray(rates, dtype=np.float32)
    return np.broadcast_to(rates_arr, (n_test, len(rates))).copy()
