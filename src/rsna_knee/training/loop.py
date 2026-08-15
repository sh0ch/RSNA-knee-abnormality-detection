"""K-fold training and inference loops for Phase 1 image baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _core(model: nn.Module) -> nn.Module:
    """Unwrap ``DataParallel`` for checkpoint save/load."""
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: torch.Tensor | None,
    use_mixup: bool = True,
    use_aug: bool = True,
    use_amp: bool = False,
    scaler: Any | None = None,
    desc: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    pin = device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=pin)
        labels = batch["labels"].to(device, non_blocking=pin)
        masks = batch["mask"].to(device, non_blocking=pin)

        if use_aug:
            images = augment_study_batch(images)
        if use_mixup and images.size(0) > 1:
            images, labels, masks = mixup_batch(images, labels, masks)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = masked_bce_with_logits(logits, labels, masks, pos_weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_val = float(loss.item())
        total_loss += loss_val * bs
        total_n += bs
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: bool = False,
    use_amp: bool = False,
    desc: str = "val",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    uids: list[str] = []
    pin = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=pin)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            probs = torch.sigmoid(logits)
            if tta:
                logits_flip = model(horizontal_flip_tta(images))
                probs = 0.5 * (probs + torch.sigmoid(logits_flip))
        preds.append(probs.float().cpu().numpy())
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
    allow_random_init: bool = True,
    tta: bool = True,
    num_workers: int = 0,
    model_name: str = "convnext_tiny_mil",
    use_amp: bool = True,
) -> dict[str, Any]:
    """
    Train study-level k-fold on labeled studies only.

    ``batch_size`` is per GPU. ``DataParallel`` is used when 2+ CUDA devices
    are visible. Checkpoints are stored without a ``module.`` prefix.
    """
    device = _device()
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = bool(use_amp) and device.type == "cuda"
    loader_batch = batch_size * max(n_gpus, 1)
    pin_memory = device.type == "cuda"
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    print(
        f"{len(base)} studies | {min(n_folds, len(base))} folds × {max_epochs} epochs | "
        f"{device} x{max(n_gpus, 1)} | batch {loader_batch} ({batch_size}/GPU) | amp={use_amp}",
        flush=True,
    )

    labels_all: list[np.ndarray] = []
    masks_all_list: list[np.ndarray] = []
    for i in tqdm(range(len(base)), desc="cache volumes"):
        item = base[i]
        labels_all.append(item["labels"])
        masks_all_list.append(item["mask"])
    labels_all_np = np.stack(labels_all, axis=0)
    masks_all = np.stack(masks_all_list, axis=0)

    ds = _NumpyStudyDataset(base)
    pos_w_np = compute_pos_weight(labels_all_np, masks_all)
    pos_weight = torch.tensor(pos_w_np, device=device)

    kf = KFold(n_splits=min(n_folds, len(base)), shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []
    oof_preds = np.zeros((len(base), len(TARGET_LABELS)), dtype=np.float32)
    oof_labels = labels_all_np.copy()
    oof_masks = masks_all.copy()
    oof_filled = np.zeros(len(base), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(base)))):
        train_loader = DataLoader(
            Subset(ds, train_idx.tolist()),
            batch_size=loader_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_torch,
        )
        val_loader = DataLoader(
            Subset(ds, val_idx.tolist()),
            batch_size=loader_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_torch,
        )

        model = build_model(
            model_name,
            pretrained_path=pretrained_path,
            allow_random_init=allow_random_init,
        ).to(device)
        if n_gpus > 1:
            model = nn.DataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        best_auc = -1.0
        ckpt_path = ckpt_dir / f"fold{fold}.pt"
        best_state = None

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight=pos_weight,
                use_amp=use_amp,
                scaler=scaler,
                desc=f"fold {fold} epoch {epoch}/{max_epochs}",
            )
            scheduler.step()
            val_preds, val_labels, val_masks, _ = predict_loader(
                model,
                val_loader,
                device,
                tta=False,
                use_amp=use_amp,
                desc=f"fold {fold} val",
            )
            try:
                auc = macro_roc_auc(val_labels, val_preds)
            except ValueError:
                auc = float("nan")
            if auc == auc and auc > best_auc:
                best_auc = auc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in _core(model).state_dict().items()
                }
            auc_str = f"{auc:.4f}" if auc == auc else "nan"
            print(
                f"Fold {fold} epoch {epoch}/{max_epochs}  "
                f"loss={train_loss:.4f}  val_auc={auc_str}",
                flush=True,
            )

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in _core(model).state_dict().items()}
            best_auc = float("nan")

        torch.save({"model": best_state, "fold": fold, "val_auc": best_auc}, ckpt_path)
        _core(model).load_state_dict(best_state)
        val_preds, val_labels, val_masks, _ = predict_loader(
            model, val_loader, device, tta=tta, use_amp=use_amp, desc=f"fold {fold} oof"
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
        auc_str = f"{best_auc:.4f}" if best_auc == best_auc else "nan"
        print(f"Fold {fold}: val macro ROC-AUC = {auc_str}  → {ckpt_path}", flush=True)

    assert oof_filled.all(), "Not all studies received OOF predictions"
    try:
        overall_auc = macro_roc_auc(oof_labels, oof_preds)
    except ValueError:
        overall_auc = float("nan")

    print(f"OOF macro ROC-AUC: {overall_auc:.4f}", flush=True)
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
    use_amp: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Average fold checkpoints (+ optional TTA) on the test split."""
    device = _device()
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = bool(use_amp) and device.type == "cuda"
    loader_batch = batch_size * max(n_gpus, 1)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base = KneeStudyDataset(
        data_root,
        split="test",
        labeled_only=False,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=False,
    )
    loader = DataLoader(
        _NumpyStudyDataset(base),
        batch_size=loader_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
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
        if n_gpus > 1:
            model = nn.DataParallel(model)
        preds, _, _, uids = predict_loader(
            model,
            loader,
            device,
            tta=tta,
            use_amp=use_amp,
            desc=f"test {Path(ckpt).name}",
        )
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
