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
from rsna_knee.data.schema import labels_present_mask, load_train_table
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
            "confidence": torch.from_numpy(item["confidence"]),
            "report": item.get("report", ""),
        }


class _GpuStudyBank(Dataset):
    """All studies resident on GPU — avoids per-step CPU→GPU copies (n≈58)."""

    def __init__(
        self,
        base: KneeStudyDataset,
        device: torch.device,
        *,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        self.uids: list[str] = []
        for i in range(len(base)):
            item = base[i]
            images.append(torch.from_numpy(item["image"]))
            labels.append(torch.from_numpy(item["labels"]))
            masks.append(torch.from_numpy(item["mask"]))
            confidences.append(torch.from_numpy(item["confidence"]))
            self.uids.append(item["study_uid"])
        self.images = torch.stack(images).to(device=device, dtype=storage_dtype)
        self.labels = torch.stack(labels).to(device=device, dtype=torch.float32)
        self.masks = torch.stack(masks).to(device=device, dtype=torch.float32)
        self.confidences = torch.stack(confidences).to(device=device, dtype=torch.float32)
        self.device = device

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "study_uid": self.uids[index],
            "image": self.images[index],
            "labels": self.labels[index],
            "mask": self.masks[index],
            "confidence": self.confidences[index],
            "report": "",
        }


def _collate_torch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "study_uid": [item["study_uid"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "confidence": torch.stack([item["confidence"] for item in batch], dim=0),
        "report": [item.get("report", "") for item in batch],
    }


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _core(model: nn.Module) -> nn.Module:
    """Unwrap ``DataParallel`` for checkpoint save/load."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _maybe_data_parallel(model: nn.Module, n_gpus: int, enabled: bool) -> nn.Module:
    if enabled and n_gpus > 1:
        return nn.DataParallel(model)
    return model


def _make_loader(
    ds: Dataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_torch,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: torch.Tensor | None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    use_aug: bool = True,
    use_amp: bool = False,
    scaler: Any | None = None,
    desc: str = "train",
    images_on_device: bool = False,
    confidence_weighted_loss: bool = False,
    use_multimodal: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    pin = device.type == "cuda" and not images_on_device
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        if images_on_device:
            images = batch["image"]
            labels = batch["labels"]
            masks = batch["mask"]
            confidences = batch["confidence"]
        else:
            images = batch["image"].to(device, non_blocking=pin)
            labels = batch["labels"].to(device, non_blocking=pin)
            masks = batch["mask"].to(device, non_blocking=pin)
            confidences = batch["confidence"].to(device, non_blocking=pin)

        if use_aug:
            images = augment_study_batch(images)
        if use_mixup and images.size(0) > 1:
            images, labels, masks = mixup_batch(images, labels, masks, alpha=mixup_alpha)
            confidences = confidences * masks

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if use_multimodal:
                logits = model(images, batch["report"], use_text=True)
            else:
                logits = model(images)
            loss_kw: dict[str, Any] = {}
            if confidence_weighted_loss:
                loss_kw["confidence"] = confidences
            loss = masked_bce_with_logits(
                logits, labels, masks, pos_weight=pos_weight, **loss_kw
            )
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
    images_on_device: bool = False,
    use_multimodal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    uids: list[str] = []
    pin = device.type == "cuda" and not images_on_device
    for batch in tqdm(loader, desc=desc, leave=False):
        if images_on_device:
            images = batch["image"]
        else:
            images = batch["image"].to(device, non_blocking=pin)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if use_multimodal:
                logits = model(images, batch["report"], use_text=False)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits)
            if tta:
                if use_multimodal:
                    logits_flip = model(horizontal_flip_tta(images), use_text=False)
                else:
                    logits_flip = model(horizontal_flip_tta(images))
                probs = 0.5 * (probs + torch.sigmoid(logits_flip))
        preds.append(probs.float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        masks.append(batch["mask"].cpu().numpy())
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
    gpu_cache: bool = True,
    data_parallel: bool = False,
    slice_chunk: int = 16,
    labeled_only: bool = True,
    pseudo_labels_path: Path | str | None = None,
    min_confidence: float = 0.7,
    confidence_weighted_loss: bool = False,
    eval_labeled_only: bool = False,
    mixup_alpha: float = 0.4,
    use_multimodal: bool = False,
    text_model_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Train study-level k-fold CV.

    Phase 1: ``labeled_only=True`` on 58 explicit labels.
    Phase 2: ``labeled_only=False`` + ``pseudo_labels_path`` for report-derived
    supervision; set ``eval_labeled_only=True`` to score OOF on labeled studies only.
    """
    device = _device()
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = bool(use_amp) and device.type == "cuda"
    gpu_cache = bool(gpu_cache) and device.type == "cuda"
    use_dp = bool(data_parallel) and n_gpus > 1
    loader_batch = batch_size
    pin_memory = device.type == "cuda" and not gpu_cache
    if gpu_cache:
        num_workers = 0
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base = KneeStudyDataset(
        data_root,
        split="train",
        labeled_only=labeled_only,
        pseudo_labels_path=pseudo_labels_path,
        min_confidence=min_confidence,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=True,
    )
    if len(base) == 0:
        raise RuntimeError("No studies found for training.")

    labeled_indices: np.ndarray | None = None
    if eval_labeled_only:
        train_df = load_train_table(data_root)
        labeled_ids = set(
            train_df.loc[labels_present_mask(train_df), "StudyInstanceUID"].astype(str)
        )
        labeled_indices = np.array(
            [i for i, uid in enumerate(base.study_ids) if uid in labeled_ids],
            dtype=int,
        )
        if labeled_indices.size == 0:
            raise RuntimeError("eval_labeled_only=True but no labeled studies in dataset.")

    gpu_mode = "gpu-cache" if gpu_cache else "cpu-loader"
    dp_mode = f"dp×{n_gpus}" if use_dp else f"single-gpu"
    print(
        f"{len(base)} studies | {min(n_folds, len(base))} folds × {max_epochs} epochs | "
        f"{device} ({dp_mode}) | batch {loader_batch} | {gpu_mode} | "
        f"slice_chunk={slice_chunk} | amp={use_amp} | "
        f"labeled_only={labeled_only} | pseudo={pseudo_labels_path is not None}",
        flush=True,
    )

    for i in tqdm(range(len(base)), desc="cache volumes (cpu)"):
        _ = base[i]

    if gpu_cache:
        print("Uploading cached volumes to GPU...", flush=True)
        bank = _GpuStudyBank(base, device)
        ds = bank
        mib = bank.images.numel() * bank.images.element_size() / (1024**2)
        print(f"  GPU study bank: {mib:.0f} MiB  dtype={bank.images.dtype}", flush=True)
        labels_all_np = bank.labels.cpu().numpy()
        masks_all = bank.masks.cpu().numpy()
    else:
        ds = _NumpyStudyDataset(base)
        labels_all_np = np.stack([base[i]["labels"] for i in range(len(base))], axis=0)
        masks_all = np.stack([base[i]["mask"] for i in range(len(base))], axis=0)

    pos_w_np = compute_pos_weight(labels_all_np, masks_all)
    pos_weight = torch.tensor(pos_w_np, device=device)

    kf = KFold(n_splits=min(n_folds, len(base)), shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []
    oof_preds = np.zeros((len(base), len(TARGET_LABELS)), dtype=np.float32)
    oof_labels = labels_all_np.copy()
    oof_masks = masks_all.copy()
    oof_filled = np.zeros(len(base), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(base)))):
        train_loader = _make_loader(
            ds,
            train_idx.tolist(),
            batch_size=loader_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = _make_loader(
            ds,
            val_idx.tolist(),
            batch_size=loader_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if use_multimodal:
            from rsna_knee.models.multimodal import build_multimodal_model

            model = build_multimodal_model(
                pretrained_path=pretrained_path,
                allow_random_init=allow_random_init,
                slice_chunk=slice_chunk,
                text_model_path=text_model_path,
            ).to(device)
        else:
            model = build_model(
                model_name,
                pretrained_path=pretrained_path,
                allow_random_init=allow_random_init,
                slice_chunk=slice_chunk,
            ).to(device)
        model = _maybe_data_parallel(model, n_gpus, use_dp)
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
                mixup_alpha=mixup_alpha,
                desc=f"fold {fold} epoch {epoch}/{max_epochs}",
                images_on_device=gpu_cache,
                confidence_weighted_loss=confidence_weighted_loss,
                use_multimodal=use_multimodal,
            )
            scheduler.step()
            val_preds, val_labels, val_masks, _ = predict_loader(
                model,
                val_loader,
                device,
                tta=False,
                use_amp=use_amp,
                desc=f"fold {fold} val",
                images_on_device=gpu_cache,
                use_multimodal=use_multimodal,
            )
            try:
                if eval_labeled_only and labeled_indices is not None:
                    val_global_idx = val_idx
                    labeled_in_val = np.isin(val_global_idx, labeled_indices)
                    if labeled_in_val.any():
                        auc = macro_roc_auc(
                            val_labels[labeled_in_val],
                            val_preds[labeled_in_val],
                        )
                    else:
                        auc = float("nan")
                else:
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
            model,
            val_loader,
            device,
            tta=tta,
            use_amp=use_amp,
            desc=f"fold {fold} oof",
            images_on_device=gpu_cache,
            use_multimodal=use_multimodal,
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
        if eval_labeled_only and labeled_indices is not None:
            overall_auc = macro_roc_auc(
                oof_labels[labeled_indices],
                oof_preds[labeled_indices],
            )
        else:
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
        "labeled_indices": labeled_indices,
        "pos_weight": pos_w_np,
        "checkpoint_dir": ckpt_dir,
    }


# Phase 2 alias for clarity in notebooks.
run_phase2_training = run_kfold_training


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
    data_parallel: bool = False,
    slice_chunk: int = 16,
) -> tuple[list[str], np.ndarray]:
    """Average fold checkpoints (+ optional TTA) on the test split."""
    device = _device()
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = bool(use_amp) and device.type == "cuda"
    use_dp = bool(data_parallel) and n_gpus > 1
    loader_batch = batch_size
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
        model = build_model(
            model_name, allow_random_init=True, slice_chunk=slice_chunk
        ).to(device)
        model.load_state_dict(state)
        model = _maybe_data_parallel(model, n_gpus, use_dp)
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
