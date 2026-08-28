"""K-fold training and inference loops for Phase 1 image baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import traceback

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


def _cached_volume_nbytes(
    n_studies: int,
    volume_shape: tuple[int, int, int],
    max_series: int,
    *,
    dtype_bytes: int = 4,
) -> int:
    depth, height, width = volume_shape
    return n_studies * max_series * depth * 3 * height * width * dtype_bytes


# ~110 studies at (16, 256, 256) × 3 series float32. Phase 1 (n=58) fits; Phase 2 does not.
_CPU_CACHE_MAX_BYTES = 4 * 1024**3


def _make_loader(
    ds: Dataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        # CUDA is already initialized (pos_weight, cudnn); fork would be unsafe.
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_torch,
        **kwargs,
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
    tqdm_position: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    pin = device.type == "cuda" and not images_on_device
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    pbar = tqdm(loader, desc=desc, leave=False, position=tqdm_position)
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
    tqdm_position: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    uids: list[str] = []
    pin = device.type == "cuda" and not images_on_device
    for batch in tqdm(loader, desc=desc, leave=False, position=tqdm_position):
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


def _fold_waves(n_folds: int, n_gpus: int) -> list[list[tuple[int, int]]]:
    """Group folds into waves of ``(fold_index, gpu_id)``."""
    gpus = max(int(n_gpus), 1)
    waves: list[list[tuple[int, int]]] = []
    for start in range(0, n_folds, gpus):
        wave = [
            (start + offset, offset) for offset in range(min(gpus, n_folds - start))
        ]
        waves.append(wave)
    return waves


def _train_one_fold(
    *,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ds: Dataset,
    device: torch.device,
    loader_batch: int,
    num_workers: int,
    pin_memory: bool,
    max_epochs: int,
    learning_rate: float,
    pretrained_path: Path | str | None,
    allow_random_init: bool,
    model_name: str,
    slice_chunk: int,
    use_amp: bool,
    use_dp: bool,
    n_gpus: int,
    gpu_cache: bool,
    tta: bool,
    mixup_alpha: float,
    confidence_weighted_loss: bool,
    use_multimodal: bool,
    text_model_path: Path | str | None,
    pos_w_np: np.ndarray,
    ckpt_dir: Path,
    eval_labeled_only: bool,
    labeled_indices: np.ndarray | None,
    tqdm_position: int = 0,
) -> FoldResult:
    if device.type == "cuda":
        torch.cuda.set_device(device)

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
    pos_weight = torch.tensor(pos_w_np, device=device)

    best_auc = -1.0
    ckpt_path = ckpt_dir / f"fold{fold}.pt"
    best_state = None

    print(f"Fold {fold} [{device}] ready — first batches stream DICOMs on this thread", flush=True)

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
            tqdm_position=tqdm_position,
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
            tqdm_position=tqdm_position,
        )
        try:
            if eval_labeled_only and labeled_indices is not None:
                labeled_in_val = np.isin(val_idx, labeled_indices)
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
            f"Fold {fold} [{device}] epoch {epoch}/{max_epochs}  "
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
        tqdm_position=tqdm_position,
    )
    auc_str = f"{best_auc:.4f}" if best_auc == best_auc else "nan"
    print(f"Fold {fold}: val macro ROC-AUC = {auc_str}  → {ckpt_path}", flush=True)

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return FoldResult(
        fold=fold,
        checkpoint_path=ckpt_path,
        val_auc=best_auc,
        oof_indices=val_idx,
        oof_preds=val_preds,
        oof_labels=val_labels,
        oof_masks=val_masks,
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
    parallel_folds: bool | None = None,
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
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        for gpu_id in range(n_gpus):
            free, total = torch.cuda.mem_get_info(gpu_id)
            print(
                f"  cuda:{gpu_id}  {free / (1024**3):.1f} / {total / (1024**3):.1f} GiB free",
                flush=True,
            )

    base = KneeStudyDataset(
        data_root,
        split="train",
        labeled_only=labeled_only,
        pseudo_labels_path=pseudo_labels_path,
        min_confidence=min_confidence,
        volume_shape=volume_shape,
        max_series=max_series,
        cache=False,
    )
    if len(base) == 0:
        raise RuntimeError("No studies found for training.")

    cache_bytes = _cached_volume_nbytes(len(base), volume_shape, max_series)
    ram_cache = cache_bytes <= _CPU_CACHE_MAX_BYTES
    if gpu_cache and not ram_cache:
        print(
            f"gpu_cache disabled: {len(base)} studies would need "
            f"{cache_bytes / (1024**3):.0f} GiB",
            flush=True,
        )
        gpu_cache = False
    base.cache = bool(ram_cache or gpu_cache)
    pin_memory = device.type == "cuda" and not gpu_cache
    if gpu_cache or ram_cache:
        num_workers = 0

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

    n_splits = min(n_folds, len(base))
    want_parallel = n_gpus > 1 and not use_dp and not gpu_cache and not ram_cache
    if parallel_folds is None:
        use_parallel = want_parallel
    else:
        use_parallel = bool(parallel_folds) and want_parallel
        if parallel_folds and not want_parallel:
            print(
                "parallel_folds skipped (need 2+ GPUs, streaming, and data_parallel=false)",
                flush=True,
            )
    fold_workers = num_workers
    if use_parallel:
        # DataLoader workers must be started from the main process. Spawning
        # them from ThreadPoolExecutor threads hangs on the first batch
        # (tqdm stuck at 0/N with ?it/s).
        fold_workers = 0

    gpu_mode = "gpu-cache" if gpu_cache else ("cpu-cache" if ram_cache else "stream")
    if use_dp:
        gpu_sched = f"dp×{n_gpus}"
    elif use_parallel:
        gpu_sched = f"{n_gpus}-gpu fold-parallel"
    else:
        gpu_sched = "single-gpu"
    print(
        f"{len(base)} studies | {n_splits} folds × {max_epochs} epochs | "
        f"{device} ({gpu_sched}) | batch {loader_batch} | {gpu_mode} | "
        f"workers={fold_workers}"
        f"{' in-thread ×2' if use_parallel else ''} | slice_chunk={slice_chunk} | amp={use_amp} | "
        f"labeled_only={labeled_only} | pseudo={pseudo_labels_path is not None}",
        flush=True,
    )
    if not ram_cache and not gpu_cache:
        print(
            f"Streaming DICOMs ({cache_bytes / (1024**3):.0f} GiB RAM cache skipped).",
            flush=True,
        )

    if ram_cache or gpu_cache:
        for i in tqdm(range(len(base)), desc="cache volumes (cpu)"):
            _ = base[i]

    labels_all_np, masks_all, _ = base.stacked_labels()

    if gpu_cache:
        print("Uploading cached volumes to GPU...", flush=True)
        bank = _GpuStudyBank(base, device)
        ds = bank
        mib = bank.images.numel() * bank.images.element_size() / (1024**2)
        print(f"  GPU study bank: {mib:.0f} MiB  dtype={bank.images.dtype}", flush=True)
    else:
        ds = _NumpyStudyDataset(base)

    pos_w_np = compute_pos_weight(labels_all_np, masks_all)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = list(enumerate(kf.split(np.arange(len(base)))))
    fold_results: list[FoldResult] = []
    oof_preds = np.zeros((len(base), len(TARGET_LABELS)), dtype=np.float32)
    oof_labels = labels_all_np.copy()
    oof_masks = masks_all.copy()
    oof_filled = np.zeros(len(base), dtype=bool)

    fold_kwargs: dict[str, Any] = {
        "ds": ds,
        "loader_batch": loader_batch,
        "num_workers": fold_workers,
        "pin_memory": pin_memory,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "pretrained_path": pretrained_path,
        "allow_random_init": allow_random_init,
        "model_name": model_name,
        "slice_chunk": slice_chunk,
        "use_amp": use_amp,
        "use_dp": use_dp and not use_parallel,
        "n_gpus": n_gpus,
        "gpu_cache": gpu_cache,
        "tta": tta,
        "mixup_alpha": mixup_alpha,
        "confidence_weighted_loss": confidence_weighted_loss,
        "use_multimodal": use_multimodal,
        "text_model_path": text_model_path,
        "pos_w_np": pos_w_np,
        "ckpt_dir": ckpt_dir,
        "eval_labeled_only": eval_labeled_only,
        "labeled_indices": labeled_indices,
    }

    def _launch(fold: int, train_idx: np.ndarray, val_idx: np.ndarray, gpu_id: int) -> FoldResult:
        fold_device = (
            torch.device(f"cuda:{gpu_id}") if device.type == "cuda" else device
        )
        try:
            return _train_one_fold(
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                device=fold_device,
                tqdm_position=gpu_id if use_parallel else 0,
                **fold_kwargs,
            )
        except Exception:
            print(f"Fold {fold} [{fold_device}] crashed:", flush=True)
            traceback.print_exc()
            raise

    if use_parallel:
        for wave in _fold_waves(n_splits, n_gpus):
            ids = ", ".join(f"fold {f}→cuda:{g}" for f, g in wave)
            print(f"Starting {ids}", flush=True)
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = []
                for fold_i, gpu_id in wave:
                    fold, (train_idx, val_idx) = fold_splits[fold_i]
                    futures.append(pool.submit(_launch, fold, train_idx, val_idx, gpu_id))
                wave_results = [fut.result() for fut in futures]
            for result in wave_results:
                oof_preds[result.oof_indices] = result.oof_preds
                oof_filled[result.oof_indices] = True
                fold_results.append(result)
    else:
        for fold, (train_idx, val_idx) in fold_splits:
            result = _launch(fold, train_idx, val_idx, 0)
            oof_preds[result.oof_indices] = result.oof_preds
            oof_filled[result.oof_indices] = True
            fold_results.append(result)

    fold_results.sort(key=lambda r: r.fold)

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
