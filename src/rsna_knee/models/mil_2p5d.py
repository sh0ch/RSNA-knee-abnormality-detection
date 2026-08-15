"""2.5D ConvNeXt + gated attention MIL for study-level multilabel classification."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.models.weights import resolve_pretrained_weights


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over slice embeddings (Ilse et al., 2018)."""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, S, D]`` slice embeddings.

        Returns:
            pooled ``[B, D]`` and attention weights ``[B, S]``.
        """
        a = self.attention_w(self.attention_v(x) * self.attention_u(x)).squeeze(-1)
        weights = torch.softmax(a, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ConvNeXtTinyMIL(nn.Module):
    """
    Encode each 2.5D slice with ConvNeXt-Tiny, then gated-attention pool to 12 logits.

    Input shape: ``[B, S, 3, H, W]``.
    """

    def __init__(
        self,
        num_labels: int = len(TARGET_LABELS),
        *,
        pretrained_path: Path | str | None = None,
        allow_random_init: bool = True,
        attention_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

        backbone = convnext_tiny(weights=None)
        # Drop classification head; keep features + avgpool → 768-d vector.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[2].in_features

        weights_path = resolve_pretrained_weights(
            explicit=pretrained_path,
            allow_missing=allow_random_init,
        )
        if weights_path is not None:
            try:
                raw = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                raw = torch.load(weights_path, map_location="cpu")
            feature_state = _feature_state_dict(raw)
            missing, unexpected = self.features.load_state_dict(feature_state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys loading ConvNeXt features from {weights_path}: {unexpected}"
                )
            if missing:
                raise RuntimeError(
                    f"Missing ConvNeXt feature keys from {weights_path}: {missing}"
                )

        self.pool = GatedAttentionPool(feat_dim, hidden=attention_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_labels),
        )

    def encode_slices(self, slices: torch.Tensor) -> torch.Tensor:
        """``[B*S, 3, H, W]`` → ``[B*S, D]``."""
        x = self.features(slices)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: ``[B, S, 3, H, W]`` float tensor in roughly [0, 1].

        Returns:
            logits ``[B, num_labels]``.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        batch, num_slices, channels, height, width = images.shape
        flat = images.reshape(batch * num_slices, channels, height, width)
        # ImageNet normalization (offline; matches torchvision ConvNeXt).
        mean = flat.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = flat.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        flat = (flat - mean) / std
        feats = self.encode_slices(flat).view(batch, num_slices, -1)
        pooled, _ = self.pool(feats)
        return self.head(pooled)


def _feature_state_dict(state: dict) -> dict[str, torch.Tensor]:
    """Map a full convnext_tiny state dict to ``features.*`` keys only."""
    # torchvision save may be raw state_dict or {"model": ...} / {"state_dict": ...}
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("features."):
            out[key.removeprefix("features.")] = value
        elif key.startswith("features"):
            # unlikely
            continue
    if out:
        return out

    # If the file already contains only feature weights (no prefix).
    return {k: v for k, v in state.items() if not k.startswith("classifier")}


def build_model(
    name: str = "convnext_tiny_mil",
    *,
    num_labels: int = len(TARGET_LABELS),
    pretrained_path: Path | str | None = None,
    allow_random_init: bool = True,
) -> nn.Module:
    """Factory used by configs / notebooks."""
    if name in {"convnext_tiny_mil", "baseline"}:
        return ConvNeXtTinyMIL(
            num_labels=num_labels,
            pretrained_path=pretrained_path,
            allow_random_init=allow_random_init,
        )
    raise ValueError(f"Unknown model name: {name}")


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixup on study tensors; masks become intersection (both must be valid)."""
    if alpha <= 0:
        return images, labels, masks
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_labels = lam * labels + (1.0 - lam) * labels[index]
    mixed_masks = masks * masks[index]
    return mixed_images, mixed_labels, mixed_masks


def horizontal_flip_tta(images: torch.Tensor) -> torch.Tensor:
    """Flip width dimension for TTA (``[B,S,3,H,W]``)."""
    return torch.flip(images, dims=[-1])
