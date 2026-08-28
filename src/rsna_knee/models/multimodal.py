"""Multimodal image + report model (train-only text, image-only inference)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.models.mil_2p5d import ConvNeXtTinyMIL
from rsna_knee.reports.encoding import ReportEncoder


class MultimodalMIL(nn.Module):
    """
    ConvNeXt-Tiny MIL with optional report fusion at training time.

    At inference (``use_text=False``), only the image branch is used — competition compliant.
    """

    def __init__(
        self,
        *,
        num_labels: int = len(TARGET_LABELS),
        pretrained_path: Path | str | None = None,
        allow_random_init: bool = True,
        slice_chunk: int = 16,
        text_model_path: Path | str | None = None,
        text_embed_dim: int = 256,
        fusion_hidden: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_branch = ConvNeXtTinyMIL(
            num_labels=num_labels,
            pretrained_path=pretrained_path,
            allow_random_init=allow_random_init,
            slice_chunk=slice_chunk,
            dropout=0.0,
        )
        # Reuse MIL internals but replace head with fusion.
        feat_dim = self.image_branch.head[1].in_features
        self.image_branch.head = nn.Identity()

        self.report_encoder = ReportEncoder(
            embed_dim=text_embed_dim,
            model_path=text_model_path,
        )
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + text_embed_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_labels),
        )
        self.image_only_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_labels),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected [B,S,3,H,W], got {tuple(images.shape)}")
        batch, num_slices, channels, height, width = images.shape
        flat = images.reshape(batch * num_slices, channels, height, width)
        mean = self.image_branch._imagenet_mean
        std = self.image_branch._imagenet_std
        flat = (flat - mean) / std
        feats = self.image_branch._encode_slices_chunked(flat).view(batch, num_slices, -1)
        pooled, _ = self.image_branch.pool(feats)
        return pooled

    def forward(
        self,
        images: torch.Tensor,
        reports: list[str] | None = None,
        *,
        use_text: bool = True,
    ) -> torch.Tensor:
        image_feat = self.encode_image(images)
        if use_text and reports is not None and len(reports) > 0:
            text_feat = self.report_encoder(reports)
            fused = torch.cat([image_feat, text_feat], dim=-1)
            return self.fusion(fused)
        return self.image_only_head(image_feat)


def build_multimodal_model(
    *,
    num_labels: int = len(TARGET_LABELS),
    pretrained_path: Path | str | None = None,
    allow_random_init: bool = True,
    slice_chunk: int = 16,
    text_model_path: Path | str | None = None,
) -> MultimodalMIL:
    return MultimodalMIL(
        num_labels=num_labels,
        pretrained_path=pretrained_path,
        allow_random_init=allow_random_init,
        slice_chunk=slice_chunk,
        text_model_path=text_model_path,
    )
