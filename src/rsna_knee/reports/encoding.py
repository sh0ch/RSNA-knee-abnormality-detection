"""Report text encoding for multimodal training (train-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ReportEncoder(nn.Module):
    """
    Lightweight text encoder for train-time fusion.

  Uses mean-pooled token embeddings from a HuggingFace encoder when available;
  falls back to a bag-of-characters embedding for tests without transformers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        model_path: Path | str | None = None,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.model_path = Path(model_path) if model_path is not None else None
        self._tokenizer: Any = None
        self._encoder: Any = None
        self._fallback = nn.Embedding(128, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        if self.model_path is not None and self.model_path.exists():
            self._init_transformers()

    def _init_transformers(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._encoder = AutoModel.from_pretrained(str(self.model_path))
        hidden = int(self._encoder.config.hidden_size)
        if hidden != self.embed_dim:
            self.proj = nn.Linear(hidden, self.embed_dim)

    def _fallback_embed(self, texts: list[str], device: torch.device) -> torch.Tensor:
        batch = []
        for text in texts:
            chars = [min(ord(c), 127) for c in text[: self.max_length]]
            if not chars:
                chars = [0]
            idx = torch.tensor(chars, device=device, dtype=torch.long)
            emb = self._fallback(idx).mean(dim=0)
            batch.append(emb)
        return torch.stack(batch, dim=0)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._encoder is None or self._tokenizer is None:
            return self.proj(self._fallback_embed(texts, device))

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self._encoder(**encoded)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.proj(pooled)


def encode_reports(
    reports: list[str],
    encoder: ReportEncoder,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Encode a batch of report strings to numpy embeddings."""
    dev = device or torch.device("cpu")
    encoder = encoder.to(dev)
    encoder.eval()
    with torch.no_grad():
        emb = encoder(reports)
    return emb.cpu().numpy().astype(np.float32)
