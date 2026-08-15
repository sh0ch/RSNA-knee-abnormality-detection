"""Model definitions."""

from rsna_knee.models.weights import (
    CONVNEXT_TINY_FILENAME,
    default_convnext_tiny_path,
    resolve_pretrained_weights,
)

__all__ = [
    "CONVNEXT_TINY_FILENAME",
    "ConvNeXtTinyMIL",
    "build_model",
    "default_convnext_tiny_path",
    "mixup_batch",
    "resolve_pretrained_weights",
]


def __getattr__(name: str):
    """Lazy-import torch / torchvision model code."""
    if name in {"ConvNeXtTinyMIL", "build_model", "mixup_batch"}:
        from rsna_knee.models import mil_2p5d as _mil

        return getattr(_mil, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
