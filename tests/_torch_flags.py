"""Helpers for tests that check a backend gives the global PyTorch flags back."""

from __future__ import annotations

from zvisiongenerator.utils.torch_flags import TorchFlags, capture_torch_flags

__all__ = ["current_flags", "set_app_flags", "set_kit_flags"]


def set_kit_flags() -> None:
    """Set the flags the way the audio kits do when they build a model: deterministic, no TF32, strictest precision."""
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")


def set_app_flags() -> None:
    """Set the flags the way the image and video backends leave them: benchmark on, TF32 on, ``high`` precision."""
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")


def current_flags() -> TorchFlags:
    """Return the global torch flags as they are now."""
    return capture_torch_flags()
