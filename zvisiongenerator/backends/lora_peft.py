"""Detect and report LoRA files that match no module of the model (peft)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any


def is_unmatched_adapter_error(exc: BaseException) -> bool:
    """Return whether *exc* is peft's error for a LoRA none of whose tensors match the model.

    Looks peft up in ``sys.modules`` instead of importing it: diffusers has imported peft by the time it raises this.
    """
    peft = sys.modules.get("peft")
    return peft is not None and isinstance(exc, getattr(peft, "NoMatchingPeftModuleError", ()))


def warn_unmatched_lora(path: str) -> None:
    """Warn that the LoRA at *path* loaded nothing because none of its tensors match the model.

    The warning points at the caller of the function that calls this one.
    """
    warnings.warn(f"LoRA {Path(path).name}: none of its LoRA tensors match this model; they were not applied.", stacklevel=3)


def restore_cpu_offload(pipeline: Any, *, enabled: bool) -> None:
    """Re-enable model CPU offload on *pipeline* when loading a LoRA removed its hooks and left them off.

    diffusers removes the offload hooks before it loads a LoRA and does not put them back when the load raises,
    which a LoRA that matches no module does. Without them the pipeline would run with its components on the CPU.

    Args:
        pipeline: A diffusers pipeline with a ``transformer``.
        enabled: Whether the pipeline was set up with model CPU offload.
    """
    transformer = getattr(pipeline, "transformer", None)
    if enabled and transformer is not None and getattr(transformer, "_hf_hook", None) is None and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
