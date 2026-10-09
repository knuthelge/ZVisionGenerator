"""CUDA memory for the diffusers backends: the allocator options, and release after a job."""

from __future__ import annotations

import gc
import os

_ALLOCATOR_CONFIG = "expandable_segments:True,garbage_collection_threshold:0.8"


def configure_allocator() -> None:
    """Set the CUDA allocator options the diffusers backends run with, unless the user set their own.

    PyTorch reads them once, when CUDA first starts, so this must run before anything touches ``torch.cuda``.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOCATOR_CONFIG)


def release_memory() -> None:
    """Free a finished model's memory: collect it, then return PyTorch's cached GPU memory to the driver.

    Pipelines with offload hooks hold reference cycles, so an unused model stays in system and GPU memory until
    the garbage collector runs.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
