"""GPU and system memory for the CUDA backends: release after a job, and the machine's sizes."""

from __future__ import annotations

import gc
import os
import sys


_ALLOCATOR_CONFIG = "expandable_segments:True,garbage_collection_threshold:0.8"


def configure_allocator() -> None:
    """Set the CUDA allocator options the diffusers backends run with, unless the user set their own.

    PyTorch reads them once, when CUDA first starts, so this must run before anything touches ``torch.cuda``.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOCATOR_CONFIG)


def release_memory() -> None:
    """Free a finished model's memory: collect it, then return the freed system and GPU memory.

    Pipelines with offload hooks hold reference cycles, so an unused model stays in system and GPU memory until
    the garbage collector runs. Streamed weights live in pinned memory, which PyTorch caches for reuse instead of
    returning it; that cache is emptied too.
    """
    import torch

    gc.collect()
    _trim_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        empty_host_cache = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
        if empty_host_cache is not None:
            empty_host_cache()


def _trim_heap() -> None:
    """Return the C heap's free memory to the system on glibc Linux.

    Freed tensors below glibc's mmap threshold stay in the heap, so after a model is released the process can
    keep gigabytes it no longer uses.
    """
    if sys.platform != "linux":
        return
    import ctypes

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError, AttributeError:  # not glibc
        pass


def memory_sizes() -> tuple[int, int] | None:
    """Return ``(GPU memory, system memory)`` in bytes for the first NVIDIA GPU, or ``None`` without one.

    The GPU's memory comes from ``nvidia-smi``, so the query does not start CUDA in this process.
    """
    gpu_bytes = _gpu_memory_bytes()
    if gpu_bytes is None:
        return None
    import psutil

    return gpu_bytes, int(psutil.virtual_memory().total)


def _gpu_memory_bytes() -> int | None:
    """Return the first NVIDIA GPU's memory in bytes as ``nvidia-smi`` reports it, or ``None`` when unavailable."""
    import shutil
    import subprocess

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run([executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
        return int(float(result.stdout.strip().splitlines()[0])) * 1024 * 1024  # reported in MiB
    except OSError, subprocess.SubprocessError, ValueError, IndexError:
        return None
