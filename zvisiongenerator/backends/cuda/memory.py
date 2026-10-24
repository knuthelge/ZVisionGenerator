"""GPU and system memory for the CUDA backends: release after a job, and the machine's sizes."""

from __future__ import annotations

import gc
import os
import sys


_ALLOCATOR_CONFIG = "expandable_segments:True,garbage_collection_threshold:0.8"

PIN_ROUND_SETTING = "pinned_max_round_threshold_mb:1"
"""Allocator setting that stops PyTorch rounding pinned host allocations up to a power of two."""

__all__ = ["PIN_ROUND_SETTING", "configure_allocator", "memory_sizes", "merged_alloc_conf", "release_memory"]


def merged_alloc_conf(cuda_conf: str | None, alloc_conf: str | None, add: str) -> str:
    """Merge allocator option strings into one ``PYTORCH_ALLOC_CONF`` value.

    PyTorch 2.14 reads ``PYTORCH_CUDA_ALLOC_CONF`` first and falls back to ``PYTORCH_ALLOC_CONF`` only when the
    former is unset, while other builds prefer the latter, so the combined options must live in one string. The
    pairs of ``cuda_conf`` come first, then those of ``alloc_conf``, which win on a repeated key. ``add`` is
    appended only when its key is absent, so a value the user chose is kept. Keys keep the order in which they
    were first seen.

    Args:
        cuda_conf: The ``PYTORCH_CUDA_ALLOC_CONF`` value, or ``None``.
        alloc_conf: The ``PYTORCH_ALLOC_CONF`` value, or ``None``.
        add: One ``key:value`` option to include unless a key of that name is already set.

    Returns:
        A comma-separated ``key:value`` string.
    """
    merged: dict[str, str] = {}
    for conf in (cuda_conf, alloc_conf, add):
        for pair in (conf or "").split(","):
            pair = pair.strip()
            if not pair:
                continue
            key = pair.partition(":")[0].strip()
            if conf is add and key in merged:
                continue
            merged[key] = pair
    return ",".join(merged.values())


def configure_allocator() -> None:
    """Set the CUDA allocator options the CUDA backends run with, unless the user set their own.

    Pinned host memory is no longer rounded up to a power of two, which otherwise inflates the pinned copies of
    streamed weights by up to 1.7x. A ``pinned_max_round_threshold_mb`` value the user set is kept.

    PyTorch 2.14 reads ``PYTORCH_CUDA_ALLOC_CONF`` first and ``PYTORCH_ALLOC_CONF`` only when that is unset, so
    the same merged string is written to both. The defaults apply only when the user set neither variable. A
    ``PYTORCH_CUDA_ALLOC_CONF`` the user set is left exactly as written; the merged string goes to
    ``PYTORCH_ALLOC_CONF`` alone in that case.

    PyTorch reads the options once, when CUDA first starts, so this must run before anything touches ``torch.cuda``.
    """
    user_cuda = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    user_alloc = os.environ.get("PYTORCH_ALLOC_CONF")
    base = user_cuda
    if user_cuda is None and user_alloc is None:
        base = _ALLOCATOR_CONFIG
    merged = merged_alloc_conf(base, user_alloc, PIN_ROUND_SETTING)
    if user_cuda is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = merged
    os.environ["PYTORCH_ALLOC_CONF"] = merged


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
