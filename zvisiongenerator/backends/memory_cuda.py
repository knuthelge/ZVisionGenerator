"""CUDA memory for the diffusers backends: the allocator options, and release after a job."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

from zvisiongenerator.utils.stored_quant import fsync_file

_ALLOCATOR_CONFIG = "expandable_segments:True,garbage_collection_threshold:0.8"


def configure_allocator() -> None:
    """Set the CUDA allocator options the diffusers backends run with, unless the user set their own.

    PyTorch reads them once, when CUDA first starts, so this must run before anything touches ``torch.cuda``.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOCATOR_CONFIG)


def release_memory() -> None:
    """Free a finished model's memory: collect it, then return the freed system and GPU memory.

    Pipelines with offload hooks hold reference cycles, so an unused model stays in system and GPU memory until
    the garbage collector runs.
    """
    import torch

    gc.collect()
    _trim_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def drop_cached_files(directory: Path) -> None:
    """Write the files in *directory* through to disk and drop them from the page cache, where the platform can.

    The kernel keeps every model file read or written in memory until it needs the space. Model files can be
    larger than system memory, and reclaiming them under load is memory pressure that can get processes killed.
    """
    for path in directory.rglob("*"):
        if path.is_file():
            drop_cached_file(path)


def drop_cached_file(path: Path) -> None:
    """Write *path* through to disk and drop it from the page cache, where the platform can.

    Pages a process still maps stay cached, so call this once the file is closed.
    """
    if not hasattr(os, "posix_fadvise"):  # Linux only
        return
    fsync_file(path)  # dirty pages cannot be dropped
    try:
        with open(path, "rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass  # the cache is only a memory aid


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
