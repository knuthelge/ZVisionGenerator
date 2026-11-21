"""GPU and system memory for the CUDA backends: release after a job, and the machine's sizes."""

from __future__ import annotations

import gc
import logging
import os
import sys

from zvisiongenerator.backends.cuda import nvml
from zvisiongenerator.core.memory_sampler import MemoryUsage


_logger = logging.getLogger(__name__)

_ALLOCATOR_CONFIG = "expandable_segments:True,garbage_collection_threshold:0.8"

_WINDOWS_ALLOCATOR_CONFIG = "garbage_collection_threshold:0.8"

PIN_ROUND_SETTING = "pinned_max_round_threshold_mb:1"
"""Allocator setting that stops PyTorch rounding pinned host allocations up to a power of two."""

__all__ = [
    "PIN_ROUND_SETTING",
    "card_total_bytes",
    "configure_allocator",
    "free_bytes",
    "memory_sizes",
    "memory_usage",
    "merged_alloc_conf",
    "release_memory",
]


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

    Windows runs without expandable segments: under WDDM a full card with them is a driver error in
    ``cuMemSetAccess``, which skips PyTorch's free-and-retry. Without them the cache is freed and the allocation
    retried, so jobs near the limit run, at the cost of more fragmentation (the Windows CUDA offset).

    PyTorch 2.14 reads ``PYTORCH_CUDA_ALLOC_CONF`` first and ``PYTORCH_ALLOC_CONF`` only when that is unset, so
    the same merged string is written to both. The defaults apply only when the user set neither variable. A
    ``PYTORCH_CUDA_ALLOC_CONF`` the user set is left exactly as written; the merged string goes to
    ``PYTORCH_ALLOC_CONF`` alone in that case.

    PyTorch reads the options once, when CUDA first starts, so this must run before anything touches ``torch.cuda``.
    ``zvisiongenerator.backends`` runs it when it is first imported, before any CUDA module can load; a message is
    logged when CUDA had already started by then, because the options no longer take effect.
    """
    _warn_if_cuda_started()
    user_cuda = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    user_alloc = os.environ.get("PYTORCH_ALLOC_CONF")
    base = user_cuda
    if user_cuda is None and user_alloc is None:
        base = _WINDOWS_ALLOCATOR_CONFIG if sys.platform == "win32" else _ALLOCATOR_CONFIG
    merged = merged_alloc_conf(base, user_alloc, PIN_ROUND_SETTING)
    if user_cuda is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = merged
    os.environ["PYTORCH_ALLOC_CONF"] = merged


def _warn_if_cuda_started() -> None:
    """Log that the allocator options came too late when this process has already started CUDA.

    Only a PyTorch that is already imported can have started CUDA, so nothing is imported here.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        _logger.warning("CUDA started before the allocator options were set, so expandable segments and the other options are not in effect; set PYTORCH_CUDA_ALLOC_CONF before starting Python")


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


def free_bytes() -> int | None:
    """Return the GPU memory the driver reports free, in bytes, or ``None`` when there is no CUDA device or the query fails.

    This is the driver's view (``torch.cuda.mem_get_info``), so it counts memory PyTorch has reserved but not used as
    taken. Memory the allocator holds for reuse is not free for another model.
    """
    import torch

    try:
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001 - no CUDA, a driver error or a broken install all mean "unknown"
        return None


def memory_sizes() -> tuple[int, int] | None:
    """Return ``(GPU budget, system memory)`` in bytes for the card CUDA uses as ``cuda:0`` (:func:`~zvisiongenerator.backends.cuda.nvml.card_bus_id`), or ``None`` without one.

    The GPU budget is the card's total less the memory the driver reserves, from ``nvidia-smi``, so the query does not start CUDA in
    this process.
    """
    gpu_bytes = _gpu_budget_bytes()
    if gpu_bytes is None:
        return None
    import psutil

    return gpu_bytes, int(psutil.virtual_memory().total)


def _smi_query(executable: str, fields: str) -> list[str] | None:
    """Return the *fields* of the card CUDA uses (the one at its PCI bus id) as ``nvidia-smi`` prints them (MiB, no units), or ``None`` when CUDA has no card or the query fails."""
    import subprocess

    bus_id = nvml.card_bus_id()
    if bus_id is None:
        return None
    try:
        result = subprocess.run([executable, "-i", bus_id, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
        return [value.strip() for value in result.stdout.strip().splitlines()[0].split(",")]
    except OSError, subprocess.SubprocessError, IndexError:
        return None


def _mib(value: str) -> int | None:
    """Return the MiB *value* of an ``nvidia-smi`` field, or ``None`` when it is not a number (``[N/A]``)."""
    try:
        return int(float(value))
    except ValueError:
        return None


def _gpu_budget_bytes() -> int | None:
    """Return the memory budget in bytes of the card CUDA uses, or ``None`` when ``nvidia-smi`` is unavailable.

    The budget is ``memory.total`` less ``memory.reserved``, the memory the driver and firmware keep for themselves. A driver
    that does not report the reserve leaves the total.
    """
    sizes = _smi_card_sizes()
    return None if sizes is None else sizes[0] - sizes[1]


def card_total_bytes() -> int | None:
    """Return the card's whole memory in bytes as ``nvidia-smi`` reports it (``memory.total``, the driver's reserve included), or ``None`` without ``nvidia-smi``.

    Features tuned on the card's full size (SeedVR2's tier, the Smart prompt enhancer, the first-run model pick) read this; the memory
    engine's budget is :func:`memory_sizes`, which leaves the reserve out.
    """
    sizes = _smi_card_sizes()
    return None if sizes is None else sizes[0]


def _smi_card_sizes() -> tuple[int, int] | None:
    """Return ``(memory.total, memory.reserved)`` of the card CUDA uses in bytes (the reserve is 0 when the driver does not report it), or ``None`` without ``nvidia-smi``."""
    import shutil

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    fields = _smi_query(executable, "memory.total,memory.reserved")
    total = _mib(fields[0]) if fields else None
    if total is None:
        return None
    reserved = _mib(fields[1]) if len(fields) > 1 else None
    return total * 1024 * 1024, (reserved or 0) * 1024 * 1024


def memory_usage() -> MemoryUsage | None:
    """Return the machine's system memory in use and the memory in use on the card CUDA uses, with their totals, or ``None`` when system memory cannot be read.

    Both figures are the load on the machine, not this app's share: system memory is the total minus what is available, and
    GPU memory is the card's total minus what the driver reports free, which includes the CUDA context, the driver's own
    share and other programs' allocations. The card's total is ``nvidia-smi``'s budget (its total less the memory the driver reserves),
    the one the memory tag judges against; without ``nvidia-smi`` it is the driver's own. PyTorch is never imported here, so reading
    memory cannot start a CUDA context. Without NVML the GPU figure is unknown (``gpu_used_bytes`` is ``None`` and ``gpu_unknown`` is set) until
    something else has imported torch and initialized CUDA, and the driver's total is unknown until then.
    """
    system = _system_memory()
    if system is None:
        return None
    gpu_used, gpu_total = _gpu_reading()
    return MemoryUsage(system_used_bytes=system[0], gpu_used_bytes=gpu_used, system_total_bytes=system[1], gpu_total_bytes=gpu_total, gpu_unknown=gpu_used is None)


def _initialized_torch():  # noqa: ANN202 - torch is never imported at module level
    """Return ``torch`` when it is already imported and CUDA is initialized, else ``None``."""
    torch = sys.modules.get("torch")
    return torch if torch is not None and torch.cuda.is_initialized() else None


_smi_budget: int | None = None


def _gpu_total_bytes() -> int | None:
    """Return the budget of the card CUDA uses as ``nvidia-smi`` reports it, asked once (it is fixed hardware), or ``None`` without ``nvidia-smi``."""
    global _smi_budget
    if _smi_budget is None:
        _smi_budget = _gpu_budget_bytes()
    return _smi_budget


def _gpu_reading() -> tuple[int | None, int | None]:
    """Return ``(memory in use on the card CUDA uses, the budget it is measured against)``: the card's total minus what the driver reports free.

    The free memory is NVML's (:mod:`~zvisiongenerator.backends.cuda.nvml`), which counts other programs on every platform and needs no CUDA
    context, so the reading is known before this process starts CUDA. It is taken from the budget, not from NVML's own total, because that total still
    holds the driver's reserve, which the budget leaves out: on Linux this is the figure ``torch.cuda.mem_get_info`` gives. Without NVML it is ``torch.cuda.mem_get_info``'s, which ignores other programs on
    Windows; before this process has initialized CUDA, or when that query fails, it is ``None`` (unknown), because reading it would need a CUDA
    context, which this module never creates. ``cudaMemGetInfo`` is a driver call that takes microseconds and is safe to make from the sampler
    thread while the job thread uses the same device.

    The budget is ``nvidia-smi``'s, else the driver's own (``None`` before CUDA is initialized and without NVML).
    """
    total = _gpu_total_bytes()
    card = nvml.card_memory()
    if card is not None:
        return max(0, (total or card.total) - card.free), total or card.total
    torch = _initialized_torch()
    if torch is None:
        return None, total
    try:
        free, driver_total = torch.cuda.mem_get_info(0)
    except Exception:  # noqa: BLE001 - a driver error means "unknown"
        return None, total
    return max(0, int(driver_total - free)), total or int(driver_total)


def _system_memory() -> tuple[int, int] | None:
    """Return ``(system memory in use on the whole machine (total minus available), the machine's total)``, or ``None`` when it cannot be read."""
    import psutil

    try:
        memory = psutil.virtual_memory()
        return int(memory.total - memory.available), int(memory.total)
    except OSError, AttributeError:
        return None
