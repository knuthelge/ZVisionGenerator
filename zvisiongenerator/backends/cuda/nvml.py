"""The card's memory as the NVIDIA driver (NVML) counts it, which includes other programs' allocations on every platform.

``torch.cuda.mem_get_info`` reports the memory free *for CUDA*. On Windows (WDDM) that figure ignores what other programs hold, while NVML's ``used`` tracks it. This module
reads NVML directly through ``ctypes`` (no dependency, no CUDA context): ``nvml.dll`` on Windows and ``libnvidia-ml.so.1`` on Linux.

The card is the one ``cuda:0`` names. CUDA numbers its cards by its own order (``CUDA_DEVICE_ORDER``, the fastest first unless the user set it) and ``CUDA_VISIBLE_DEVICES`` indices follow that order, while
NVML and ``nvidia-smi`` number them by PCI bus, so an index means different cards to the two. The two sides are joined by the card's PCI bus id: :func:`card_bus_id` asks the CUDA driver for the bus id
of its device 0 (``cuInit``, ``cuDeviceGet``, ``cuDeviceGetPCIBusId``, which start no context and no PyTorch, so the budget is known before a job starts CUDA), and NVML and ``nvidia-smi`` are asked for
the card with that bus id. ``CUDA_VISIBLE_DEVICES`` set to the empty string hides every card: there is none.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass

__all__ = ["CardMemory", "card_bus_id", "card_memory"]


@dataclass(frozen=True)
class CardMemory:
    """The card's memory in bytes as the driver counts it: *free* is what no program, this one or another, holds."""

    total: int
    free: int


class _NvmlMemory(ctypes.Structure):
    """``nvmlMemory_t``."""

    _fields_ = (("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong))


_WINDOWS_LIBRARIES = ("nvml.dll",)
_LINUX_LIBRARIES = ("libnvidia-ml.so.1",)
_DRIVER_LIBRARY = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
_BUS_ID_LENGTH = 64

_bus_id_lock = threading.Lock()
_bus_id_known = False
_bus_id: str | None = None


def card_bus_id() -> str | None:
    """Return the PCI bus id (``0000:01:00.0``) of the card CUDA names ``cuda:0``, or ``None`` when CUDA has no card here; asked once, then kept.

    The id comes from the CUDA driver, which applies ``CUDA_DEVICE_ORDER`` and ``CUDA_VISIBLE_DEVICES`` as PyTorch will, without creating a context or importing PyTorch. ``None`` means
    ``CUDA_VISIBLE_DEVICES`` is the empty string (no card is visible), there is no CUDA driver, or the driver sees no card.
    """
    global _bus_id_known, _bus_id
    with _bus_id_lock:
        if not _bus_id_known:
            _bus_id, _bus_id_known = _ask_driver_for_bus_id(), True
        return _bus_id


def _ask_driver_for_bus_id() -> str | None:
    """Ask the CUDA driver for the PCI bus id of its device 0, or return ``None`` when it has none."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return None  # CUDA reads an empty list as no card, and the driver is not asked
    try:
        driver = ctypes.CDLL(_DRIVER_LIBRARY)
        device, count, name = ctypes.c_int(), ctypes.c_int(), ctypes.create_string_buffer(_BUS_ID_LENGTH)
        if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
            return None
        if driver.cuDeviceGet(ctypes.byref(device), 0) != 0 or driver.cuDeviceGetPCIBusId(name, _BUS_ID_LENGTH, device) != 0:
            return None
    except AttributeError, OSError:
        return None
    return name.value.decode() or None


_lock = threading.Lock()
_started = False
_library: ctypes.CDLL | None = None
_handle = ctypes.c_void_p()


def _open_library() -> ctypes.CDLL | None:
    """Return the loaded NVML library, or ``None`` when this platform or machine has none."""
    names = _WINDOWS_LIBRARIES if sys.platform == "win32" else _LINUX_LIBRARIES if sys.platform.startswith("linux") else ()
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _card_handle(library: ctypes.CDLL, bus_id: str) -> bool:
    """Take the handle of the card at PCI bus *bus_id* into ``_handle``; return whether NVML knew it."""
    return library.nvmlDeviceGetHandleByPciBusId_v2(bus_id.encode(), ctypes.byref(_handle)) == 0


def _start() -> None:
    """Load NVML once, initialize it and take the handle of the card CUDA uses; leave ``_library`` unset on any failure (a failure is not retried)."""
    global _started, _library
    with _lock:
        if _started:
            return
        _started = True
        bus_id = card_bus_id()
        library = None if bus_id is None else _open_library()
        if bus_id is None or library is None:
            return
        try:
            if library.nvmlInit_v2() != 0 or not _card_handle(library, bus_id):
                return
        except AttributeError, OSError:
            return
        _library = library


def card_memory() -> CardMemory | None:
    """Return the memory of the card CUDA uses as the driver counts it, or ``None`` when NVML is unavailable or the query fails."""
    if not _started:
        _start()
    if _library is None:
        return None
    info = _NvmlMemory()
    try:
        if _library.nvmlDeviceGetMemoryInfo(_handle, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return CardMemory(total=int(info.total), free=int(info.free))
