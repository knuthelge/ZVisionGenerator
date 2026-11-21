"""Hold GPU memory in the job's own process, the way another program on the card would (:func:`hold_in_process`, ``run --hold-in-process-gib``).

Before the job, the process takes the memory through the CUDA driver API (``ctypes`` on ``nvcuda.dll`` or ``libcuda.so.1``), in the same primary context PyTorch uses and outside
PyTorch's allocator, so the allocator never sees it and cannot return it. The memory belongs to the one process and WDDM has no other process to evict, so it stays resident.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from collections.abc import Iterator

__all__ = ["DriverHold", "hold_in_process", "open_driver"]

GIB = 1024**3
_CUDA_SUCCESS = 0
_held: list[DriverHold] = []
"""The holds this process keeps until it exits."""


def open_driver() -> ctypes.CDLL:
    """Return the CUDA driver library (``nvcuda.dll`` on Windows, ``libcuda.so.1`` on Linux) with the argument types of the calls the holds use.

    Raises:
        OSError: The platform has no CUDA driver library.
    """
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    void_p, size_t = ctypes.c_void_p, ctypes.c_size_t
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(void_p), ctypes.c_int]
    driver.cuCtxPushCurrent_v2.argtypes = [void_p]
    driver.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(void_p)]
    driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t]
    driver.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, size_t]
    return driver


class DriverHold:
    """GPU memory taken through the CUDA driver API in the primary context of device *device* (the ordinal PyTorch uses), outside PyTorch's allocator."""

    def __init__(self, device: int) -> None:
        self._driver = open_driver()
        self._check(self._driver.cuInit(0), "cuInit")
        handle = ctypes.c_int()
        self._check(self._driver.cuDeviceGet(ctypes.byref(handle), device), "cuDeviceGet")
        self._context = ctypes.c_void_p()
        self._check(self._driver.cuDevicePrimaryCtxRetain(ctypes.byref(self._context), handle.value), "cuDevicePrimaryCtxRetain")
        self._pointer = ctypes.c_uint64(0)
        self.nbytes = 0

    @staticmethod
    def _check(code: int, call: str) -> None:
        if code != _CUDA_SUCCESS:
            raise RuntimeError(f"{call} failed with CUDA driver error {code}")

    @contextlib.contextmanager
    def _current(self) -> Iterator[None]:
        """Make the retained primary context current on this thread while the block runs."""
        self._check(self._driver.cuCtxPushCurrent_v2(self._context), "cuCtxPushCurrent")
        try:
            yield
        finally:
            popped = ctypes.c_void_p()
            self._driver.cuCtxPopCurrent_v2(ctypes.byref(popped))

    def allocate(self, nbytes: int) -> None:
        """Take *nbytes* and write every byte once, so the driver has made all of it resident (``cuMemAlloc``, ``cuMemsetD8``, ``cuCtxSynchronize``)."""
        with self._current():
            self._check(self._driver.cuMemAlloc_v2(ctypes.byref(self._pointer), nbytes), "cuMemAlloc")
            self.nbytes = nbytes
            self._check(self._driver.cuMemsetD8_v2(self._pointer.value, 1, self.nbytes), "cuMemsetD8")
            self._check(self._driver.cuCtxSynchronize(), "cuCtxSynchronize")


def take_hold(gib: float) -> tuple[DriverHold, int]:
    """Take *gib* GiB in this process's own CUDA context and return the hold and the bytes it took from the card as ``torch.cuda.mem_get_info`` counts them.

    The charge is read after the context exists and before the allocation, so it is the hold alone (the job's own context is the job's). It is the figure a bench row records and ``fit``
    subtracts, not the nominal size: the driver rounds an allocation up.

    Raises:
        RuntimeError: The driver refused a call, the card has no room for the hold included.
    """
    import torch

    device = torch.cuda.current_device()
    hold = DriverHold(device)
    free_before = int(torch.cuda.mem_get_info(device)[0])
    hold.allocate(round(gib * GIB))
    return hold, free_before - int(torch.cuda.mem_get_info(device)[0])


def hold_in_process(gib: float) -> dict[str, int]:
    """Take *gib* GiB in this process's own CUDA context for the rest of the process's life and return the row field ``hold_charged_bytes``."""
    hold, charged = take_hold(gib)
    _held.append(hold)
    return {"hold_charged_bytes": charged}
