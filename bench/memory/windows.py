"""Windows process memory for a bench case: the private working set, the full working set and the commit charge.

Linux rows record the process's ``RssAnon`` and ``RssShmem``. Windows has no such counters, so a case records the nearest equivalents, all read in one ``psutil`` call
once per poll (the interval is the wait between polls), like the Linux counters, because a coarser poll missed short RAM peaks:

- **Private working set** (``private_working_set_peak_bytes``): the process's resident pages that no other process shares, which is what ``psutil`` calls ``uss``.
  Pinned host memory is in it. ``fit`` judges a Windows RAM estimate against its peak, except where the estimate counts weights mapped from disk.
- **Full working set** (``working_set_sampled_peak_bytes``, the highest reading of the poll; ``working_set_peak_bytes``, the OS's own ``PeakWorkingSetSize`` read at the end):
  every resident page, shared and file-mapped ones included. Mapped weights are in it and not in the private working set, so ``fit`` judges a RAM estimate that
  counts mapped weights against ``working_set_peak_bytes``.
- **Private commit** (``private_commit_peak_bytes``): the process's peak commit charge (``PeakPagefileUsage``), the memory the system promised it, touched or not.
  ``fit`` shows it and does not judge by it: a job whose commit charge outruns the commit limit (RAM plus the page file) fails although its RAM fits.
- **Physical RAM** (``physical_ram_bytes``): the machine's installed RAM (``GetPerformanceInfo``'s ``PhysicalTotal``), read once. A job that needs more than the machine has gets its
  working set trimmed, so its full working set then reads about the free RAM and not what the job needs; ``fit`` leaves such a row's RAM check out.
- **System commit rise** (``system_commit_rise_bytes``): how far the whole system's committed memory (``GetPerformanceInfo``'s ``CommitTotal``) rose at its highest above its
  first reading, which includes whatever else ran meanwhile.

Each reader returns ``None`` where Windows or the counter is unavailable, and the row then leaves the figure out.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable

__all__ = ["WindowsSampler", "physical_ram", "process_memory", "system_commit"]


def process_memory() -> dict[str, int] | None:
    """Return this process's ``private`` (``uss``), ``working_set``, ``peak_working_set`` and ``peak_commit`` in bytes, or ``None`` off Windows or when they cannot be read."""
    if sys.platform != "win32":
        return None
    try:
        import psutil

        info = psutil.Process().memory_full_info()
        return {"private": int(info.uss), "working_set": int(info.wset), "peak_working_set": int(info.peak_wset), "peak_commit": int(info.peak_pagefile)}
    except Exception:  # noqa: BLE001 -- psutil missing or the query refused: no sample
        return None


class _PerformanceInformation(ctypes.Structure):
    """``PERFORMANCE_INFORMATION`` of ``GetPerformanceInfo``; the counts are in pages."""

    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    )


def _performance_info() -> _PerformanceInformation | None:
    """Return the system's ``GetPerformanceInfo`` reading, or ``None`` off Windows or when the call fails."""
    if sys.platform != "win32":
        return None
    try:
        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        return info
    except Exception:  # noqa: BLE001 -- psapi missing or the call refused: no sample
        return None


def system_commit() -> int | None:
    """Return the memory committed system-wide in bytes (``CommitTotal``), or ``None`` off Windows or when the call fails."""
    return None if (info := _performance_info()) is None else int(info.CommitTotal * info.PageSize)


def physical_ram() -> int | None:
    """Return the machine's physical RAM in bytes (``PhysicalTotal``), or ``None`` off Windows or when the call fails."""
    return None if (info := _performance_info()) is None else int(info.PhysicalTotal * info.PageSize)


class WindowsSampler:
    """Poll this process's memory on a daemon thread and keep the peaks.

    The interval is the wait between polls, so a poll that takes longer than it only stretches the period (polls never overlap or pile up). The sampler reads nothing off
    Windows, so every figure is ``None`` there.
    """

    def __init__(
        self,
        interval: float = 0.1,
        *,
        memory: Callable[[], dict[str, int] | None] = process_memory,
        commit: Callable[[], int | None] = system_commit,
        physical: Callable[[], int | None] = physical_ram,
    ) -> None:
        self._interval = interval
        self._memory = memory
        self._commit = commit
        self._physical = physical
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._private: int | None = None
        self._sampled: int | None = None
        self._os_peak: int | None = None
        self._peak_commit: int | None = None
        self._commit_first: int | None = None
        self._commit_peak: int | None = None

    def start(self) -> None:
        """Take the first reading and start polling."""
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        """Stop polling, take a last reading, and return the figures (each ``None`` when it was never read)."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._sample()
        rise = None if self._commit_first is None or self._commit_peak is None else self._commit_peak - self._commit_first
        return {
            "private_working_set_peak_bytes": self._private,
            "working_set_sampled_peak_bytes": self._sampled,
            "working_set_peak_bytes": self._os_peak,
            "private_commit_peak_bytes": self._peak_commit,
            "system_commit_rise_bytes": rise,
            "physical_ram_bytes": self._physical(),
        }

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        if (reading := self._memory()) is not None:
            self._private = _higher(self._private, reading["private"])
            self._sampled = _higher(self._sampled, reading["working_set"])
            self._os_peak = _higher(self._os_peak, reading["peak_working_set"])
            self._peak_commit = _higher(self._peak_commit, reading["peak_commit"])
        if (committed := self._commit()) is not None:
            self._commit_first = committed if self._commit_first is None else self._commit_first
            self._commit_peak = _higher(self._commit_peak, committed)


def _higher(peak: int | None, value: int) -> int:
    return value if peak is None else max(peak, value)
