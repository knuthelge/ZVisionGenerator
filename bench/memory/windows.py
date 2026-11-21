"""Windows process memory for a bench case: the private working set.

Linux rows record the process's ``RssAnon`` and ``RssShmem``. Windows has no such counters, so a case records the nearest equivalent: the **private working set**
(``private_working_set_peak_bytes``), the process's resident pages that no other process shares, which is what ``psutil`` calls ``uss``. It is read once a polling round (the
interval is the wait between rounds), like the Linux counters, because a coarser poll missed short RAM peaks. Pinned host memory is in it. ``fit`` judges a Windows RAM
estimate against its peak.

The reader returns ``None`` where Windows or ``psutil`` is unavailable, and the row then leaves the figure out.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

__all__ = ["WindowsSampler", "private_working_set"]


def private_working_set() -> int | None:
    """Return this process's private working set in bytes (resident pages no other process shares), or ``None`` off Windows or when it cannot be read."""
    if sys.platform != "win32":
        return None
    try:
        import psutil

        return int(psutil.Process().memory_full_info().uss)
    except Exception:  # noqa: BLE001 -- psutil missing or the query refused: no sample
        return None


class WindowsSampler:
    """Poll this process's private working set on a daemon thread and keep its peak.

    The interval is the wait between rounds, so a round that takes longer than it only stretches the period (rounds never overlap or queue up). The sampler reads nothing off
    Windows, so the figure is ``None`` there.
    """

    def __init__(self, interval: float = 0.1, *, working_set: Callable[[], int | None] = private_working_set) -> None:
        self._interval = interval
        self._read = working_set
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._peak: int | None = None

    def start(self) -> None:
        """Take the first reading and start polling."""
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        """Stop polling, take a last reading, and return ``private_working_set_peak_bytes`` (``None`` when it was never read)."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._sample()
        return {"private_working_set_peak_bytes": self._peak}

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        value = self._read()
        if value is not None:
            self._peak = value if self._peak is None else max(self._peak, value)
