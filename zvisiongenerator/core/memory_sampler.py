"""Live memory readings and a sampler that polls them while a job runs."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["MemorySampler", "MemoryUsage"]


@dataclass(frozen=True)
class MemoryUsage:
    """One reading of the machine's memory load.

    Attributes:
        system_used_bytes: System memory in use on the whole machine (its total minus what is available), every program's included.
            On Apple Silicon, which has one memory, this is the whole of it that is in use.
        gpu_used_bytes: Memory in use on the card the app uses (the one CUDA names ``cuda:0``), every program's included (its total minus what the driver reports
            free). ``None`` when the platform has no separate GPU memory or (see ``gpu_unknown``) it cannot be read yet.
        system_total_bytes: The machine's RAM, which ``system_used_bytes`` is a part of; ``None`` when not read.
        gpu_total_bytes: The card's budget that ``gpu_used_bytes`` is measured against (the one the memory tag judges
            against), or ``None`` when it is unknown.
        gpu_unknown: Whether the machine has a separate GPU whose use cannot be read yet: ``gpu_used_bytes`` is ``None`` then too.
    """

    system_used_bytes: int
    gpu_used_bytes: int | None = None
    system_total_bytes: int | None = None
    gpu_total_bytes: int | None = None
    gpu_unknown: bool = False


class MemorySampler:
    """Read and publish memory usage at an interval until stopped.

    ``start`` takes a reading and publishes it in the caller's thread, then a daemon thread repeats that every
    interval. ``stop`` never joins that thread: a reading taken after it is called is dropped, though one that is already
    being published may still land. A ``read`` or ``publish`` that raises ends the sampling quietly.
    """

    def __init__(
        self,
        read: Callable[[], MemoryUsage | None],
        publish: Callable[[MemoryUsage], None],
        *,
        interval: float = 1.0,
    ) -> None:
        """Store the callables; nothing is read until ``start``.

        Args:
            read: Return the current usage, or ``None`` when it cannot be read right now.
            publish: Receive each reading the sampler takes before ``stop``.
            interval: Seconds between readings on the sampling thread.
        """
        self._read = read
        self._publish = publish
        self._interval = interval
        self._halt = threading.Event()

    def start(self) -> None:
        """Take and publish one reading, then start the sampling thread."""
        try:
            if not self._sample():
                return
        except Exception:  # noqa: BLE001 - memory readout is best-effort and must never fail a job
            logger.debug("memory sampling stopped at start", exc_info=True)
            return
        threading.Thread(target=self._loop, name="memory-sampler", daemon=True).start()

    def stop(self) -> None:
        """Stop sampling: a reading taken after this is called is dropped, though one already being published may still land. Calling it again does nothing."""
        self._halt.set()

    def _loop(self) -> None:
        """Sample every interval until stopped or a read or publish raises."""
        try:
            while not self._halt.wait(self._interval):
                if not self._sample():
                    return
        except Exception:  # noqa: BLE001 - memory readout is best-effort and must never fail a job
            logger.debug("memory sampling stopped", exc_info=True)

    def _sample(self) -> bool:
        """Read once and publish; return ``False`` once sampling should end."""
        reading = self._read()
        if reading is None:
            return True
        if self._halt.is_set():
            return False
        self._publish(reading)
        return True
