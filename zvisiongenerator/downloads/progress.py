"""Byte counting for a running Hugging Face download and the speed and ETA derived from it."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from zvisiongenerator.downloads.target import DownloadProgress

_MIN_SPEED_SPAN_SECS = 2.0
_DESC_POSITION = 1


class ByteCounts:
    """Thread-safe counters for the two aggregate byte bars of a snapshot download.

    ``wire`` counts the bytes fetched from the network (the "Downloading bytes" bar) and ``written`` the bytes
    reconstructed on disk (the "Reconstructing" bar). The two overlap and must never be summed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wire = 0
        self._written = 0

    def add_wire(self, n: int) -> None:
        """Add *n* bytes fetched from the network."""
        with self._lock:
            self._wire += n

    def add_written(self, n: int) -> None:
        """Add *n* bytes written to disk."""
        with self._lock:
            self._written += n

    def snapshot(self) -> tuple[int, int]:
        """Return the ``(wire, written)`` counts read together."""
        with self._lock:
            return self._wire, self._written


def make_byte_sink(counts: ByteCounts) -> type:
    """Build a ``tqdm`` subclass, for ``snapshot_download(tqdm_class=...)``, that feeds *counts*.

    tqdm is imported here so that importing this module stays light. The sink draws nothing.
    Only byte bars are counted: "downloading" bars add to ``wire``, "reconstruct" bars to ``written``;
    file-count bars and everything else are ignored.

    Args:
        counts: The counters every bar created from the returned class updates.
    """
    from tqdm import tqdm

    class ByteSink(tqdm):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Stored before the base class runs: with TQDM_DISABLE set it returns before assigning desc and unit.
            desc = kwargs.get("desc", args[_DESC_POSITION] if len(args) > _DESC_POSITION else None)
            self._ziv_desc = str(desc or "").lower()
            self._ziv_unit = kwargs.get("unit", "it")
            super().__init__(*args, **{**kwargs, "disable": True})

        def update(self, n: float | None = 1) -> bool | None:
            amount = n or 0
            try:
                if amount > 0 and getattr(self, "_ziv_unit", None) == "B":
                    desc = getattr(self, "_ziv_desc", "")
                    if "downloading" in desc:
                        counts.add_wire(int(amount))
                    elif "reconstruct" in desc:
                        counts.add_written(int(amount))
            except Exception:  # noqa: BLE001 - a sink must never break the download
                pass
            return super().update(amount)

    return ByteSink


class SpeedMeter:
    """Turn a growing byte count into a bytes-per-second average over a sliding window."""

    def __init__(self, window_secs: float = 10.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._window_secs = window_secs
        self._clock = clock
        self._samples: deque[tuple[float, int]] = deque()

    def add(self, bytes_seen: int) -> float | None:
        """Record the total bytes seen so far and return the speed, or ``None`` before two samples 2 s apart."""
        now = self._clock()
        self._samples.append((now, bytes_seen))
        cutoff = now - self._window_secs
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        first_time, first_bytes = self._samples[0]
        span = now - first_time
        if span < _MIN_SPEED_SPAN_SECS:
            return None
        return max(0.0, (bytes_seen - first_bytes) / span)


def progress_from_counts(wire: int, written: int, total: int | None, speed: float | None) -> DownloadProgress:
    """Combine byte counts into a :class:`DownloadProgress`.

    The done count is the larger of *wire* and *written*, capped at *total* when it is known.

    Args:
        wire: Bytes fetched from the network.
        written: Bytes reconstructed on disk.
        total: Bytes the download will move, or ``None`` when unknown.
        speed: Bytes per second, or ``None`` when not measured yet.
    """
    done = max(wire, written)
    if total is not None:
        done = min(total, done)
    eta = (total - done) / speed if total is not None and speed else None
    return DownloadProgress(bytes_done=done, bytes_total=total, bytes_per_sec=speed, eta_secs=eta)
