"""Single-flight registry: one owner downloads a destination while others wait."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget

__all__ = ["DownloadFlights", "Flight"]

_RUNNING = "running"


@dataclass
class Flight:
    """One running download that owners update and waiters observe.

    Attributes:
        target: The target the owner downloads.
        progress: The latest progress, ``None`` until the owner's first update.
        status: ``"running"``, then ``"done"``, ``"stopped"`` or ``"failed"``.
        message: The failure or stop reason, set by :meth:`DownloadFlights.finish`.
    """

    target: DownloadTarget
    progress: DownloadProgress | None = None
    status: str = _RUNNING
    message: str | None = None
    _finished: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def update(self, progress: DownloadProgress) -> None:
        """Store the latest *progress* for waiters to read."""
        self.progress = progress

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the flight finishes or *timeout* seconds pass.

        Returns:
            True once the flight has finished, False on timeout.
        """
        return self._finished.wait(timeout)


class DownloadFlights:
    """Thread-safe registry of running downloads, keyed by ``DownloadTarget.key``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[tuple[str, str, str], Flight] = {}

    def claim(self, target: DownloadTarget) -> tuple[Flight, bool]:
        """Join the running flight for *target*'s key, or register a new one.

        Returns:
            ``(flight, owner)``; ``owner`` is True when the caller must run the download.
        """
        with self._lock:
            flight = self._flights.get(target.key)
            if flight is not None:
                return flight, False
            flight = Flight(target)
            self._flights[target.key] = flight
            return flight, True

    def finish(self, flight: Flight, status: str, message: str | None = None) -> None:
        """End *flight*: set its outcome, unregister it and wake its waiters, atomically."""
        with self._lock:
            flight.status = status
            flight.message = message
            if self._flights.get(flight.target.key) is flight:
                del self._flights[flight.target.key]
            flight._finished.set()

    def active(self) -> tuple[DownloadTarget, ...]:
        """Return the targets of all running flights."""
        with self._lock:
            return tuple(flight.target for flight in self._flights.values())

    def conflict(
        self,
        *,
        repo_ids: Iterable[str] = (),
        paths: Iterable[str | os.PathLike[str]] = (),
        any_cache: bool = False,
    ) -> str | None:
        """Return the repo id of the first running flight that touches the given scope.

        A flight conflicts when its repo id is in *repo_ids* (case-insensitive), when its
        ``local_dir`` equals or lies inside one of *paths*, or, with *any_cache*, when it
        downloads into the Hub cache.

        Returns:
            The conflicting repo id, or None when nothing running touches the scope.
        """
        wanted = {repo_id.lower() for repo_id in repo_ids}
        roots = [_normalize(path) for path in paths]
        with self._lock:
            targets = [flight.target for flight in self._flights.values()]
        for target in targets:
            if target.repo_id.lower() in wanted:
                return target.repo_id
            if target.local_dir is None:
                if any_cache:
                    return target.repo_id
            elif any(_is_within(_normalize(target.local_dir), root) for root in roots):
                return target.repo_id
        return None


def _normalize(path: str | os.PathLike[str]) -> str:
    """Return *path* as an absolute, normalized, case-folded-on-Windows string."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def _is_within(path: str, root: str) -> bool:
    """Return whether *path* equals *root* or lies inside it."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)
