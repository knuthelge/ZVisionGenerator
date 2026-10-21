"""Run Models-page downloads one at a time on a thread of their own, beside the generation job queue."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, ensure_downloaded

if TYPE_CHECKING:
    from zvisiongenerator.downloads.ensure import DownloadReport
    from zvisiongenerator.downloads.flights import DownloadFlights
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = ["DownloadLane"]

_QUEUED = "queued"
_RUNNING = "running"
_DONE = "done"
_FAILED = "failed"
_CANCELLED = "cancelled"
_FINISHED = frozenset({_DONE, _FAILED, _CANCELLED})


@dataclass(eq=False)
class _Item:
    """One lane download: what it fetches, where it stands and its latest progress report."""

    id: str
    kind: str
    name: str
    label: str
    plan: Callable[[], tuple[DownloadTarget, ...] | None]
    state: str = _QUEUED
    message: str | None = None
    report: DownloadReport | None = None
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


def _finished_at(item: _Item) -> float:
    """Return when *item* finished (``0`` for one that has not)."""
    return 0.0 if item.finished_at is None else item.finished_at


class DownloadLane:
    """Queue Models-page downloads and run them one at a time on a single worker thread.

    The lane never touches the job queue's executor or its exclusive slot, so generation jobs are admitted
    while it downloads. It shares *flights* with the job runner: a download of a repo that a job (or an earlier
    lane item) is already fetching is mirrored instead of repeated.

    Args:
        flights: The registry that keeps one download per destination, shared with the job runner.
        ensure: Runs one item's targets, like :func:`~zvisiongenerator.downloads.ensure.ensure_downloaded`.
        retention_secs: How long a finished item stays in the snapshot, so a poll sees its last state.
        max_finished: The most finished items the snapshot keeps (the newest).
        clock: Monotonic clock for the retention.
    """

    def __init__(
        self,
        *,
        flights: DownloadFlights,
        ensure: Callable[..., str] = ensure_downloaded,
        retention_secs: float = 600.0,
        max_finished: int = 20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._flights = flights
        self._ensure = ensure
        self._retention_secs = retention_secs
        self._max_finished = max_finished
        self._clock = clock
        self._cond = threading.Condition()
        self._items: dict[str, _Item] = {}
        self._queue: deque[_Item] = deque()
        self._running: _Item | None = None
        self._ids = count(1)
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, *, kind: str, name: str, label: str, plan: Callable[[], tuple[DownloadTarget, ...] | None]) -> dict[str, Any]:
        """Queue the download of the row *name* of *kind*, or return its queued or running item.

        Args:
            kind: ``"model"`` or ``"enhancer"``.
            name: The row's name or tier.
            label: How the row reads in messages.
            plan: Returns the targets to fetch (called on the lane thread when the item starts), or ``None`` when
                the plan cannot be made now.

        Returns:
            The item as JSON.

        Raises:
            RuntimeError: If the lane has been shut down.
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("The download lane has shut down.")
            existing = next((item for item in self._live_items() if (item.kind, item.name) == (kind, name)), None)
            if existing is not None:
                return self._item_json(existing)
            item = _Item(id=f"dl-{next(self._ids)}", kind=kind, name=name, label=label, plan=plan)
            self._items[item.id] = item
            self._queue.append(item)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="download-lane", daemon=True)
                self._thread.start()
            self._cond.notify_all()
            return self._item_json(item)

    def cancel(self, item_id: str) -> dict[str, Any]:
        """Cancel a queued item at once, or ask the running one to stop; a finished item is returned unchanged.

        Raises:
            KeyError: If no item has the id.
        """
        with self._cond:
            item = self._items[item_id]
            if item.state == _QUEUED:
                self._queue.remove(item)
                self._finish(item, _CANCELLED)
            elif item.state == _RUNNING:
                item.cancel.set()
            return self._item_json(item)

    def snapshot(self) -> dict[str, Any]:
        """Return the running item, the queued ones in order and the recently finished ones (newest first)."""
        with self._cond:
            self._prune()
            finished = sorted((item for item in self._items.values() if item.state in _FINISHED), key=_finished_at, reverse=True)
            live = [*([self._running] if self._running is not None else []), *self._queue]
            return {"items": [self._item_json(item) for item in (*live, *finished)]}

    def shutdown(self, *, join_timeout: float = 0.0) -> None:
        """Refuse new submits, cancel the queued items and signal the running one to stop.

        Args:
            join_timeout: Seconds to wait for the worker to finish stopping its download; ``0`` returns at once.
        """
        with self._cond:
            self._closed = True
            while self._queue:
                self._finish(self._queue.popleft(), _CANCELLED)
            if self._running is not None:
                self._running.cancel.set()
            thread = self._thread
            self._cond.notify_all()
        if thread is not None and join_timeout > 0:
            thread.join(join_timeout)

    def _work(self) -> None:
        """Run queued items one after another until the lane shuts down."""
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                item = self._queue.popleft()
                item.state = _RUNNING
                self._running = item
            state, message = self._execute(item)
            with self._cond:
                self._running = None
                self._finish(item, state, message)

    def _execute(self, item: _Item) -> tuple[str, str | None]:
        """Run *item* and return its final ``(state, message)``; never raises."""
        unreachable = f"Could not reach Hugging Face to download {item.label}."
        try:
            targets = item.plan()
            if targets is None:
                return _FAILED, unreachable
            outcome = self._ensure(targets, flights=self._flights, report=lambda report: self._record(item, report), cancelled=item.cancel.is_set)
        except Exception as exc:  # noqa: BLE001 - one failing item must not stop the lane
            return _FAILED, str(exc) or f"Could not download {item.label}."
        if outcome == READY:
            return _DONE, None
        if outcome in (STOPPED, DETACHED):
            return _CANCELLED, None
        if outcome == UNCHECKED:
            return _FAILED, unreachable
        return _FAILED, f"Could not download {item.label}."

    def _record(self, item: _Item, report: DownloadReport) -> None:
        """Store the latest progress report of *item*."""
        with self._cond:
            item.report = report

    def _finish(self, item: _Item, state: str, message: str | None = None) -> None:
        """Move *item* to a final state (the caller holds the lock) and drop finished items past the retention."""
        item.state = state
        item.message = message
        item.finished_at = self._clock()
        self._prune()

    def _prune(self) -> None:
        """Forget finished items older than the retention, then all but the newest ``max_finished`` (the caller holds the lock)."""
        now = self._clock()
        finished = sorted((item for item in self._items.values() if item.state in _FINISHED), key=_finished_at, reverse=True)
        for position, item in enumerate(finished):
            if position >= self._max_finished or now - _finished_at(item) > self._retention_secs:
                del self._items[item.id]

    def _live_items(self) -> list[_Item]:
        """Return the queued and running items (the caller holds the lock)."""
        return [item for item in self._items.values() if item.state in (_QUEUED, _RUNNING)]

    def _item_json(self, item: _Item) -> dict[str, Any]:
        """Return *item* as the API's ``DownloadItem`` (the caller holds the lock)."""
        data: dict[str, Any] = {"id": item.id, "kind": item.kind, "name": item.name, "label": item.label, "state": item.state, "message": item.message}
        if item.state == _QUEUED:
            data["queue_position"] = list(self._queue).index(item) + 1
        report = item.report
        if report is not None:
            progress = report.progress
            data.update(
                repo=report.repo,
                part=report.part,
                parts=report.parts,
                bytes_done=progress.bytes_done,
                bytes_total=progress.bytes_total,
                bytes_per_sec=progress.bytes_per_sec,
                download_eta_secs=progress.eta_secs,
                shared=report.shared,
            )
        return data
