"""Make sure a plan's downloads are on disk before a model loads: skip, wait for, or run each part."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.child import run_in_child
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.progress import SpeedMeter, progress_from_counts
from zvisiongenerator.downloads.target import DownloadProgress
from zvisiongenerator.utils.model_files import disk_size_label

if TYPE_CHECKING:
    from zvisiongenerator.downloads.child import ChildResult
    from zvisiongenerator.downloads.flights import DownloadFlights, Flight
    from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck

__all__ = ["DETACHED", "READY", "STOPPED", "UNCHECKED", "DownloadReport", "ensure_downloaded"]

READY = "ready"
UNCHECKED = "unchecked"
STOPPED = "stopped"
DETACHED = "detached"


@dataclass(frozen=True)
class DownloadReport:
    """One progress report for the caller.

    Attributes:
        repo: The repo being downloaded.
        part: The 1-based position of this target in the plan.
        parts: The number of targets in the plan.
        progress: Bytes done and total, speed and ETA of this part.
        shared: True when mirroring a download that another caller started.
    """

    repo: str
    part: int
    parts: int
    progress: DownloadProgress
    shared: bool = False


class _Reporter:
    """Pass reports to the caller at most once per interval, except a part's first and forced ones."""

    def __init__(self, report: Callable[[DownloadReport], None], clock: Callable[[], float], interval_secs: float) -> None:
        self._report = report
        self._clock = clock
        self._interval_secs = interval_secs
        self._last_passed: float | None = None
        self._repo = ""
        self._part = 0
        self._parts = 0
        self._part_reported = False
        self._mirrored: DownloadProgress | None = None

    def begin_part(self, repo: str, part: int, parts: int) -> None:
        """Start a new part, whose first report always passes."""
        self._repo, self._part, self._parts = repo, part, parts
        self._part_reported = False
        self._mirrored = None

    @property
    def part_reported(self) -> bool:
        """Return whether a report of the current part has passed."""
        return self._part_reported

    def emit(self, progress: DownloadProgress, *, shared: bool = False, force: bool = False) -> None:
        """Report *progress* when it is the part's first, *force*d, or an interval after the last passed report."""
        now = self._clock()
        due = self._last_passed is None or now - self._last_passed >= self._interval_secs
        if not (force or due or not self._part_reported):
            return
        self._last_passed = now
        self._part_reported = True
        self._report(DownloadReport(self._repo, self._part, self._parts, progress, shared))

    def mirror(self, progress: DownloadProgress) -> None:
        """Report a progress snapshot of someone else's download once per new snapshot, throttled."""
        if progress is self._mirrored:
            return
        self._mirrored = progress
        self.emit(progress, shared=True)


def _size_text(num_bytes: int | None) -> str:
    """Return *num_bytes* as a size label, or "an unknown size"."""
    return "an unknown size" if num_bytes is None else disk_size_label(num_bytes)


def _failure_message(target: DownloadTarget, last: DownloadProgress | None, total: int | None, message: str | None) -> str:
    """Build the failure text: the repo, what was fetched and the reason."""
    done = 0 if last is None else last.bytes_done
    return f"Download of {target.repo_id} failed after {_size_text(done)} of {_size_text(total)}: {message or 'unknown error'}"


def _final_progress(last: DownloadProgress | None, total: int | None) -> DownloadProgress:
    """Return the last progress snapped to 100 % of *total* (or kept as is when the total is unknown)."""
    if last is None:
        return DownloadProgress(bytes_done=total or 0, bytes_total=total, bytes_per_sec=None, eta_secs=None)
    if total is None:
        return last
    return dataclasses.replace(last, bytes_done=total, bytes_total=total, eta_secs=0.0)


def _run_owned(
    target: DownloadTarget,
    dry_run: TargetCheck,
    *,
    flight: Flight,
    reporter: _Reporter,
    cancelled: Callable[[], bool],
    run: Callable[..., ChildResult],
    clock: Callable[[], float],
) -> tuple[str, str | None]:
    """Download *target* in the child, feeding the flight and the reporter; return the flight outcome ``(status, message)``."""
    total = dry_run.missing_bytes
    meter = SpeedMeter(clock=clock)
    latest: list[DownloadProgress | None] = [None]
    if not reporter.part_reported:
        reporter.emit(DownloadProgress(0, total, None, None), force=True)

    def on_counts(wire: int, written: int) -> None:
        # A raising callback would make the child runner abandon the child without cleanup.
        try:
            speed = meter.add(wire if wire else written)
            progress = progress_from_counts(wire, written, total, speed)
            latest[0] = progress
            flight.update(progress)
            reporter.emit(progress)
        except Exception:  # noqa: BLE001 - progress reporting must never break the download
            pass

    result = run(target, on_counts=on_counts, cancelled=cancelled)
    if result.status == "stopped":
        return "stopped", None
    if result.status == "failed":
        return "failed", _failure_message(target, latest[0], total, result.message)
    final = _final_progress(latest[0], total)
    flight.update(final)
    reporter.emit(final, force=True)
    return "done", None


def _own(
    target: DownloadTarget,
    flight: Flight,
    flights: DownloadFlights,
    *,
    reporter: _Reporter,
    cancelled: Callable[[], bool],
    check: Callable[[DownloadTarget], TargetCheck | None],
    run: Callable[..., ChildResult],
    clock: Callable[[], float],
) -> str | None:
    """Run the download of a part this caller owns; return ``None`` to continue or an early result.

    The flight is finished exactly once, whatever happens; waiters of a flight that could not be checked or
    was stopped re-check their own target.
    """
    status, message = "stopped", None
    try:
        dry_run = check(target)
        if dry_run is None:
            return UNCHECKED
        if dry_run.missing_files == 0:
            status = "done"
            return None
        if cancelled():
            return STOPPED
        status, message = _run_owned(target, dry_run, flight=flight, reporter=reporter, cancelled=cancelled, run=run, clock=clock)
        if status == "failed":
            raise RuntimeError(message)
        return STOPPED if status == "stopped" else None
    except Exception as exc:
        status, message = "failed", str(exc)
        raise
    finally:
        flights.finish(flight, status, message)


def _wait_for(flight: Flight, *, reporter: _Reporter, cancelled: Callable[[], bool], wait_secs: float) -> bool:
    """Mirror another caller's download until it ends; return False when *cancelled* turned true first."""
    while True:
        if cancelled():
            return False
        progress = flight.progress
        if progress is not None:
            reporter.mirror(progress)
        if flight.wait(wait_secs):
            return True


def ensure_downloaded(
    targets: Sequence[DownloadTarget],
    *,
    flights: DownloadFlights,
    report: Callable[[DownloadReport], None],
    cancelled: Callable[[], bool],
    present: Callable[[DownloadTarget], bool] = target_present_offline,
    check: Callable[[DownloadTarget], TargetCheck | None] = check_target,
    run: Callable[..., ChildResult] = run_in_child,
    clock: Callable[[], float] = time.monotonic,
    interval_secs: float = 1.0,
    wait_secs: float = 0.5,
) -> str:
    """Download whatever *targets* still miss, in order, sharing a running download of the same destination.

    Each part is skipped when it is present offline. Otherwise the caller either owns the download (dry-run check,
    then a child process) or waits for the owner and mirrors its progress. Reports reach *report* at most once
    per *interval_secs*, apart from the first report of a part and an owned part's final one.

    Args:
        targets: The parts of the plan, in download order.
        flights: The registry that keeps one download per destination.
        report: Receives :class:`DownloadReport` updates.
        cancelled: Polled while downloading or waiting; true stops an owned download or detaches a waiter.
        present: Tells whether a target is fully on disk without using the network.
        check: Dry-runs a target; ``None`` means the Hub could not be reached.
        run: Runs a download in a child process.
        clock: Monotonic clock for the report throttle and the speed meter.
        interval_secs: Minimum seconds between passed reports.
        wait_secs: Seconds a waiter sleeps between looks at the owner's flight.

    Returns:
        ``READY`` when every part is on disk; ``UNCHECKED`` when a part could not be checked (the rest is
        skipped and the caller loads as before); ``STOPPED`` when this caller owned a download and stopped it;
        ``DETACHED`` when this caller stopped waiting for someone else's download.

    Raises:
        RuntimeError: When a part cannot be checked for a reason other than being offline, or its download fails.
    """
    reporter = _Reporter(report, clock, interval_secs)
    parts = len(targets)
    for index, target in enumerate(targets, start=1):
        reporter.begin_part(target.repo_id, index, parts)
        while not present(target):
            flight, owner = flights.claim(target)
            if owner:
                early = _own(target, flight, flights, reporter=reporter, cancelled=cancelled, check=check, run=run, clock=clock)
                if early is not None:
                    return early
                break
            if not _wait_for(flight, reporter=reporter, cancelled=cancelled, wait_secs=wait_secs):
                return DETACHED
            if flight.status == "failed":
                raise RuntimeError(flight.message or f"Download of {target.repo_id} failed.")
    return READY
