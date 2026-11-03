"""Print a running download for a command-line tool: one live line on a terminal, a line now and then otherwise."""

from __future__ import annotations

import time
from collections.abc import Callable

from zvisiongenerator.downloads.ensure import DownloadReport
from zvisiongenerator.utils.live_line import LiveLine, shared_line
from zvisiongenerator.utils.model_files import disk_size_label

__all__ = ["CliDownloadReporter", "eta_text", "progress_line"]

_SECS_PER_MIN = 60
_MINS_PER_HOUR = 60


def progress_line(report: DownloadReport) -> str:
    """Return the line for *report*, such as ``Downloading org/repo (1 of 2): 2.1/7.0 GB, 45 MB/s, about 2 min left``."""
    progress = report.progress
    pieces = [_amount_text(progress.bytes_done, progress.bytes_total)]
    if progress.bytes_per_sec:
        pieces.append(f"{disk_size_label(round(progress.bytes_per_sec))}/s")
    finished = progress.bytes_total is not None and progress.bytes_done >= progress.bytes_total
    if progress.eta_secs is not None and not finished:
        pieces.append(eta_text(progress.eta_secs))
    return f"Downloading {report.repo} ({report.part} of {report.parts}): {', '.join(pieces)}"


def _amount_text(done: int, total: int | None) -> str:
    """Return ``2.1/7.0 GB`` (in the unit of *total*), or just the size done when the total is unknown."""
    if total is None:
        return disk_size_label(done)
    if total >= 1e9:
        return f"{done / 1e9:.1f}/{total / 1e9:.1f} GB"
    return f"{done / 1e6:.0f}/{total / 1e6:.0f} MB"


def eta_text(eta_secs: float) -> str:
    """Return the time left in whole minutes, or hours and minutes, such as ``about 2 min left``."""
    minutes = round(eta_secs / _SECS_PER_MIN)
    if minutes < 1:
        return "less than a minute left"
    if minutes < _MINS_PER_HOUR:
        return f"about {minutes} min left"
    hours, rest = divmod(minutes, _MINS_PER_HOUR)
    return f"about {hours} h {rest} min left"


class CliDownloadReporter:
    """Report callback for :func:`~zvisiongenerator.downloads.ensure.ensure_downloaded` that prints to the console.

    On a terminal one line is rewritten in place, and ends when its part is done. Anywhere else (a pipe, a log file) a
    line is printed for the first and last report of each part and at most once per *interval_secs* between them.
    Call :meth:`close` when the downloads end, so a stopped one does not leave the cursor on the line.

    Args:
        interval_secs: The least time between two lines printed off a terminal.
        clock: The monotonic clock, in seconds.
        line: The console line to write through; the process console's when ``None``.
    """

    def __init__(self, *, interval_secs: float = 30.0, clock: Callable[[], float] = time.monotonic, line: LiveLine | None = None) -> None:
        self._interval_secs = interval_secs
        self._clock = clock
        self._line = line or shared_line()
        self._part = 0
        self._last_printed: float | None = None

    def __call__(self, report: DownloadReport) -> None:
        """Show *report*."""
        if report.part != self._part:
            self.close()
            self._part, self._last_printed = report.part, None
        text = progress_line(report)
        finished = report.progress.bytes_total is not None and report.progress.bytes_done >= report.progress.bytes_total
        if self._line.is_terminal():
            self._line.rewrite(text)
            if finished:
                self.close()
            return
        now = self._clock()
        if self._last_printed is None or finished or now - self._last_printed >= self._interval_secs:
            self._last_printed = now
            self._line.print_line(text)

    def close(self) -> None:
        """End the live line, if one is open."""
        self._line.end()
