"""Child-process entry point that downloads one target and reports byte counts on stdout.

Run as ``python -m zvisiongenerator.downloads.worker '<target json>'``. Every protocol line is
``ZIV-DOWNLOAD `` followed by a JSON object:

- ``{"wire": n, "written": m}``: counts so far, at most every 0.5 s while they change.
- ``{"done": true, "wire": n, "written": m}``: the download finished (exit code 0).
- ``{"error": "<Type>: <message>"}``: the download failed (exit code 1); a Hub refusal is one line saying what to do instead.
"""

from __future__ import annotations

import json
import os
import sys
import threading

from zvisiongenerator.downloads.hub_access import describe_refusal
from zvisiongenerator.downloads.progress import ByteCounts, make_byte_sink
from zvisiongenerator.downloads.protocol import PROTOCOL_PREFIX
from zvisiongenerator.downloads.target import DownloadTarget

REPORT_INTERVAL_SECS = 0.5
_LIFELINE_EXIT_CODE = 3


def _emit(payload: dict[str, object], lock: threading.Lock) -> None:
    """Print one protocol line and flush it so the parent sees it immediately."""
    with lock:
        print(PROTOCOL_PREFIX + json.dumps(payload), flush=True)


def _report_counts(counts: ByteCounts, stop: threading.Event, lock: threading.Lock) -> None:
    """Print the counts every report interval while they change, until *stop* is set."""
    last: tuple[int, int] | None = None
    while True:
        current = counts.snapshot()
        if current != last and current != (0, 0):
            _emit({"wire": current[0], "written": current[1]}, lock)
            last = current
        if stop.wait(REPORT_INTERVAL_SECS):
            return


def _exit_when_stdin_closes() -> None:
    """Block until stdin reaches EOF, then end the process: the parent that owned it is gone."""
    try:
        sys.stdin.read()
    except OSError, ValueError:
        pass
    os._exit(_LIFELINE_EXIT_CODE)


def main(argv: list[str] | None = None, *, start_lifeline: bool = False) -> int:
    """Download the target given as JSON in ``argv[0]`` and print protocol lines.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        start_lifeline: Exit the process when stdin closes, so an orphaned child never keeps downloading.

    Returns:
        ``0`` when the download finished, ``1`` when it failed.
    """
    args = sys.argv[1:] if argv is None else argv
    if start_lifeline:
        threading.Thread(target=_exit_when_stdin_closes, name="ziv-download-lifeline", daemon=True).start()
    lock = threading.Lock()
    counts = ByteCounts()
    stop = threading.Event()
    reporter = threading.Thread(target=_report_counts, args=(counts, stop, lock), name="ziv-download-report", daemon=True)
    reporter.start()
    target: DownloadTarget | None = None
    try:
        if len(args) != 1:
            raise ValueError("expected one argument: the download target as JSON")
        target = DownloadTarget.from_json(args[0])
        from huggingface_hub import snapshot_download

        snapshot_download(
            target.repo_id,
            revision=target.revision,
            allow_patterns=None if target.allow_patterns is None else list(target.allow_patterns),
            ignore_patterns=None if target.ignore_patterns is None else list(target.ignore_patterns),
            local_dir=target.local_dir,
            tqdm_class=make_byte_sink(counts),
        )
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the parent as a protocol line
        failure: dict[str, object] | None = {"error": describe_refusal(exc, None if target is None else target.repo_id) or f"{type(exc).__name__}: {exc}"}
    else:
        failure = None
    stop.set()
    reporter.join()
    if failure is not None:
        _emit(failure, lock)
        return 1
    wire, written = counts.snapshot()
    _emit({"done": True, "wire": wire, "written": written}, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(start_lifeline=True))
