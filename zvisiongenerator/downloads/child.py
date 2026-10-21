"""Run one download in a child process, relay its byte counts and stop it on request."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any, Literal

from zvisiongenerator.downloads.presence import remove_partial_files
from zvisiongenerator.downloads.protocol import PROTOCOL_PREFIX
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.subprocess_env import utf8_child_env

_XET_BUFFER_VAR = "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"
_XET_BUFFER_DEFAULT = "512mb"
_TAIL_LINES = 20
_READER_JOIN_SECS = 2.0


@dataclass(frozen=True)
class ChildResult:
    """How a child download ended; ``message`` explains a failure."""

    status: Literal["done", "failed", "stopped"]
    message: str | None = None


def child_env() -> dict[str, str]:
    """Return the environment for the download child: UTF-8, unbuffered, with progress bars switchable.

    ``TQDM_DISABLE`` is removed because the parent may run inside a worker runtime that sets it,
    which would hide the bar attributes the byte sink reads. hf_xet otherwise sizes its in-flight download buffer
    from host RAM, so ``HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE`` defaults to 512mb to keep memory bounded beside a
    running generation; a value already set in the parent environment is kept.
    """
    env = utf8_child_env(HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
    env.pop("TQDM_DISABLE", None)
    env.setdefault(_XET_BUFFER_VAR, _XET_BUFFER_DEFAULT)
    return env


def _group_options() -> dict[str, Any]:
    """Return the ``Popen`` option that puts the child in its own process group, so a terminal Ctrl+C skips it."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _read_lines(stream: IO[str], lines: queue.Queue[str]) -> None:
    """Move every line of *stream* onto *lines* until it closes."""
    try:
        for line in stream:
            lines.put(line.rstrip("\r\n"))
    except OSError, ValueError:
        pass


def _parse_protocol(line: str) -> dict[str, Any] | None:
    """Return the JSON payload of a protocol line, or ``None`` for any other line."""
    start = line.find(PROTOCOL_PREFIX)
    if start < 0:
        return None
    try:
        payload = json.loads(line[start + len(PROTOCOL_PREFIX) :])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class _ChildState:
    """What the parent has learned from the child's output so far."""

    done: bool = False
    error: str | None = None
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=_TAIL_LINES))


def _handle_lines(lines: queue.Queue[str], state: _ChildState, on_counts: Callable[[int, int], None]) -> None:
    """Process every queued output line: protocol lines update *state* and counts, others go to the tail."""
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return
        payload = _parse_protocol(line)
        if payload is None:
            if line.strip():
                state.tail.append(line)
            continue
        if "error" in payload:
            state.error = str(payload["error"])
        if payload.get("done") is True:
            state.done = True
        if "wire" in payload and "written" in payload:
            on_counts(int(payload["wire"]), int(payload["written"]))


def _terminate(process: subprocess.Popen[str], grace_secs: float) -> None:
    """Ask the process to end, kill it when it outlives *grace_secs*, and wait for it."""
    process.terminate()
    try:
        process.wait(timeout=grace_secs)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    try:
        process.wait(timeout=grace_secs)
    except subprocess.TimeoutExpired:
        pass


def _run_cleanup(cleanup: Callable[[DownloadTarget], object], target: DownloadTarget) -> None:
    """Remove the target's partial files; a filesystem error only warns."""
    try:
        cleanup(target)
    except OSError as exc:
        warnings.warn(f"Could not remove partial files of {target.repo_id}: {exc}", stacklevel=3)


def _close(stream: IO[str] | None) -> None:
    """Close a pipe, ignoring a peer that already went away."""
    if stream is None:
        return
    try:
        stream.close()
    except OSError:
        pass


def _failure_message(state: _ChildState, code: int | None) -> str:
    """Pick the best explanation for a failed child: its error line, its output tail, or the exit code."""
    if state.error:
        return state.error
    tail = "\n".join(state.tail)
    return tail or f"The download process exited with code {code}."


def run_in_child(
    target: DownloadTarget,
    *,
    on_counts: Callable[[int, int], None],
    cancelled: Callable[[], bool],
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    poll_secs: float = 0.25,
    grace_secs: float = 5.0,
    cleanup: Callable[[DownloadTarget], object] = remove_partial_files,
) -> ChildResult:
    """Download *target* in a child process and block until it ends or *cancelled* turns true.

    Args:
        target: The download to run.
        on_counts: Called with ``(wire, written)`` byte counts as the child reports them.
        cancelled: Polled every *poll_secs*; once true the child is terminated.
        popen: Starts the child process; replaceable for tests.
        poll_secs: Seconds between output drains and cancel checks.
        grace_secs: Seconds a terminated child gets before it is killed.
        cleanup: Removes the partial files of a stopped or failed download.

    Returns:
        ``done`` when the child finished, ``stopped`` after a cancel, ``failed`` otherwise.
    """
    command = [sys.executable, "-m", "zvisiongenerator.downloads.worker", target.to_json()]
    try:
        process = popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env(),
            **_group_options(),
        )
    except OSError as exc:
        return ChildResult("failed", f"Could not start the download process: {exc}")
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_read_lines, args=(process.stdout, lines), name="ziv-download-reader", daemon=True)
    reader.start()
    state = _ChildState()
    try:
        while True:
            _handle_lines(lines, state, on_counts)
            if cancelled():
                _terminate(process, grace_secs)
                _run_cleanup(cleanup, target)
                return ChildResult("stopped")
            code = process.poll()
            if code is not None:
                break
            time.sleep(poll_secs)
        reader.join(timeout=_READER_JOIN_SECS)
        _handle_lines(lines, state, on_counts)
    finally:
        if process.poll() is None:
            _terminate(process, grace_secs)
        _close(process.stdin)
        _close(process.stdout)
    if code == 0 and state.done:
        return ChildResult("done")
    _run_cleanup(cleanup, target)
    return ChildResult("failed", _failure_message(state, code))
