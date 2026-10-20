"""Keep a bounded, persisted log of finished Web UI jobs."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.atomic_write import write_text_atomic
from zvisiongenerator.utils.paths import get_ziv_data_dir

JOB_LOG_FILENAME = "jobs.jsonl"
JOB_LOG_MAX_LINES = 2000
JOB_LOG_KEEP_LINES = 1500
JOB_LOG_PAGE_SIZE = 50
JOB_LOG_PROMPT_LIMIT = 500
LOGGED_JOB_TYPES = frozenset({"image", "video", "upscale"})
JOB_LOG_STATUS_FILTERS = ("all", "failed", "completed", "cancelled")

type JobLogEntry = dict[str, Any]

__all__ = [
    "JOB_LOG_FILENAME",
    "JOB_LOG_KEEP_LINES",
    "JOB_LOG_MAX_LINES",
    "JOB_LOG_PAGE_SIZE",
    "JOB_LOG_PROMPT_LIMIT",
    "JOB_LOG_STATUS_FILTERS",
    "LOGGED_JOB_TYPES",
    "JobLog",
    "JobLogEntry",
    "build_job_log_entry",
    "is_loggable",
    "job_log_path",
    "matches_status",
    "page_entries",
]


def job_log_path() -> Path:
    """Return the job log file inside the ZIV data directory, resolved at call time."""
    return get_ziv_data_dir() / JOB_LOG_FILENAME


def is_loggable(job_type: str, started_at: float | None) -> bool:
    """Return whether a job of this type that started at *started_at* belongs in the log."""
    return job_type in LOGGED_JOB_TYPES and started_at is not None


def _safe_text(value: object) -> str:
    """Return ``str(value)``, or an empty string when the value cannot be rendered."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - the entry builder must never raise
        return ""


def _safe_epoch(value: object) -> float | None:
    """Return *value* as a float timestamp, or ``None`` when it is not a number."""
    try:
        return float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None


def _safe_count(value: object) -> int:
    """Return *value* as a non-negative int, or 0 when it is not a number."""
    try:
        return max(0, int(value))  # type: ignore[call-overload]
    except TypeError, ValueError, OverflowError:
        return 0


def _safe_settings(value: object) -> dict[str, Any]:
    """Return a shallow copy of *value* when it is a mapping, else an empty dict."""
    if not isinstance(value, Mapping):
        return {}
    try:
        return dict(value)
    except Exception:  # noqa: BLE001
        return {}


def _safe_output_ids(outputs: object) -> list[str]:
    """Return the ``id`` of every mapping in *outputs* that has one."""
    ids: list[str] = []
    try:
        for output in outputs:  # type: ignore[attr-defined]
            if isinstance(output, Mapping) and "id" in output:
                ids.append(_safe_text(output["id"]))
    except Exception:  # noqa: BLE001
        pass
    return ids


def _duration_secs(started_at: float | None, completed_at: float | None) -> float:
    """Return the run time in seconds, rounded to milliseconds and never negative."""
    if started_at is None or completed_at is None:
        return 0.0
    return max(0.0, round(completed_at - started_at, 3))


def build_job_log_entry(
    *,
    job_id: str,
    job_type: str,
    context: Mapping[str, Any],
    status: str,
    created_at: float | None,
    started_at: float | None,
    completed_at: float | None,
    outputs: list[Mapping[str, Any]],
    error: object,
    failed_runs: int,
) -> JobLogEntry:
    """Build the log entry of a finished job; never raises, whatever the context holds."""
    ctx: Mapping[str, Any] = context if isinstance(context, Mapping) else {}
    try:
        prompt = _safe_text(ctx.get("prompt") or "")[:JOB_LOG_PROMPT_LIMIT]
        model = _safe_text(ctx.get("model") or "")
        workflow = _safe_text(ctx.get("workflow") or job_type)
        settings = _safe_settings(ctx.get("settings"))
    except Exception:  # noqa: BLE001
        prompt, model, workflow, settings = "", "", _safe_text(job_type), {}
    started = _safe_epoch(started_at)
    completed = _safe_epoch(completed_at)
    return {
        "job_id": _safe_text(job_id),
        "job_type": _safe_text(job_type),
        "workflow": workflow,
        "model": model,
        "prompt": prompt,
        "settings": settings,
        "status": _safe_text(status),
        "created_at": _safe_epoch(created_at),
        "started_at": started,
        "completed_at": completed,
        "duration_secs": _duration_secs(started, completed),
        "outputs": _safe_output_ids(outputs),
        "error": None if error is None else _safe_text(error),
        "failed_runs": _safe_count(failed_runs),
    }


def matches_status(entry: Mapping[str, Any], status: str) -> bool:
    """Return whether *entry* belongs under the *status* filter (``failed`` includes partial failures)."""
    if status == "all":
        return True
    if status == "failed":
        failed_runs = entry.get("failed_runs")
        return entry.get("status") == "failed" or (isinstance(failed_runs, int) and failed_runs > 0)
    return entry.get("status") == status


def page_entries(entries_oldest_first: list[JobLogEntry], *, status: str, page: int, page_size: int) -> dict[str, Any]:
    """Filter entries by *status* and return one newest-first page with paging facts; *page* is clamped."""
    size = max(1, page_size)
    matching = [entry for entry in reversed(entries_oldest_first) if matches_status(entry, status)]
    total_pages = max(1, math.ceil(len(matching) / size))
    current = min(max(1, page), total_pages)
    start = (current - 1) * size
    return {
        "entries": matching[start : start + size],
        "page": current,
        "total_pages": total_pages,
        "total_count": len(matching),
        "page_size": size,
    }


def _parse_lines(text: str) -> list[JobLogEntry]:
    """Return the entries in *text*, skipping lines that are blank, not JSON, not objects or without a ``job_id``."""
    entries: list[JobLogEntry] = []
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+0085 and others, which JSON output keeps raw.
    for line in text.split("\n"):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("job_id"), str):
            entries.append(entry)
    return entries


def _serialize(entries: list[JobLogEntry]) -> str:
    """Return *entries* as JSON lines, one per entry."""
    return "".join(json.dumps(entry, ensure_ascii=False, default=str) + "\n" for entry in entries)


class JobLog:
    """Store finished jobs in a JSON-lines file, oldest first, behind one lock.

    The path is resolved on every call. Each change reads the file, edits the entries and replaces the file
    atomically, so readers never see a partial line. ``OSError`` from the file system propagates to the caller.
    """

    def __init__(self, path_provider: Callable[[], Path]) -> None:
        """Create a store that finds its file through *path_provider*."""
        self._path_provider = path_provider
        self._lock = threading.Lock()

    def _read(self, path: Path) -> list[JobLogEntry]:
        """Return the entries in *path*; a missing file is empty and bad lines are skipped."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        return _parse_lines(text)

    def append(self, entry: JobLogEntry) -> None:
        """Add *entry* as the newest line; past the cap only the newest lines are kept."""
        with self._lock:
            path = self._path_provider()
            entries = [*self._read(path), entry]
            if len(entries) > JOB_LOG_MAX_LINES:
                entries = entries[-JOB_LOG_KEEP_LINES:]
            write_text_atomic(path, _serialize(entries))

    def page(self, status: str, page: int, page_size: int = JOB_LOG_PAGE_SIZE) -> dict[str, Any]:
        """Return one newest-first page of entries matching *status*."""
        with self._lock:
            entries = self._read(self._path_provider())
        return page_entries(entries, status=status, page=page, page_size=page_size)

    def delete(self, job_id: str) -> bool:
        """Remove the entries of *job_id*; return whether there were any."""
        with self._lock:
            path = self._path_provider()
            entries = self._read(path)
            kept = [entry for entry in entries if entry["job_id"] != job_id]
            if len(kept) == len(entries):
                return False
            write_text_atomic(path, _serialize(kept))
            return True

    def clear(self) -> int:
        """Remove every entry and return how many there were."""
        with self._lock:
            path = self._path_provider()
            entries = self._read(path)
            if path.exists():
                write_text_atomic(path, "")
            return len(entries)
