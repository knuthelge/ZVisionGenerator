"""Keep a log of finished Web UI jobs in the app database."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils import app_db

JOB_LOG_PAGE_SIZE = 50
JOB_LOG_PROMPT_LIMIT = 500
LOGGED_JOB_TYPES = frozenset({"image", "video", "upscale"})
JOB_LOG_STATUS_FILTERS = ("all", "failed", "completed", "cancelled")

_VERSION = 1
_TABLE = f"jobs_entries_v{_VERSION}"

type JobLogEntry = dict[str, Any]

__all__ = [
    "JOB_LOG_PAGE_SIZE",
    "JOB_LOG_PROMPT_LIMIT",
    "JOB_LOG_STATUS_FILTERS",
    "LOGGED_JOB_TYPES",
    "JobLog",
    "JobLogEntry",
    "JobLogError",
    "build_job_log_entry",
    "is_loggable",
    "matches_status",
    "page_entries",
]


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


class JobLogError(RuntimeError):
    """Report that the job log could not be read or written."""


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Create the job log table and index, refusing a database written by a newer ziv."""
    if from_version > _VERSION:
        raise JobLogError("it was written by a newer version of ziv; update ziv to read it")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_TABLE} (seq INTEGER PRIMARY KEY, job_id TEXT NOT NULL, entry TEXT NOT NULL)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {_TABLE}_job_id ON {_TABLE} (job_id)")


_COMPONENT = app_db.Component("jobs", _VERSION, _migrate)


class JobLog:
    """Store finished jobs as rows of the ``jobs`` component of the app database, oldest first.

    The database path is resolved on every call and each operation uses its own short-lived connection, so
    servers sharing one data folder share the log; SQLite serialises the writers. Storage failures, and a
    jobs schema written by a newer ziv, raise :class:`JobLogError`.
    """

    def __init__(self, path_provider: Callable[[], Path]) -> None:
        """Create a store that finds its database through *path_provider*."""
        self._path_provider = path_provider

    def _run[T](self, operation: Callable[[app_db.AppConnection], T]) -> T:
        """Run *operation* on a new connection; quarantine a corrupt file and wrap storage errors."""
        return app_db.run(self._path_provider(), (_COMPONENT,), operation, error=JobLogError)

    def append(self, entry: JobLogEntry) -> None:
        """Add *entry* as the newest row; rows stay until :meth:`delete` or :meth:`clear` removes them."""
        job_id = entry.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("A job log entry needs a string job_id.")
        text = json.dumps(entry, ensure_ascii=False, default=str)

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.execute(f"INSERT INTO {_TABLE} (job_id, entry) VALUES (?, ?)", (job_id, text))

        self._run(write)

    def page(self, status: str, page: int, page_size: int = JOB_LOG_PAGE_SIZE) -> dict[str, Any]:
        """Return one newest-first page of entries matching *status*."""
        rows = self._run(lambda conn: conn.execute(f"SELECT entry FROM {_TABLE} ORDER BY seq").fetchall())
        entries = [json.loads(text) for (text,) in rows]
        return page_entries(entries, status=status, page=page, page_size=page_size)

    def delete(self, job_id: str) -> bool:
        """Remove the entries of *job_id*; return whether there were any."""
        return self._delete(f"DELETE FROM {_TABLE} WHERE job_id = ?", (job_id,)) > 0

    def clear(self) -> int:
        """Remove every entry and return how many there were."""
        return self._delete(f"DELETE FROM {_TABLE}", ())

    def _delete(self, statement: str, params: tuple[str, ...]) -> int:
        """Run one DELETE in its own write transaction and return how many rows it removed."""

        def remove(conn: app_db.AppConnection) -> int:
            with app_db.transaction(conn):
                return conn.execute(statement, params).rowcount

        return self._run(remove)
