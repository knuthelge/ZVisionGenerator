"""Define Web UI job lifecycle constants and snapshot helpers."""

from __future__ import annotations

from typing import Any


SUCCESS_TERMINAL_EVENT = "job_completed"
FAILED_TERMINAL_EVENT = "job_failed"
CANCELLED_TERMINAL_EVENT = "job_cancelled"

TERMINAL_EVENT_TYPES = frozenset({SUCCESS_TERMINAL_EVENT, FAILED_TERMINAL_EVENT, CANCELLED_TERMINAL_EVENT})
TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed"})
QUEUED_STATUS = "queued"
ACTIVE_STATUSES = frozenset({QUEUED_STATUS, "running", "paused"})
STARTED_EVENT = "job_started"
# ``reason`` of a ``job_cancelled`` event for a job removed from the queue before it started.
REMOVED_REASON = "removed"
IMAGE_SUPPORTED_CONTROLS = ("next", "pause", "resume", "repeat", "quit")
VIDEO_SUPPORTED_CONTROLS: tuple[str, ...] = ()
UPSCALE_SUPPORTED_CONTROLS = ("quit",)


def terminal_event_for_status(status: str) -> str | None:
    """Return the frontend-visible terminal event for a terminal status."""
    if status == "completed":
        return SUCCESS_TERMINAL_EVENT
    if status == "failed":
        return FAILED_TERMINAL_EVENT
    if status == "cancelled":
        return CANCELLED_TERMINAL_EVENT
    return None


def public_job_snapshot(
    *,
    job_id: str,
    status: str,
    workflow: str,
    supported_controls: tuple[str, ...],
    context: dict[str, Any],
    created_at: float,
    completed_at: float | None,
    event_count: int,
    last_event: dict[str, Any] | None,
    paused: bool,
    result_path: str | None,
    outputs: list[dict[str, Any]] | None = None,
    preview_version: int = 0,
    started_at: float | None = None,
    queue_position: int | None = None,
) -> dict[str, Any]:
    """Build the public snake_case job snapshot payload."""
    snapshot = {
        "id": job_id,
        "job_id": job_id,
        "workflow": workflow,
        "job_type": context.get("job_type", workflow),
        "status": status,
        "created_at": context.get("created_at", created_at),
        "started_at": started_at,
        "completed_at": completed_at,
        "queue_position": queue_position,
        "event_count": event_count,
        "last_event": last_event,
        "supported_controls": list(supported_controls),
        "supports_controls": list(supported_controls),
        "paused": paused,
        "result_path": result_path,
        "outputs": outputs or [],
        "preview_version": preview_version,
        "prompt": context.get("prompt", ""),
        "model": context.get("model", ""),
        "runs": context.get("runs", 1),
        "meta": context.get("meta", ""),
        "settings": context.get("settings", {}),
        "notices": list(context.get("notices", ())),
    }
    terminal_event = terminal_event_for_status(status)
    if terminal_event is not None:
        snapshot["terminal_event"] = terminal_event
    return snapshot
