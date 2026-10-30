"""Run synchronous generation batches in background workers for the Web UI."""

from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from zvisiongenerator.backends import get_backend, get_seedvr2_upscaler, get_video_backend, release_accelerator_memory
from zvisiongenerator.built_models import ensure_built_model
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.progress_events import exception_message
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.downloads import DETACHED, STOPPED, DownloadFlights, DownloadReport, DownloadTarget, describe_refusal, enhancer_plan, ensure_downloaded, image_plan, seedvr2_plan, video_plan
from zvisiongenerator.image_model_loader import load_image_model
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.preflight import run_preflight
from zvisiongenerator.review import review_enabled
from zvisiongenerator.seedvr2_loader import load_seedvr2
from zvisiongenerator.upscale_runner import run_upscale, upscale_prompts
from zvisiongenerator.video_file_upscale_runner import run_video_file_upscale
from zvisiongenerator.utils.config import video_load_options
from zvisiongenerator.utils.ffmpeg import require_ffmpeg
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_SEEDVR2
from zvisiongenerator.utils.seedvr2 import SEEDVR2_BITS, seedvr2_tier_label
from zvisiongenerator.web.config import load_web_config
from zvisiongenerator.web.gallery import gallery_asset_for_output_path, gallery_asset_to_json
from zvisiongenerator.web.job_log import JobLog, build_job_log_entry, is_loggable
from zvisiongenerator.web.job_contract import (
    CANCELLED_TERMINAL_EVENT,
    DOWNLOADING_EVENT,
    FAILED_TERMINAL_EVENT,
    QUEUED_STATUS,
    REMOVED_REASON,
    REVIEW_DECISION_CONTROLS,
    STARTED_EVENT,
    SUCCESS_TERMINAL_EVENT,
    TERMINAL_EVENT_TYPES,
    TERMINAL_STATUSES,
    UPSCALE_SUPPORTED_CONTROLS,
    generation_supported_controls,
    public_job_snapshot,
)
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import build_video_workflow

type EventPayload = dict[str, Any]

logger = logging.getLogger(__name__)

# Events that end the generation a live preview belongs to, so the stale preview is dropped. A new workflow stage
# (e.g. the upscale pass) keeps the previous stage's preview until it sends its own.
_PREVIEW_RESET_EVENT_TYPES = frozenset({"prompt_started", "generation_finished", *TERMINAL_EVENT_TYPES})


_REVIEW_RESET_EVENT_TYPES = frozenset({"job_resumed", "prompt_started", "generation_finished", *TERMINAL_EVENT_TYPES})


class JobConflictError(RuntimeError):
    """Raised when a job or exclusive action is refused because other work holds the model memory."""


class UnsupportedJobControlError(RuntimeError):
    """Raised when a job cannot accept the requested control action."""


class _ThreadAwareTextStream:
    """Proxy a process-global text stream while muting selected worker threads."""

    def __init__(self, stream: Any, *, mode: str) -> None:
        self._stream = stream
        self._lock = threading.RLock()
        self._muted_threads: set[int] = set()
        self._devnull = open(os.devnull, mode, encoding="utf-8", errors="ignore")

    def mute(self, thread_id: int) -> None:
        with self._lock:
            self._muted_threads.add(thread_id)

    def unmute(self, thread_id: int) -> None:
        with self._lock:
            self._muted_threads.discard(thread_id)

    def write(self, data: str) -> int:
        if self._is_muted():
            return len(data)
        return self._stream.write(data)

    def flush(self) -> None:
        if self._is_muted():
            self._devnull.flush()
            return
        self._stream.flush()

    def isatty(self) -> bool:
        if self._is_muted():
            return False
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        if self._is_muted():
            return self._devnull.fileno()
        return self._stream.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _is_muted(self) -> bool:
        with self._lock:
            return threading.get_ident() in self._muted_threads


class _MutedWorkerStreams:
    """Install thread-aware stdio wrappers and mute generation workers only."""

    def __init__(self) -> None:
        self.stdout = _ThreadAwareTextStream(sys.stdout, mode="w")
        self.stderr = _ThreadAwareTextStream(sys.stderr, mode="w")
        if not isinstance(sys.stdout, _ThreadAwareTextStream):
            sys.stdout = self.stdout
        else:
            self.stdout = sys.stdout
        if not isinstance(sys.stderr, _ThreadAwareTextStream):
            sys.stderr = self.stderr
        else:
            self.stderr = sys.stderr

    @contextmanager
    def mute_current_thread(self):
        thread_id = threading.get_ident()
        self.stdout.mute(thread_id)
        self.stderr.mute(thread_id)
        try:
            yield
        finally:
            self.stdout.unmute(thread_id)
            self.stderr.unmute(thread_id)


_WORKER_STREAMS: _MutedWorkerStreams | None = None
_WORKER_STREAMS_LOCK = threading.RLock()


def _get_worker_streams() -> _MutedWorkerStreams:
    """Install worker stream wrappers only when a worker actually runs."""
    global _WORKER_STREAMS
    with _WORKER_STREAMS_LOCK:
        if _WORKER_STREAMS is None:
            _WORKER_STREAMS = _MutedWorkerStreams()
        return _WORKER_STREAMS


@contextmanager
def worker_runtime_context():
    """Suppress noisy worker stdio and progress bars inside worker execution only."""
    previous_env = {key: os.environ.get(key) for key in ("HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE")}
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    with _get_worker_streams().mute_current_thread():
        try:
            yield
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@dataclass(slots=True)
class _JobRecord:
    """Hold background job state and SSE subscribers."""

    job_id: str
    job_type: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    future: Future[None] | None = None
    history: list[EventPayload] = field(default_factory=list)
    event_count: int = 0
    last_event: EventPayload | None = None
    prompt_progress: EventPayload = field(default_factory=dict)
    subscribers: set[queue.Queue[EventPayload]] = field(default_factory=set)
    next_event_id: int = 1
    exclusive: bool = False
    control_signal: SkipSignal | None = None
    supported_controls: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    result_path: str | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    paused: bool = False
    # ``{"iteration": n, "attempt": k}`` from the pause of a render until ``job_resumed``; reconnecting browsers see it in the snapshot.
    review: dict[str, Any] | None = None
    review_decided: bool = False
    # Temporary video of the small render under review; server-side only, never in events or snapshots.
    review_path: str | None = None
    last_eta_secs: float | None = None
    preview_jpeg: bytes | None = None
    preview_version: int = 0
    # Generations that ended ``failed`` and the reason of the last one that said why; they survive a job that still ends
    # ``completed`` or ``cancelled``, and feed the job log.
    failed_runs: int = 0
    last_failure: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class WebRunner:
    """Execute synchronous runners on one worker thread and surface SSE progress.

    Generation (exclusive) jobs wait in a FIFO queue. The worker claims each one when it starts, so a job removed
    from the queue never runs. The *active* job is the claimed one, or, between jobs, the head of the queue.
    """

    _TERMINAL_EVENT_TYPES = TERMINAL_EVENT_TYPES
    _TERMINAL_STATUSES = TERMINAL_STATUSES

    def __init__(
        self,
        *,
        max_workers: int = 1,
        heartbeat_seconds: float = 10.0,
        max_history_events: int = 200,
        terminal_retention_seconds: float = 300.0,
        max_terminal_jobs: int = 25,
        download_flights: DownloadFlights | None = None,
        job_log: JobLog | None = None,
    ) -> None:
        """Create the thread-backed web runner facade.

        *download_flights* is the registry that lets jobs (and the server's download lane) share one download per
        destination; it defaults to a registry of its own. A started image, video or upscale job is logged to *job_log*
        when it ends.
        """
        self._download_flights = download_flights if download_flights is not None else DownloadFlights()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ziv-web-runner")
        self._heartbeat_seconds = heartbeat_seconds
        self._max_history_events = max(1, max_history_events)
        self._terminal_retention_seconds = max(0.0, terminal_retention_seconds)
        self._max_terminal_jobs = max(0, max_terminal_jobs)
        self._job_log = job_log
        self._jobs: dict[str, _JobRecord] = {}
        # Final snapshot + terminal event of pruned jobs, so late clients (e.g. a laptop waking up) still learn the
        # real outcome instead of a 404. Bounded; only a server restart makes a job truly unknown.
        self._pruned: OrderedDict[str, tuple[dict[str, Any], EventPayload | None]] = OrderedDict()
        self._max_pruned_jobs = 500
        self._jobs_lock = threading.RLock()
        # Waiting generation job ids, oldest first, and the job the worker is running; both guarded by _jobs_lock.
        self._queue: deque[str] = deque()
        self._claimed_job_id: str | None = None

    @property
    def download_flights(self) -> DownloadFlights:
        """Return the registry of running model downloads that jobs share."""
        return self._download_flights

    def submit_image_request_job(
        self,
        *,
        request: ImageGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        quantize: int | None = None,
        enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
        admission_check: Callable[[], None] | None = None,
        context: dict[str, Any] | None = None,
        supported_controls: tuple[str, ...] | None = None,
    ) -> str:
        """Queue the image job; the worker loads the model and runs the batch loop.

        *supported_controls* defaults to the controls the job's ``args`` imply, with a review when one is requested.
        """
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="image",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=supported_controls if supported_controls is not None else generation_supported_controls(review=review_enabled(args)),
            context={"output_dir": getattr(args, "output", None), **(context or {})},
            admission_check=admission_check,
            target_factory=lambda progress_callback: self._run_image_request(
                request=request,
                prompts_data=prompts_data,
                config=config,
                args=args,
                model_ref=model_ref,
                quantize=quantize,
                progress_callback=progress_callback,
                control_signal=control_signal,
                enhance_by_set=enhance_by_set,
            ),
        )

    def submit_video_request_job(
        self,
        *,
        request: VideoGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
        scene_by_set: dict[str, list[str | None]] | None = None,
        admission_check: Callable[[], None] | None = None,
        context: dict[str, Any] | None = None,
        supported_controls: tuple[str, ...] | None = None,
    ) -> str:
        """Queue the video job; the worker loads the model and runs the batch loop.

        The job supports the same controls as an image job, and with a review also approve and discard.
        *supported_controls* defaults to the controls the job's ``args`` imply.
        """
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="video",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=supported_controls if supported_controls is not None else generation_supported_controls(review=review_enabled(args)),
            context={"output_dir": getattr(args, "output", None), **(context or {})},
            admission_check=admission_check,
            target_factory=lambda progress_callback: self._run_video_request(
                request=request,
                prompts_data=prompts_data,
                config=config,
                args=args,
                model_ref=model_ref,
                progress_callback=progress_callback,
                control_signal=control_signal,
                enhance_by_set=enhance_by_set,
                scene_by_set=scene_by_set,
            ),
        )

    def submit_upscale_job(
        self,
        *,
        request: ImageGenerationRequest,
        workflow: GenerationWorkflow,
        model_ref: str,
        config: dict[str, Any],
        admission_check: Callable[[], None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Queue the upscale job; the worker loads the refinement model (at ``request.quantize``) and upscales one image.

        *config* is the app config, from which a model built on first use (MOSS-Image, Ideogram 4 Instant) is planned and built.
        """
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="upscale",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=UPSCALE_SUPPORTED_CONTROLS,
            context=dict(context or {}),
            admission_check=admission_check,
            target_factory=lambda progress_callback: self._run_upscale_request(
                request=request,
                workflow=workflow,
                model_ref=model_ref,
                config=config,
                progress_callback=progress_callback,
                control_signal=control_signal,
            ),
        )

    def submit_video_file_upscale_job(
        self,
        *,
        request: VideoFileUpscaleRequest,
        admission_check: Callable[[], None] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Queue the job that upscales an existing video file; the worker downloads and loads SeedVR2 and streams the clip through it."""
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="upscale",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=UPSCALE_SUPPORTED_CONTROLS,
            context=dict(context or {}),
            admission_check=admission_check,
            target_factory=lambda progress_callback: self._run_video_file_upscale_request(request=request, progress_callback=progress_callback, control_signal=control_signal),
        )

    def submit_dummy_job(self, *, total_steps: int = 5, delay_seconds: float = 0.25) -> str:
        """Run a dummy background job that emits example progress updates."""

        def _run_dummy(progress_callback: Callable[[EventPayload], None]) -> None:
            progress_callback(
                {
                    "type": "batch_started",
                    "mode": "dummy",
                    "total_iterations": total_steps,
                    "total_runs": 1,
                }
            )
            for step in range(1, total_steps + 1):
                time.sleep(delay_seconds)
                progress_callback(
                    {
                        "type": "progress",
                        "mode": "dummy",
                        "current": step,
                        "total": total_steps,
                        "message": f"Dummy progress {step}/{total_steps}",
                    }
                )
            progress_callback(
                {
                    "type": "batch_completed",
                    "mode": "dummy",
                    "completed_iterations": total_steps,
                    "total_iterations": total_steps,
                }
            )

        return self._submit_job(job_type="dummy", target_factory=_run_dummy)

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        """Return serializable state for a tracked job (or the final state of a pruned one)."""
        with self._jobs_lock:
            self._prune_terminal_jobs_locked()
            record = self._jobs.get(job_id)
            if record is None:
                if job_id not in self._pruned:
                    raise KeyError(job_id)
                return copy.deepcopy(self._pruned[job_id][0])
            return self._snapshot_record(record, queue_position=self._queue_positions_locked().get(job_id))

    def list_jobs(self) -> dict[str, Any]:
        """Return the active generation job and the queued ones, oldest first; a job is never in both."""
        with self._jobs_lock:
            self._prune_terminal_jobs_locked()
            active_id = self._active_job_id_locked()
            positions = self._queue_positions_locked()
            active = self._snapshot_record(self._jobs[active_id]) if active_id is not None else None
            queued = [self._snapshot_record(self._jobs[job_id], queue_position=position) for job_id, position in positions.items()]
        return {"active_job": active, "queued_jobs": queued}

    def cancel_queued(self, job_id: str) -> bool:
        """Remove a job that has not started from the queue; return whether it was queued."""
        with self._jobs_lock:
            if job_id not in self._queue:
                return False
            self._queue.remove(job_id)
            future = self._jobs[job_id].future
        self._finish_removed([(job_id, future)])
        return True

    def clear_queue(self) -> list[str]:
        """Remove every queued job, never the active one; return the removed job ids."""
        with self._jobs_lock:
            removed = list(self._queue_positions_locked())
            for job_id in removed:
                self._queue.remove(job_id)
            futures = [(job_id, self._jobs[job_id].future) for job_id in removed]
        self._finish_removed(futures)
        return removed

    def _finish_removed(self, jobs: list[tuple[str, Future[None] | None]]) -> None:
        """Publish the terminal event of jobs taken off the queue; the claim step keeps them from running."""
        for job_id, future in jobs:
            if future is not None:
                future.cancel()
            self._publish_event(job_id, {"type": CANCELLED_TERMINAL_EVENT, "reason": REMOVED_REASON})

    def _active_job_id_locked(self) -> str | None:
        """Return the claimed job, else the job waiting at the head of the queue."""
        if self._claimed_job_id is not None:
            return self._claimed_job_id
        return self._queue[0] if self._queue else None

    def _queue_positions_locked(self) -> dict[str, int]:
        """Map each queued job id to its 1-based position, skipping the head while it is the active job."""
        waiting = list(self._queue)
        if self._claimed_job_id is None:
            waiting = waiting[1:]
        return {job_id: index for index, job_id in enumerate(waiting, start=1)}

    def _snapshot_record(self, record: _JobRecord, *, queue_position: int | None = None) -> dict[str, Any]:
        with record.lock:
            last_event = dict(record.last_event) if record.last_event is not None else None
            workflow = str(record.context.get("workflow") or record.job_type)
            return public_job_snapshot(
                job_id=record.job_id,
                status=record.status,
                workflow=workflow,
                supported_controls=record.supported_controls,
                context=record.context,
                created_at=record.created_at,
                completed_at=record.completed_at,
                event_count=record.event_count,
                last_event=last_event,
                paused=record.paused,
                review=record.review,
                result_path=record.result_path,
                outputs=[dict(output) for output in record.outputs],
                preview_version=record.preview_version if record.preview_jpeg is not None else 0,
                started_at=record.started_at,
                queue_position=queue_position,
            )

    def get_job_result_path(self, job_id: str) -> str | None:
        """Return the latest successful output path recorded for a job."""
        record = self._get_job(job_id)
        with record.lock:
            return record.result_path

    def get_job_review_path(self, job_id: str) -> str | None:
        """Return the temporary video of the small render under review, or ``None`` when no review of this job is pending."""
        record = self._get_job(job_id)
        with record.lock:
            return record.review_path if record.review is not None else None

    def get_job_preview(self, job_id: str) -> bytes | None:
        """Return the latest in-memory live preview JPEG for a job, if one is current."""
        record = self._get_job(job_id)
        with record.lock:
            return record.preview_jpeg

    def admit_exclusive(self, admit: Callable[[], None], *, busy_message: str) -> None:
        """Run *admit* atomically with the exclusive-job check; raise JobConflictError while a job is active."""
        with self._jobs_lock:
            if self._find_active_exclusive_job_id() is not None:
                raise JobConflictError(busy_message)
            admit()

    def has_active_jobs(self) -> bool:
        """Return whether any generation job is running or queued."""
        with self._jobs_lock:
            return self._find_active_exclusive_job_id() is not None

    def get_active_exclusive_job_snapshot(self) -> dict[str, Any] | None:
        """Return the active generation job (running, paused, or next in line), if one exists."""
        return self.list_jobs()["active_job"]

    def queue_job_control(self, job_id: str, action: str) -> dict[str, Any]:
        """Queue a supported control action for an active job; quitting a queued job removes it."""
        normalized = action.strip().lower()
        if normalized == "quit" and self.cancel_queued(job_id):
            return {"job_id": job_id, "action": normalized, "status": "cancelled"}
        record = self._get_job(job_id)
        with record.lock:
            if record.status in self._TERMINAL_STATUSES:
                raise UnsupportedJobControlError("This job is no longer running.")
            if record.status == QUEUED_STATUS:
                raise UnsupportedJobControlError("This job has not started yet.")
            if record.review is not None and not record.review_decided:
                if normalized not in (*REVIEW_DECISION_CONTROLS, "quit"):
                    raise UnsupportedJobControlError("A render is waiting for review: approve or discard it, or quit.")
                if record.control_signal is None or normalized not in record.supported_controls:
                    raise UnsupportedJobControlError(f"'{action}' is not available for this job.")
                # The first decision latches, so any later one is a conflict. The review itself stays on the record
                # until job_resumed, so a snapshot taken in between still shows the render instead of a plain pause.
                record.review_decided = True
                if normalized == "quit":
                    record.control_signal.queue_action("quit")
                else:
                    record.control_signal.set_review_decision(normalized)
            elif normalized in REVIEW_DECISION_CONTROLS:
                if record.review is not None:
                    raise UnsupportedJobControlError("This review has already been decided.")
                raise UnsupportedJobControlError("No render is waiting for review.")
            elif normalized == "resume":
                if normalized not in record.supported_controls or not record.paused or record.review is not None or record.control_signal is None:
                    raise UnsupportedJobControlError("This job is not paused.")
                record.control_signal.resume()
            else:
                if normalized not in record.supported_controls or record.control_signal is None:
                    raise UnsupportedJobControlError(f"'{action}' is not available for this job.")
                if record.paused and normalized != "quit":
                    raise UnsupportedJobControlError("Resume or quit the paused job before sending another control.")
                mapped = "skip" if normalized == "next" else normalized
                record.control_signal.queue_action(mapped)
                if record.paused and normalized == "quit":
                    record.control_signal.resume()

        self._publish_event(job_id, {"type": "control_queued", "action": normalized})
        return {"job_id": job_id, "action": normalized, "status": "queued"}

    async def stream_job_events(self, job_id: str, *, after_event_id: int | None = None) -> AsyncIterator[str]:
        """Yield a job's progress events as SSE frames."""
        try:
            record = self._get_job(job_id)
        except KeyError:
            with self._jobs_lock:
                if job_id not in self._pruned:
                    raise
                terminal_event = self._pruned[job_id][1]
            # Pruned job: replay only its terminal event so the client finishes normally.
            if terminal_event is not None:
                yield self._format_sse(terminal_event)
            return
        subscriber: queue.Queue[EventPayload] | None = None
        with record.lock:
            history = [dict(event) for event in record.history if after_event_id is None or event["event_id"] > after_event_id]
            if record.status not in self._TERMINAL_STATUSES:
                subscriber = queue.Queue()
                record.subscribers.add(subscriber)

        try:
            for event in history:
                yield self._format_sse(event)
            if subscriber is None:
                return

            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, self._heartbeat_seconds)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield self._format_sse(event)
                if event["type"] in self._TERMINAL_EVENT_TYPES:
                    return
        finally:
            if subscriber is not None:
                with record.lock:
                    record.subscribers.discard(subscriber)

    def shutdown(self) -> None:
        """Stop accepting work, cancel queued jobs, and tell the running job to quit; a paused or reviewing job is woken so it can end."""
        with self._jobs_lock:
            removed = [(job_id, self._jobs[job_id].future) for job_id in self._queue]
            self._queue.clear()
            running = self._jobs.get(self._claimed_job_id) if self._claimed_job_id is not None else None
        self._finish_removed(removed)
        if running is not None and running.control_signal is not None:
            running.control_signal.queue_action("quit")
            running.control_signal.resume()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _submit_job(
        self,
        *,
        job_type: str,
        target_factory: Callable[[Callable[[EventPayload], None]], None],
        exclusive: bool = False,
        control_signal: SkipSignal | None = None,
        supported_controls: tuple[str, ...] = (),
        context: dict[str, Any] | None = None,
        admission_check: Callable[[], None] | None = None,
    ) -> str:
        """Register and dispatch a background task; *admission_check* may raise JobConflictError under the job lock.

        Exclusive jobs join the queue. Registering, queueing and dispatching happen under one lock, so the worker
        receives jobs in queue order even when several requests submit at once.
        """
        record = _JobRecord(
            job_id=uuid.uuid4().hex,
            job_type=job_type,
            exclusive=exclusive,
            control_signal=control_signal,
            supported_controls=supported_controls,
            context=context or {},
        )
        progress_callback = self._make_progress_callback(record.job_id)
        with self._jobs_lock:
            self._prune_terminal_jobs_locked()
            if admission_check is not None:
                admission_check()
            self._jobs[record.job_id] = record
            self._publish_event(record.job_id, {"type": "job_submitted", "mode": job_type})
            if exclusive:
                self._queue.append(record.job_id)
            record.future = self._executor.submit(self._run_target, record.job_id, lambda: target_factory(progress_callback), release_memory=exclusive)
        return record.job_id

    def _run_target(self, job_id: str, target: Callable[[], None], *, release_memory: bool = True) -> None:
        """Claim the job, run its synchronous target, free accelerator memory, and publish terminal events."""
        if not self._claim(job_id):
            return
        failure_message: str | None = None
        try:
            with worker_runtime_context():
                target()
        except (Exception, SystemExit) as exc:
            full_text = str(exc).strip()
            failure_message = describe_refusal(exc) or exception_message(full_text) or f"{type(exc).__name__} stopped the generation worker."
            logger.exception("Job %s failed: %s", job_id, full_text or failure_message)

        # Released outside the except block: the active traceback pins the worker frames (and the loaded model).
        # Only generation jobs hold models; skipping the rest avoids clearing caches under a running generation.
        if release_memory:
            _release_accelerator_memory()
        if failure_message is not None:
            self._publish_event(job_id, {"type": FAILED_TERMINAL_EVENT, "message": failure_message})
            return

        record = self._get_job(job_id)
        with record.lock:
            status = record.status
        if status not in self._TERMINAL_STATUSES:
            self._publish_event(job_id, {"type": SUCCESS_TERMINAL_EVENT, "mode": record.job_type})

    def _claim(self, job_id: str) -> bool:
        """Take a generation job off the queue as it starts; return False when it was removed meanwhile."""
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if not record.exclusive:
                return True
            if job_id not in self._queue:
                return False
            self._queue.remove(job_id)
            self._claimed_job_id = job_id
            with record.lock:
                record.started_at = time.time()
                # Running from the moment it leaves the queue, so a Stop sent before job_started is published
                # reaches the job instead of being refused as "not started".
                record.status = "running"
        self._publish_event(job_id, {"type": STARTED_EVENT})
        return True

    def _run_image_request(
        self,
        *,
        request: ImageGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        quantize: int | None,
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
        enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
    ) -> None:
        """Run preflight, then load the image model inside the worker thread and run the batch."""
        plan = run_preflight(
            prompts_data,
            config,
            args,
            mode="image",
            model_family=request.model_family,
            enhance_by_set=enhance_by_set,
            control=control_signal,
            progress_callback=progress_callback,
            ensure_downloads=self._enhancer_download_hook(mode="image", prompts_data=prompts_data, args=args, progress_callback=progress_callback, control_signal=control_signal),
        )
        if plan.cancelled:
            return
        if not self._ensure_downloads(
            image_plan(model_ref, family=request.model_family, config=config, platform_key=sys.platform),
            mode="image",
            role="model",
            model=request.model_name or model_ref,
            total_iterations=len(plan.iterations),
            progress_callback=progress_callback,
            control_signal=control_signal,
        ):
            return
        backend, model, model_info = _load_image_model_with_progress(request, model_ref, quantize, progress_callback, control_signal, config=config)
        run_batch(
            backend,
            model,
            prompts_data,
            config,
            args,
            model_info=model_info,
            plan=plan,
            progress_callback=progress_callback,
            skip_signal=control_signal,
        )

    def _run_upscale_request(
        self,
        *,
        request: ImageGenerationRequest,
        workflow: GenerationWorkflow,
        model_ref: str,
        config: dict[str, Any],
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
    ) -> None:
        """Load the upscale model inside the worker thread (SeedVR2 or the recorded image model) and run the upscale workflow once."""
        if control_signal.pending() == "quit":
            progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 1})
            return
        if not self._ensure_downloads(
            self._upscale_download_plan(request, model_ref, config),
            mode="image",
            role="model",
            model=self._upscale_model_label(request, model_ref),
            total_iterations=1,
            progress_callback=progress_callback,
            control_signal=control_signal,
        ):
            return
        if request.upscale_method == UPSCALE_METHOD_SEEDVR2:
            backend, model = _load_seedvr2_with_progress(request.upscale_model, progress_callback, mode="image")
        else:
            backend, model, _model_info = _load_image_model_with_progress(request, model_ref, request.quantize, progress_callback, control_signal, config=config)
            if model is not None:
                backend.prepare_prompts(model, upscale_prompts(request), width=request.width, height=request.height, cancelled=lambda: control_signal.pending() == "quit")
        run_upscale(backend, model, request, workflow, progress_callback=progress_callback, skip_signal=control_signal)

    def _run_video_file_upscale_request(
        self,
        *,
        request: VideoFileUpscaleRequest,
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
    ) -> None:
        """Download and load SeedVR2 inside the worker thread, then upscale the video file once; Quit stops any step."""
        if control_signal.pending() == "quit":
            progress_callback({"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 1})
            return
        if not self._ensure_downloads(
            self._seedvr2_download_plan(request.upscale_model),
            mode="video",
            role="model",
            model=self._seedvr2_model_label(request.upscale_model),
            total_iterations=1,
            progress_callback=progress_callback,
            control_signal=control_signal,
        ):
            return
        upscaler, model = _load_seedvr2_with_progress(request.upscale_model, progress_callback, mode="video")
        run_video_file_upscale(upscaler, model, request, progress_callback=progress_callback, skip_signal=control_signal)

    def _run_video_request(
        self,
        *,
        request: VideoGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
        enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
        scene_by_set: dict[str, list[str | None]] | None = None,
    ) -> None:
        """Run preflight, then load the video model inside the worker thread and run the batch; Quit stops either."""
        require_ffmpeg()
        plan = run_preflight(
            prompts_data,
            config,
            args,
            mode="video",
            model_family=request.model_family,
            enhance_by_set=enhance_by_set,
            scene=getattr(args, "scene", None),
            scene_by_set=scene_by_set,
            control=control_signal,
            progress_callback=progress_callback,
            ensure_downloads=self._enhancer_download_hook(mode="video", prompts_data=prompts_data, args=args, progress_callback=progress_callback, control_signal=control_signal),
        )
        if plan.cancelled:
            return
        if control_signal.pending() == "quit":
            progress_callback({"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": len(plan.iterations)})
            return
        if not self._ensure_downloads(
            video_plan(model_ref, family=request.model_family, config=config, model_name=request.model_name or model_ref, upscale=bool(request.upscale)),
            mode="video",
            role="model",
            model=request.model_name or model_ref,
            total_iterations=len(plan.iterations),
            progress_callback=progress_callback,
            control_signal=control_signal,
        ):
            return
        progress_callback({"type": "model_loading", "mode": "video", "model": request.model_name or model_ref})
        backend = get_video_backend(request.model_family)
        workflow = build_video_workflow(args, enhance=plan.has_rewrites)
        lora_paths = request.lora_paths or []
        lora_weights = request.lora_weights or []
        loras = list(zip(lora_paths, lora_weights, strict=False)) or None
        model, model_info = backend.load_model(
            model_ref,
            mode="extend" if getattr(args, "extend_source", None) is not None else "i2v" if request.image_path else "t2v",
            end_image=getattr(args, "end_image_path", None) is not None,
            low_memory=getattr(args, "low_memory", True),
            loras=loras,
            **video_load_options(request.model_family, config, request.model_name or model_ref, upscale=bool(request.upscale), quiet=True),
        )
        run_video_batch(
            backend=backend,
            model=model,
            model_info=model_info,
            workflow=workflow,
            prompts_data=prompts_data,
            config=config,
            args=args,
            plan=plan,
            progress_callback=progress_callback,
            skip_signal=control_signal,
        )

    @staticmethod
    def _upscale_download_plan(request: ImageGenerationRequest, model_ref: str, config: dict[str, Any]) -> tuple[DownloadTarget, ...] | None:
        """Return the downloads an upscale job needs: the SeedVR2 tier's files or the creative refinement model's."""
        if request.upscale_method != UPSCALE_METHOD_SEEDVR2:
            return image_plan(model_ref, family=request.model_family, config=config, platform_key=sys.platform)
        return WebRunner._seedvr2_download_plan(request.upscale_model)

    @staticmethod
    def _upscale_model_label(request: ImageGenerationRequest, model_ref: str) -> str:
        """Return the model name an upscale job's download events show."""
        if request.upscale_method != UPSCALE_METHOD_SEEDVR2:
            return request.model_name or model_ref
        return WebRunner._seedvr2_model_label(request.upscale_model)

    @staticmethod
    def _seedvr2_download_plan(model_name: str) -> tuple[DownloadTarget, ...] | None:
        """Return the files SeedVR2 tier *model_name* needs on this machine, or ``None`` when it has no engine."""
        upscaler = get_seedvr2_upscaler()
        if upscaler is None:
            return None  # the load reports that this platform has no SeedVR2 engine
        return seedvr2_plan(model_name, models_dir=get_ziv_data_dir() / "models", weights=upscaler.weights)

    @staticmethod
    def _seedvr2_model_label(model_name: str) -> str:
        """Return the name of SeedVR2 tier *model_name* that a download event shows."""
        upscaler = get_seedvr2_upscaler()
        return seedvr2_tier_label(model_name, upscaler.weights) if upscaler is not None else str(model_name)

    def _enhancer_download_hook(
        self,
        *,
        mode: str,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        args: argparse.Namespace,
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
    ) -> Callable[[str, str | None], bool]:
        """Return the preflight hook that downloads the prompt enhancer ``(repo, revision)`` with ``role: "enhancer"`` events.

        The hook returns False when the job was cancelled (stopped or detached) while it downloaded; the cancellation
        event is already published by then.
        """
        total_iterations = sum(len(prompts) for prompts in prompts_data.values()) * args.runs

        def ensure(repo: str, revision: str | None) -> bool:
            return self._ensure_downloads(
                enhancer_plan(repo, revision),
                mode=mode,
                role="enhancer",
                model=repo,
                total_iterations=total_iterations,
                progress_callback=progress_callback,
                control_signal=control_signal,
            )

        return ensure

    def _ensure_downloads(
        self,
        targets: tuple[DownloadTarget, ...] | None,
        *,
        mode: str,
        role: str,
        model: str,
        total_iterations: int,
        progress_callback: Callable[[EventPayload], None],
        control_signal: SkipSignal,
    ) -> bool:
        """Download what *targets* still miss, publishing ``model_downloading`` events; return False when the job was cancelled.

        ``None`` (the plan could not be made) and no targets skip the step. A Stop while this job owns the download
        ends the job with ``download: "stopped"``; a Stop while it only waits on another download, with
        ``download: "detached"`` (the other download continues).

        Raises:
            RuntimeError: If a download fails; the message names the repo.
        """
        if not targets:
            return True

        def report(update: DownloadReport) -> None:
            progress_callback(
                {
                    "type": DOWNLOADING_EVENT,
                    "mode": mode,
                    "role": role,
                    "model": model,
                    "repo": update.repo,
                    "part": update.part,
                    "parts": update.parts,
                    "bytes_done": update.progress.bytes_done,
                    "bytes_total": update.progress.bytes_total,
                    "bytes_per_sec": update.progress.bytes_per_sec,
                    "download_eta_secs": update.progress.eta_secs,
                    "shared": update.shared,
                }
            )

        outcome = ensure_downloaded(targets, flights=self._download_flights, report=report, cancelled=lambda: control_signal.pending() == "quit")
        if outcome in (STOPPED, DETACHED):
            progress_callback({"type": "batch_cancelled", "mode": mode, "completed_iterations": 0, "total_iterations": total_iterations, "download": outcome})
            return False
        return True

    def _make_progress_callback(self, job_id: str) -> Callable[[EventPayload], None]:
        """Bind a job id to a runner progress callback."""
        return lambda event: self._publish_event(job_id, event)

    def _publish_event(self, job_id: str, event: EventPayload) -> None:
        """Record an event and fan it out to current subscribers."""
        record = self._get_job(job_id)
        preview_jpeg = None
        if "preview" in event:
            event = dict(event)
            preview_jpeg = _encode_preview_jpeg(event.pop("preview"))
        review_path = None
        if "review_path" in event:
            # A private key of the video runner: the file stays on the record and is never published.
            event = dict(event)
            review_path = event.pop("review_path")
        with record.lock:
            event = self._normalize_event(event)
            timestamp = time.time()
            if preview_jpeg is not None:
                record.preview_version += 1
                record.preview_jpeg = preview_jpeg
                event["preview_version"] = record.preview_version
            elif event["type"] in _PREVIEW_RESET_EVENT_TYPES:
                record.preview_jpeg = None
            record.review = self._review_after_event(record.review, event)
            if record.review is None:
                record.review_path = None
                record.review_decided = False
            elif event["type"] == "job_paused":
                record.review_decided = False
                if review_path is not None:
                    record.review_path = review_path
            if event["type"] == "prompt_started":
                record.prompt_progress = {key: event[key] for key in ("prompt", "run_index", "total_runs", "ran_iterations", "total_iterations", "enhance_status") if key in event}
            elif event["type"] == "prompt_enhanced" and "enhanced_prompt" in event:
                # Kept with the prompt progress so later events and reconnecting clients still see it.
                record.prompt_progress = {**record.prompt_progress, "enhanced_prompt": event["enhanced_prompt"]}
            enriched_event = {
                **record.prompt_progress,
                "event_id": record.next_event_id,
                "job_id": job_id,
                "job_type": record.job_type,
                "timestamp": timestamp,
                "elapsed_secs": event.get("elapsed_secs", max(0.0, timestamp - (record.started_at or record.created_at))),
                **event,
            }
            if enriched_event.get("eta_secs") is not None:
                record.last_eta_secs = enriched_event["eta_secs"]
            elif record.last_eta_secs is not None and enriched_event["type"] not in self._TERMINAL_EVENT_TYPES:
                enriched_event["eta_secs"] = record.last_eta_secs
            if enriched_event["type"] == "generation_finished" and enriched_event.get("status") == "success":
                record.result_path = enriched_event.get("output_path")
                asset_payload = _output_asset_payload(record.context, enriched_event.get("output_path"))
                if asset_payload is not None and all(output.get("id") != asset_payload.get("id") for output in record.outputs):
                    record.outputs.append(asset_payload)
                    enriched_event["asset"] = asset_payload
            if enriched_event["type"] == SUCCESS_TERMINAL_EVENT and "outputs" not in enriched_event:
                enriched_event["outputs"] = [dict(output) for output in record.outputs]
            record.next_event_id += 1
            if enriched_event["type"] == DOWNLOADING_EVENT:
                # Only the latest download progress is worth replaying to a client that connects later.
                record.history[:] = [past for past in record.history if past["type"] != DOWNLOADING_EVENT]
            record.history.append(enriched_event)
            if len(record.history) > self._max_history_events:
                del record.history[: len(record.history) - self._max_history_events]
            record.event_count += 1
            record.last_event = dict(enriched_event)
            record.status = self._status_from_event(record.status, enriched_event["type"])
            if enriched_event["type"] == "job_paused":
                record.paused = True
            elif enriched_event["type"] == "job_resumed":
                record.paused = False
            if record.status in self._TERMINAL_STATUSES:
                record.paused = False
                record.last_eta_secs = None
            if enriched_event["type"] == "generation_finished" and enriched_event.get("status") == "failed":
                record.failed_runs += 1
                if enriched_event.get("message"):
                    record.last_failure = str(enriched_event["message"])
            first_terminal = record.status in self._TERMINAL_STATUSES and record.completed_at is None
            if first_terminal:
                record.completed_at = enriched_event["timestamp"]
            log_args: dict[str, Any] | None = None
            if first_terminal and self._job_log is not None and is_loggable(record.job_type, record.started_at):
                # Plain copies only: the entry is built after every lock is released.
                log_args = {
                    "job_id": record.job_id,
                    "job_type": record.job_type,
                    "context": dict(record.context),
                    "status": record.status,
                    "created_at": record.created_at,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "outputs": [dict(output) for output in record.outputs],
                    "error": (enriched_event.get("message") if record.status == "failed" else None) or record.last_failure,
                    "failed_runs": record.failed_runs,
                }
            subscribers = list(record.subscribers)

        for subscriber in subscribers:
            subscriber.put(enriched_event)

        if enriched_event["type"] in self._TERMINAL_EVENT_TYPES:
            with self._jobs_lock:
                if self._claimed_job_id == job_id:
                    self._claimed_job_id = None
                self._prune_terminal_jobs_locked(exclude_job_id=job_id)

        self._write_job_log(log_args)

    def _write_job_log(self, log_args: dict[str, Any] | None) -> None:
        """Append the finished job to the log; any failure becomes a warning and never touches the job.

        Runs last in ``_publish_event``, holding no lock, so a slow or failing write cannot delay the terminal event,
        the release of the claim or the start of the next queued job.
        """
        if log_args is None or self._job_log is None:
            return
        try:
            self._job_log.append(build_job_log_entry(**log_args))
        except Exception as exc:  # noqa: BLE001 - best effort by design
            warnings.warn(f"Could not write the job log: {exc}", stacklevel=2)

    @staticmethod
    def _review_after_event(current: dict[str, Any] | None, event: EventPayload) -> dict[str, Any] | None:
        """Return the review a job shows after *event*: ``job_paused`` may start one; the events that end the paused render clear it."""
        if event["type"] == "job_paused" and isinstance(event.get("review"), dict):
            return dict(event["review"])
        if event["type"] in _REVIEW_RESET_EVENT_TYPES:
            return None
        return current

    def _get_job(self, job_id: str) -> _JobRecord:
        """Look up a job record or raise KeyError."""
        with self._jobs_lock:
            self._prune_terminal_jobs_locked()
            return self._jobs[job_id]

    def _find_active_exclusive_job_id(self) -> str | None:
        """Return the currently active exclusive job id, if one exists."""
        for record in self._jobs.values():
            if record.exclusive and record.status not in self._TERMINAL_STATUSES:
                return record.job_id
        return None

    def _prune_terminal_jobs_locked(self, *, exclude_job_id: str | None = None) -> None:
        """Remove expired terminal jobs without touching active jobs or live subscribers."""
        now = time.time()
        removable = [
            record for record in self._jobs.values() if record.job_id != exclude_job_id and record.status in self._TERMINAL_STATUSES and not record.subscribers and record.completed_at is not None
        ]
        expired_ids = {record.job_id for record in removable if now - record.completed_at >= self._terminal_retention_seconds}
        retained_terminal = [record for record in removable if record.job_id not in expired_ids]
        if self._max_terminal_jobs < len(retained_terminal):
            overflow = len(retained_terminal) - self._max_terminal_jobs
            retained_terminal.sort(key=lambda record: (record.completed_at or 0.0, record.created_at, record.job_id))
            expired_ids.update(record.job_id for record in retained_terminal[:overflow])
        for job_id in expired_ids:
            record = self._jobs.pop(job_id, None)
            if record is None:
                continue
            terminal_event = next((dict(event) for event in reversed(record.history) if event["type"] in self._TERMINAL_EVENT_TYPES), None)
            self._pruned[job_id] = (self._snapshot_record(record), terminal_event)
            while len(self._pruned) > self._max_pruned_jobs:
                self._pruned.popitem(last=False)

    def _format_sse(self, event: EventPayload) -> str:
        """Serialize a structured event as a single SSE frame."""
        payload = json.dumps(event, default=str)
        return f"id: {event['event_id']}\nevent: {event['type']}\ndata: {payload}\n\n"

    def _status_from_event(self, current_status: str, event_type: str) -> str:
        """Map runner events to coarse job states."""
        if event_type == FAILED_TERMINAL_EVENT:
            return "failed"
        if event_type == CANCELLED_TERMINAL_EVENT:
            return "cancelled"
        if event_type == SUCCESS_TERMINAL_EVENT:
            return "completed"
        if event_type in {"job_submitted"}:
            return QUEUED_STATUS
        if event_type == STARTED_EVENT:
            return "running"
        if event_type == "job_paused":
            return "paused"
        if event_type == "job_resumed":
            return "running"
        if event_type in {"control_queued"}:
            return current_status
        if current_status in self._TERMINAL_STATUSES:
            return current_status
        return "running"

    def _normalize_event(self, event: EventPayload) -> EventPayload:
        """Translate internal runner event names to the public lifecycle contract."""
        if event.get("type") == "batch_cancelled":
            return {**event, "type": CANCELLED_TERMINAL_EVENT}
        if event.get("type") == "batch_failed":
            return {**event, "type": FAILED_TERMINAL_EVENT}
        return event


def _load_image_model_with_progress(
    request: ImageGenerationRequest,
    model_ref: str,
    quantize: int | None,
    progress_callback: Callable[[EventPayload], None],
    control_signal: SkipSignal,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Load the request's image model (stored quants, LoRAs) and report ``model_loading`` events.

    With *config*, a model built on first use (Ideogram 4 Instant, MOSS-Image) is built first when it is missing (the ``building`` phase).

    Returns:
        ``(backend, model, model_info)``; the model is ``None`` when the user quit during the build.
    """
    model_label = request.model_name or model_ref
    progress_callback({"type": "model_loading", "mode": "image", "model": model_label})
    backend = get_backend()
    if config is not None and not ensure_built_model(
        model_ref,
        config,
        platform_key=sys.platform,
        on_phase=lambda phase: progress_callback({"type": "model_loading", "mode": "image", "model": model_label, "phase": phase}),
        cancelled=lambda: control_signal.pending() == "quit",
    ):
        return backend, None, detect_image_model(model_ref)
    model, model_info = load_image_model(
        backend,
        model_ref,
        quantize=quantize,
        models_dir=get_ziv_data_dir() / "models",
        model_name=request.model_name,
        lora_paths=request.lora_paths,
        lora_weights=request.lora_weights,
        on_phase=lambda phase: progress_callback({"type": "model_loading", "mode": "image", "model": model_label, "phase": phase, "quantize": quantize}),
        cancelled=lambda: control_signal.pending() == "quit",
        release_memory=_release_accelerator_memory,
    )
    return backend, model, model_info


def _load_seedvr2_with_progress(model_name: str, progress_callback: Callable[[EventPayload], None], *, mode: str) -> tuple[Any, Any]:
    """Load SeedVR2 tier *model_name* the way the engine keeps it and report ``model_loading`` events for *mode*.

    The events carry ``replaces_download`` where the engine stores a copy, because the first load deletes the
    downloaded files once the copy is saved. There is no cancel hook: a quit during the load is consumed when
    the workflow starts.

    Returns:
        ``(upscaler, model)``.

    Raises:
        RuntimeError: If this platform has no SeedVR2 engine.
    """
    upscaler = get_seedvr2_upscaler()
    if upscaler is None:
        raise RuntimeError("SeedVR2 upscaling is not available on this platform.")
    model_label = seedvr2_tier_label(model_name, upscaler.weights)
    replaces_download = upscaler.weights.stores_copy
    model = load_seedvr2(
        upscaler,
        model_name,
        models_dir=get_ziv_data_dir() / "models",
        on_phase=lambda phase: progress_callback({"type": "model_loading", "mode": mode, "model": model_label, "phase": phase, "quantize": SEEDVR2_BITS, "replaces_download": replaces_download}),
    )
    return upscaler, model


def _release_accelerator_memory() -> None:
    """Free the finished job's model memory; a cleanup failure must never change the job outcome."""
    try:
        release_accelerator_memory()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not release accelerator memory after a web job: {exc}", stacklevel=2)


def _encode_preview_jpeg(preview: Any) -> bytes | None:
    """Encode a live preview image as JPEG bytes; anything else, or a failed encode, yields ``None``.

    Previews are best-effort: this runs inside the denoising loop, so an encode error must not fail the job.
    """
    if not isinstance(preview, Image.Image):
        return None
    buffer = io.BytesIO()
    try:
        preview.convert("RGB").save(buffer, format="JPEG", quality=85)
    except Exception:  # noqa: BLE001 - previews are best-effort
        return None
    return buffer.getvalue()


def _output_asset_payload(context: dict[str, Any], output_path: Any) -> dict[str, Any] | None:
    if not isinstance(output_path, str) or not output_path.strip():
        return None
    output_dir = context.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        output_dir = str(Path(output_path).expanduser().parent)
    asset = gallery_asset_for_output_path(output_dir, output_path)
    if asset is None:
        return None
    return gallery_asset_to_json(asset, load_web_config())
