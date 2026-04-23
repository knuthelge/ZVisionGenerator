"""Run synchronous generation batches in background workers for the Web UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from zvisiongenerator.backends import get_backend, get_video_backend
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import build_video_workflow

type EventPayload = dict[str, Any]


class JobConflictError(RuntimeError):
    """Raised when a new exclusive generation job is submitted while one is active."""


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


_WORKER_STREAMS = _MutedWorkerStreams()


@dataclass(slots=True)
class _JobRecord:
    """Hold background job state and SSE subscribers."""

    job_id: str
    job_type: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    future: Future[None] | None = None
    history: list[EventPayload] = field(default_factory=list)
    subscribers: set[queue.Queue[EventPayload]] = field(default_factory=set)
    next_event_id: int = 1
    exclusive: bool = False
    control_signal: SkipSignal | None = None
    supported_controls: tuple[str, ...] = ()
    result_path: str | None = None
    paused: bool = False
    last_eta_secs: float | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class WebRunner:
    """Execute synchronous runners in a thread pool and surface SSE progress."""

    _TERMINAL_EVENT_TYPES = frozenset({"batch_completed", "batch_cancelled", "job_completed", "job_failed"})
    _TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed"})

    def __init__(self, *, max_workers: int = 2, heartbeat_seconds: float = 10.0) -> None:
        """Create the thread-backed web runner facade."""
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ziv-web-runner")
        self._heartbeat_seconds = heartbeat_seconds
        self._jobs: dict[str, _JobRecord] = {}
        self._jobs_lock = threading.RLock()

    def submit_image_job(
        self,
        *,
        backend: ImageBackend,
        model: Any,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_info: ImageModelInfo,
    ) -> str:
        """Run the image batch loop on a worker thread."""
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="image",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=("next", "pause", "repeat", "quit"),
            target_factory=lambda progress_callback: run_batch(
                backend,
                model,
                prompts_data,
                config,
                args,
                model_info,
                progress_callback=progress_callback,
                enable_interactive_controls=False,
                skip_signal=control_signal,
            ),
        )

    def submit_video_job(
        self,
        *,
        backend: Any,
        model: Any,
        model_info: VideoModelInfo,
        workflow: GenerationWorkflow,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
    ) -> str:
        """Run the video batch loop on a worker thread."""
        return self._submit_job(
            job_type="video",
            exclusive=True,
            target_factory=lambda progress_callback: run_video_batch(
                backend=backend,
                model=model,
                model_info=model_info,
                workflow=workflow,
                prompts_data=prompts_data,
                config=config,
                args=args,
                progress_callback=progress_callback,
            ),
        )

    def submit_image_request_job(
        self,
        *,
        request: ImageGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        quantize: int | None = None,
    ) -> str:
        """Load the image model and run the batch loop on a worker thread."""
        control_signal = SkipSignal()
        return self._submit_job(
            job_type="image",
            exclusive=True,
            control_signal=control_signal,
            supported_controls=("next", "pause", "repeat", "quit"),
            target_factory=lambda progress_callback: self._run_image_request(
                request=request,
                prompts_data=prompts_data,
                config=config,
                args=args,
                model_ref=model_ref,
                quantize=quantize,
                progress_callback=progress_callback,
                control_signal=control_signal,
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
    ) -> str:
        """Load the video model and run the batch loop on a worker thread."""
        return self._submit_job(
            job_type="video",
            exclusive=True,
            target_factory=lambda progress_callback: self._run_video_request(
                request=request,
                prompts_data=prompts_data,
                config=config,
                args=args,
                model_ref=model_ref,
                progress_callback=progress_callback,
            ),
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
        """Return serializable state for a tracked job."""
        record = self._get_job(job_id)
        with record.lock:
            last_event = dict(record.history[-1]) if record.history else None
            return {
                "job_id": record.job_id,
                "job_type": record.job_type,
                "status": record.status,
                "created_at": record.created_at,
                "completed_at": record.completed_at,
                "event_count": len(record.history),
                "last_event": last_event,
                "supports_controls": list(record.supported_controls),
                "paused": record.paused,
                "result_path": record.result_path,
            }

    def get_job_result_path(self, job_id: str) -> str | None:
        """Return the latest successful output path recorded for a job."""
        record = self._get_job(job_id)
        with record.lock:
            return record.result_path

    def get_active_exclusive_job_snapshot(self) -> dict[str, Any] | None:
        """Return the currently running exclusive job, if one exists."""
        with self._jobs_lock:
            for record in self._jobs.values():
                if not record.exclusive or record.status in self._TERMINAL_STATUSES:
                    continue
                with record.lock:
                    return {
                        "job_id": record.job_id,
                        "job_type": record.job_type,
                        "status": record.status,
                    }
        return None

    def queue_job_control(self, job_id: str, action: str) -> dict[str, Any]:
        """Queue a supported control action for an active image job."""
        record = self._get_job(job_id)
        normalized = action.strip().lower()
        with record.lock:
            if record.status in self._TERMINAL_STATUSES:
                raise UnsupportedJobControlError("This job is no longer running.")
            if normalized == "resume":
                if not record.paused or record.control_signal is None:
                    raise UnsupportedJobControlError("This job is not paused.")
                record.control_signal.resume()
            else:
                if normalized not in record.supported_controls or record.control_signal is None:
                    raise UnsupportedJobControlError(f"'{action}' is not available for this job.")
                if record.paused and normalized != "quit":
                    raise UnsupportedJobControlError("Resume or quit the paused job before sending another control.")
                mapped = "skip" if normalized == "next" else normalized
                record.control_signal.queue_action(mapped)

        self._publish_event(job_id, {"type": "control_queued", "action": normalized})
        return {"job_id": job_id, "action": normalized, "status": "queued"}

    async def stream_job_events(self, job_id: str) -> AsyncIterator[str]:
        """Yield a job's progress events as SSE frames."""
        record = self._get_job(job_id)
        subscriber: queue.Queue[EventPayload] | None = None
        with record.lock:
            history = [dict(event) for event in record.history]
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
        """Stop accepting work and tear down worker threads."""
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _submit_job(
        self,
        *,
        job_type: str,
        target_factory: Callable[[Callable[[EventPayload], None]], None],
        exclusive: bool = False,
        control_signal: SkipSignal | None = None,
        supported_controls: tuple[str, ...] = (),
    ) -> str:
        """Register and dispatch a background task."""
        record = _JobRecord(
            job_id=uuid.uuid4().hex,
            job_type=job_type,
            exclusive=exclusive,
            control_signal=control_signal,
            supported_controls=supported_controls,
        )
        with self._jobs_lock:
            if exclusive:
                active_job_id = self._find_active_exclusive_job_id()
                if active_job_id is not None:
                    raise JobConflictError(f"Job '{active_job_id}' is already running. Wait for it to finish before starting another.")
            self._jobs[record.job_id] = record
        self._publish_event(record.job_id, {"type": "job_submitted", "mode": job_type})
        progress_callback = self._make_progress_callback(record.job_id)
        record.future = self._executor.submit(self._run_target, record.job_id, lambda: target_factory(progress_callback))
        return record.job_id

    def _run_target(self, job_id: str, target: Callable[[], None]) -> None:
        """Wrap a synchronous worker target and publish terminal events."""
        try:
            with _WORKER_STREAMS.mute_current_thread():
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TQDM_DISABLE", "1")
                target()
        except Exception as exc:
            self._publish_event(job_id, {"type": "job_failed", "message": str(exc)})
            return

        record = self._get_job(job_id)
        with record.lock:
            status = record.status
        if status not in self._TERMINAL_STATUSES:
            self._publish_event(job_id, {"type": "job_completed", "mode": record.job_type})

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
    ) -> None:
        """Load the image model inside the worker thread, then run the batch."""
        progress_callback({"type": "model_loading", "mode": "image", "model": request.model_name or model_ref})
        backend = get_backend()
        model, model_info = backend.load_model(
            model_ref,
            quantize=quantize,
            precision="bfloat16",
            lora_paths=request.lora_paths,
            lora_weights=request.lora_weights,
        )
        run_batch(
            backend,
            model,
            prompts_data,
            config,
            args,
            model_info=model_info,
            progress_callback=progress_callback,
            enable_interactive_controls=False,
            skip_signal=control_signal,
        )

    def _run_video_request(
        self,
        *,
        request: VideoGenerationRequest,
        prompts_data: dict[str, list[tuple[str, str | None]]],
        config: dict[str, Any],
        args: argparse.Namespace,
        model_ref: str,
        progress_callback: Callable[[EventPayload], None],
    ) -> None:
        """Load the video model inside the worker thread, then run the batch."""
        progress_callback({"type": "model_loading", "mode": "video", "model": request.model_name or model_ref})
        ensure_ffmpeg()
        backend = get_video_backend(request.model_family)
        workflow = build_video_workflow(args)
        lora_paths = request.lora_paths or []
        lora_weights = request.lora_weights or []
        loras = list(zip(lora_paths, lora_weights, strict=False)) or None
        load_kwargs: dict[str, Any] = {}
        if request.upscale:
            load_kwargs["upscale"] = True
        model, model_info = backend.load_model(
            model_ref,
            mode="i2v" if request.image_path else "t2v",
            low_memory=getattr(args, "low_memory", True),
            loras=loras,
            **load_kwargs,
        )
        run_video_batch(
            backend=backend,
            model=model,
            model_info=model_info,
            workflow=workflow,
            prompts_data=prompts_data,
            config=config,
            args=args,
            progress_callback=progress_callback,
        )

    def _make_progress_callback(self, job_id: str) -> Callable[[EventPayload], None]:
        """Bind a job id to a runner progress callback."""
        return lambda event: self._publish_event(job_id, event)

    def _publish_event(self, job_id: str, event: EventPayload) -> None:
        """Record an event and fan it out to current subscribers."""
        record = self._get_job(job_id)
        with record.lock:
            timestamp = time.time()
            enriched_event = {
                "event_id": record.next_event_id,
                "job_id": job_id,
                "job_type": record.job_type,
                "timestamp": timestamp,
                "elapsed_secs": event.get("elapsed_secs", max(0.0, timestamp - record.created_at)),
                **event,
            }
            if enriched_event.get("eta_secs") is not None:
                record.last_eta_secs = enriched_event["eta_secs"]
            elif record.last_eta_secs is not None and enriched_event["type"] not in self._TERMINAL_EVENT_TYPES:
                enriched_event["eta_secs"] = record.last_eta_secs
            record.next_event_id += 1
            record.history.append(enriched_event)
            record.status = self._status_from_event(record.status, enriched_event["type"])
            if enriched_event["type"] == "generation_finished" and enriched_event.get("status") == "success":
                record.result_path = enriched_event.get("output_path")
            if enriched_event["type"] == "job_paused":
                record.paused = True
            elif enriched_event["type"] == "job_resumed":
                record.paused = False
            if record.status in self._TERMINAL_STATUSES:
                record.paused = False
                record.last_eta_secs = None
            if record.status in self._TERMINAL_STATUSES and record.completed_at is None:
                record.completed_at = enriched_event["timestamp"]
            subscribers = list(record.subscribers)

        for subscriber in subscribers:
            subscriber.put(enriched_event)

    def _get_job(self, job_id: str) -> _JobRecord:
        """Look up a job record or raise KeyError."""
        with self._jobs_lock:
            return self._jobs[job_id]

    def _find_active_exclusive_job_id(self) -> str | None:
        """Return the currently active exclusive job id, if one exists."""
        for record in self._jobs.values():
            if record.exclusive and record.status not in self._TERMINAL_STATUSES:
                return record.job_id
        return None

    def _format_sse(self, event: EventPayload) -> str:
        """Serialize a structured event as a single SSE frame."""
        payload = json.dumps(event, default=str)
        return f"id: {event['event_id']}\nevent: {event['type']}\ndata: {payload}\n\n"

    def _status_from_event(self, current_status: str, event_type: str) -> str:
        """Map runner events to coarse job states."""
        if event_type in {"job_failed"}:
            return "failed"
        if event_type in {"batch_cancelled"}:
            return "cancelled"
        if event_type in {"batch_completed", "job_completed"}:
            return "completed"
        if event_type in {"job_submitted"}:
            return "queued"
        if event_type in {"control_queued"}:
            return current_status
        if current_status in self._TERMINAL_STATUSES:
            return current_status
        return "running"
