"""Share structured workflow progress event helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.ffmpeg import strip_ffmpeg_banner


type ProgressCallback = Callable[[dict[str, Any]], None]

# Longest failure message, in characters, that a runner sends to clients.
FAILURE_MESSAGE_LIMIT = 500

# Longest "<head>" of an exception message whose head is kept when the message is shortened.
_EXCEPTION_HEAD_MAX = 200


def one_line_message(text: str, *, limit: int = FAILURE_MESSAGE_LIMIT) -> str:
    """Collapse *text* to a single line of at most *limit* characters.

    Whitespace runs become single spaces and the ends are stripped. Text longer
    than *limit* is cut to ``limit - 1`` characters followed by an ellipsis.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def failure_with_detail(prefix: str, detail: str, *, limit: int = FAILURE_MESSAGE_LIMIT) -> str:
    """Return ``"<prefix>: <detail>"`` as one line of at most *limit* characters, keeping the end of *detail*.

    Tool output states the cause last, so a *detail* that does not fit is cut from the front
    and starts with an ellipsis. FFmpeg banner lines are dropped unless nothing else remains.
    The *prefix* is always kept whole.
    """
    tail = " ".join(strip_ffmpeg_banner(detail).split())
    head = one_line_message(prefix, limit=limit)
    if not tail:
        return head
    room = limit - len(head) - 2
    if room < 2:
        return head
    if len(tail) > room:
        tail = "…" + tail[-(room - 1) :]
    return f"{head}: {tail}"


def exception_message(text: str, *, limit: int = FAILURE_MESSAGE_LIMIT) -> str:
    """Shorten exception *text* to one line of at most *limit* characters, keeping its ending.

    Text that fits is only collapsed to one line. Longer text shaped ``"<head>: <rest>"``
    with a short first-line head keeps the head and the end of *rest*, since tool errors
    state the cause last. The split is on the first ``": "`` (colon and space), so URL schemes
    and Windows drive letters stay in the head. Any other long text keeps its beginning.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    first_line = text.strip().split("\n", 1)[0]
    head, separator, _ = first_line[:_EXCEPTION_HEAD_MAX].partition(": ")
    head = head.strip()
    if separator and head:
        rest = text.strip().partition(": ")[2]
        return failure_with_detail(head, rest, limit=limit)
    return one_line_message(text, limit=limit)


def preview_steps(total_steps: int) -> frozenset[int]:
    """Return the 1-based steps that should emit a live preview: every step but the last.

    The final step is excluded because the finished image follows immediately.
    """
    return frozenset(range(1, total_steps))


def emit_progress(progress_callback: ProgressCallback | None, event_type: str, **payload: Any) -> None:
    """Send a structured progress event when a callback is configured."""
    if progress_callback is None:
        return
    progress_callback({"type": event_type, **payload})


def make_prompt_enhanced_callback(progress_callback: ProgressCallback | None, **context: Any) -> Callable[[str], None] | None:
    """Return a callback that emits ``prompt_enhanced`` with the enhanced text and *context*."""
    if progress_callback is None:
        return None

    def _callback(enhanced_prompt: str) -> None:
        emit_progress(progress_callback, "prompt_enhanced", **context, enhanced_prompt=enhanced_prompt)

    return _callback


def emit_generation_finished(
    progress_callback: ProgressCallback | None,
    *,
    event_context: dict[str, Any],
    status: str,
    filename: str | None = None,
    generation_time: float | None = None,
    output_path: Any = None,
    message: str | None = None,
) -> None:
    """Emit the shared terminal payload for one generation attempt.

    *message* is included only when it is a non-empty string.
    """
    payload = {**event_context, "status": status}
    if filename is not None:
        payload["filename"] = filename
    if generation_time is not None:
        payload["generation_time"] = generation_time
    if output_path is not None:
        payload["output_path"] = output_path
    if message:
        payload["message"] = message
    emit_progress(
        progress_callback,
        "generation_finished",
        **payload,
    )


def make_step_progress_callback(
    progress_callback: ProgressCallback | None,
    *,
    mode: str,
    run_index: int,
    total_runs: int,
    ran_iterations: int,
    total_iterations: int,
    set_name: str,
    prompt_index: int,
    total_prompts: int,
) -> ProgressCallback | None:
    """Bind generation context to low-level denoising step events."""
    if progress_callback is None:
        return None

    def _callback(payload: dict[str, Any]) -> None:
        emit_progress(
            progress_callback,
            "step_progress",
            mode=mode,
            run_index=run_index,
            total_runs=total_runs,
            ran_iterations=ran_iterations,
            total_iterations=total_iterations,
            set_name=set_name,
            prompt_index=prompt_index,
            total_prompts=total_prompts,
            **payload,
        )

    return _callback


def run_workflow_with_progress(
    workflow: Any,
    request: Any,
    artifacts: Any,
    *,
    progress_callback: ProgressCallback | None,
    event_context: dict[str, Any],
) -> StageOutcome:
    """Run a workflow while emitting structured stage progress events."""
    stages = workflow.stages if isinstance(getattr(workflow, "stages", None), list | tuple) else None
    if stages is None:
        stage_name = getattr(workflow, "name", "workflow")
        emit_progress(progress_callback, "workflow_started", total_stages=1, stage_name=stage_name, **event_context)
        emit_progress(progress_callback, "workflow_stage_started", stage_index=1, total_stages=1, stage_name=stage_name, **event_context)
        outcome = workflow.run(request, artifacts)
        emit_progress(progress_callback, "workflow_stage_completed", stage_index=1, total_stages=1, stage_name=stage_name, outcome=outcome.name.lower(), **event_context)
        emit_progress(progress_callback, "workflow_finished", total_stages=1, completed_stages=1, stage_name=stage_name, status=outcome.name.lower(), **event_context)
        return outcome

    total_stages = len(stages)
    if total_stages == 0:
        return StageOutcome.success

    # stage_names lets clients show stages that have not started yet (e.g. an upscale pass after generation).
    stage_names = [_stage_label(stage) for stage in stages]
    emit_progress(progress_callback, "workflow_started", total_stages=total_stages, stage_name=stage_names[0], stage_names=stage_names, **event_context)

    for stage_index, stage in enumerate(stages, start=1):
        stage_name = _stage_label(stage)
        stage_request = replace(
            request,
            step_callback=_wrap_stage_step_callback(
                request.step_callback,
                stage_index=stage_index,
                total_stages=total_stages,
                stage_name=stage_name,
            ),
        )
        emit_progress(progress_callback, "workflow_stage_started", stage_index=stage_index, total_stages=total_stages, stage_name=stage_name, **event_context)
        outcome = stage(stage_request, artifacts)
        emit_progress(progress_callback, "workflow_stage_completed", stage_index=stage_index, total_stages=total_stages, stage_name=stage_name, outcome=outcome.name.lower(), **event_context)
        if outcome is not StageOutcome.success:
            emit_progress(progress_callback, "workflow_finished", total_stages=total_stages, completed_stages=stage_index, stage_name=stage_name, status=outcome.name.lower(), **event_context)
            return outcome

    emit_progress(progress_callback, "workflow_finished", total_stages=total_stages, completed_stages=total_stages, stage_name=_stage_label(stages[-1]), status="success", **event_context)
    return StageOutcome.success


def _stage_label(stage: Callable[..., StageOutcome]) -> str:
    """Convert a stage callable name into a stable progress label."""
    stage_name = getattr(stage, "__name__", None) or getattr(stage, "_mock_name", None) or stage.__class__.__name__
    return stage_name.removesuffix("_stage")


def _wrap_stage_step_callback(
    step_callback: ProgressCallback | None,
    *,
    stage_index: int,
    total_stages: int,
    stage_name: str,
) -> ProgressCallback | None:
    """Attach workflow-stage metadata to low-level denoiser step events."""
    if step_callback is None:
        return None

    def _callback(payload: dict[str, Any]) -> None:
        step_callback(
            {
                "workflow_stage_index": stage_index,
                "workflow_total_stages": total_stages,
                "workflow_stage_name": stage_name,
                **payload,
            }
        )

    return _callback
