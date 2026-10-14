"""Video batch orchestration — runs × sets × prompts loop for video generation."""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.core.progress_events import emit_generation_finished as _emit_generation_finished
from zvisiongenerator.core.progress_events import emit_progress as _emit_progress
from zvisiongenerator.core.progress_events import make_prompt_enhanced_callback as _make_prompt_enhanced_callback
from zvisiongenerator.core.progress_events import make_step_progress_callback as _make_step_progress_callback
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.review import REVIEW_PROMPT, await_review_decision, review_enabled
from zvisiongenerator.utils.video_model_detect import VideoModelInfo

if TYPE_CHECKING:
    from pathlib import Path

    from zvisiongenerator.utils.interactive import SkipSignal

logger = logging.getLogger(__name__)


class _VideoCancelled(Exception):
    """Raised from a step callback to stop a running video when Quit is queued."""


class _VideoDiscarded(Exception):
    """Raised from the review callable when the user discards the small render, so the upscale is skipped."""


def _make_video_review(
    skip_signal: SkipSignal,
    *,
    progress_callback: ProgressCallback | None,
    ran_iterations: int,
    completed_iterations: int,
    total_iterations: int,
) -> Callable[[Path], None]:
    """Build the ``review`` callable of one video generation.

    The callable is called with the path of the decoded half-size render, shows it and waits for the decision. It
    returns on approve, raises ``_VideoDiscarded`` on discard and ``_VideoCancelled`` on Quit (which stays queued).

    Args:
        skip_signal: The job's control signal; its review state carries the decision.
        progress_callback: Receives ``job_paused`` (with ``review`` and, under the private key ``review_path``, the file
            for the web runner to serve) and ``job_resumed``. Without one the job runs in a terminal: the path is printed.
        ran_iterations: The 1-based iteration under review.
        completed_iterations: Finished iterations when the review starts.
        total_iterations: Iterations in the batch.
    """

    def _review(path: Path) -> None:
        if progress_callback is None:
            print(f"\nSmall render: {path}", flush=True)
            print(REVIEW_PROMPT, flush=True)
        decision = await_review_decision(
            skip_signal,
            progress_callback=progress_callback,
            mode="video",
            completed_iterations=completed_iterations,
            total_iterations=total_iterations,
            review={"iteration": ran_iterations},
            **({} if progress_callback is None else {"review_path": str(path)}),
        )
        if decision == "quit":
            raise _VideoCancelled
        if decision == "discard":
            raise _VideoDiscarded

    return _review


def _quit_requested(skip_signal: SkipSignal | None) -> bool:
    """Return whether Quit is queued on *skip_signal*."""
    return skip_signal is not None and skip_signal.pending() == "quit"


def _cancellable_step_callback(step_callback: ProgressCallback | None, skip_signal: SkipSignal | None) -> ProgressCallback | None:
    """Wrap *step_callback* so a queued Quit stops the denoising loop at its next step.

    Both video backends call the step callback from every denoising step, including the upscale refinement, so raising
    there ends the generation. Only the latent upsample between the two passes is a single short step that runs to
    completion, so a Quit during it applies at the first refinement step.
    """
    if skip_signal is None:
        return step_callback

    def _callback(payload: dict[str, Any]) -> None:
        if _quit_requested(skip_signal):
            raise _VideoCancelled
        if step_callback is not None:
            step_callback(payload)

    return _callback


def run_video_batch(
    backend: Any,
    model: Any,
    model_info: VideoModelInfo,
    workflow: GenerationWorkflow,
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    plan: JobPlan,
    progress_callback: ProgressCallback | None = None,
    skip_signal: SkipSignal | None = None,
) -> None:
    """Run the video batch generation loop.

    A Quit queued on *skip_signal* stops the batch before the next video, or the running video at its next step; the
    batch then ends with ``batch_cancelled``. Other queued actions are not supported for video and are ignored.

    With ``args.review`` and an upscale, every video pauses after its half-size render until *skip_signal* carries a
    decision. Approve continues with the upscale. Discard skips the video (``generation_finished`` with status
    ``skipped``, not a failure). Quit cancels the batch.

    Args:
        backend: Video backend instance (satisfies VideoBackend Protocol).
        model: Loaded model handle from backend.load_model().
        model_info: VideoModelInfo from detect_video_model().
        workflow: Built GenerationWorkflow from build_video_workflow() (``enhance=plan.has_rewrites``).
        prompts_data: Dict of set_name -> list of (prompt, negative_prompt) tuples.
        config: Loaded config.yaml dict.
        args: Parsed video CLI arguments.
        plan: Preflight plan with one iteration per run × prompt (seeds, resolved and enhanced prompts).
        progress_callback: Receives structured progress events.
        skip_signal: Control signal of the job; Quit and, with a review, its decisions are honoured.

    Raises:
        ValueError: When *plan* was cancelled or does not match the batch size, or a review is requested without
            *skip_signal*.
        RuntimeError: When *plan*'s iteration order differs from the batch loop.
    """
    extend = _extend_spec(args)
    total_prompts = sum(len(p) for p in prompts_data.values())
    total_iterations = args.runs * total_prompts
    plan.require_runnable(total_iterations)
    with_review = review_enabled(args)
    if with_review and skip_signal is None:
        raise ValueError("A review before upscaling needs a skip_signal to carry the decision.")
    if total_iterations == 0:
        print("No active prompt sets found. Exiting.")
        _emit_progress(progress_callback, "batch_completed", mode="video", total_iterations=0, completed_iterations=0)
        return

    ran_iterations = 0
    completed_iterations = 0
    batch_start = time.time()
    gen_times: list[float] = []
    failed_generations = 0
    discarded_generations = 0

    print(f"Total video iterations to run: {total_iterations}\n")
    _emit_progress(progress_callback, "batch_started", mode="video", total_iterations=total_iterations, total_runs=args.runs)

    for run_idx in range(args.runs):
        for set_name, prompts in prompts_data.items():
            for prompt_idx, (prompt, _) in enumerate(prompts):
                if _quit_requested(skip_signal):
                    _emit_cancelled(progress_callback, len(gen_times), total_iterations)
                    return
                ran_iterations += 1
                avg = sum(gen_times) / len(gen_times) if gen_times else None
                remaining = total_iterations - completed_iterations
                eta = avg * remaining if avg is not None else None

                planned = plan.iteration_for(ran_iterations, run_index=run_idx, set_name=set_name, prompt_index=prompt_idx)
                seed = planned.seed
                _emit_progress(
                    progress_callback,
                    "prompt_started",
                    mode="video",
                    run_index=run_idx,
                    total_runs=args.runs,
                    ran_iterations=ran_iterations,
                    total_iterations=total_iterations,
                    set_name=set_name,
                    prompt_index=prompt_idx,
                    total_prompts=len(prompts),
                    prompt=prompt,
                    seed=seed,
                    enhance_status=planned.enhance_status.value,
                    elapsed_secs=time.time() - batch_start,
                    avg_secs=avg,
                    eta_secs=eta,
                )

                _emit_progress(
                    progress_callback,
                    "generation_started",
                    mode="video",
                    run_index=run_idx,
                    total_runs=args.runs,
                    ran_iterations=ran_iterations,
                    total_iterations=total_iterations,
                    set_name=set_name,
                    prompt_index=prompt_idx,
                    total_prompts=len(prompts),
                    prompt=prompt,
                    seed=seed,
                    elapsed_secs=time.time() - batch_start,
                    avg_secs=avg,
                    eta_secs=eta,
                )

                # Display progress
                eta_str = f"  ETA: {eta:.0f}s" if eta is not None else ""
                print(f"\n[{ran_iterations}/{total_iterations}] Run {run_idx + 1}/{args.runs} | Set: {set_name} | Prompt {prompt_idx + 1}/{len(prompts)}{eta_str}")
                if extend is not None:
                    print(f"  {args.width}x{args.height}, +{extend.frames} frames from {extend.source.path.name}, seed={seed}")
                else:
                    print(f"  {args.width}x{args.height}, {args.num_frames} frames, seed={seed}")
                if args.upscale:
                    print("  Upscale: 2x" + (" (review)" if with_review else ""))
                if getattr(args, "no_audio", False):
                    print("  Audio: off")

                request = VideoGenerationRequest(
                    backend=backend,
                    model=model,
                    prompt=prompt,
                    model_name=getattr(args, "model", None),
                    filename_base=set_name,
                    model_family=model_info.family,
                    lora_paths=getattr(args, "lora_paths", None) or [],
                    lora_weights=getattr(args, "lora_weights", None) or [],
                    width=args.width,
                    height=args.height,
                    num_frames=args.num_frames,
                    seed=seed,
                    steps=args.steps,
                    step_callback=_cancellable_step_callback(
                        _make_step_progress_callback(
                            progress_callback,
                            mode="video",
                            run_index=run_idx,
                            total_runs=args.runs,
                            ran_iterations=ran_iterations,
                            total_iterations=total_iterations,
                            set_name=set_name,
                            prompt_index=prompt_idx,
                            total_prompts=len(prompts),
                        ),
                        skip_signal,
                    ),
                    image_path=getattr(args, "image_path", None),
                    upscale=getattr(args, "upscale", None),
                    upscale_steps=getattr(args, "upscale_steps", None),
                    review=(
                        _make_video_review(
                            skip_signal,
                            progress_callback=progress_callback,
                            ran_iterations=ran_iterations,
                            completed_iterations=completed_iterations,
                            total_iterations=total_iterations,
                        )
                        if with_review
                        else None
                    ),
                    no_audio=getattr(args, "no_audio", False),
                    output_dir=args.output,
                    output_format=getattr(args, "format", "mp4"),
                    resolved_prompt=planned.resolved_prompt,
                    enhanced_prompt=planned.enhanced_prompt,
                    scene=planned.scene,
                    resolved_scene=planned.resolved_scene,
                    enhanced_scene=planned.enhanced_scene,
                    scene_input=getattr(args, "scene_input", None),
                    prompt_input=getattr(args, "prompt_input", None),
                    extend=extend,
                    on_prompt_enhanced=_make_prompt_enhanced_callback(
                        progress_callback,
                        mode="video",
                        run_index=run_idx,
                        ran_iterations=ran_iterations,
                        total_iterations=total_iterations,
                        set_name=set_name,
                        prompt_index=prompt_idx,
                        seed=seed,
                    ),
                )
                artifacts = VideoWorkingArtifacts()
                started = time.monotonic()
                try:
                    event_context = {
                        "mode": "video",
                        "run_index": run_idx,
                        "total_runs": args.runs,
                        "ran_iterations": ran_iterations,
                        "total_iterations": total_iterations,
                        "set_name": set_name,
                        "prompt_index": prompt_idx,
                        "total_prompts": len(prompts),
                        "prompt": prompt,
                        "seed": seed,
                    }
                    outcome = _run_workflow_with_progress(
                        workflow,
                        request,
                        artifacts,
                        progress_callback=progress_callback,
                        event_context=event_context,
                    )
                except _VideoCancelled:
                    print("\n⏹ Video cancelled.")
                    _emit_cancelled(progress_callback, len(gen_times), total_iterations)
                    return
                except _VideoDiscarded:
                    print("  Small render discarded; skipping the upscale.")
                    outcome = StageOutcome.skipped
                    discarded_generations += 1
                    # The small render took real time, so it counts toward the ETA; the wait for the decision does not.
                    artifacts.generation_time = time.monotonic() - started - artifacts.review_wait
                except Exception as exc:
                    logger.exception("Video generation failed")
                    warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
                    outcome = StageOutcome.failed
                completed_iterations += 1

                if outcome is StageOutcome.success:
                    gen_times.append(artifacts.generation_time)
                    _emit_generation_finished(
                        progress_callback,
                        event_context=event_context,
                        status="success",
                        filename=artifacts.filename,
                        generation_time=artifacts.generation_time,
                        output_path=str(artifacts.video_path) if artifacts.video_path is not None else None,
                    )
                elif outcome is StageOutcome.skipped:
                    gen_times.append(artifacts.generation_time)
                    _emit_generation_finished(
                        progress_callback,
                        event_context=event_context,
                        status="skipped",
                        generation_time=artifacts.generation_time,
                    )
                elif outcome is StageOutcome.failed:
                    print("  Video generation failed.")
                    failed_generations += 1
                    _emit_generation_finished(
                        progress_callback,
                        event_context=event_context,
                        status="failed",
                        output_path=str(artifacts.video_path) if artifacts.video_path is not None else None,
                    )

    total_time = time.time() - batch_start
    generated = len(gen_times) - discarded_generations
    discarded_note = f", {discarded_generations} discarded" if discarded_generations else ""
    print(f"\nBatch complete: {generated}/{total_iterations} videos generated{discarded_note} in {total_time:.1f}s")
    terminal_event = "batch_failed" if len(gen_times) == 0 and failed_generations > 0 else "batch_completed"
    _emit_progress(
        progress_callback,
        terminal_event,
        mode="video",
        completed_iterations=completed_iterations,
        total_iterations=total_iterations,
        total_time=total_time,
    )


def _extend_spec(args: argparse.Namespace) -> ExtendSpec | None:
    """Return what to append to the source clip when the job extends one (``extend_source`` set), else ``None``."""
    source = getattr(args, "extend_source", None)
    if source is None:
        return None
    return ExtendSpec(source, args.extend_frames)


def _emit_cancelled(progress_callback: ProgressCallback | None, completed: int, total: int) -> None:
    """Report that the batch stopped on Quit after *completed* of *total* videos."""
    print("\n⏹ Quitting batch...")
    _emit_progress(progress_callback, "batch_cancelled", mode="video", completed_iterations=completed, total_iterations=total)
