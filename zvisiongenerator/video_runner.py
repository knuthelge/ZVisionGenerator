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
from zvisiongenerator.core.progress_events import exception_message as _exception_message
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.job_controls import announce_repeat, pause_until_resumed, seed_range
from zvisiongenerator.review import REVIEW_PROMPT, await_review_decision, review_enabled
from zvisiongenerator.utils.enhance_record import planned_enhance_record
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.video_model_detect import VideoModelInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class _VideoCancelled(Exception):
    """Raised from a step callback to stop a running video when Quit is queued."""


class _VideoSkipped(Exception):
    """Raised from a step callback or the review callable to stop a running video when Next is queued."""


class _VideoDiscarded(Exception):
    """Raised from the review callable when the user discards the small render, so the upscale is skipped."""


def _make_video_review(
    skip_signal: SkipSignal,
    *,
    progress_callback: ProgressCallback | None,
    ran_iterations: int,
    attempt: int,
    completed_iterations: int,
    total_iterations: int,
) -> Callable[[Path], None]:
    """Build the ``review`` callable of one video generation.

    The callable is called with the path of the decoded half-size render, shows it and waits for the decision. It
    returns on approve only: it raises ``_VideoDiscarded`` on discard, ``_VideoSkipped`` on Next (which declines the upscale)
    and ``_VideoCancelled`` on Quit (which stays queued).

    Args:
        skip_signal: The job's control signal; its review state carries the decision.
        progress_callback: Receives ``job_paused`` (with ``review`` and, under the private key ``review_path``, the file
            for the web runner to serve) and ``job_resumed``. Without one the job runs in a terminal: the path is printed.
        ran_iterations: The 1-based iteration under review.
        attempt: The 1-based attempt within that iteration (a Repeat starts the next one).
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
            review={"iteration": ran_iterations, "attempt": attempt},
            **({} if progress_callback is None else {"review_path": str(path)}),
        )
        if decision == "quit":
            raise _VideoCancelled
        if decision == "discard":
            raise _VideoDiscarded
        if decision == "skip":
            raise _VideoSkipped

    return _review


def _interruptible_step_callback(step_callback: ProgressCallback | None, skip_signal: SkipSignal) -> ProgressCallback:
    """Wrap *step_callback* so a queued Quit or Next stops the denoising loop at its next step.

    The backends call it from every denoising step; the latent upsample between the two passes runs to completion, so a
    Quit or Next during it applies at the first refinement step. Pause and Repeat apply after the video.

    The step that reports the end of the render (``current_step`` reaches ``total_steps``) never raises, so a Quit or
    Next pressed meanwhile lets the clip be written; the runner applies it after the video. A backend that reports
    only the start and the end of a clip is thereby uninterruptible.
    """

    def _callback(payload: dict[str, Any]) -> None:
        if not payload.get("current_step", 0) >= payload.get("total_steps", 0) > 0:
            pending = skip_signal.pending()
            if pending == "quit":
                raise _VideoCancelled
            if pending == "skip":
                raise _VideoSkipped
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

    The controls are those of the image batch; Next and Quit stop the running video at its next step. After a failed
    video Pause is honoured and Repeat dropped; after a Next-stopped video both are dropped.

    With ``args.review`` and an upscale, every video pauses after its half-size render until *skip_signal* carries a
    decision. Approve continues with the upscale. Discard and Next skip the video (``generation_finished`` with status
    ``skipped``, not a failure) without the upscale. Quit cancels the batch.

    Args:
        backend: Video backend instance (satisfies VideoBackend Protocol).
        model: Loaded model handle from backend.load_model().
        model_info: VideoModelInfo from detect_video_model().
        workflow: Built GenerationWorkflow from build_video_workflow() (``enhance=plan.has_rewrites``).
        prompts_data: Dict of set_name -> list of (prompt, negative_prompt) tuples.
        config: Loaded config.yaml dict.
        args: Parsed video CLI arguments.
        plan: Preflight plan with one iteration per run x prompt (seeds, resolved and enhanced prompts).
        progress_callback: Receives structured progress events.
        skip_signal: Control signal of the job; its queued actions and, with a review, its decisions are honoured.

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

    skip = skip_signal or SkipSignal()
    seed_min, seed_max = seed_range(config)
    ran_iterations = 0
    completed_iterations = 0
    batch_start = time.time()
    gen_times: list[float] = []
    successful_generations = 0
    repeats = 0
    failed_generations = 0
    last_failure: str | None = None
    skipped_generations = 0

    print(f"Total video iterations to run: {total_iterations}\n")
    _emit_progress(progress_callback, "batch_started", mode="video", total_iterations=total_iterations, total_runs=args.runs)

    for run_idx in range(args.runs):
        for set_name, prompts in prompts_data.items():
            for prompt_idx, (prompt, _) in enumerate(prompts):
                ran_iterations += 1
                avg = sum(gen_times) / len(gen_times) if gen_times else None
                remaining = total_iterations - ran_iterations + 1
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

                eta_str = f"  ETA: {eta:.0f}s" if eta is not None else ""
                print(f"\n[{ran_iterations}/{total_iterations}] Run {run_idx + 1}/{args.runs} | Set: {set_name} | Prompt {prompt_idx + 1}/{len(prompts)}{eta_str}")

                attempt = 0
                while True:
                    # Honour Pause and Quit queued before this attempt (e.g. during the model load); Next and Repeat
                    # have no video to act on yet and are dropped.
                    pending_action = skip.consume()
                    if pending_action == "pause":
                        pending_action = pause_until_resumed(skip, progress_callback=progress_callback, mode="video", completed_iterations=completed_iterations, total_iterations=total_iterations)
                    if pending_action == "quit":
                        _emit_cancelled(progress_callback, completed_iterations, total_iterations)
                        return

                    attempt += 1
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
                        model_name=getattr(args, "model_name", None) or getattr(args, "model", None),
                        filename_base=set_name,
                        model_family=model_info.family,
                        lora_paths=getattr(args, "lora_paths", None) or [],
                        lora_weights=getattr(args, "lora_weights", None) or [],
                        width=args.width,
                        height=args.height,
                        num_frames=args.num_frames,
                        seed=seed,
                        steps=args.steps,
                        step_callback=_interruptible_step_callback(
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
                            skip,
                        ),
                        image_path=getattr(args, "image_path", None),
                        end_image_path=getattr(args, "end_image_path", None),
                        upscale=getattr(args, "upscale", None),
                        upscale_steps=getattr(args, "upscale_steps", None),
                        review=(
                            _make_video_review(
                                skip,
                                progress_callback=progress_callback,
                                ran_iterations=ran_iterations,
                                attempt=attempt,
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
                        enhance_record=planned_enhance_record(planned, plan.enhancer, mode="video") or getattr(args, "enhance_record", None),
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
                    next_stopped = False
                    started = time.monotonic()
                    try:
                        outcome = _run_workflow_with_progress(
                            workflow,
                            request,
                            artifacts,
                            progress_callback=progress_callback,
                            event_context=event_context,
                        )
                    except _VideoCancelled:
                        print("\n⏹ Video cancelled.")
                        _emit_cancelled(progress_callback, completed_iterations, total_iterations)
                        return
                    except (_VideoDiscarded, _VideoSkipped) as stop:
                        next_stopped = isinstance(stop, _VideoSkipped)
                        print("  Video skipped." if next_stopped else "  Small render discarded; skipping the upscale.")
                        outcome = StageOutcome.skipped
                        skipped_generations += 1
                        # A discarded small render took real time, so it counts toward the ETA (without the wait for the
                        # decision); the partial time of a Next-stopped video does not.
                        artifacts.generation_time = time.monotonic() - started - artifacts.review_wait
                    except Exception as exc:
                        logger.exception("Video generation failed")
                        warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
                        artifacts.failure_message = _exception_message(str(exc)) or type(exc).__name__
                        outcome = StageOutcome.failed
                    completed_iterations += 1

                    if outcome is StageOutcome.success:
                        successful_generations += 1
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
                        if not next_stopped:
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
                        last_failure = artifacts.failure_message or "Video generation failed."
                        _emit_generation_finished(
                            progress_callback,
                            event_context=event_context,
                            status="failed",
                            message=last_failure,
                            output_path=str(artifacts.video_path) if artifacts.video_path is not None else None,
                        )

                    # Quit always wins. Pause also follows a failed or discarded video, Repeat a saved or discarded one;
                    # a Next-stopped video drops both.
                    action = skip.consume()
                    if action == "pause" and not next_stopped:
                        action = pause_until_resumed(skip, progress_callback=progress_callback, mode="video", completed_iterations=completed_iterations, total_iterations=total_iterations)
                        if action != "quit":
                            break
                    if action == "quit":
                        _emit_cancelled(progress_callback, completed_iterations, total_iterations)
                        return
                    if action != "repeat" or next_stopped or outcome is StageOutcome.failed:
                        break
                    repeats += 1
                    seed = announce_repeat(progress_callback, mode="video", completed_iterations=completed_iterations, total_iterations=total_iterations, seed_min=seed_min, seed_max=seed_max)

    total_time = time.time() - batch_start
    total_attempts = total_iterations + repeats
    notes = f", {skipped_generations} skipped" if skipped_generations else ""
    print(f"\nBatch complete: {successful_generations}/{total_attempts} videos generated{notes} in {total_time:.1f}s")
    batch_failed = successful_generations == 0 and failed_generations > 0
    terminal_extra = {"message": last_failure} if batch_failed and last_failure else {}
    _emit_progress(
        progress_callback,
        "batch_failed" if batch_failed else "batch_completed",
        mode="video",
        completed_iterations=completed_iterations,
        total_iterations=total_attempts,
        total_time=total_time,
        **terminal_extra,
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
