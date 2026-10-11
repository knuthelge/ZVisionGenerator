"""Video batch orchestration — runs × sets × prompts loop for video generation."""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.core.progress_events import emit_generation_finished as _emit_generation_finished
from zvisiongenerator.core.progress_events import emit_progress as _emit_progress
from zvisiongenerator.core.progress_events import make_prompt_enhanced_callback as _make_prompt_enhanced_callback
from zvisiongenerator.core.progress_events import make_step_progress_callback as _make_step_progress_callback
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.utils.video_model_detect import VideoModelInfo

if TYPE_CHECKING:
    from zvisiongenerator.utils.interactive import SkipSignal

logger = logging.getLogger(__name__)


class _VideoCancelled(Exception):
    """Raised from a step callback to stop a running video when Quit is queued."""


def _quit_requested(skip_signal: SkipSignal | None) -> bool:
    """Return whether Quit is queued on *skip_signal*."""
    return skip_signal is not None and skip_signal.pending() == "quit"


def _cancellable_step_callback(step_callback: ProgressCallback | None, skip_signal: SkipSignal | None) -> ProgressCallback | None:
    """Wrap *step_callback* so a queued Quit stops the denoising loop at its next step.

    Both video backends call the step callback from their denoising loop, so raising there ends the generation. The
    diffusers latent upscaler reports only once, after it finishes, so a Quit during that pass applies when it ends.
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
    batch then ends with ``batch_cancelled``. Other actions are not supported for video and are ignored.

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
        skip_signal: Control signal of a Web UI job; only Quit is honoured.

    Raises:
        ValueError: When *plan* was cancelled or does not match the batch size.
        RuntimeError: When *plan*'s iteration order differs from the batch loop.
    """
    total_prompts = sum(len(p) for p in prompts_data.values())
    total_iterations = args.runs * total_prompts
    plan.require_runnable(total_iterations)
    if total_iterations == 0:
        print("No active prompt sets found. Exiting.")
        _emit_progress(progress_callback, "batch_completed", mode="video", total_iterations=0, completed_iterations=0)
        return

    ran_iterations = 0
    completed_iterations = 0
    batch_start = time.time()
    gen_times: list[float] = []
    failed_generations = 0

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
                print(f"  {args.width}x{args.height}, {args.num_frames} frames, seed={seed}")
                if args.upscale:
                    print("  Upscale: 2x")
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
                    no_audio=getattr(args, "no_audio", False),
                    output_dir=args.output,
                    output_format=getattr(args, "format", "mp4"),
                    resolved_prompt=planned.resolved_prompt,
                    enhanced_prompt=planned.enhanced_prompt,
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
    print(f"\nBatch complete: {len(gen_times)}/{total_iterations} videos generated in {total_time:.1f}s")
    terminal_event = "batch_failed" if len(gen_times) == 0 and failed_generations > 0 else "batch_completed"
    _emit_progress(
        progress_callback,
        terminal_event,
        mode="video",
        completed_iterations=completed_iterations,
        total_iterations=total_iterations,
        total_time=total_time,
    )


def _emit_cancelled(progress_callback: ProgressCallback | None, completed: int, total: int) -> None:
    """Report that the batch stopped on Quit after *completed* of *total* videos."""
    print("\n⏹ Quitting batch...")
    _emit_progress(progress_callback, "batch_cancelled", mode="video", completed_iterations=completed, total_iterations=total)
