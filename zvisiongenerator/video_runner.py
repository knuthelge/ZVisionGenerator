"""Video batch orchestration — runs × sets × prompts loop for video generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import random
import time
import warnings
from collections.abc import Callable
from typing import Any

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.utils.video_model_detect import VideoModelInfo


type ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(progress_callback: ProgressCallback | None, event_type: str, **payload: Any) -> None:
    """Send a structured progress event when a callback is configured."""
    if progress_callback is None:
        return
    progress_callback({"type": event_type, **payload})


def _make_step_progress_callback(
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
        _emit_progress(
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


def _run_workflow_with_progress(
    workflow: GenerationWorkflow,
    request: VideoGenerationRequest,
    artifacts: VideoWorkingArtifacts,
    *,
    progress_callback: ProgressCallback | None,
    event_context: dict[str, Any],
) -> StageOutcome:
    """Run a video workflow while emitting structured stage progress events."""
    stages = workflow.stages if isinstance(getattr(workflow, "stages", None), list | tuple) else None
    if stages is None:
        stage_name = getattr(workflow, "name", "workflow")
        _emit_progress(
            progress_callback,
            "workflow_started",
            total_stages=1,
            stage_name=stage_name,
            **event_context,
        )
        _emit_progress(
            progress_callback,
            "workflow_stage_started",
            stage_index=1,
            total_stages=1,
            stage_name=stage_name,
            **event_context,
        )
        outcome = workflow.run(request, artifacts)
        _emit_progress(
            progress_callback,
            "workflow_stage_completed",
            stage_index=1,
            total_stages=1,
            stage_name=stage_name,
            outcome=outcome.name.lower(),
            **event_context,
        )
        _emit_progress(
            progress_callback,
            "workflow_finished",
            total_stages=1,
            completed_stages=1,
            stage_name=stage_name,
            status=outcome.name.lower(),
            **event_context,
        )
        return outcome

    total_stages = len(stages)
    if total_stages == 0:
        return StageOutcome.success

    _emit_progress(
        progress_callback,
        "workflow_started",
        total_stages=total_stages,
        stage_name=_stage_label(stages[0]),
        **event_context,
    )

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
        _emit_progress(
            progress_callback,
            "workflow_stage_started",
            stage_index=stage_index,
            total_stages=total_stages,
            stage_name=stage_name,
            **event_context,
        )
        outcome = stage(stage_request, artifacts)
        _emit_progress(
            progress_callback,
            "workflow_stage_completed",
            stage_index=stage_index,
            total_stages=total_stages,
            stage_name=stage_name,
            outcome=outcome.name.lower(),
            **event_context,
        )
        if outcome is not StageOutcome.success:
            _emit_progress(
                progress_callback,
                "workflow_finished",
                total_stages=total_stages,
                completed_stages=stage_index,
                stage_name=stage_name,
                status=outcome.name.lower(),
                **event_context,
            )
            return outcome

    _emit_progress(
        progress_callback,
        "workflow_finished",
        total_stages=total_stages,
        completed_stages=total_stages,
        stage_name=_stage_label(stages[-1]),
        status="success",
        **event_context,
    )
    return StageOutcome.success


def run_video_batch(
    backend: Any,
    model: Any,
    model_info: VideoModelInfo,
    workflow: GenerationWorkflow,
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Run the video batch generation loop.

    Args:
        backend: Video backend instance (satisfies VideoBackend Protocol).
        model: Loaded model handle from backend.load_model().
        model_info: VideoModelInfo from detect_video_model().
        workflow: Built GenerationWorkflow from build_video_workflow().
        prompts_data: Dict of set_name -> list of (prompt, negative_prompt) tuples.
        config: Loaded config.yaml dict.
        args: Parsed video CLI arguments.
    """
    # Seed range from config
    seed_min = config.get("generation", {}).get("seed_min", 4)
    seed_max = config.get("generation", {}).get("seed_max", 2**32 - 1)

    total_prompts = sum(len(p) for p in prompts_data.values())
    total_iterations = args.runs * total_prompts
    if total_iterations == 0:
        print("No active prompt sets found. Exiting.")
        _emit_progress(progress_callback, "batch_completed", mode="video", total_iterations=0, completed_iterations=0)
        return

    ran_iterations = 0
    completed_iterations = 0
    batch_start = time.time()
    gen_times: list[float] = []

    print(f"Total video iterations to run: {total_iterations}\n")
    _emit_progress(progress_callback, "batch_started", mode="video", total_iterations=total_iterations, total_runs=args.runs)

    for run_idx in range(args.runs):
        for set_name, prompts in prompts_data.items():
            for prompt_idx, (prompt, _) in enumerate(prompts):
                ran_iterations += 1
                avg = sum(gen_times) / len(gen_times) if gen_times else None
                remaining = total_iterations - completed_iterations
                eta = avg * remaining if avg is not None else None

                seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)
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
                    step_callback=_make_step_progress_callback(
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
                    image_path=getattr(args, "image_path", None),
                    upscale=getattr(args, "upscale", None),
                    upscale_steps=getattr(args, "upscale_steps", None),
                    no_audio=getattr(args, "no_audio", False),
                    output_dir=args.output,
                    output_format=getattr(args, "format", "mp4"),
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
                except Exception as exc:
                    warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
                    outcome = StageOutcome.failed
                completed_iterations += 1

                if outcome is StageOutcome.success:
                    gen_times.append(artifacts.generation_time)
                    _emit_progress(
                        progress_callback,
                        "generation_finished",
                        mode="video",
                        status="success",
                        run_index=run_idx,
                        total_runs=args.runs,
                        ran_iterations=ran_iterations,
                        total_iterations=total_iterations,
                        set_name=set_name,
                        prompt_index=prompt_idx,
                        total_prompts=len(prompts),
                        filename=artifacts.filename,
                        generation_time=artifacts.generation_time,
                        output_path=str(artifacts.video_path) if artifacts.video_path is not None else None,
                    )
                elif outcome is StageOutcome.failed:
                    print("  Video generation failed.")
                    _emit_progress(
                        progress_callback,
                        "generation_finished",
                        mode="video",
                        status="failed",
                        run_index=run_idx,
                        total_runs=args.runs,
                        ran_iterations=ran_iterations,
                        total_iterations=total_iterations,
                        set_name=set_name,
                        prompt_index=prompt_idx,
                        total_prompts=len(prompts),
                        output_path=str(artifacts.video_path) if artifacts.video_path is not None else None,
                    )

    total_time = time.time() - batch_start
    print(f"\nBatch complete: {len(gen_times)}/{total_iterations} videos generated in {total_time:.1f}s")
    _emit_progress(
        progress_callback,
        "batch_completed",
        mode="video",
        completed_iterations=completed_iterations,
        total_iterations=total_iterations,
        total_time=total_time,
    )
