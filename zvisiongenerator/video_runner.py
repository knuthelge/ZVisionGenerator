"""Video batch orchestration — runs × sets × prompts loop for video generation."""

from __future__ import annotations

import argparse
import random
import time
import warnings
from typing import Any

from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.core.progress_events import emit_generation_finished as _emit_generation_finished
from zvisiongenerator.core.progress_events import emit_progress as _emit_progress
from zvisiongenerator.core.progress_events import make_step_progress_callback as _make_step_progress_callback
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.utils.video_model_detect import VideoModelInfo


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
    failed_generations = 0

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
