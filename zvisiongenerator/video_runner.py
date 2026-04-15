"""Video batch orchestration — runs × sets × prompts loop for video generation."""

from __future__ import annotations

import argparse
import random
import time
import warnings
from typing import Any

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
        return

    ran_iterations = 0
    completed_iterations = 0
    batch_start = time.time()
    gen_times: list[float] = []

    print(f"Total video iterations to run: {total_iterations}\n")

    for run_idx in range(args.runs):
        for set_name, prompts in prompts_data.items():
            for prompt_idx, (prompt, _) in enumerate(prompts):
                ran_iterations += 1
                avg = sum(gen_times) / len(gen_times) if gen_times else None
                remaining = total_iterations - completed_iterations
                eta = avg * remaining if avg is not None else None

                seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)

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
                    image_path=getattr(args, "image_path", None),
                    upscale=getattr(args, "upscale", None),
                    upscale_steps=getattr(args, "upscale_steps", None),
                    no_audio=getattr(args, "no_audio", False),
                    output_dir=args.output,
                    output_format=getattr(args, "format", "mp4"),
                )
                artifacts = VideoWorkingArtifacts()
                try:
                    outcome = workflow.run(request, artifacts)
                except Exception as exc:
                    warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
                    outcome = StageOutcome.failed
                completed_iterations += 1

                if outcome is StageOutcome.success:
                    gen_times.append(artifacts.generation_time)
                elif outcome is StageOutcome.failed:
                    print("  Video generation failed.")

    total_time = time.time() - batch_start
    print(f"\nBatch complete: {len(gen_times)}/{total_iterations} videos generated in {total_time:.1f}s")
