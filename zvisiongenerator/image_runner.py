"""Batch orchestration — extracted from cli.py::main().

Handles the triple-nested run/set/prompt batch loop, timing/ETA,
interactive controls (skip/quit/pause/repeat via SkipSignal), and
_QuitBatch exception for clean multi-level loop exit.
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from typing import Any

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils import generate_filename, format_generation_info
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.workflows import build_workflow
from zvisiongenerator.utils.alignment import round_to_alignment


def _resolve_scheduler_class(scheduler: str | None, config: dict, backend_name: str) -> str | None:
    """Resolve scheduler name to backend-specific class path from config."""
    if scheduler is None:
        return None
    sched_cfg = config.get("schedulers", {}).get(scheduler, {})
    class_key = f"{backend_name}_class"
    return sched_cfg.get(class_key, scheduler)


class _QuitBatch(Exception):
    """Sentinel exception to break out of nested batch loops.

    Raised by the interactive control handler (SkipSignal) when the user
    presses 'q' to quit the current batch. Caught at the outermost loop
    level in run_batch(). This pattern exists because Python has no
    labeled break — the exception provides clean multi-level loop exit.
    """


def run_batch(
    backend: ImageBackend,
    model: Any,
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    model_info: ImageModelInfo,
) -> None:
    """Run the batch generation loop.

    Args:
        backend: ImageBackend instance (satisfies ImageBackend Protocol).
        model: Loaded model handle from ``backend.load_model()``.
        prompts_data: Dict of set_name → list of (prompt, negative_prompt) tuples.
        config: Loaded config.yaml dict.
        args: Parsed CLI arguments (argparse Namespace).
        model_info: ImageModelInfo from ``backend.load_model()``.
    """

    # Resolve supports_negative_prompt from config
    _family = model_info.family
    _preset = config.get("model_presets", {}).get(_family, {})
    supports_neg = _preset.get("supports_negative_prompt", False)

    # Resolve size from config
    sizes = config.get("sizes", {})
    dims = sizes[args.ratio][args.size]
    width = dims["width"]
    height = dims["height"]
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height

    # Determine whether to propagate ratio/size — only when user didn't specify explicit dims
    use_ratio = args.ratio if args.width is None and args.height is None else None
    use_size = args.size if args.width is None and args.height is None else None

    # Warn if preset dimensions drift under upscale round-trip
    if args.upscale is not None and args.width is None and args.height is None:
        for dim_name, dim_val in [("width", width), ("height", height)]:
            base = dim_val // args.upscale
            final = round_to_alignment(base) * args.upscale
            if final != dim_val:
                warnings.warn(
                    f"Preset ratio '{args.ratio}' size '{args.size}' ({width}x{height}) drifts {dim_name} {dim_val} to {final} with {args.upscale}x upscale. Use explicit --width/--height for exact control.",
                    stacklevel=2,
                )

    # Seed range from config
    seed_min = config.get("generation", {}).get("seed_min", 4)
    seed_max = config.get("generation", {}).get("seed_max", 2**32 - 1)

    # Sharpening amounts from config
    sharpening = config.get("sharpening", {})
    sharpen_normal = sharpening.get("normal", 0.8)
    sharpen_upscaled = sharpening.get("upscaled", 1.2)
    sharpen_pre_upscale = sharpening.get("pre_upscale", 0.4)

    # Warn about negative prompts for models that don't support them (single location)
    if not supports_neg:
        has_negatives = any(neg and neg.strip() for prompts in prompts_data.values() for _, neg in prompts)
        if has_negatives:
            warnings.warn(
                f"{model_info.family} models do not support negative prompts — ignoring.",
                stacklevel=2,
            )

    # Calculate total iterations for progress tracking
    total_iterations = args.runs * sum(len(prompts) for prompts in prompts_data.values())
    if total_iterations == 0:
        print("No active prompt sets found. Exiting.")
        return
    ran_iterations = 0
    print(f"Total iterations to run: {total_iterations}\n")

    skip = SkipSignal()
    batch_start_time = time.time()
    image_times: list[float] = []

    # Resolve sharpen override — only when user provided explicit float
    sharpen_amount_override = args.sharpen if isinstance(args.sharpen, float) else None

    # Resolve contrast_amount from explicit value or config default
    contrast_cfg = config.get("contrast", {})
    if isinstance(args.contrast, float):
        contrast_amount = args.contrast
    else:
        contrast_amount = contrast_cfg.get("default_amount", 1.0)

    # Resolve saturation_amount from explicit value or config default
    saturation_cfg = config.get("saturation", {})
    if isinstance(args.saturation, float):
        saturation_amount = args.saturation
    else:
        saturation_amount = saturation_cfg.get("default_amount", 1.0)

    workflow = build_workflow(args)

    try:
        skip.start()
        for run_idx in range(args.runs):
            for set_name, prompts in prompts_data.items():
                for prompt_idx, (prompt, negative_prompt) in enumerate(prompts):
                    print("\033[2J\033[H", end="", flush=True)
                    print("\n\nZ-Vision Generator - Batch Image Generation")
                    ran_iterations += 1
                    _elapsed = time.time() - batch_start_time
                    _completed = len(image_times)
                    _avg = sum(image_times) / _completed if _completed > 0 else None
                    _remaining = total_iterations - _completed
                    _eta = _avg * _remaining if _avg is not None else None

                    # Suppress negative prompt if model doesn't support it
                    effective_negative = negative_prompt
                    if not supports_neg:
                        effective_negative = None

                    # Generate seed before display so the header shows the real value
                    seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)

                    # Build a lightweight request just for display info
                    # (the real request is built below with seed etc.)
                    _display_request = ImageGenerationRequest(
                        backend=backend,
                        model=model,
                        model_name=args.model,
                        model_family=_family,
                        supports_negative_prompt=supports_neg,
                        prompt=prompt,
                        negative_prompt=effective_negative if supports_neg else None,
                        ratio=use_ratio,
                        size=use_size,
                        width=round_to_alignment(width // args.upscale) if args.upscale else width,
                        height=round_to_alignment(height // args.upscale) if args.upscale else height,
                        steps=args.steps,
                        guidance=args.guidance,
                        seed=seed,
                        upscale_factor=args.upscale,
                        lora_paths=getattr(args, "lora_paths", None),
                        lora_weights=getattr(args, "lora_weights", None),
                    )
                    _display_artifacts = ImageWorkingArtifacts()
                    print(
                        format_generation_info(
                            _display_request,
                            _display_artifacts,
                            run_number=run_idx,
                            total_runs=args.runs,
                            ran_iterations=ran_iterations,
                            total_iterations=total_iterations,
                            set_name=set_name,
                            prompt_idx=prompt_idx,
                            total_prompts=len(prompts),
                            elapsed_secs=_elapsed,
                            avg_secs=_avg,
                            eta_secs=_eta,
                        )
                    )
                    print("Commands: [n] skip  [q] quit  [p] pause  [r] repeat\n")

                    retries = 0
                    max_retries = 3
                    while True:
                        if retries > 0:
                            seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)
                        gen_filename = generate_filename(
                            set_name,
                            width=width,
                            height=height,
                            seed=seed,
                            steps=args.steps,
                            guidance=args.guidance,
                            scheduler=args.scheduler,
                            model=args.model,
                            lora_paths=getattr(args, "lora_paths", None),
                            lora_weights=getattr(args, "lora_weights", None),
                        )
                        skip.reset()
                        _img_start = time.time()

                        # Resolve upscale_denoise from config if not explicitly set
                        if args.upscale and args.upscale_denoise is None:
                            upscale_cfg = config.get("upscale", {})
                            if args.upscale == 4:
                                resolved_denoise = upscale_cfg.get("default_denoise_4x", 0.4)
                            else:
                                resolved_denoise = upscale_cfg.get("default_denoise_2x", 0.3)
                        else:
                            resolved_denoise = args.upscale_denoise if args.upscale else None

                        request = ImageGenerationRequest(
                            backend=backend,
                            model=model,
                            model_name=args.model,
                            model_family=_family,
                            supports_negative_prompt=supports_neg,
                            lora_paths=getattr(args, "lora_paths", None),
                            lora_weights=getattr(args, "lora_weights", None),
                            prompt=prompt,
                            negative_prompt=effective_negative,
                            ratio=use_ratio,
                            size=use_size,
                            width=round_to_alignment(width // args.upscale) if args.upscale else width,
                            height=round_to_alignment(height // args.upscale) if args.upscale else height,
                            seed=seed,
                            steps=args.steps,
                            guidance=args.guidance,
                            scheduler=_resolve_scheduler_class(args.scheduler, config, backend.name),
                            skip_signal=skip,
                            upscale_factor=args.upscale,
                            upscale_denoise=resolved_denoise,
                            upscale_steps=(args.upscale_steps if args.upscale else None),
                            upscale_guidance=args.upscale_guidance,
                            upscale_sharpen=args.upscale_sharpen,
                            upscale_save_pre=(args.upscale_save_pre if args.upscale else False),
                            image_path=args.image_path,
                            image_strength=args.image_strength,
                            sharpen_amount_normal=sharpen_normal,
                            sharpen_amount_upscaled=sharpen_upscaled,
                            sharpen_amount_pre_upscale=sharpen_pre_upscale,
                            sharpen=args.sharpen is not False,
                            sharpen_amount_override=sharpen_amount_override,
                            contrast=args.contrast is not False,
                            contrast_amount=contrast_amount,
                            saturation=args.saturation is not False,
                            saturation_amount=saturation_amount,
                            output_dir=args.output,
                            filename_base=gen_filename,
                        )
                        artifacts = ImageWorkingArtifacts(filename=gen_filename)

                        outcome = workflow.run(request, artifacts)
                        if outcome is StageOutcome.skipped:
                            warnings.warn("Generation skipped by workflow stage.")
                            image_times.append(time.time() - _img_start)
                            if skip.consume() == "quit":
                                raise _QuitBatch()
                            break
                        elif outcome is StageOutcome.failed:
                            warnings.warn("Generation failed in workflow stage.")
                            image_times.append(time.time() - _img_start)
                            if skip.consume() == "quit":
                                raise _QuitBatch()
                            break
                        elif outcome is StageOutcome.retry:
                            retries += 1
                            if retries > max_retries:
                                warnings.warn(f"Generation failed after {max_retries} retries; skipping.")
                                image_times.append(time.time() - _img_start)
                                break
                            warnings.warn(f"Workflow stage requested retry ({retries}/{max_retries}).")
                            continue

                        retries = 0
                        action = skip.consume()
                        if action == "quit":
                            print("\n⏹ Quitting batch...")
                            raise _QuitBatch()
                        elif action == "pause":
                            image_times.append(time.time() - _img_start)
                            print("\n⏸ Paused. Press any key to continue...")
                            skip.wait_for_key()
                            print("▶ Resumed.\n")
                            break
                        elif action == "repeat":
                            image_times.append(time.time() - _img_start)
                            seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)
                            print("\n🔁 Repeating prompt with new seed...\n")
                            continue
                        elif action == "skip":
                            image_times.append(time.time() - _img_start)
                            break
                        else:
                            image_times.append(time.time() - _img_start)
                            break
            print(f"\nCompleted run {run_idx + 1}/{args.runs}\n{'#' * 30}\n")
        print("\nAll runs completed!\n")
    except _QuitBatch:
        pass
    finally:
        skip.stop()
