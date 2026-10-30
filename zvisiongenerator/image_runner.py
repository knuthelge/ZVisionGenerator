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
from pathlib import Path
from typing import Any

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.core.progress_events import emit_generation_finished as _emit_generation_finished
from zvisiongenerator.core.progress_events import emit_progress as _emit_progress
from zvisiongenerator.core.progress_events import make_prompt_enhanced_callback as _make_prompt_enhanced_callback
from zvisiongenerator.core.progress_events import make_step_progress_callback as _make_step_progress_callback
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.image_review import make_image_review
from zvisiongenerator.job_controls import announce_repeat, pause_until_resumed, seed_range
from zvisiongenerator.review import ReviewOutcome, review_enabled
from zvisiongenerator.utils import generate_filename, format_generation_info
from zvisiongenerator.utils.alignment import round_to_alignment
from zvisiongenerator.utils.config import get_variant_key, model_capabilities, model_display_name, resolve_scheduler_class, sharpening_amounts
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.upscale import default_upscale_denoise
from zvisiongenerator.workflows import build_workflow, generation_negative
from zvisiongenerator.utils.enhance_record import planned_enhance_record


class _QuitBatch(Exception):
    """Sentinel exception to break out of nested batch loops.

    Raised by the interactive control handler (SkipSignal) when the user
    presses 'q' to quit the current batch. Caught at the outermost loop
    level in run_batch(). This pattern exists because Python has no
    labeled break — the exception provides clean multi-level loop exit.
    """


def _active_seconds(started: float, review: ReviewOutcome) -> float:
    """Return the seconds since *started* without the time spent waiting on a review."""
    return time.time() - started - review.wait_seconds


def _negative_prompt_warning(config: dict[str, Any], args: argparse.Namespace, model_info: ImageModelInfo) -> str:
    """Return the warning for negative prompts a model ignores; a variant of a supporting family gets its own wording."""
    variant = get_variant_key(model_info)
    if variant is None or not model_capabilities(config, model_info.family)["supports_negative_prompt"]:
        return f"{model_info.family} models do not support negative prompts — ignoring."
    # args.model is the resolved models-folder path by now; its folder name is the alias.
    name = model_display_name(config, Path(args.model).name)
    hint = model_capabilities(config, model_info.family, variant)["negative_prompt_hint"]
    return f"{name} does not support negative prompts; it was ignored.{f' {hint}' if hint else ''}"


def run_batch(
    backend: ImageBackend,
    model: Any,
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    model_info: ImageModelInfo,
    *,
    plan: JobPlan,
    progress_callback: ProgressCallback | None = None,
    skip_signal: SkipSignal | None = None,
) -> None:
    """Run the batch generation loop.

    Args:
        backend: ImageBackend instance (satisfies ImageBackend Protocol).
        model: Loaded model handle from ``backend.load_model()``.
        prompts_data: Dict of set_name → list of (prompt, negative_prompt) tuples.
        config: Loaded config.yaml dict.
        args: Parsed CLI arguments (argparse Namespace).
        model_info: ImageModelInfo from ``backend.load_model()``.
        plan: Preflight plan with one iteration per run × prompt (seeds, resolved and enhanced prompts).
        progress_callback: Receives structured progress events.
        skip_signal: The job's control signal; the caller owns its key listener (``start``/``stop``).

    Raises:
        ValueError: When *plan* was cancelled or does not match the batch size, or a review is requested without
            *skip_signal*.
        RuntimeError: When *plan*'s iteration order differs from the batch loop.
    """

    # Resolve supports_negative_prompt from config; a preset variant may override the family's value
    _family = model_info.family
    capabilities = model_capabilities(config, _family, get_variant_key(model_info))
    supports_neg = capabilities["supports_negative_prompt"]
    max_references = capabilities["max_reference_images"]

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
    seed_min, seed_max = seed_range(config)

    # Sharpening amounts from config
    sharpening = sharpening_amounts(config)
    sharpen_normal = sharpening["normal"]
    sharpen_upscaled = sharpening["upscaled"]
    sharpen_pre_upscale = sharpening["pre_upscale"]

    # Warn about negative prompts for models that don't support them (single location)
    if not supports_neg:
        has_negatives = any(neg and neg.strip() for prompts in prompts_data.values() for _, neg in prompts)
        if has_negatives:
            warnings.warn(_negative_prompt_warning(config, args, model_info), stacklevel=2)

    # Calculate total iterations for progress tracking
    total_iterations = args.runs * sum(len(prompts) for prompts in prompts_data.values())
    plan.require_runnable(total_iterations)
    if review_enabled(args) and skip_signal is None:
        raise ValueError("A review before upscaling needs a skip_signal to carry the decision.")
    if total_iterations == 0:
        print("No active prompt sets found. Exiting.")
        _emit_progress(progress_callback, "batch_completed", mode="image", total_iterations=0, completed_iterations=0)
        return
    reference_paths = tuple(getattr(args, "reference_paths", None) or ())
    ran_iterations = 0
    print(f"Total iterations to run: {total_iterations}\n")
    _emit_progress(progress_callback, "batch_started", mode="image", total_iterations=total_iterations, total_runs=args.runs)

    skip = skip_signal or SkipSignal()
    batch_start_time = time.time()
    image_times: list[float] = []
    completed_iterations = 0
    repeats = 0
    successful_generations = 0
    failed_generations = 0
    last_failure: str | None = None

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

    workflow = build_workflow(args, enhance=plan.has_rewrites)

    try:
        for run_idx in range(args.runs):
            for set_name, prompts in prompts_data.items():
                for prompt_idx, (prompt, negative_prompt) in enumerate(prompts):
                    print("\033[2J\033[H", end="", flush=True)
                    print("\n\nZ-Vision Generator - Batch Image Generation")
                    ran_iterations += 1
                    _elapsed = time.time() - batch_start_time
                    _avg = sum(image_times) / len(image_times) if image_times else None
                    _remaining = total_iterations - ran_iterations + 1
                    _eta = _avg * _remaining if _avg is not None else None

                    planned = plan.iteration_for(ran_iterations, run_index=run_idx, set_name=set_name, prompt_index=prompt_idx)

                    effective_negative = generation_negative(negative_prompt, supports_negative=supports_neg)

                    seed = planned.seed
                    _emit_progress(
                        progress_callback,
                        "prompt_started",
                        mode="image",
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
                        elapsed_secs=_elapsed,
                        avg_secs=_avg,
                        eta_secs=_eta,
                    )

                    # Build a lightweight request just for display info
                    # (the real request is built below with seed etc.)
                    _display_request = ImageGenerationRequest(
                        backend=backend,
                        model=model,
                        model_name=args.model,
                        model_family=_family,
                        supports_negative_prompt=supports_neg,
                        prompt=prompt,
                        negative_prompt=effective_negative,
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
                    attempt = 0
                    while True:
                        if retries > 0:
                            # Retries keep the planned text and always draw a new seed, even when the job's seed is fixed.
                            seed = random.randint(seed_min, seed_max)
                        gen_filename = generate_filename(set_name)
                        # Honour pause/quit queued before this generation started (e.g. during model load).
                        # skip/repeat have no generation to act on yet, so they are dropped.
                        pending_action = skip.consume()
                        if pending_action == "pause":
                            pending_action = pause_until_resumed(skip, progress_callback=progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                        if pending_action == "quit":
                            print("\n⏹ Quitting batch...")
                            _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                            raise _QuitBatch()
                        attempt += 1
                        _img_start = time.time()
                        review_outcome = ReviewOutcome()
                        _emit_progress(
                            progress_callback,
                            "generation_started",
                            mode="image",
                            run_index=run_idx,
                            total_runs=args.runs,
                            ran_iterations=ran_iterations,
                            total_iterations=total_iterations,
                            set_name=set_name,
                            prompt_index=prompt_idx,
                            total_prompts=len(prompts),
                            prompt=prompt,
                            seed=seed,
                            filename=gen_filename,
                            retry=retries,
                            elapsed_secs=_elapsed,
                            avg_secs=_avg,
                            eta_secs=_eta,
                        )

                        # Resolve upscale_denoise from config if not explicitly set
                        if args.upscale and args.upscale_denoise is None:
                            resolved_denoise = default_upscale_denoise(config, args.upscale)
                        else:
                            resolved_denoise = args.upscale_denoise if args.upscale else None

                        request = ImageGenerationRequest(
                            backend=backend,
                            model=model,
                            model_name=args.model,
                            model_family=_family,
                            supports_negative_prompt=supports_neg,
                            max_reference_images=max_references,
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
                            scheduler=resolve_scheduler_class(args.scheduler, config, backend.name),
                            scheduler_name=args.scheduler,
                            quantize=getattr(args, "quantize", None),
                            steps_explicit=getattr(args, "steps_explicit", False),
                            guidance_explicit=getattr(args, "guidance_explicit", False),
                            first_sigma=getattr(args, "first_sigma", None),
                            json_prompt=getattr(args, "json_prompt_enabled", False),
                            skip_signal=skip,
                            review=(
                                make_image_review(
                                    skip,
                                    review_outcome,
                                    progress_callback=progress_callback,
                                    ran_iterations=ran_iterations,
                                    attempt=attempt,
                                    completed_iterations=lambda: completed_iterations,
                                    total_iterations=total_iterations,
                                )
                                if review_enabled(args)
                                else None
                            ),
                            step_callback=_make_step_progress_callback(
                                progress_callback,
                                mode="image",
                                run_index=run_idx,
                                total_runs=args.runs,
                                ran_iterations=ran_iterations,
                                total_iterations=total_iterations,
                                set_name=set_name,
                                prompt_index=prompt_idx,
                                total_prompts=len(prompts),
                            ),
                            upscale_factor=args.upscale,
                            upscale_denoise=resolved_denoise,
                            upscale_steps=(args.upscale_steps if args.upscale else None),
                            upscale_guidance=args.upscale_guidance,
                            upscale_sharpen=args.upscale_sharpen,
                            upscale_save_pre=(args.upscale_save_pre if args.upscale else False),
                            image_path=args.image_path,
                            image_strength=args.image_strength,
                            reference_paths=reference_paths,
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
                            resolved_prompt=planned.resolved_prompt,
                            enhanced_prompt=planned.enhanced_prompt,
                            prompt_input=getattr(args, "prompt_input", None),
                            enhance_record=planned_enhance_record(planned, plan.enhancer, mode="image") or getattr(args, "enhance_record", None),
                            on_prompt_enhanced=_make_prompt_enhanced_callback(
                                progress_callback,
                                mode="image",
                                run_index=run_idx,
                                ran_iterations=ran_iterations,
                                total_iterations=total_iterations,
                                set_name=set_name,
                                prompt_index=prompt_idx,
                                seed=seed,
                            ),
                        )
                        artifacts = ImageWorkingArtifacts(filename=gen_filename)

                        event_context = {
                            "mode": "image",
                            "run_index": run_idx,
                            "total_runs": args.runs,
                            "ran_iterations": ran_iterations,
                            "total_iterations": total_iterations,
                            "set_name": set_name,
                            "prompt_index": prompt_idx,
                            "total_prompts": len(prompts),
                            "prompt": prompt,
                            "seed": seed,
                            "filename": gen_filename,
                        }
                        outcome = _run_workflow_with_progress(
                            workflow,
                            request,
                            artifacts,
                            progress_callback=progress_callback,
                            event_context=event_context,
                        )
                        if outcome is StageOutcome.skipped:
                            _emit_generation_finished(
                                progress_callback,
                                event_context=event_context,
                                status="skipped",
                                filename=artifacts.filename,
                                generation_time=_active_seconds(_img_start, review_outcome),
                                output_path=artifacts.filepath,
                            )
                            if not review_outcome.declined:
                                warnings.warn("Generation skipped by workflow stage.")
                            completed_iterations += 1
                            # A discarded render measured a real render; a Next-stopped one only a partial time.
                            if review_outcome.decision == "discard":
                                image_times.append(_active_seconds(_img_start, review_outcome))
                            skipped_action = skip.consume()
                            if skipped_action == "quit":
                                _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                                raise _QuitBatch()
                            # A discarded render ends the image like a finished one: a Pause or Repeat queued
                            # meanwhile takes effect now, as after a saved image.
                            if review_outcome.decision == "discard" and skipped_action == "pause":
                                if pause_until_resumed(skip, progress_callback=progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations) == "quit":
                                    _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                                    raise _QuitBatch()
                            elif review_outcome.decision == "discard" and skipped_action == "repeat":
                                retries = 0
                                repeats += 1
                                seed = announce_repeat(
                                    progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations, seed_min=seed_min, seed_max=seed_max
                                )
                                continue
                            break
                        elif outcome is StageOutcome.failed:
                            last_failure = "Generation failed in workflow stage."
                            _emit_generation_finished(
                                progress_callback,
                                event_context=event_context,
                                status="failed",
                                message=last_failure,
                                filename=artifacts.filename,
                                generation_time=_active_seconds(_img_start, review_outcome),
                                output_path=artifacts.filepath,
                            )
                            failed_generations += 1
                            warnings.warn("Generation failed in workflow stage.")
                            completed_iterations += 1
                            image_times.append(_active_seconds(_img_start, review_outcome))
                            failed_action = skip.consume()
                            if failed_action == "pause":
                                failed_action = pause_until_resumed(
                                    skip, progress_callback=progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations
                                )
                            if failed_action == "quit":
                                _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                                raise _QuitBatch()
                            break
                        elif outcome is StageOutcome.retry:
                            retries += 1
                            _emit_progress(
                                progress_callback,
                                "generation_retry",
                                mode="image",
                                run_index=run_idx,
                                total_runs=args.runs,
                                ran_iterations=ran_iterations,
                                total_iterations=total_iterations,
                                set_name=set_name,
                                prompt_index=prompt_idx,
                                total_prompts=len(prompts),
                                retry=retries,
                                max_retries=max_retries,
                            )
                            if retries > max_retries:
                                last_failure = f"Generation failed after {max_retries} retries."
                                _emit_generation_finished(
                                    progress_callback,
                                    event_context=event_context,
                                    status="failed",
                                    message=last_failure,
                                    filename=artifacts.filename,
                                    generation_time=_active_seconds(_img_start, review_outcome),
                                    output_path=artifacts.filepath,
                                )
                                failed_generations += 1
                                warnings.warn(f"Generation failed after {max_retries} retries; skipping.")
                                completed_iterations += 1
                                image_times.append(_active_seconds(_img_start, review_outcome))
                                break
                            warnings.warn(f"Workflow stage requested retry ({retries}/{max_retries}).")
                            continue

                        retries = 0
                        generation_time = _active_seconds(_img_start, review_outcome)
                        _emit_generation_finished(
                            progress_callback,
                            event_context=event_context,
                            status="success",
                            filename=artifacts.filename,
                            generation_time=generation_time,
                            output_path=artifacts.filepath,
                        )
                        successful_generations += 1
                        completed_iterations += 1
                        image_times.append(generation_time)
                        action = skip.consume()
                        if action == "quit":
                            print("\n⏹ Quitting batch...")
                            _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                            raise _QuitBatch()
                        if action == "pause":
                            if pause_until_resumed(skip, progress_callback=progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations) == "quit":
                                _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations)
                                raise _QuitBatch()
                            break
                        if action == "repeat":
                            repeats += 1
                            seed = announce_repeat(progress_callback, mode="image", completed_iterations=completed_iterations, total_iterations=total_iterations, seed_min=seed_min, seed_max=seed_max)
                            continue
                        break
            print(f"\nCompleted run {run_idx + 1}/{args.runs}\n{'#' * 30}\n")
        print("\nAll runs completed!\n")
        batch_failed = successful_generations == 0 and failed_generations > 0
        terminal_extra = {"message": last_failure} if batch_failed and last_failure else {}
        _emit_progress(
            progress_callback,
            "batch_failed" if batch_failed else "batch_completed",
            mode="image",
            completed_iterations=completed_iterations,
            total_iterations=total_iterations + repeats,
            **terminal_extra,
        )
    except _QuitBatch:
        pass
