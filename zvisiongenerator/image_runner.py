"""Batch orchestration — extracted from cli.py::main().

Handles the triple-nested run/set/prompt batch loop, timing/ETA,
interactive controls (skip/quit/pause/repeat via SkipSignal), and
_QuitBatch exception for clean multi-level loop exit.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import random
import time
import warnings
from collections.abc import Callable
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
    workflow,
    request: ImageGenerationRequest,
    artifacts: ImageWorkingArtifacts,
    *,
    progress_callback: ProgressCallback | None,
    event_context: dict[str, Any],
) -> StageOutcome:
    """Run an image workflow while emitting structured stage progress events."""
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


def run_batch(
    backend: ImageBackend,
    model: Any,
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    model_info: ImageModelInfo,
    progress_callback: ProgressCallback | None = None,
    enable_interactive_controls: bool = True,
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
        _emit_progress(progress_callback, "batch_completed", mode="image", total_iterations=0, completed_iterations=0)
        return
    ran_iterations = 0
    print(f"Total iterations to run: {total_iterations}\n")
    _emit_progress(progress_callback, "batch_started", mode="image", total_iterations=total_iterations, total_runs=args.runs)

    skip = skip_signal or SkipSignal()
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
        if enable_interactive_controls:
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
                            _emit_progress(
                                progress_callback,
                                "generation_finished",
                                mode="image",
                                status="skipped",
                                run_index=run_idx,
                                total_runs=args.runs,
                                ran_iterations=ran_iterations,
                                total_iterations=total_iterations,
                                set_name=set_name,
                                prompt_index=prompt_idx,
                                total_prompts=len(prompts),
                                filename=artifacts.filename,
                                generation_time=time.time() - _img_start,
                                output_path=artifacts.filepath,
                            )
                            warnings.warn("Generation skipped by workflow stage.")
                            image_times.append(time.time() - _img_start)
                            if skip.consume() == "quit":
                                raise _QuitBatch()
                            break
                        elif outcome is StageOutcome.failed:
                            _emit_progress(
                                progress_callback,
                                "generation_finished",
                                mode="image",
                                status="failed",
                                run_index=run_idx,
                                total_runs=args.runs,
                                ran_iterations=ran_iterations,
                                total_iterations=total_iterations,
                                set_name=set_name,
                                prompt_index=prompt_idx,
                                total_prompts=len(prompts),
                                filename=artifacts.filename,
                                generation_time=time.time() - _img_start,
                                output_path=artifacts.filepath,
                            )
                            warnings.warn("Generation failed in workflow stage.")
                            image_times.append(time.time() - _img_start)
                            if skip.consume() == "quit":
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
                                warnings.warn(f"Generation failed after {max_retries} retries; skipping.")
                                image_times.append(time.time() - _img_start)
                                break
                            warnings.warn(f"Workflow stage requested retry ({retries}/{max_retries}).")
                            continue

                        retries = 0
                        generation_time = time.time() - _img_start
                        _emit_progress(
                            progress_callback,
                            "generation_finished",
                            mode="image",
                            status="success",
                            run_index=run_idx,
                            total_runs=args.runs,
                            ran_iterations=ran_iterations,
                            total_iterations=total_iterations,
                            set_name=set_name,
                            prompt_index=prompt_idx,
                            total_prompts=len(prompts),
                            filename=artifacts.filename,
                            generation_time=generation_time,
                            output_path=artifacts.filepath,
                        )
                        action = skip.consume()
                        if action == "quit":
                            print("\n⏹ Quitting batch...")
                            _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=len(image_times) + 1, total_iterations=total_iterations)
                            raise _QuitBatch()
                        elif action == "pause":
                            image_times.append(time.time() - _img_start)
                            _emit_progress(progress_callback, "job_paused", mode="image", completed_iterations=len(image_times), total_iterations=total_iterations)
                            print("\n⏸ Paused. Press any key to continue...")
                            skip.wait_for_key()
                            _emit_progress(progress_callback, "job_resumed", mode="image", completed_iterations=len(image_times), total_iterations=total_iterations)
                            print("▶ Resumed.\n")
                            break
                        elif action == "repeat":
                            image_times.append(time.time() - _img_start)
                            seed = args.seed if args.seed is not None else random.randint(seed_min, seed_max)
                            _emit_progress(progress_callback, "generation_repeat", mode="image", completed_iterations=len(image_times), total_iterations=total_iterations)
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
        _emit_progress(progress_callback, "batch_completed", mode="image", completed_iterations=len(image_times), total_iterations=total_iterations)
    except _QuitBatch:
        pass
    finally:
        if enable_interactive_controls:
            skip.stop()
