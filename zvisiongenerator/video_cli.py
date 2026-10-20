"""Video CLI — entry point for ziv-video command."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends import accelerator_memory_budget_or_none, get_video_backend, supports_video_audio, supports_video_extend, supports_video_upscale_review
from zvisiongenerator.preflight import run_preflight
from zvisiongenerator.utils.alignment import aligned_video_dims
from zvisiongenerator.utils.app_log import setup_logging
from zvisiongenerator.utils.config import load_config, ltx_load_options, resolve_video_defaults, select_ratio_size_defaults, validate_video_size_choice
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.interactive import SkipSignal, stdin_is_interactive
from zvisiongenerator.utils.lora import parse_lora_arg, resolve_lora_references
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.model_memory import clip_memory_fit
from zvisiongenerator.utils.paths import resolve_lora_path, resolve_model_path
from zvisiongenerator.utils.prompt_compose import is_original_prompt
from zvisiongenerator.utils.prompts import inspect_prompts_file
from zvisiongenerator.utils.video_extend import DEFAULT_EXTEND_SECONDS, EXTEND_FRAME_STEP, apply_extend_args, load_extend_source, resolve_extend_frames
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.enhance_cli import add_enhance_arguments, parse_enhance_args
from zvisiongenerator.workflows import build_video_workflow

if TYPE_CHECKING:
    from zvisiongenerator.core.video_types import ExtendSource

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_FILE = "prompts.yaml"
# Options that choose a size or a frame count; an extension keeps the source's size and takes its own length.
_EXTEND_REFUSED_OPTIONS = (
    ("image_path", "--image", "an extension continues the source video, not an image"),
    ("upscale", "--upscale", "an extension is generated at the source's size"),
    ("ratio", "--ratio", "an extension keeps the source video's size"),
    ("size", "--size", "an extension keeps the source video's size"),
    ("width", "--width", "an extension keeps the source video's size"),
    ("height", "--height", "an extension keeps the source video's size"),
    ("frames", "--frames", "use --extend-seconds or --extend-frames for the length of an extension"),
)


def _build_video_parser(*, prog: str = "ziv-video") -> argparse.ArgumentParser:
    """Build the argument parser for ziv-video."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Z-Vision Video Generator — text-to-video and image-to-video.",
        epilog=f"Example usage: {prog} -m models/ltx-mlx --ratio 16:9 --size m --prompt 'a sunset'",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Model alias, local path, or supported/configured HuggingFace repo ID. With --extend, defaults to the source video's model.")
    parser.add_argument("-p", "--prompts-file", type=str, default=None, help=f"Path to YAML prompts file (default: {_DEFAULT_PROMPTS_FILE}).")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of batch runs.")
    parser.add_argument("--prompt", type=str, default=None, help="Inline prompt: what happens (overrides --prompts-file). May be empty when --scene or --extend is given.")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene: who and where, look, style, camera. Sent before the prompt. Alone it makes a scene-only run; with --prompts-file it fills entries without their own 'scene:'. With --extend, defaults to the source video's scene.",
    )
    parser.add_argument(
        "--extend",
        dest="extend_path",
        type=str,
        default=None,
        metavar="SRC",
        help="Continue the video SRC: generate what happens next and append it, writing a new file. The prompt may be empty. macOS only.",
    )
    extend_length = parser.add_mutually_exclusive_group()
    extend_length.add_argument(
        "--extend-seconds",
        type=float,
        default=None,
        metavar="S",
        help=f"Length to add with --extend in seconds, converted at the source's frame rate (default: {DEFAULT_EXTEND_SECONDS:g}).",
    )
    extend_length.add_argument(
        "--extend-frames",
        type=int,
        default=None,
        metavar="N",
        help=f"Frames to add with --extend, rounded to a multiple of {EXTEND_FRAME_STEP} (minimum {EXTEND_FRAME_STEP}).",
    )
    parser.add_argument("--image", dest="image_path", type=str, default=None, help="Input image for image-to-video.")
    parser.add_argument("--ratio", type=str, default=None, help="Aspect ratio for generated video (e.g. 16:9, 9:16, 1:1).")
    parser.add_argument("-s", "--size", type=str, default=None, help="Resolution scale (e.g. s, m, l). Default from config.")
    parser.add_argument("-W", "--width", type=int, default=None, help="Override video width. LTX: must be divisible by 32.")
    parser.add_argument("-H", "--height", type=int, default=None, help="Override video height. LTX: must be divisible by 32.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames. LTX: must follow 8k+1 pattern (9,17,...,97,121).")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random).")
    parser.add_argument("-o", "--output", type=str, default=".", help="Output directory.")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4"], help="Output format.")
    parser.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=True, help="Low-memory mode for LTX (default: enabled).")
    parser.add_argument("--lora", type=str, default=None, help="Comma-separated LoRAs with optional weights: name1:0.8,name2:0.5. Bare names resolve from ~/.ziv/loras/.")
    parser.add_argument("--upscale", type=int, default=None, help="Upscale factor (only 2 accepted).")
    parser.add_argument(
        "--review",
        action="store_true",
        default=False,
        help="Pause after each half-size render, before the upscale: press y to upscale it, n to discard it or q to quit. Needs --upscale and a terminal.",
    )
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True, help="Include audio in output (default: enabled).")
    add_enhance_arguments(parser, mode="video")
    return parser


def _resolve_upscale_steps(args: argparse.Namespace, config: dict[str, Any], model_family: str, *, steps_explicitly_set: bool) -> None:
    """Apply video upscale step defaults and model step caps to parsed args."""
    if args.upscale:
        upscale_cfg = config.get("video_model_presets", {}).get("ltx", {}).get("upscale", {})
        if not steps_explicitly_set:
            args.steps = upscale_cfg.get("default_upscale_steps", 8)
        args.upscale_steps = args.steps
    else:
        args.upscale_steps = None

    if model_family == "ltx":
        max_steps = 8
        if args.steps > max_steps:
            warnings.warn(
                f"LTX distilled model supports max {max_steps} denoising steps; capping {args.steps} -> {max_steps}",
                stacklevel=2,
            )
            args.steps = max_steps
            if args.upscale_steps is not None:
                args.upscale_steps = args.steps


def _check_extend_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse extend options that cannot be used as given: no capability, options that choose a size, a bad length, a length without ``--extend``."""
    if args.extend_path is None:
        for value, flag in ((args.extend_seconds, "--extend-seconds"), (args.extend_frames, "--extend-frames")):
            if value is not None:
                parser.error(f"{flag} requires --extend")
        return
    if not args.extend_path.strip():
        parser.error("--extend needs a video path")
    if not supports_video_extend():
        parser.error("--extend is not available on this platform.")
    for attribute, flag, reason in _EXTEND_REFUSED_OPTIONS:
        if getattr(args, attribute) is not None:
            parser.error(f"{flag} cannot be used with --extend: {reason}")
    if args.extend_frames is not None and args.extend_frames < EXTEND_FRAME_STEP:
        parser.error(f"--extend-frames must be at least {EXTEND_FRAME_STEP}")
    if args.extend_seconds is not None and (not math.isfinite(args.extend_seconds) or args.extend_seconds <= 0):
        parser.error("--extend-seconds must be a finite number greater than 0")


def _load_extend_source(parser: argparse.ArgumentParser, path: str) -> ExtendSource:
    """Return the checked source clip, or exit with every problem found before any model loads."""
    ensure_ffmpeg()
    try:
        return load_extend_source(path)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))


def _model_available_locally(model: str, config: dict[str, Any]) -> bool:
    """Return whether a model reference resolves to a fully downloaded local model."""
    try:
        resolved = resolve_model_path(model, aliases=config.get("model_aliases", {}), platform_key=sys.platform)
    except ValueError:
        return False
    return find_local_model_dir(resolved) is not None


def _extend_model(model: str | None, source: ExtendSource, config: dict[str, Any]) -> str:
    """Return the model for an extension: the one given, else the source's when available locally, else the configured default.

    Raises:
        ValueError: If no model is given and neither the source nor the config names one.
    """
    if model is not None:
        return model
    if source.model and _model_available_locally(source.model, config):
        return source.model
    if source.model:
        warnings.warn(f"The model this video was made with ({source.model}) is not available locally; using the configured default model.", stacklevel=2)
    default = ((config.get("ui") or {}).get("default_models") or {}).get("video")
    if not default:
        raise ValueError("--model is required: the source video records no model available locally and no default video model is configured.")
    return default


def _extend_lora(lora: str | None, source: ExtendSource) -> str | None:
    """Return the ``--lora`` value for an extension: the one given, else the source's LoRAs that still exist (others are dropped with a warning)."""
    if lora is not None or not source.lora:
        return lora
    try:
        recorded = parse_lora_arg(source.lora)
    except ValueError:
        warnings.warn(f"The LoRAs recorded in the source video could not be read ({source.lora}); using none.", stacklevel=2)
        return None
    kept: list[str] = []
    for name, weight in recorded:
        if os.path.exists(resolve_lora_path(name)):
            kept.append(f"{name}:{weight:g}")
        else:
            warnings.warn(f"The LoRA {name} used by the source video was not found; skipping it.", stacklevel=2)
    return ",".join(kept) or None


def _extend_prompt(prompt: str | None, source: ExtendSource) -> str | None:
    """Return the inline prompt for an extension: the one given, else a legacy source's recorded prompt, else ``None`` (an empty prompt).

    A source with a recorded scene or no recorded prompt continues from its scene alone. A legacy source has only its
    prompt, so omitting ``--prompt`` keeps it; the run then strips quoted sentences and continues naturally.
    """
    if prompt is None and source.legacy:
        return source.prompt
    return prompt


def _warn_if_still_original_prompt(prompt: str | None, source: ExtendSource) -> None:
    """Warn when the prompt for a source without a recorded scene is still that source's own prompt (it would repeat its action)."""
    if source.legacy and prompt is not None and source.prompt is not None and is_original_prompt(prompt, source.prompt):
        warnings.warn("Still the original prompt. Change it to what happens next.", stacklevel=2)


def _clip_fit(args: argparse.Namespace, config: dict[str, Any], *, upscale: bool) -> dict[str, Any] | None:
    """Return how the run's clip fits this machine's memory (``status``, ``required_gb``, ``budget_gb``), or ``None`` when unknown."""
    return clip_memory_fit(
        args.model,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        low_memory=args.low_memory,
        upscale=upscale,
        budget=accelerator_memory_budget_or_none(),
        model_name=getattr(args, "model_name", None),
        app_config=config,
    )


def _warn_if_too_large(args: argparse.Namespace, config: dict[str, Any], *, extending: bool) -> None:
    """Warn when the clip, or the window of an extension, may not fit in memory; the run continues either way."""
    fit = _clip_fit(args, config, upscale=bool(args.upscale))
    if fit is None or fit["status"] == "fits":
        return
    if fit.get("resource") == "system":
        # The weights, not the clip, are what fill system memory: a shorter or smaller clip does not help.
        warnings.warn(
            f"The model needs about {fit['required_gb']:g} GB of system memory; this machine has {fit['budget_gb']:g} GB ({fit['status'].replace('_', ' ')}). Continuing; it may be slow or fail. ltx-4 needs less than ltx-8, and --upscale adds about 4 GB.",
            stacklevel=2,
        )
        return
    if extending:
        subject = f"The {args.num_frames}-frame extension window"
        advice = "Use --extend-seconds or --extend-frames for a shorter extension"
    else:
        subject = f"At {args.width}x{args.height}, {args.num_frames} frames this clip"
        advice = "Use fewer --frames or a smaller --size" + ("" if args.low_memory else ", and keep --low-memory on")
    warnings.warn(
        f"{subject} needs about {fit['required_gb']:g} GB; this machine's budget is {fit['budget_gb']:g} GB ({fit['status'].replace('_', ' ')}). Continuing; it may be slow or fail. {advice}.",
        stacklevel=2,
    )


def _apply_ratio_size_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Fill ``--ratio`` and ``--size`` from the config defaults and refuse values the size table does not have."""
    vgen_cfg = config.get("video_generation", {})
    vsizes = config.get("video_sizes", {})
    default_ratio, default_size = select_ratio_size_defaults(
        vgen_cfg.get("default_ratio"),
        tuple(vsizes.keys()),
        {ratio: tuple(size_map.keys()) for ratio, size_map in vsizes.items()},
        vgen_cfg.get("default_size"),
        fallback_ratio="16:9",
        fallback_size="s",
    )
    if args.ratio is None:
        args.ratio = default_ratio
    if args.size is None:
        args.size = default_size

    # Validate ratio/size from flat video size config. Model family remains authoritative for presets/backend behavior.
    try:
        validate_video_size_choice(config, args.ratio, args.size)
    except ValueError as e:
        parser.error(str(e))


def _unknown_video_model_guidance() -> str:
    """Return guidance for unknown video model selections."""

    return "Use a supported LTX model: 'ltx-4' or 'ltx-8', a converted LTX model (ziv model model --model-type ltx-2.3), known LTX repo prefixes, supported/configured HuggingFace LTX repo IDs, or a local path containing 'ltx'."


def main(*, prog: str = "ziv-video") -> None:
    """Entry point for ziv-video CLI."""
    parser = _build_video_parser(prog=prog)
    args = parser.parse_args()
    setup_logging("video")

    # Validation (matching image CLI pattern)
    if args.model is None and args.extend_path is None:
        parser.error("--model is required. Provide a model alias, local path, or supported/configured HuggingFace repo ID.")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.upscale is not None and args.upscale != 2:
        parser.error("--upscale only supports factor 2 (LTX spatial upscaler)")
    if args.review and args.upscale is None:
        parser.error("--review needs --upscale: without an upscale there is no half-size render to review.")
    if args.review and not supports_video_upscale_review():
        parser.error("--review is not supported by the video backend of this platform.")
    if args.review and not stdin_is_interactive():
        parser.error("--review needs an interactive terminal to ask for each decision.")
    parse_enhance_args(parser, args, mode="video")
    _check_extend_options(parser, args)

    # Expand ~ in filesystem-only path arguments. Model/LoRA tokens are resolved by shared helpers.
    if args.image_path:
        args.image_path = str(Path(args.image_path).expanduser())
    if args.extend_path is not None:
        args.extend_path = str(Path(args.extend_path).expanduser())
    args.output = str(Path(args.output).expanduser())
    scene_given = args.scene is not None
    args.scene = (args.scene or "").strip() or None
    args.scene_input = args.prompt_input = None
    args.extend_source = None
    # One inline iteration: no prompt source was given, and there is a scene or a clip to continue (the prompt is then empty).
    inline_run = args.prompt is None and args.prompts_file is None and (args.scene is not None or args.extend_path is not None)
    args.prompts_file = str(Path(args.prompts_file or _DEFAULT_PROMPTS_FILE).expanduser())

    # Load config early so aliases are available for model resolution
    try:
        config = load_config()
    except ValueError as e:
        parser.error(str(e))

    # An extension takes its scene, model and LoRAs from the source clip unless the options say otherwise.
    source = _load_extend_source(parser, args.extend_path) if args.extend_path is not None else None
    if source is not None:
        if not scene_given:
            args.scene = source.scene
        _warn_if_still_original_prompt(args.prompt, source)
        if inline_run:
            args.prompt = _extend_prompt(args.prompt, source)
        try:
            args.model = _extend_model(args.model, source, config)
        except ValueError as e:
            parser.error(str(e))
        args.lora = _extend_lora(args.lora, source)

    # Resolve friendly model names (e.g. "ltx-2-mlx" → ~/.ziv/models/...); the picked name chooses the quantize level
    # and is what the video records, as the Web UI does: on Windows/Linux ltx-4 and ltx-8 resolve to the same repo.
    model_name = args.model
    args.model_name = model_name
    try:
        args.model = resolve_model_path(args.model, aliases=config.get("model_aliases", {}), platform_key=sys.platform)
    except ValueError as e:
        parser.error(str(e))

    # Validate prompt source BEFORE heavy operations (model loading, ffmpeg check)
    if args.prompt is not None and not args.prompt.strip() and args.scene is None and source is None:
        parser.error("--prompt must not be empty")
    if args.prompt is None and not inline_run:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.is_file():
            parser.error(f"Prompts file not found: {args.prompts_file}. Provide --prompt or a valid --prompts-file.")

    # Check ffmpeg is available (required by video backends)
    ensure_ffmpeg()

    # Detect video model family
    model_info = detect_video_model(args.model)
    if model_info.family == "unknown":
        parser.error(f"Could not detect video model family for '{args.model}'. {_unknown_video_model_guidance()}")

    # Validate --image file exists
    if args.image_path and not os.path.isfile(args.image_path):
        parser.error(f"Image file not found: {args.image_path}")

    # Validate T2V-only model with --image
    if args.image_path and not model_info.supports_i2v:
        parser.error(f"Model '{args.model}' does not support image-to-video.")

    # An extension keeps the source's size, so it has no ratio or size.
    if source is None:
        _apply_ratio_size_defaults(parser, args, config)

    # Parse LoRA args (matching image CLI pattern)
    lora_paths, lora_weights = None, None
    if args.lora is not None:
        try:
            # The diffusers backend also accepts a LoRA directory, so require existence rather than a file.
            lora_paths, lora_weights = resolve_lora_references(args.lora, require_file=False)
        except ValueError as e:
            parser.error(str(e))
    args.lora_paths = lora_paths
    args.lora_weights = lora_weights

    # Resolve defaults (CLI > preset > global)
    cli_overrides: dict[str, Any] = {}
    if args.ratio is not None:
        cli_overrides["ratio"] = args.ratio
    if args.size is not None:
        cli_overrides["size"] = args.size
    if args.steps is not None:
        cli_overrides["steps"] = args.steps
    if args.width is not None:
        cli_overrides["width"] = args.width
    if args.height is not None:
        cli_overrides["height"] = args.height
    if args.frames is not None:
        cli_overrides["num_frames"] = args.frames

    defaults = resolve_video_defaults(model_info.family, config, cli_overrides)

    # Apply resolved defaults to args
    args.steps = defaults["steps"]

    steps_explicitly_set = "steps" in cli_overrides
    _resolve_upscale_steps(args, config, model_info.family, steps_explicitly_set=steps_explicitly_set)

    # Audio flag normalization; a backend without an audio track records and rewrites for silent video.
    args.no_audio = not getattr(args, "audio", True) or not supports_video_audio()

    if source is not None:
        # The source decides size and frame rate; its length was checked on load, so there is no resolution or frame alignment.
        extend_frames, note = resolve_extend_frames(
            frames=args.extend_frames, seconds=args.extend_seconds, fps=source.fps, strict=False, frames_name="--extend-frames", seconds_name="--extend-seconds"
        )
        if note:
            warnings.warn(note, stacklevel=2)
        apply_extend_args(args, source, extend_frames)
    else:
        args.width = defaults["width"]
        args.height = defaults["height"]
        args.num_frames = defaults["num_frames"]
        args.fps = model_info.default_fps

        # Alignment corrections using model metadata (an upscaled clip uses 64-alignment so its half-size stage stays 32-aligned)
        args.width, args.height, args.num_frames = aligned_video_dims(
            args.width,
            args.height,
            args.num_frames,
            upscale=bool(args.upscale),
            resolution_alignment=model_info.resolution_alignment,
            frame_alignment=model_info.frame_alignment,
            family=model_info.family.upper(),
        )

        if args.width < 64 or args.height < 64:
            parser.error(f"Resolved dimensions {args.width}x{args.height} are too small (minimum 64x64)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Select backend
    try:
        backend = get_video_backend(model_info.backend)
    except RuntimeError as e:
        parser.error(str(e))

    _warn_if_too_large(args, config, extending=source is not None)

    # Determine mode hint
    mode = "extend" if source is not None else "i2v" if args.image_path else "t2v"

    # Prepare LoRA tuples for backend
    loras: list[tuple[str, float]] | None = None
    if lora_paths:
        loras = list(zip(lora_paths, lora_weights, strict=False))

    # Load prompts and run preflight before the model: prompt-file and enhancer errors fail fast, and the enhancer is unloaded first
    enhance_by_set = scene_by_set = None
    if args.prompt is not None or inline_run:
        prompts_data: dict[str, list[tuple[str, str | None]]] = {"prompt": [(args.prompt or "", None)]}
    else:
        try:
            inspection = inspect_prompts_file(args.prompts_file)
        except (FileNotFoundError, ValueError) as e:
            parser.error(str(e))
        scene_by_set = inspection.scene_by_set
        prompts_data = inspection.prompts_data
        enhance_by_set = inspection.enhance_by_set
    try:
        plan = run_preflight(prompts_data, config, args, mode="video", model_family=model_info.family, enhance_by_set=enhance_by_set, scene=args.scene, scene_by_set=scene_by_set)
    except (ValueError, RuntimeError) as e:
        logger.exception("Prompt preflight failed")
        parser.error(str(e))
    if plan.cancelled:
        return

    # The key listener only runs for a review: it would otherwise take over the terminal for nothing. It starts after
    # ensure_ffmpeg and the preflight, which can both prompt with input().
    skip = SkipSignal(quit_only=True) if args.review else None
    if skip is not None:
        skip.start()
        print("Commands: [q] quit")
    try:
        # Load model
        print(f"Loading {model_info.family.upper()} video model: {args.model}")
        try:
            model, model_info = backend.load_model(
                args.model,
                mode=mode,
                low_memory=args.low_memory,
                loras=loras,
                **ltx_load_options(config, model_name, upscale=bool(args.upscale)),
            )
        except (RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
            logger.exception("Failed to load video model %s", args.model)
            parser.error(str(e))

        # Build workflow
        workflow = build_video_workflow(args, enhance=plan.has_rewrites)

        run_video_batch(
            backend=backend,
            model=model,
            model_info=model_info,
            workflow=workflow,
            prompts_data=prompts_data,
            config=config,
            args=args,
            plan=plan,
            skip_signal=skip,
        )
    finally:
        if skip is not None:
            skip.stop()


if __name__ == "__main__":
    main()
