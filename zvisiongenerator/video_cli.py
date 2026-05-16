"""Video CLI — entry point for ziv-video command."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_video_backend
from zvisiongenerator.utils.alignment import align_ltx_frames, align_resolution
from zvisiongenerator.utils.config import load_config, resolve_video_defaults, select_ratio_size_defaults
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import is_remote_lora_reference, resolve_lora_path, resolve_model_path
from zvisiongenerator.utils.prompts import load_prompts_file
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import build_video_workflow


def _build_video_parser(*, prog: str = "ziv-video") -> argparse.ArgumentParser:
    """Build the argument parser for ziv-video."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Z-Vision Video Generator — text-to-video and image-to-video.",
        epilog=f"Example usage: {prog} -m models/ltx-mlx --ratio 16:9 --size m --prompt 'a sunset'",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Model alias, local path, or supported/configured HuggingFace repo ID.")
    parser.add_argument("-p", "--prompts-file", type=str, default="prompts.yaml", help="Path to YAML prompts file.")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of batch runs.")
    parser.add_argument("--prompt", type=str, default=None, help="Inline prompt (overrides --prompts-file).")
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
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True, help="Include audio in output (default: enabled).")
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


def _unknown_video_model_guidance(platform_key: str) -> str:
    """Return platform-aware guidance for unknown video model selections."""

    suffix = "supported/configured HuggingFace LTX repo IDs, or a local path containing 'ltx'."
    if platform_key == "darwin":
        return f"Use a supported LTX model: 'ltx-4' or 'ltx-8' on macOS, known LTX repo prefixes, {suffix}"
    if platform_key == "win32":
        return f"Use a supported LTX model: 'ltx-2.3' on Windows, known LTX repo prefixes, {suffix}"
    if platform_key.startswith("linux"):
        return f"Use a supported LTX model: 'ltx-2.3' on Linux, known LTX repo prefixes, {suffix}"
    return f"Use a supported LTX model: 'ltx-4' or 'ltx-8' on macOS, 'ltx-2.3' on Windows/Linux, known LTX repo prefixes, {suffix}"


def main(*, prog: str = "ziv-video") -> None:
    """Entry point for ziv-video CLI."""
    parser = _build_video_parser(prog=prog)
    args = parser.parse_args()

    # Validation (matching image CLI pattern)
    if args.model is None:
        parser.error("--model is required. Provide a model alias, local path, or supported/configured HuggingFace repo ID.")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.upscale is not None and args.upscale != 2:
        parser.error("--upscale only supports factor 2 (LTX spatial upscaler)")

    # Expand ~ in filesystem-only path arguments. Model/LoRA tokens are resolved by shared helpers.
    if args.image_path:
        args.image_path = str(Path(args.image_path).expanduser())
    args.output = str(Path(args.output).expanduser())
    args.prompts_file = str(Path(args.prompts_file).expanduser())

    # Load config early so aliases are available for model resolution
    try:
        config = load_config()
    except ValueError as e:
        parser.error(str(e))

    # Resolve friendly model names (e.g. "ltx-2-mlx" → ~/.ziv/models/...)
    try:
        args.model = resolve_model_path(args.model, aliases=config.get("model_aliases", {}), platform_key=sys.platform)
    except ValueError as e:
        parser.error(str(e))

    # Validate prompt source BEFORE heavy operations (model loading, ffmpeg check)
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.prompt is None:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.is_file():
            parser.error(f"Prompts file not found: {args.prompts_file}. Provide --prompt or a valid --prompts-file.")

    # Check ffmpeg is available (required by video backends)
    ensure_ffmpeg()

    # Detect video model family
    model_info = detect_video_model(args.model)
    if model_info.family == "unknown":
        parser.error(f"Could not detect video model family for '{args.model}'. {_unknown_video_model_guidance(sys.platform)}")

    # Validate --image file exists
    if args.image_path and not os.path.isfile(args.image_path):
        parser.error(f"Image file not found: {args.image_path}")

    # Validate T2V-only model with --image
    if args.image_path and not model_info.supports_i2v:
        parser.error(f"Model '{args.model}' does not support image-to-video.")

    # Default ratio/size from shared config-backed semantics.
    vgen_cfg = config.get("video_generation", {})
    vsizes = config.get("video_sizes", {})
    default_ratio, default_size = select_ratio_size_defaults(
        vgen_cfg.get("default_ratio"),
        tuple(vsizes.keys()),
        {ratio: tuple(size_map.keys()) for ratio, size_map in vsizes.items()},
        vgen_cfg.get("default_size"),
        fallback_ratio="16:9",
        fallback_size="m",
    )
    if args.ratio is None:
        args.ratio = default_ratio
    if args.size is None:
        args.size = default_size

    # Validate ratio/size from flat video size config. Model family remains authoritative for presets/backend behavior.
    if vsizes:
        if args.ratio not in vsizes:
            parser.error(f"Unknown ratio '{args.ratio}'. Valid: {list(vsizes.keys())}")
        if args.size not in vsizes.get(args.ratio, {}):
            parser.error(f"Unknown size '{args.size}' for ratio '{args.ratio}'. Valid: {list(vsizes.get(args.ratio, {}).keys())}")

    # Parse LoRA args (matching image CLI pattern)
    lora_paths, lora_weights = None, None
    if args.lora is not None:
        try:
            parsed = parse_lora_arg(args.lora)
        except ValueError as e:
            parser.error(str(e))
        remote_loras = [name for name, _ in parsed if is_remote_lora_reference(name)]
        if remote_loras:
            parser.error(f"Remote HuggingFace LoRA references are not supported: {', '.join(remote_loras)}. Import the LoRA locally or pass a local LoRA path.")
        lora_paths = [resolve_lora_path(name) for name, _ in parsed]
        lora_weights = [weight for _, weight in parsed]
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
    args.width = defaults["width"]
    args.height = defaults["height"]
    args.num_frames = defaults["num_frames"]

    steps_explicitly_set = "steps" in cli_overrides
    _resolve_upscale_steps(args, config, model_info.family, steps_explicitly_set=steps_explicitly_set)

    # Audio flag normalization
    args.no_audio = not getattr(args, "audio", True)

    # Alignment corrections using model metadata
    # When upscaling, use 64-alignment so half-res (dim//2) stays 32-aligned
    alignment = 64 if args.upscale else model_info.resolution_alignment
    args.width, args.height = align_resolution(
        args.width,
        args.height,
        alignment,
        model_info.family.upper(),
    )
    args.num_frames = align_ltx_frames(args.num_frames, model_info.frame_alignment)

    if args.width < 64 or args.height < 64:
        parser.error(f"Resolved dimensions {args.width}x{args.height} are too small (minimum 64x64)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Select backend
    try:
        backend = get_video_backend(model_info.backend)
    except RuntimeError as e:
        parser.error(str(e))

    # Determine mode hint
    mode = "i2v" if args.image_path else "t2v"

    # Prepare LoRA tuples for backend
    loras: list[tuple[str, float]] | None = None
    if lora_paths:
        loras = list(zip(lora_paths, lora_weights, strict=False))

    # Load model
    print(f"Loading {model_info.family.upper()} video model: {args.model}")
    load_kwargs: dict[str, Any] = {}
    if args.upscale:
        load_kwargs["upscale"] = True
    try:
        model, model_info = backend.load_model(
            args.model,
            mode=mode,
            low_memory=args.low_memory,
            loras=loras,
            **load_kwargs,
        )
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
        parser.error(str(e))

    # Load prompts
    if args.prompt is not None:
        prompts_data: dict[str, list[tuple[str, str | None]]] = {"prompt": [(args.prompt, None)]}
    else:
        prompts_data = load_prompts_file(args.prompts_file)

    # Build workflow
    workflow = build_video_workflow(args)

    # Run batch
    run_video_batch(
        backend=backend,
        model=model,
        model_info=model_info,
        workflow=workflow,
        prompts_data=prompts_data,
        config=config,
        args=args,
    )


if __name__ == "__main__":
    main()
