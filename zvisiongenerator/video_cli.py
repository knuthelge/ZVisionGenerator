"""Video CLI — entry point for ziv-video command."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_video_backend
from zvisiongenerator.utils.config import load_config, resolve_video_defaults
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import resolve_lora_path, resolve_model_path
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
    parser.add_argument("-m", "--model", type=str, default=None, help="Model path or HF repo ID.")
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


def _align_resolution(width: int, height: int, divisor: int, label: str = "Video") -> tuple[int, int]:
    """Round width/height to nearest multiple of divisor."""
    half = divisor // 2
    aligned_w = ((width + half) // divisor) * divisor
    aligned_h = ((height + half) // divisor) * divisor
    if aligned_w != width or aligned_h != height:
        warnings.warn(
            f"{label} requires dimensions divisible by {divisor}. Adjusted {width}x{height} -> {aligned_w}x{aligned_h}",
            stacklevel=2,
        )
    return aligned_w, aligned_h


def _align_ltx_frames(frames: int, alignment: int = 8) -> int:
    """Round frame count to nearest valid alignment*k+1 value."""
    # For LTX (alignment=8): 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
    if alignment <= 0:
        return frames
    k = max(1, round((frames - 1) / alignment))
    aligned = alignment * k + 1
    if aligned != frames:
        warnings.warn(f"Video model requires frames = {alignment}k+1. Adjusted {frames} -> {aligned}", stacklevel=2)
    return aligned


def main(*, prog: str = "ziv-video") -> None:
    """Entry point for ziv-video CLI."""
    parser = _build_video_parser(prog=prog)
    args = parser.parse_args()

    # Validation (matching image CLI pattern)
    if args.model is None:
        parser.error("--model is required. Provide a model path or HuggingFace repo ID.")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.upscale is not None and args.upscale != 2:
        parser.error("--upscale only supports factor 2 (LTX spatial upscaler)")

    # Expand ~ in path arguments
    args.model = str(Path(args.model).expanduser())
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
    args.model = resolve_model_path(args.model, aliases=config.get("model_aliases", {}))

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
        parser.error(f"Could not detect video model family for '{args.model}'. Use a supported model: dgrauet/ltx-*")

    # Validate --image file exists
    if args.image_path and not os.path.isfile(args.image_path):
        parser.error(f"Image file not found: {args.image_path}")

    # Validate T2V-only model with --image
    if args.image_path and not model_info.supports_i2v:
        parser.error(f"Model '{args.model}' does not support image-to-video.")

    # Default ratio/size from config (matching image CLI pattern)
    vgen_cfg = config.get("video_generation", {})
    if args.ratio is None:
        args.ratio = vgen_cfg.get("default_ratio", "16:9")
    if args.size is None:
        args.size = vgen_cfg.get("default_size", "m")

    # Validate ratio/size for this model family
    vsizes = config.get("video_sizes", {})
    family_sizes = vsizes.get(model_info.family, {})
    if family_sizes:
        if args.ratio not in family_sizes:
            parser.error(f"Unknown ratio '{args.ratio}' for {model_info.family}. Valid: {list(family_sizes.keys())}")
        if args.size not in family_sizes.get(args.ratio, {}):
            parser.error(f"Unknown size '{args.size}' for ratio '{args.ratio}'. Valid: {list(family_sizes.get(args.ratio, {}).keys())}")

    # Parse LoRA args (matching image CLI pattern)
    lora_paths, lora_weights = None, None
    if args.lora is not None:
        try:
            parsed = parse_lora_arg(args.lora)
        except ValueError as e:
            parser.error(str(e))
        lora_paths = [resolve_lora_path(str(Path(name).expanduser())) for name, _ in parsed]
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

    # Resolve upscale defaults from config (before step cap so cap applies to final value)
    steps_explicitly_set = "steps" in cli_overrides
    if args.upscale:
        upscale_cfg = config.get("video_model_presets", {}).get("ltx", {}).get("upscale", {})
        if not steps_explicitly_set:
            args.steps = upscale_cfg.get("default_upscale_steps", 8)
        args.upscale_steps = args.steps
    else:
        args.upscale_steps = None

    # LTX: cap steps at max supported value (after upscale defaults)
    if model_info.family == "ltx":
        _LTX_MAX_STEPS = 8
        if args.steps > _LTX_MAX_STEPS:
            warnings.warn(
                f"LTX distilled model supports max {_LTX_MAX_STEPS} denoising steps; capping {args.steps} \u2192 {_LTX_MAX_STEPS}",
                stacklevel=2,
            )
            args.steps = _LTX_MAX_STEPS
            if args.upscale_steps is not None:
                args.upscale_steps = args.steps

    # Audio flag normalization
    args.no_audio = not getattr(args, "audio", True)

    # Alignment corrections using model metadata
    # When upscaling, use 64-alignment so half-res (dim//2) stays 32-aligned
    alignment = 64 if args.upscale else model_info.resolution_alignment
    args.width, args.height = _align_resolution(
        args.width,
        args.height,
        alignment,
        model_info.family.upper(),
    )
    args.num_frames = _align_ltx_frames(args.num_frames, model_info.frame_alignment)

    if args.width < 64 or args.height < 64:
        parser.error(f"Resolved dimensions {args.width}x{args.height} are too small (minimum 64x64)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Select backend
    backend = get_video_backend(model_info.backend)

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
    model, model_info = backend.load_model(
        args.model,
        mode=mode,
        low_memory=args.low_memory,
        loras=loras,
        **load_kwargs,
    )

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
