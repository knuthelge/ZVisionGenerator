"""Image CLI — entry point for the ziv command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from zvisiongenerator.backends import get_backend
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.config import load_config, resolve_defaults, validate_scheduler
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import resolve_model_path, resolve_lora_path
from zvisiongenerator.utils.prompts import load_prompts_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ziv",
        description="Batch image generator with configurable settings.",
        epilog="Example usage: ziv -m models/my-model -p prompts.yaml -r 3 --ratio 16:9 --size l --steps 20",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the base model (Hugging Face format).")
    parser.add_argument("-q", "--quantize", type=int, default=None, help="Quantization level (4 or 8) for faster generation at the cost of quality.")
    parser.add_argument("-p", "--prompts-file", type=str, default="prompts.yaml", help="Path to the YAML file containing prompts. Ignored when --prompt is set.")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of runs to execute (each run will generate images for all prompts).")
    parser.add_argument("--ratio", type=str, default=None, choices=["1:1", "16:9", "9:16", "3:2", "2:3"], help="Aspect ratio for generated images (default: 2:3).")
    parser.add_argument("-s", "--size", type=str, default=None, choices=["xs", "s", "m", "l", "xl"], help="Resolution scale: xs, s, m, l, xl. Default: m.")
    parser.add_argument("-W", "--width", type=int, default=None, help="Override image width (integer). If omitted, uses size preset.")
    parser.add_argument("-H", "--height", type=int, default=None, help="Override image height (integer). If omitted, uses size preset.")
    parser.add_argument("-o", "--output", type=str, default=".", help="Output directory for generated images (default: current directory).")
    parser.add_argument("--prompt", type=str, default=None, help="Inline prompt string for quick one-off generation. Overrides --prompts-file when set.")
    parser.add_argument("--lora", type=str, default=None, help="Comma-separated LoRAs with optional weights: name1:0.8,name2:0.5. Bare names resolve from ~/.ziv/loras/.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps for image generation (e.g., 10, 20).")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale for image generation (float).")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler to use (e.g., 'beta').")
    parser.add_argument("--upscale", type=int, default=None, help="Upscale factor (2 or 4). Default: disabled.")
    parser.add_argument("--upscale-denoise", type=float, default=None, help="Denoising strength for upscaling (float). Defaults to config value based on upscale factor.")
    parser.add_argument("--upscale-steps", type=int, default=None, help="Number of steps for upscaling (integer).")
    parser.add_argument("--upscale-save-pre", action=argparse.BooleanOptionalAction, default=False, help="Save the image before upscaling (with '_pre' suffix).")
    parser.add_argument("--upscale-guidance", type=float, default=None, help="Override guidance scale for the upscale refinement step (float). Falls back to --guidance when not set.")
    parser.add_argument("--upscale-sharpen", action=argparse.BooleanOptionalAction, default=True, help="Apply contrast-adaptive sharpening before upscale refinement.")
    parser.add_argument("--sharpen", nargs="?", const=True, type=float, default=True, help="Apply CAS sharpening. Optional amount overrides config (default: enabled).")
    parser.add_argument("--no-sharpen", dest="sharpen", action="store_const", const=False, help="Disable CAS sharpening.")
    parser.add_argument("--contrast", nargs="?", const=True, type=float, default=False, help="Apply contrast adjustment. Optional amount (1.0 = no change). Default: disabled.")
    parser.add_argument("--no-contrast", dest="contrast", action="store_const", const=False, help="Disable contrast adjustment.")
    parser.add_argument("--saturation", nargs="?", const=True, type=float, default=False, help="Apply saturation adjustment. Optional amount (1.0 = no change). Default: disabled.")
    parser.add_argument("--no-saturation", dest="saturation", action="store_const", const=False, help="Disable saturation adjustment.")
    parser.add_argument("--image", dest="image_path", type=str, default=None, help="Path to a reference image for img2img steering.")
    parser.add_argument("--image-strength", dest="image_strength", type=float, default=0.5, help="Denoising strength for reference image steering (0.0–1.0). Default: 0.5.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible image generation. If not set, a random seed is used each run.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.quantize is not None and args.quantize not in (4, 8):
        parser.error("--quantize must be 4 or 8")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.width is not None and args.width % 16 != 0:
        parser.error(f"--width must be a multiple of 16 (got {args.width})")
    if args.height is not None and args.height % 16 != 0:
        parser.error(f"--height must be a multiple of 16 (got {args.height})")
    if args.upscale is not None and args.upscale not in (2, 4):
        parser.error("--upscale must be 2 or 4")
    if args.upscale_denoise is not None and not (0.0 <= args.upscale_denoise <= 1.0):
        parser.error("--upscale-denoise must be between 0.0 and 1.0")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.upscale_steps is not None and args.upscale_steps < 1:
        parser.error("--upscale-steps must be at least 1")
    if args.guidance is not None and args.guidance < 0:
        parser.error("--guidance must be non-negative")
    if args.upscale_guidance is not None and args.upscale_guidance < 0:
        parser.error("--upscale-guidance must be non-negative")
    if isinstance(args.sharpen, float) and args.sharpen < 0:
        parser.error("--sharpen amount must be non-negative")
    if isinstance(args.contrast, float) and args.contrast < 0:
        parser.error("--contrast amount must be non-negative")
    if isinstance(args.saturation, float) and args.saturation < 0:
        parser.error("--saturation amount must be non-negative")

    # Validate that explicit dimensions survive the upscale round-trip
    if args.upscale is not None:

        def _round16(n: int) -> int:
            return ((n + 15) // 16) * 16

        for dim_name, dim_val in [("--width", args.width), ("--height", args.height)]:
            if dim_val is not None:
                base = dim_val // args.upscale
                final = _round16(base) * args.upscale
                if final != dim_val:
                    parser.error(
                        f"{dim_name} {dim_val} is not compatible with --upscale {args.upscale}: "
                        f"base size {base} rounds to {_round16(base)}, giving final size {final} instead of {dim_val}. "
                        f"Choose a {dim_name} where value // {args.upscale} is already a multiple of 16."
                    )

    try:
        config = load_config()
    except ValueError as e:
        parser.error(str(e))

    gen_cfg = config.get("generation", {})
    if args.ratio is None:
        args.ratio = gen_cfg.get("default_ratio", "2:3")
    if args.size is None:
        args.size = gen_cfg.get("default_size", "m")

    sizes = config.get("sizes", {})
    if args.ratio not in sizes:
        parser.error(f"Unknown ratio '{args.ratio}'. Valid: {list(sizes.keys())}")
    if args.size not in sizes[args.ratio]:
        parser.error(f"Unknown size '{args.size}'. Valid: {list(sizes[args.ratio].keys())}")

    # Expand ~ in path arguments
    if args.model:
        args.model = str(Path(args.model).expanduser())
    if args.image_path:
        args.image_path = str(Path(args.image_path).expanduser())
    if args.output:
        args.output = str(Path(args.output).expanduser())
    if args.prompts_file:
        args.prompts_file = str(Path(args.prompts_file).expanduser())

    if args.model is None:
        parser.error("--model is required. Provide a model name, path, or HuggingFace repo ID.")
    args.model = resolve_model_path(args.model, aliases=config.get("model_aliases", {}))

    lora_paths, lora_weights = None, None
    if args.lora is not None:
        try:
            parsed = parse_lora_arg(args.lora)
        except ValueError as e:
            parser.error(str(e))
        lora_paths = [resolve_lora_path(str(Path(name).expanduser())) for name, _ in parsed]
        lora_weights = [weight for _, weight in parsed]
    if lora_paths:
        for p in lora_paths:
            if not os.path.isfile(p):
                parser.error(f"LoRA file not found: {p}")

    try:
        model_info = detect_image_model(args.model)
    except Exception as e:
        parser.error(f"Could not detect model type: {e}")
    try:
        backend = get_backend()
    except (RuntimeError, ImportError) as e:
        parser.error(str(e))
    cli_overrides = {
        k: v
        for k, v in {
            "steps": args.steps,
            "guidance": args.guidance,
            "scheduler": args.scheduler,
        }.items()
        if v is not None
    }
    defaults = resolve_defaults(
        model_info,
        config,
        cli_overrides,
        backend.name,
    )
    args.steps, args.guidance, args.scheduler = defaults["steps"], defaults["guidance"], defaults["scheduler"]
    try:
        validate_scheduler(args.scheduler, config)
    except ValueError as e:
        parser.error(str(e))
    if args.upscale and args.upscale_steps is None:
        args.upscale_steps = max(1, args.steps // 2)
    if args.image_path is not None and not os.path.isfile(args.image_path):
        parser.error(f"Reference image not found: {args.image_path}")
    if not (0.0 <= args.image_strength <= 1.0):
        parser.error(f"--image-strength must be between 0.0 and 1.0, got {args.image_strength}")

    if args.prompt is not None:
        if not args.prompt.strip():
            parser.error("--prompt must not be empty")
        prompts_data = {"prompt": [(args.prompt, None)]}
    else:
        try:
            prompts_data = load_prompts_file(args.prompts_file)
        except (FileNotFoundError, ValueError) as e:
            parser.error(str(e))
    model_suffix = f" ({model_info.size or 'unknown size'})" if model_info.family == "flux2_klein" else ""
    print(f"Model type: {model_info.family}{model_suffix}")

    args.lora_paths, args.lora_weights = lora_paths, lora_weights
    try:
        loaded_model, loaded_model_info = backend.load_model(
            args.model,
            quantize=args.quantize,
            precision="bfloat16",
            lora_paths=lora_paths,
            lora_weights=lora_weights,
        )
    except (RuntimeError, ImportError, OSError, ValueError) as e:
        parser.error(f"Failed to load model: {e}")
    run_batch(backend, loaded_model, prompts_data, config, args, model_info=loaded_model_info)
