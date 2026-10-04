"""Shared ``--enhance`` / ``--no-enhance`` / ``--enhance-model`` handling for the image and video CLIs."""

from __future__ import annotations

import argparse

from zvisiongenerator.utils.prompt_enhance import parse_enhance_spec, validate_settings


def add_enhance_arguments(parser: argparse.ArgumentParser, *, mode: str) -> None:
    """Register the prompt-enhancement flags on *parser*."""
    motion = ",motion=action+camera-move" if mode == "video" else ""
    parser.add_argument(
        "--enhance",
        nargs="?",
        const="",
        default=None,
        metavar="SPEC",
        help=(
            "Rewrite each prompt with a local LLM before the model loads. Optional SPEC sets the options, e.g. "
            f"'style=cinematic,mood=dramatic,details=lighting+camera,length=longer{motion}'. Applies to every prompt and overrides prompt-file 'enhance:' entries."
        ),
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable prompt enhancement, including prompt-file 'enhance:' entries.")
    parser.add_argument("--enhance-model", type=str, default=None, metavar="REPO[@REVISION]", help="Hugging Face repo or local path of the enhancer LLM (overrides config).")


def parse_enhance_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, mode: str) -> None:
    """Replace ``args.enhance`` (raw SPEC) with ``EnhanceSettings`` or ``None``; report errors via *parser*."""
    spec = args.enhance
    if args.no_enhance and spec is not None:
        parser.error("--enhance and --no-enhance cannot be combined.")
    if spec is None:
        args.enhance = None
        return
    try:
        settings = parse_enhance_spec(spec, mode=mode)
        validate_settings(settings, mode=mode)
    except ValueError as exc:
        parser.error(f"--enhance: {exc}")
    args.enhance = settings
