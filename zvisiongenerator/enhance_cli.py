"""Shared ``--enhance`` / ``--enhance-note`` / ``--no-enhance`` / ``--enhance-model`` handling for the image and video CLIs."""

from __future__ import annotations

import argparse
from dataclasses import replace

from zvisiongenerator.utils.prompt_enhance import parse_enhance_spec, settings_from_mapping, validate_settings


def add_enhance_arguments(parser: argparse.ArgumentParser, *, mode: str) -> None:
    """Register the prompt-enhancement flags on *parser*."""
    video = ",motion=action+camera-move,sound=ambient+effects" if mode == "video" else ""
    parser.add_argument(
        "--enhance",
        nargs="?",
        const="",
        default=None,
        metavar="SPEC",
        help=(
            "Rewrite each prompt with a local LLM before the model loads. Optional SPEC sets the options, e.g. "
            f"'style=cinematic,mood=dramatic,details=lighting+camera,length=longer{video}'. Applies to every prompt and overrides prompt-file 'enhance:' entries."
        ),
    )
    parser.add_argument(
        "--enhance-note",
        type=str,
        default=None,
        metavar="TEXT",
        help="Free-text direction for the enhancer, e.g. 'at night, in the rain'. Implies --enhance.",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable prompt enhancement, including prompt-file 'enhance:' entries.")
    parser.add_argument("--enhance-model", type=str, default=None, metavar="REPO[@REVISION]", help="Hugging Face repo or local path of the enhancer LLM (overrides config).")


def parse_enhance_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, mode: str, split: bool = False) -> None:
    """Replace ``args.enhance`` (raw SPEC plus ``--enhance-note``) with ``EnhanceSettings`` or ``None``; report errors via *parser*.

    Args:
        split: A video job whose prompts are split into scene and action (no scene given, or an extension), so
            settings that keep everything still change the prompt and are not refused as a no-op.
    """
    spec = args.enhance
    note = getattr(args, "enhance_note", None)
    if args.no_enhance and (spec is not None or note is not None):
        parser.error("--enhance and --enhance-note cannot be combined with --no-enhance.")
    if spec is None and note is None:
        args.enhance = None
        return
    try:
        settings = parse_enhance_spec(spec, mode=mode)
        if note is not None:
            settings = replace(settings, note=settings_from_mapping({"note": note}, mode=mode).note)
        validate_settings(settings, mode=mode, split=mode == "video" and split)
    except ValueError as exc:
        parser.error(f"--enhance: {exc}")
    args.enhance = settings
