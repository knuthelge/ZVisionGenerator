"""Shared ``--enhance`` / ``--enhance-note`` / ``--no-enhance`` / ``--enhance-model`` / ``--enhance-format`` handling for the image and video CLIs."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace

from zvisiongenerator.utils.prompt_enhance import ENHANCE_FORMATS, EnhanceSettings, parse_enhance_spec, settings_from_mapping, validate_settings


def add_enhance_arguments(parser: argparse.ArgumentParser, *, mode: str) -> None:
    """Register the prompt-enhancement flags on *parser*."""
    video = ",motion=action+camera-move,sound=ambient+effects" if mode == "video" else ""
    parts_help = " parts=action keeps the scene and rewrites only the action (parts=both is the default; with --extend and a scene, action)." if mode == "video" else ""
    parser.add_argument(
        "--enhance",
        nargs="?",
        const="",
        default=None,
        metavar="SPEC",
        help=(
            "Rewrite each prompt with a local LLM before the model loads. Optional SPEC sets the options, e.g. "
            f"'style=cinematic,mood=dramatic,details=lighting+camera,length=longer{video}'. Applies to every prompt and overrides prompt-file 'enhance:' entries.{parts_help}"
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
    parser.add_argument(
        "--enhance-model",
        type=str,
        default=None,
        metavar="fast|smart|REPO[@REVISION]",
        help="Enhancer LLM: fast (default), smart (larger and follows the options more closely; needs more memory), or a Hugging Face repo or local path (overrides config).",
    )
    if mode == "image":
        parser.add_argument(
            "--enhance-format",
            choices=ENHANCE_FORMATS,
            default=None,
            help="text (default) rewrites the prompt; json writes a structured JSON caption (models that take JSON captions, such as ideo). Implies --enhance.",
        )


def parse_enhance_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, mode: str, split: bool = False) -> None:
    """Replace ``args.enhance`` (raw SPEC plus ``--enhance-note``) with ``EnhanceSettings`` or ``None``; report errors via *parser*.

    Args:
        split: A video job whose prompts are split into scene and action (no scene given, or an extension), so
            settings that keep everything still change the prompt and are not refused as a no-op.
    """
    spec = args.enhance
    note = getattr(args, "enhance_note", None)
    caption = getattr(args, "enhance_format", None) == "json"
    # Only --enhance-format asked for enhancement: a prompt file's own enhance: entries keep their settings.
    args.enhance_implied = caption and spec is None and note is None
    if args.no_enhance and (spec is not None or note is not None or caption):
        parser.error("--enhance, --enhance-note and --enhance-format cannot be combined with --no-enhance.")
    if spec is None and note is None and not caption:
        args.enhance = None
        return
    try:
        settings = parse_enhance_spec(spec, mode=mode)
        if note is not None:
            settings = replace(settings, note=settings_from_mapping({"note": note}, mode=mode).note)
        validate_settings(settings, mode=mode, split=mode == "video" and split, caption=caption)
    except ValueError as exc:
        parser.error(f"--enhance: {exc}")
    args.enhance = settings


def defer_format_to_entries(args: argparse.Namespace, enhance_by_set: dict[str, list[EnhanceSettings | None]] | None) -> None:
    """Let ``--enhance-format`` alone set the output format of a prompt file's own ``enhance:`` entries, as the Web UI does.

    Without ``--enhance`` or ``--enhance-note`` it enhances no other prompt, and warns when no entry has settings.
    """
    if not getattr(args, "enhance_implied", False):
        return
    args.enhance = None
    if not any(settings is not None for entries in (enhance_by_set or {}).values() for settings in entries):
        warnings.warn("--enhance-format json applies to prompt-file entries with their own 'enhance:'; none have one. Add --enhance to enhance every prompt.", stacklevel=2)
