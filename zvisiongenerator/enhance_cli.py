"""Shared ``--enhance`` / ``--no-enhance`` / ``--enhance-model`` handling for the image and video CLIs."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, parse_enhance_spec, validate_settings

_PHASE_MESSAGES = {
    "downloading": "Downloading prompt enhancer model (first use only)...",
    "loading": "Loading prompt enhancer...",
    "cpu": "Prompt enhancer runs on the CPU (no CUDA GPU); each prompt may take a few minutes.",
}


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
            "Rewrite each prompt with a local LLM before generating. Optional SPEC sets the options, e.g. "
            f"'style=cinematic,details=lighting+camera,length=longer{motion}'. Applies to every prompt and overrides prompt-file 'enhance:' entries."
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


def job_enhance_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: dict[str, Any],
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
) -> tuple[str, str | None] | None:
    """Return the enhancer ``(repo, revision)`` when any prompt is enhanced, after a preflight check."""
    from zvisiongenerator.backends import prompt_enhancer_session

    try:
        return prompt_enhancer_session.plan_job_enhancer(
            config, platform_key=sys.platform, disabled=args.no_enhance, override=args.enhance, enhance_by_set=enhance_by_set, cli_model=args.enhance_model
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


def print_enhancer_phase(phase: str) -> None:
    """Print a status line while the enhancer downloads or loads."""
    print(_PHASE_MESSAGES.get(phase, phase))
