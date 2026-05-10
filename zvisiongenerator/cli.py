"""Unified CLI — entry point for the ziv parent command."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser for the unified CLI."""
    return argparse.ArgumentParser(
        prog="ziv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Z-Vision Generator — AI image and video generation.\n"
            "\n"
            "commands:\n"
            "  image   Generate images (also available as ziv-image)\n"
            "  video   Generate videos (also available as ziv-video)\n"
            "  model   Manage models and LoRAs (also available as ziv-model)\n"
            "  ui      Launch the local Web UI (also available as ziv-ui)"
        ),
        epilog="Run 'ziv <command> --help' for command-specific options.",
    )


def main() -> None:
    """Entry point for the unified ``ziv`` command.

    Dispatches to image, video, model, or Web UI entrypoints.
    With no subcommand, prints command discovery help and exits.
    With --help/-h, prints usage and exits.
    With --version, prints version and exits.
    """
    parser = _build_parser()
    argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return

    if argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return

    if argv[0] == "--version":
        from zvisiongenerator import __version__

        print(f"ziv {__version__}")
        return

    command = argv[0]
    remaining_args = argv[1:]

    if command == "image":
        from zvisiongenerator.image_cli import main as image_main

        sys.argv = ["ziv image", *remaining_args]
        image_main(prog="ziv image")
    elif command == "video":
        from zvisiongenerator.video_cli import main as video_main

        sys.argv = ["ziv video", *remaining_args]
        video_main(prog="ziv video")
    elif command == "model":
        from zvisiongenerator.converters.convert_checkpoint import main as model_main

        sys.argv = ["ziv model", *remaining_args]
        model_main(prog="ziv model")
    elif command == "ui":
        from zvisiongenerator.web import main as web_main

        web_main(remaining_args, prog="ziv ui")
    else:
        parser.error(f"unknown command '{command}'")
