"""Unified CLI — entry point for the ziv parent command."""

from __future__ import annotations

import sys


def _print_help() -> None:
    """Print usage information for the unified ziv command."""
    print(
        "usage: ziv <command> [options]\n"
        "\n"
        "Z-Vision Generator — AI image and video generation.\n"
        "\n"
        "commands:\n"
        "  image   Generate images (also available as ziv-image)\n"
        "  video   Generate videos (also available as ziv-video)\n"
        "  model   Manage models and LoRAs (also available as ziv-model)\n"
        "\n"
        "Run 'ziv <command> --help' for command-specific options."
    )


def main() -> None:
    """Entry point for the unified ``ziv`` command.

    Dispatches to image, video, or model subcommands.
    With no subcommand or with --help/-h, prints usage and exits.
    With --version, prints version and exits.
    """
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return

    if sys.argv[1] == "--version":
        from zvisiongenerator import __version__

        print(f"ziv {__version__}")
        return

    command = sys.argv[1]
    # Remove the subcommand from argv so the delegated main() sees correct args
    sys.argv = sys.argv[1:]

    if command == "image":
        from zvisiongenerator.image_cli import main as image_main

        image_main(prog="ziv image")
    elif command == "video":
        from zvisiongenerator.video_cli import main as video_main

        video_main(prog="ziv video")
    elif command == "model":
        from zvisiongenerator.converters.convert_checkpoint import main as model_main

        model_main(prog="ziv model")
    else:
        print(f"ziv: unknown command '{command}'")
        print()
        _print_help()
        sys.exit(2)
