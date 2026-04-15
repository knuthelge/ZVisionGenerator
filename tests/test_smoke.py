"""Smoke test — verify the CLI entry points are reachable."""

from __future__ import annotations

import subprocess
import sys


def test_ziv_help_exits_zero():
    """``ziv --help`` exits 0 and prints usage text."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['ziv', '--help']; from zvisiongenerator.image_cli import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "ziv" in result.stdout.lower()


def test_ziv_convert_help_exits_zero():
    """``ziv-convert --help`` exits 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['ziv-convert', '--help']; from zvisiongenerator.converters.convert_checkpoint import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
