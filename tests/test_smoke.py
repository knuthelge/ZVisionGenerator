"""Smoke test — verify the CLI entry points are reachable."""

from __future__ import annotations

import subprocess
import sys


def test_ziv_help_exits_zero():
    """``ziv`` (no args) reaches the Web UI launcher path and exits 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\nfrom unittest.mock import patch\nsys.argv = ['ziv']\nwith patch('zvisiongenerator.web.run_server') as run_server:\n    from zvisiongenerator.cli import main\n    main()\n    print(run_server.call_count)\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_ziv_image_help_exits_zero():
    """``ziv-image --help`` exits 0 and prints usage text."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['ziv-image', '--help']; from zvisiongenerator.image_cli import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "ziv-image" in result.stdout.lower()


def test_ziv_model_help_exits_zero():
    """``ziv-model --help`` exits 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['ziv-model', '--help']; from zvisiongenerator.converters.convert_checkpoint import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
