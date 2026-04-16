"""Tests for the unified ziv CLI dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from zvisiongenerator.cli import main


class TestUnifiedCli:
    """Test the unified ziv command dispatcher."""

    def test_no_args_prints_help(self, capsys):
        """ziv with no args prints help and exits 0."""
        with patch("sys.argv", ["ziv"]):
            main()
        captured = capsys.readouterr()
        assert "usage: ziv" in captured.out
        assert "image" in captured.out
        assert "video" in captured.out
        assert "model" in captured.out

    def test_help_flag(self, capsys):
        """ziv --help prints help and exits 0."""
        with patch("sys.argv", ["ziv", "--help"]):
            main()
        captured = capsys.readouterr()
        assert "usage: ziv" in captured.out

    def test_h_flag(self, capsys):
        """ziv -h prints help and exits 0."""
        with patch("sys.argv", ["ziv", "-h"]):
            main()
        captured = capsys.readouterr()
        assert "usage: ziv" in captured.out

    def test_version_flag(self, capsys):
        """ziv --version prints version and exits 0."""
        with patch("sys.argv", ["ziv", "--version"]):
            main()
        captured = capsys.readouterr()
        assert captured.out.startswith("ziv ")

    def test_unknown_command_exits_2(self):
        """ziv unknown-cmd exits with code 2."""
        with patch("sys.argv", ["ziv", "bogus"]):
            with pytest.raises(SystemExit, match="2"):
                main()

    def test_image_subcommand_delegates(self):
        """ziv image --help delegates to image_cli.main."""
        with patch("sys.argv", ["ziv", "image", "--help"]):
            with pytest.raises(SystemExit, match="0"):
                main()

    def test_video_subcommand_delegates(self):
        """ziv video --help delegates to video_cli.main."""
        with patch("sys.argv", ["ziv", "video", "--help"]):
            with pytest.raises(SystemExit, match="0"):
                main()

    def test_model_subcommand_delegates(self, capsys):
        """ziv model (no args) delegates to convert_checkpoint.main which prints help."""
        with patch("sys.argv", ["ziv", "model"]):
            with pytest.raises(SystemExit, match="0"):
                main()
