"""Tests for the unified ziv CLI dispatcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from zvisiongenerator.cli import main
from zvisiongenerator.web import main as web_main


class TestUnifiedCli:
    """Test the unified ziv command dispatcher."""

    def test_no_args_launches_web_ui(self):
        """ziv with no args launches the Web UI server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv"]):
                main()

        run_server.assert_called_once_with()

    def test_help_flag_prints_help_without_launching_web_ui(self, capsys):
        """ziv --help prints help and does not launch the Web UI server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        captured = capsys.readouterr()
        assert exc_info.value.code == 0
        assert "usage: ziv" in captured.out
        run_server.assert_not_called()

    def test_h_flag_prints_help_without_launching_web_ui(self, capsys):
        """ziv -h prints help and does not launch the Web UI server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "-h"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        captured = capsys.readouterr()
        assert exc_info.value.code == 0
        assert "usage: ziv" in captured.out
        run_server.assert_not_called()

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

    def test_model_subcommand_delegates(self):
        """ziv model (no args) delegates to convert_checkpoint.main which prints help."""
        with patch("sys.argv", ["ziv", "model"]):
            with pytest.raises(SystemExit, match="0"):
                main()

    def test_ui_subcommand_delegates_to_web_cli(self):
        """ziv ui forwards arguments into the Web UI CLI."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "ui", "--host", "0.0.0.0", "--port", "9000", "--no-browser"]):
                main()

        run_server.assert_called_once_with(host="0.0.0.0", port=9000, open_browser=False)


class TestWebCli:
    """Test the standalone Web UI CLI."""

    def test_standalone_entrypoint_launches_server(self):
        """ziv-ui launches the FastAPI server via the Web UI CLI."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            web_main(["--host", "0.0.0.0", "--port", "9090"], prog="ziv-ui")

        run_server.assert_called_once_with(host="0.0.0.0", port=9090, open_browser=True)

    def test_standalone_entrypoint_rejects_invalid_port(self):
        """ziv-ui rejects out-of-range ports before launching the server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with pytest.raises(SystemExit) as exc_info:
                web_main(["--port", "70000"], prog="ziv-ui")

        assert exc_info.value.code == 2
        run_server.assert_not_called()

    def test_standalone_entrypoint_help_bypasses_server(self, capsys):
        """ziv-ui --help prints launcher help without starting uvicorn."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with pytest.raises(SystemExit) as exc_info:
                web_main(["--help"], prog="ziv-ui")

        captured = capsys.readouterr()
        assert exc_info.value.code == 0
        assert "usage: ziv-ui" in captured.out
        run_server.assert_not_called()
