"""Tests for the unified ziv CLI dispatcher."""

from __future__ import annotations

import sys
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

import zvisiongenerator.image_cli as image_cli
from tests.test_lora_unification import _patch_image_main_dependencies
from zvisiongenerator.cli import main
from zvisiongenerator.web import main as web_main


class TestUnifiedCli:
    """Test the unified ziv command dispatcher."""

    def test_no_args_prints_help_without_launching_web_ui(self, capsys):
        """ziv with no args prints command discovery help."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv"]):
                main()

        captured = capsys.readouterr()
        assert captured.err == ""
        assert all(command in captured.out for command in ("image", "video", "model", "ui"))
        run_server.assert_not_called()

    def test_help_flag_prints_help_without_launching_web_ui(self, capsys):
        """ziv --help prints help and does not launch the Web UI server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        captured = capsys.readouterr()
        assert exc_info.value.code == 0
        assert all(token in captured.out for token in ("ziv", "image", "video", "model", "ui"))
        run_server.assert_not_called()

    def test_h_flag_prints_help_without_launching_web_ui(self, capsys):
        """ziv -h prints help and does not launch the Web UI server."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "-h"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        captured = capsys.readouterr()
        assert exc_info.value.code == 0
        assert all(token in captured.out for token in ("ziv", "image", "video", "model", "ui"))
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
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 2

    def test_image_subcommand_delegates(self):
        """ziv image --help delegates to image_cli.main."""
        with patch("sys.argv", ["ziv", "image", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_video_subcommand_delegates(self):
        """ziv video --help delegates to video_cli.main."""
        with patch("sys.argv", ["ziv", "video", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_model_subcommand_delegates(self):
        """ziv model (no args) delegates to convert_checkpoint.main which prints help."""
        with patch("sys.argv", ["ziv", "model"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

    def test_ui_subcommand_delegates_to_web_cli(self):
        """ziv ui forwards arguments into the Web UI CLI."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with patch("sys.argv", ["ziv", "ui", "--host", "0.0.0.0", "--port", "9000", "--no-browser"]):
                main()

        run_server.assert_called_once_with(host="0.0.0.0", port=9000, open_browser=False)

    def test_ui_subcommand_reports_broken_web_ui_install(self, capsys):
        """ziv ui should explain how to recover from a broken Web UI install."""
        with patch("zvisiongenerator.web._missing_web_runtime_dependencies", return_value=["fastapi", "uvicorn"]):
            with patch("zvisiongenerator.web.run_server") as run_server:
                with patch("sys.argv", ["ziv", "ui", "--no-browser"]):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert all(token in captured.err for token in ("fastapi", "uvicorn", "ziv ui", "z-vision-generator"))
        run_server.assert_not_called()


class TestWebCli:
    """Test the standalone Web UI CLI."""

    def test_standalone_entrypoint_launches_server(self):
        """ziv-ui launches the FastAPI server via the Web UI CLI."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            web_main(["--host", "0.0.0.0", "--port", "9090"], prog="ziv-ui")

        run_server.assert_called_once_with(host="0.0.0.0", port=9090, open_browser=True)

    @pytest.mark.parametrize("alias", ["local", "LOCAL"])
    def test_local_host_alias_binds_all_interfaces(self, alias):
        """--host local is short for 0.0.0.0."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            web_main(["--host", alias, "--no-browser"], prog="ziv-ui")

        run_server.assert_called_once_with(host="0.0.0.0", port=8080, open_browser=False)

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
        assert all(token in captured.out for token in ("ziv-ui", "--host", "--port", "--no-browser"))
        run_server.assert_not_called()

    def test_standalone_entrypoint_reports_broken_web_ui_install(self, capsys):
        """ziv-ui should explain how to recover from a broken Web UI install."""
        with patch("zvisiongenerator.web._missing_web_runtime_dependencies", return_value=["fastapi", "uvicorn"]):
            with patch("zvisiongenerator.web.run_server") as run_server:
                with pytest.raises(SystemExit) as exc_info:
                    web_main(["--no-browser"], prog="ziv-ui")

        captured = capsys.readouterr()
        assert exc_info.value.code == 1
        assert all(token in captured.err for token in ("fastapi", "uvicorn", "ziv-ui", "z-vision-generator"))
        run_server.assert_not_called()

    def test_background_flag_starts_background_server(self):
        """--background hands off to the background launcher instead of running the server here."""
        with patch("zvisiongenerator.web.run_server") as run_server, patch("zvisiongenerator.web.run_background", return_value=0) as run_background:
            with pytest.raises(SystemExit) as exc_info:
                web_main(["--background", "--host", "local", "--port", "9100"], prog="ziv ui")

        assert exc_info.value.code == 0
        run_background.assert_called_once_with(host="0.0.0.0", port=9100, open_browser=True)
        run_server.assert_not_called()

    def test_background_stop_flag_stops_servers_without_touching_the_log(self):
        """--background-stop stops the servers and leaves their error log alone."""
        with (
            patch("zvisiongenerator.web.run_server") as run_server,
            patch("zvisiongenerator.web.stop_all_servers", return_value=1) as stop_all_servers,
            patch("zvisiongenerator.web.setup_logging") as setup_logging,
        ):
            with pytest.raises(SystemExit) as exc_info:
                web_main(["--background-stop"], prog="ziv ui")

        assert exc_info.value.code == 1
        stop_all_servers.assert_called_once_with()
        run_server.assert_not_called()
        setup_logging.assert_not_called()

    def test_background_flags_are_mutually_exclusive(self):
        """--background and --background-stop cannot be combined."""
        with patch("zvisiongenerator.web.run_server") as run_server:
            with pytest.raises(SystemExit) as exc_info:
                web_main(["--background", "--background-stop"], prog="ziv ui")

        assert exc_info.value.code == 2
        run_server.assert_not_called()


def _gated() -> GatedRepoError:
    return GatedRepoError("boom", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/model")))


class TestImageCliHubRefusals:
    """A Hub refusal ends ziv-image with a usage error that says what to do, not a traceback."""

    @pytest.mark.parametrize("stage", ["detect", "load"])
    def test_gated_model_exits_2_with_the_next_step(self, monkeypatch, capsys, stage):
        backend = _patch_image_main_dependencies(monkeypatch)
        if stage == "detect":
            monkeypatch.setattr(image_cli, "detect_image_model", lambda _model: (_ for _ in ()).throw(_gated()))
        else:
            backend.load_model.side_effect = _gated()
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "owner/model", "--prompt", "a cat"])

        with pytest.raises(SystemExit) as exc_info:
            image_cli.main()

        error = capsys.readouterr().err.strip().splitlines()[-1]
        assert exc_info.value.code == 2
        assert "hf auth login" in error
        assert "https://huggingface.co" in error
        assert "Traceback" not in error

    def test_other_load_errors_keep_their_text(self, monkeypatch, capsys):
        backend = _patch_image_main_dependencies(monkeypatch)
        backend.load_model.side_effect = RuntimeError("out of memory")
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "owner/model", "--prompt", "a cat"])

        with pytest.raises(SystemExit) as exc_info:
            image_cli.main()

        assert exc_info.value.code == 2
        assert "out of memory" in capsys.readouterr().err
