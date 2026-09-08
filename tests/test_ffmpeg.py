"""Tests for zvisiongenerator.utils.ffmpeg — ffmpeg availability check."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg, require_ffmpeg


class TestRequireFfmpeg:
    """The Web UI prerequisite check must never enter the CLI install flow."""

    @patch("zvisiongenerator.utils.ffmpeg.shutil.which", return_value="/usr/local/bin/ffmpeg")
    def test_ffmpeg_found_returns_without_interaction(self, mock_which):
        require_ffmpeg()

        mock_which.assert_called_once_with("ffmpeg")

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run")
    @patch("zvisiongenerator.utils.ffmpeg.input")
    @patch("zvisiongenerator.utils.ffmpeg.shutil.which", return_value=None)
    def test_missing_ffmpeg_raises_without_prompting_or_installing(self, mock_which, mock_input, mock_run):
        with pytest.raises(RuntimeError, match=r"ffmpeg.*Install.*retry"):
            require_ffmpeg()

        mock_which.assert_called_once_with("ffmpeg")
        mock_input.assert_not_called()
        mock_run.assert_not_called()


class TestEnsureFfmpegFound:
    """When ffmpeg is already on PATH, return immediately."""

    @patch("zvisiongenerator.utils.ffmpeg.shutil.which", return_value="/usr/local/bin/ffmpeg")
    def test_ffmpeg_found_returns_immediately(self, mock_which):
        ensure_ffmpeg()
        mock_which.assert_called_once_with("ffmpeg")


class TestEnsureFfmpegMissingBrewInstall:
    """ffmpeg missing + brew available + user accepts → install succeeds."""

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run")
    @patch("zvisiongenerator.utils.ffmpeg.input", return_value="y")
    @patch("zvisiongenerator.utils.ffmpeg.shutil.which")
    def test_brew_install_success(self, mock_which, _mock_input, mock_run):
        # First call: ffmpeg not found; second: brew found; third: post-install check passes
        mock_which.side_effect = [None, "/opt/homebrew/bin/brew", "/usr/local/bin/ffmpeg"]
        mock_run.return_value = MagicMock(returncode=0)

        with patch("zvisiongenerator.utils.ffmpeg.sys.platform", "darwin"):
            ensure_ffmpeg()

        mock_run.assert_called_once_with(["brew", "install", "ffmpeg"], check=False)


class TestEnsureFfmpegUserDeclinesInstall:
    """ffmpeg missing + user says no → SystemExit."""

    @patch("zvisiongenerator.utils.ffmpeg.input", return_value="n")
    @patch("zvisiongenerator.utils.ffmpeg.shutil.which")
    def test_user_declines_raises_system_exit(self, mock_which, _mock_input):
        mock_which.side_effect = [None, "/opt/homebrew/bin/brew"]

        with patch("zvisiongenerator.utils.ffmpeg.sys.platform", "darwin"), pytest.raises(SystemExit):
            ensure_ffmpeg()


class TestEnsureFfmpegNoPackageManager:
    """ffmpeg missing + no brew/port on macOS → SystemExit with URL message."""

    @patch("zvisiongenerator.utils.ffmpeg.shutil.which")
    def test_no_manager_raises_system_exit(self, mock_which, capsys):
        # ffmpeg not found, brew not found, port not found
        mock_which.side_effect = [None, None, None]

        with patch("zvisiongenerator.utils.ffmpeg.sys.platform", "darwin"), pytest.raises(SystemExit) as exc_info:
            ensure_ffmpeg()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ffmpeg.org" in captured.out


class TestEnsureFfmpegInstallFails:
    """ffmpeg missing + install fails → SystemExit."""

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run")
    @patch("zvisiongenerator.utils.ffmpeg.input", return_value="yes")
    @patch("zvisiongenerator.utils.ffmpeg.shutil.which")
    def test_install_failure_raises_system_exit(self, mock_which, _mock_input, mock_run):
        mock_which.side_effect = [None, "/opt/homebrew/bin/brew"]
        mock_run.return_value = MagicMock(returncode=1)

        with patch("zvisiongenerator.utils.ffmpeg.sys.platform", "darwin"), pytest.raises(SystemExit) as exc_info:
            ensure_ffmpeg()

        assert exc_info.value.code == 1
