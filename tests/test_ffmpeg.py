"""Tests for zvisiongenerator.utils.ffmpeg — ffmpeg availability check."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg, require_ffmpeg, strip_ffmpeg_banner, tool_error


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


class TestEnsureFfmpegWindowsAndLinux:
    """ffmpeg missing on Windows and Linux → offer the platform's package manager."""

    @staticmethod
    def _which(available):
        """Return a ``shutil.which`` stand-in: ffmpeg is found only after install, listed managers are found."""
        state = {"installed": False}

        def which(name):
            if name == "ffmpeg":
                return "/bin/ffmpeg" if state["installed"] else None
            return f"/bin/{name}" if name in available else None

        return which, state

    @pytest.mark.parametrize(
        ("available", "expected"),
        [
            ({"apt", "dnf"}, ["sudo", "apt", "install", "ffmpeg"]),
            ({"dnf", "pacman"}, ["sudo", "dnf", "install", "ffmpeg"]),
            ({"pacman"}, ["sudo", "pacman", "-S", "ffmpeg"]),
        ],
    )
    def test_linux_uses_the_first_package_manager_found(self, available, expected):
        which, state = self._which(available)

        def run(command, check):
            state["installed"] = True
            return MagicMock(returncode=0)

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "linux"),
            patch("zvisiongenerator.utils.ffmpeg.os.geteuid", return_value=1000, create=True),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            patch("zvisiongenerator.utils.ffmpeg.input", return_value="y"),
            patch("zvisiongenerator.utils.ffmpeg.subprocess.run", side_effect=run) as mock_run,
        ):
            ensure_ffmpeg()

        mock_run.assert_called_once_with(expected, check=False)

    def test_linux_root_runs_without_sudo(self):
        which, state = self._which({"apt"})

        def run(command, check):
            state["installed"] = True
            return MagicMock(returncode=0)

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "linux"),
            patch("zvisiongenerator.utils.ffmpeg.os.geteuid", return_value=0, create=True),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            patch("zvisiongenerator.utils.ffmpeg.input", return_value="y"),
            patch("zvisiongenerator.utils.ffmpeg.subprocess.run", side_effect=run) as mock_run,
        ):
            ensure_ffmpeg()

        mock_run.assert_called_once_with(["apt", "install", "ffmpeg"], check=False)

    def test_linux_without_a_known_package_manager_exits_with_the_download_page(self, capsys):
        which, _ = self._which(set())

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "linux"),
            patch("zvisiongenerator.utils.ffmpeg.os.geteuid", return_value=1000, create=True),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            patch("zvisiongenerator.utils.ffmpeg.subprocess.run") as mock_run,
            pytest.raises(SystemExit),
        ):
            ensure_ffmpeg()

        mock_run.assert_not_called()
        assert "ffmpeg.org" in capsys.readouterr().out

    def test_windows_installs_with_winget(self):
        which, state = self._which({"winget"})

        def run(command, check):
            state["installed"] = True
            return MagicMock(returncode=0)

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "win32"),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            patch("zvisiongenerator.utils.ffmpeg.input", return_value="y"),
            patch("zvisiongenerator.utils.ffmpeg.subprocess.run", side_effect=run) as mock_run,
        ):
            ensure_ffmpeg()

        assert mock_run.call_args.args[0][:2] == ["winget", "install"]

    def test_windows_exits_when_path_is_not_updated_yet(self):
        which, _ = self._which({"winget"})

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "win32"),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            patch("zvisiongenerator.utils.ffmpeg.input", return_value="y"),
            patch("zvisiongenerator.utils.ffmpeg.subprocess.run", return_value=MagicMock(returncode=0)),
            pytest.raises(SystemExit) as exc_info,
        ):
            ensure_ffmpeg()

        assert exc_info.value.code == 1

    def test_windows_without_winget_exits_with_the_download_page(self, capsys):
        which, _ = self._which(set())

        with (
            patch("zvisiongenerator.utils.ffmpeg.sys.platform", "win32"),
            patch("zvisiongenerator.utils.ffmpeg.shutil.which", side_effect=which),
            pytest.raises(SystemExit),
        ):
            ensure_ffmpeg()

        assert "ffmpeg.org" in capsys.readouterr().out


class TestToolError:
    @pytest.mark.parametrize("stderr", [b"  boom\n", "  boom\n"])
    def test_ends_with_the_stripped_stderr(self, stderr):
        error = tool_error("ffmpeg failed", stderr)

        assert isinstance(error, RuntimeError)
        assert str(error) == "ffmpeg failed: boom"

    def test_keeps_only_the_last_characters(self):
        assert str(tool_error("ffmpeg failed", "x" * 10 + "tail", limit=4)) == "ffmpeg failed: tail"

    def test_leaves_out_the_banner_and_ends_with_the_real_error(self):
        banner = (
            "ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers\n"
            "  built with gcc 14 (GCC)\n"
            "  configuration: " + "--enable-foo " * 80 + "\n"
            "  libavutil      59. 39.100 / 59. 39.100\n"
            "  libpostproc    57.  1.100 / 57.  1.100\n"
        )

        message = str(tool_error("ffmpeg failed", banner + "[mov @ 0x1] stuff\nout.mp4: Invalid data found\n"))

        assert message == "ffmpeg failed: [mov @ 0x1] stuff\nout.mp4: Invalid data found"

    def test_banner_only_stderr_is_kept(self):
        assert str(tool_error("ffmpeg failed", "ffmpeg version 7.1")) == "ffmpeg failed: ffmpeg version 7.1"

    @pytest.mark.parametrize("stderr", [None, b""])
    def test_missing_stderr_gives_an_empty_tail(self, stderr):
        assert str(tool_error("ffmpeg failed", stderr)) == "ffmpeg failed: "


class TestStripFfmpegBanner:
    @pytest.mark.parametrize(
        "line",
        [
            "libavutil      59. 39.100 / 59. 39.100",
            "libavcodec     61.  3.100 / 61.  3.100",
            "libavformat    61.  1.100 / 61.  1.100",
            "libswscale      8.  1.100 /  8.  1.100",
            "libswresample   5.  1.100 /  5.  1.100",
            "libpostproc    58.  1.100 / 58.  1.100",
        ],
    )
    def test_library_version_lines_are_dropped(self, line):
        assert strip_ffmpeg_banner(f"{line}\nreal error") == "real error"

    @pytest.mark.parametrize("line", ["libavcodec: bad thing happened", "libpostproc: bad thing happened"])
    def test_error_lines_naming_a_library_are_kept(self, line):
        assert strip_ffmpeg_banner(f"ffmpeg version 7.1\n{line}") == line
