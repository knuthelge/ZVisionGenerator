"""Check for ffmpeg availability and offer to install it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def require_ffmpeg() -> None:
    """Require ffmpeg on PATH without prompting or attempting installation.

    This check is suitable for non-interactive callers such as the Web UI.

    Raises:
        RuntimeError: If ffmpeg is not available on PATH.
    """
    if shutil.which("ffmpeg"):
        return

    raise RuntimeError("ffmpeg is required for video generation but was not found on PATH. Install ffmpeg from https://ffmpeg.org/download.html, ensure it is on PATH, then retry.")


# Lines of the version and build banner that ffmpeg prints before any real message.
_FFMPEG_BANNER_LINE = re.compile(r"^(ffmpeg version |built with |configuration:|lib(av\w+|sw\w+|postproc)\s+\d+\.|copyright )", re.IGNORECASE)


def strip_ffmpeg_banner(text: str) -> str:
    """Return *text* as stripped, non-blank lines without ffmpeg's version and build banner.

    The banner lines are kept when nothing else remains, so banner-only output is still reported.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join([line for line in lines if not _FFMPEG_BANNER_LINE.match(line)] or lines)


def tool_error(message: str, stderr: bytes | str | None, *, limit: int = 1500) -> RuntimeError:
    """Return a ``RuntimeError`` that says *message* and ends with the last *limit* characters of a tool's stderr.

    ffmpeg's version and build banner is left out of the stderr so the real error is what remains.
    """
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else (stderr or "")
    return RuntimeError(f"{message}: {strip_ffmpeg_banner(text)[-limit:]}")


def strip_audio(video_path: Path) -> None:
    """Remove audio track from video in-place using ffmpeg.

    Copies the video stream without re-encoding, discards audio.
    Replaces the original file.
    """
    tmp = video_path.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-an", "-c:v", "copy", str(tmp)],
            check=True,
            capture_output=True,
        )
        tmp.replace(video_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _install_options() -> list[tuple[str, list[str], str]]:
    """Return the ffmpeg install options for this platform as ``(executable, command, manager)``, best first."""
    if sys.platform == "darwin":
        return [("brew", ["brew", "install", "ffmpeg"], "Homebrew"), ("port", ["sudo", "port", "install", "ffmpeg"], "MacPorts")]
    if sys.platform == "win32":
        return [("winget", ["winget", "install", "--id", "Gyan.FFmpeg", "--exact"], "winget")]
    # Package managers need root; skip sudo when already running as root (containers, for one).
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    return [
        ("apt", [*sudo, "apt", "install", "ffmpeg"], "apt"),
        ("dnf", [*sudo, "dnf", "install", "ffmpeg"], "dnf"),
        ("pacman", [*sudo, "pacman", "-S", "ffmpeg"], "pacman"),
    ]


def ensure_ffmpeg() -> None:
    """Check that ffmpeg is on PATH; offer to install it if missing.

    Raises:
        SystemExit: If ffmpeg is not found and the user declines installation,
            or if the installation fails.
    """
    if shutil.which("ffmpeg"):
        return

    print("ffmpeg is required for video generation but was not found on PATH.")

    for executable, command, manager in _install_options():
        if shutil.which(executable):
            _offer_install(command, manager=manager)
            return
    print("Install ffmpeg manually: https://ffmpeg.org/download.html")
    raise SystemExit(1)


def _offer_install(command: list[str], *, manager: str) -> None:
    """Prompt the user to install ffmpeg and run the command if accepted.

    Args:
        command: The install command and its arguments.
        manager: Name of the package manager (for display).

    Raises:
        SystemExit: If the user declines or installation fails.
    """
    shown = " ".join(command)
    try:
        answer = input(f"Install ffmpeg via {manager}? ({shown}) [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):  # fmt: skip
        print()
        raise SystemExit(1)

    if answer not in ("y", "yes"):
        print("ffmpeg is required. Install it manually and try again.")
        raise SystemExit(1)

    print(f"Running: {shown}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Installation failed (exit code {result.returncode}). Install ffmpeg manually and try again.")
        raise SystemExit(1)

    # Verify installation succeeded
    if not shutil.which("ffmpeg"):
        if sys.platform == "win32":
            print("ffmpeg is installed but this terminal does not see it yet. Open a new terminal and run the command again.")
        else:
            print("ffmpeg still not found after installation. Check your PATH and try again.")
        raise SystemExit(1)

    print("ffmpeg installed successfully.\n")
