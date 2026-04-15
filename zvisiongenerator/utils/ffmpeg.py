"""Check for ffmpeg availability and offer to install it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def strip_audio(video_path: Path) -> None:
    """Remove audio track from video in-place using ffmpeg.

    Copies the video stream without re-encoding, discards audio.
    Replaces the original file.
    """
    tmp = video_path.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-an", "-c:v", "copy", str(tmp)],
            check=True,
            capture_output=True,
        )
        tmp.replace(video_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def ensure_ffmpeg() -> None:
    """Check that ffmpeg is on PATH; offer to install it if missing.

    Raises:
        SystemExit: If ffmpeg is not found and the user declines installation,
            or if the installation fails.
    """
    if shutil.which("ffmpeg"):
        return

    print("ffmpeg is required for video generation but was not found on PATH.")

    # Determine install command based on platform
    if sys.platform == "darwin":
        if shutil.which("brew"):
            _offer_install("brew install ffmpeg", manager="Homebrew")
        elif shutil.which("port"):
            _offer_install("sudo port install ffmpeg", manager="MacPorts")
        else:
            print("Install ffmpeg manually: https://ffmpeg.org/download.html")
            raise SystemExit(1)
    else:
        print("Install ffmpeg for your platform: https://ffmpeg.org/download.html")
        raise SystemExit(1)


def _offer_install(command: str, *, manager: str) -> None:
    """Prompt the user to install ffmpeg and run the command if accepted.

    Args:
        command: The shell command to install ffmpeg.
        manager: Name of the package manager (for display).

    Raises:
        SystemExit: If the user declines or installation fails.
    """
    try:
        answer = input(f"Install ffmpeg via {manager}? ({command}) [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):  # fmt: skip
        print()
        raise SystemExit(1)

    if answer not in ("y", "yes"):
        print("ffmpeg is required. Install it manually and try again.")
        raise SystemExit(1)

    print(f"Running: {command}")
    result = subprocess.run(command.split(), check=False)
    if result.returncode != 0:
        print(f"Installation failed (exit code {result.returncode}). Install ffmpeg manually and try again.")
        raise SystemExit(1)

    # Verify installation succeeded
    if not shutil.which("ffmpeg"):
        print("ffmpeg still not found after installation. Check your PATH and try again.")
        raise SystemExit(1)

    print("ffmpeg installed successfully.\n")
