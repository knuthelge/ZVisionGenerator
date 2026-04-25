"""Centralize host-local directory and existing-file picker behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class PickerResult:
    """Represent the outcome of a host-local picker interaction."""

    status: str
    path: str | None = None
    message: str | None = None

    def to_payload(self) -> dict[str, str | None]:
        """Convert the picker result to the JSON API contract."""
        return {"status": self.status, "path": self.path, "message": self.message}


def pick_path(kind: str, *, initial_path: str | None = None) -> PickerResult:
    """Open a native directory or existing-file picker on the server host."""
    try:
        if sys.platform == "darwin":
            selected_path = _pick_macos(kind, initial_path)
        else:
            selected_path = _pick_tk(kind, initial_path)
    except ImportError:
        return PickerResult(status="unsupported", message="Native file browsing is unavailable in this environment.")
    except FileNotFoundError:
        return PickerResult(status="unsupported", message="Native file browsing is unavailable in this environment.")
    except RuntimeError as exc:
        return PickerResult(status="error", message=str(exc))

    if selected_path is None:
        return PickerResult(status="cancelled")
    return PickerResult(status="selected", path=str(Path(selected_path).expanduser().resolve()))


def _pick_macos(kind: str, initial_path: str | None) -> str | None:
    command = "choose folder"
    if kind == "existing_file":
        command = "choose file"
    elif kind != "directory":
        raise RuntimeError(f"Unsupported picker kind: {kind}")
    default_location = _macos_default_location(initial_path)
    if default_location:
        script = f'set defaultLocation to POSIX file "{default_location}"\nset chosenItem to {command} default location defaultLocation\nPOSIX path of chosenItem'
    else:
        script = f"set chosenItem to {command}\nPOSIX path of chosenItem"
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "user canceled" in stderr or "cancelled" in stderr:
            return None
        raise RuntimeError(result.stderr.strip() or "Native path picker failed.")
    selected = result.stdout.strip()
    return selected or None


def _macos_default_location(initial_path: str | None) -> str | None:
    if not initial_path:
        return None
    candidate = Path(initial_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate.resolve()).replace('"', '\\"')
    return None


def _pick_tk(kind: str, initial_path: str | None) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    initial_dir = _tk_initial_dir(initial_path)
    try:
        if kind == "directory":
            selected = filedialog.askdirectory(initialdir=initial_dir, mustexist=True)
        elif kind == "existing_file":
            selected = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*")])
        else:
            raise RuntimeError(f"Unsupported picker kind: {kind}")
    finally:
        root.destroy()
    return selected or None


def _tk_initial_dir(initial_path: str | None) -> str:
    if not initial_path:
        return str(Path.home())
    candidate = Path(initial_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate.resolve())
    return str(Path.home())
