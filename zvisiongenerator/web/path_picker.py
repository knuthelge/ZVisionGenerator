"""Centralize host-local directory and existing-file picker behavior."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import threading


@dataclass(frozen=True)
class PickerResult:
    """Represent the outcome of a host-local picker interaction."""

    status: str
    path: str | None = None
    message: str | None = None

    def to_payload(self) -> dict[str, str | None]:
        """Convert the picker result to the JSON API contract."""
        return {"status": self.status, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class PickerPurpose:
    """Describe one backend-owned host-local picker purpose."""

    name: str
    kind: str
    label: str
    accepted_extensions: tuple[str, ...] = ()


_PICKER_PURPOSES: dict[str, PickerPurpose] = {
    "output_directory": PickerPurpose(name="output_directory", kind="directory", label="Output directory"),
    "prompt_file": PickerPurpose(name="prompt_file", kind="existing_file", label="Prompt file", accepted_extensions=(".yaml", ".yml")),
    "prompt_file_folder": PickerPurpose(name="prompt_file_folder", kind="directory", label="Prompt file folder"),
    "checkpoint_file": PickerPurpose(name="checkpoint_file", kind="existing_file", label="Checkpoint file", accepted_extensions=(".safetensors",)),
    "lora_file": PickerPurpose(name="lora_file", kind="existing_file", label="LoRA file", accepted_extensions=(".safetensors",)),
}


# One native dialog at a time: requests run on worker threads, and Tk is not safe across threads.
_PICKER_LOCK = threading.Lock()


def pick_path(kind: str, *, purpose: str, initial_path: str | None = None) -> PickerResult:
    """Open a native picker for one explicit host-local trust bucket."""
    picker_purpose = _resolve_picker_purpose(purpose, kind)
    if not _PICKER_LOCK.acquire(blocking=False):
        return PickerResult(status="error", message="A file picker is already open. Close it first.")
    try:
        return _pick_path_locked(kind, initial_path, picker_purpose)
    finally:
        _PICKER_LOCK.release()


def _pick_path_locked(kind: str, initial_path: str | None, picker_purpose: PickerPurpose) -> PickerResult:
    try:
        if sys.platform == "darwin":
            selected_path = _pick_macos(kind, initial_path, picker_purpose)
        else:
            selected_path = _pick_tk(kind, initial_path, picker_purpose)
    except ImportError:
        return PickerResult(status="unsupported", message="Native file browsing is unavailable in this environment.")
    except FileNotFoundError:
        return PickerResult(status="unsupported", message="Native file browsing is unavailable in this environment.")
    except RuntimeError as exc:
        return PickerResult(status="error", message=str(exc))

    if selected_path is None:
        return PickerResult(status="cancelled")

    resolved_path = Path(selected_path).expanduser().resolve()
    validation_error = _validate_selected_path(resolved_path, picker_purpose)
    if validation_error is not None:
        return PickerResult(status="error", message=validation_error)
    return PickerResult(status="selected", path=str(resolved_path))


def _resolve_picker_purpose(purpose: str, kind: str) -> PickerPurpose:
    picker_purpose = _PICKER_PURPOSES.get(purpose)
    if picker_purpose is None:
        raise ValueError(f"Unknown picker purpose '{purpose}'.")
    if kind != picker_purpose.kind:
        raise ValueError(f"Picker purpose '{purpose}' requires kind '{picker_purpose.kind}'.")
    return picker_purpose


def _validate_selected_path(path: Path, picker_purpose: PickerPurpose) -> str | None:
    if picker_purpose.kind == "directory":
        if not path.is_dir():
            return f"{picker_purpose.label} picker must return an existing directory on the machine running the Web UI host."
        return None
    if not path.is_file():
        return f"{picker_purpose.label} picker must return an existing file on the machine running the Web UI host."
    if picker_purpose.accepted_extensions and path.suffix.lower() not in picker_purpose.accepted_extensions:
        extensions = ", ".join(picker_purpose.accepted_extensions)
        return f"{picker_purpose.label} picker only accepts host-local files using one of: {extensions}."
    return None


def _pick_macos(kind: str, initial_path: str | None, picker_purpose: PickerPurpose) -> str | None:
    command = "choose folder"
    if kind == "existing_file":
        command = "choose file"
    elif kind != "directory":
        raise RuntimeError(f"Unsupported picker kind: {kind}")
    default_location = _macos_default_location(initial_path)
    file_type_clause = _macos_file_type_clause(picker_purpose)
    # The default location is passed as an argv item, never interpolated into the script source.
    if default_location:
        script = f"on run argv\nset defaultLocation to POSIX file (item 1 of argv)\nset chosenItem to {command}{file_type_clause} default location defaultLocation\nPOSIX path of chosenItem\nend run"
        argv = [default_location]
    else:
        script = f"set chosenItem to {command}{file_type_clause}\nPOSIX path of chosenItem"
        argv = []
    result = subprocess.run(["osascript", "-e", script, *argv], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "user canceled" in stderr or "cancelled" in stderr:
            return None
        raise RuntimeError(result.stderr.strip() or "Native path picker failed.")
    selected = result.stdout.strip()
    return selected or None


def _macos_file_type_clause(picker_purpose: PickerPurpose) -> str:
    if not picker_purpose.accepted_extensions:
        return ""
    extensions = ", ".join(f'"{extension.lstrip(".")}"' for extension in picker_purpose.accepted_extensions)
    return f" of type {{{extensions}}}"


def _macos_default_location(initial_path: str | None) -> str | None:
    if not initial_path:
        return None
    candidate = Path(initial_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate.resolve())
    return None


# Runs in a child process: the web server calls the picker from worker threads, and Tk must own its thread.
# Like the macOS osascript path, a separate process keeps Tk on its own main thread.
_TK_PICKER_SCRIPT = """
import json, sys
import tkinter as tk
from tkinter import filedialog
kind, initial_dir, filetypes = sys.argv[1], sys.argv[2], [tuple(item) for item in json.loads(sys.argv[3])]
root = tk.Tk()
root.withdraw()
try:
    if kind == "directory":
        selected = filedialog.askdirectory(initialdir=initial_dir, mustexist=True)
    else:
        selected = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes)
finally:
    root.destroy()
sys.stdout.write(selected or "")
"""


def _pick_tk(kind: str, initial_path: str | None, picker_purpose: PickerPurpose) -> str | None:
    if kind not in {"directory", "existing_file"}:
        raise RuntimeError(f"Unsupported picker kind: {kind}")
    initial_dir = _tk_initial_dir(initial_path)
    filetypes = json.dumps(_tk_filetypes(picker_purpose))
    result = subprocess.run([sys.executable, "-c", _TK_PICKER_SCRIPT, kind, initial_dir, filetypes], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "No module named" in stderr and "tkinter" in stderr:
            raise ImportError("tkinter is not available")
        raise RuntimeError(stderr.splitlines()[-1] if stderr else "Native path picker failed.")
    selected = result.stdout.strip()
    return selected or None


def _tk_filetypes(picker_purpose: PickerPurpose) -> list[tuple[str, str]]:
    if not picker_purpose.accepted_extensions:
        return [("All files", "*")]
    patterns = " ".join(f"*{extension}" for extension in picker_purpose.accepted_extensions)
    return [(f"{picker_purpose.label}s", patterns), ("All files", "*")]


def _tk_initial_dir(initial_path: str | None) -> str:
    if not initial_path:
        return str(Path.home())
    candidate = Path(initial_path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate.resolve())
    return str(Path.home())
