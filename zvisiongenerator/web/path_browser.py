"""List host-local folders for the Web UI's in-app path browser."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BrowsePurpose:
    """Describe what one Browse button may choose: a folder, or a file with one of some extensions."""

    name: str
    kind: str
    label: str
    accepted_extensions: tuple[str, ...] = ()
    # The place the browser opens at when the field is empty (a ``BrowsePlace.id``).
    default_place: str = "home"

    def to_payload(self) -> dict[str, Any]:
        """Convert the purpose to the JSON API contract the browser shows (title, folder or file mode, filter)."""
        return {"name": self.name, "kind": self.kind, "label": self.label, "accepted_extensions": list(self.accepted_extensions)}


@dataclass(frozen=True)
class BrowseEntry:
    """Describe one folder or file inside a listed folder."""

    name: str
    path: str
    kind: str
    size: int | None
    modified: float | None
    selectable: bool

    def to_payload(self) -> dict[str, Any]:
        """Convert the entry to the JSON API contract."""
        return {"name": self.name, "path": self.path, "kind": self.kind, "size": self.size, "modified": self.modified, "selectable": self.selectable}


@dataclass(frozen=True)
class DirectoryListing:
    """Describe one listed folder: its entries and how many files the purpose's filter left out."""

    path: str
    parent: str | None
    focus: str | None
    entries: tuple[BrowseEntry, ...]
    hidden_count: int

    def to_payload(self) -> dict[str, Any]:
        """Convert the listing to the JSON API contract."""
        return {
            "path": self.path,
            "parent": self.parent,
            "separator": os.sep,
            "focus": self.focus,
            "entries": [entry.to_payload() for entry in self.entries],
            "hidden_count": self.hidden_count,
        }


@dataclass(frozen=True)
class BrowsePlace:
    """Describe one shortcut in the browser's sidebar."""

    id: str
    label: str
    path: str
    default: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Convert the place to the JSON API contract."""
        return {"id": self.id, "label": self.label, "path": self.path, "default": self.default}


# Containers a video can be extended from (the browser, the inspect route and job submission share this list).
SOURCE_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")

_PURPOSES: dict[str, BrowsePurpose] = {
    "output_directory": BrowsePurpose(name="output_directory", kind="directory", label="Output folder", default_place="output"),
    "prompt_file": BrowsePurpose(name="prompt_file", kind="existing_file", label="Prompt file", accepted_extensions=(".yaml", ".yml")),
    "prompt_file_folder": BrowsePurpose(name="prompt_file_folder", kind="directory", label="Prompt file folder"),
    "checkpoint_file": BrowsePurpose(name="checkpoint_file", kind="existing_file", label="Checkpoint file", accepted_extensions=(".safetensors",), default_place="models"),
    "lora_file": BrowsePurpose(name="lora_file", kind="existing_file", label="LoRA file", accepted_extensions=(".safetensors",), default_place="loras"),
    "reference_image": BrowsePurpose(name="reference_image", kind="existing_file", label="Reference image", accepted_extensions=(".png", ".jpg", ".jpeg", ".webp"), default_place="output"),
    "source_video": BrowsePurpose(name="source_video", kind="existing_file", label="Source video", accepted_extensions=SOURCE_VIDEO_EXTENSIONS, default_place="output"),
}


def resolve_purpose(name: str) -> BrowsePurpose:
    """Return the backend-owned purpose called *name*.

    Raises:
        ValueError: If no purpose has that name.
    """
    purpose = _PURPOSES.get(name)
    if purpose is None:
        raise ValueError(f"Unknown browse purpose '{name}'.")
    return purpose


def browse_places(purpose: BrowsePurpose, *, home: Path, output_root: Path, data_dir: Path, drives: tuple[str, ...] = ()) -> tuple[BrowsePlace, ...]:
    """Return the sidebar shortcuts, flagging the one *purpose* opens at.

    Args:
        purpose: The purpose being browsed for.
        home: The user's home folder.
        output_root: The configured output folder.
        data_dir: The ZIV data folder, holding ``models`` and ``loras``.
        drives: Drive roots to offer (Windows), in order.
    """
    places = _place_paths(home=home, output_root=output_root, data_dir=data_dir)
    result = [BrowsePlace(id=place_id, label=label, path=str(path), default=place_id == purpose.default_place) for place_id, (label, path) in places.items()]
    result.extend(BrowsePlace(id=f"drive:{drive}", label=drive.rstrip("\\/") or drive, path=drive) for drive in drives)
    return tuple(result)


def default_place_dir(purpose: BrowsePurpose, *, home: Path, output_root: Path, data_dir: Path) -> Path:
    """Return the folder *purpose* opens at when its field is empty."""
    return _place_paths(home=home, output_root=output_root, data_dir=data_dir)[purpose.default_place][1]


def _place_paths(*, home: Path, output_root: Path, data_dir: Path) -> dict[str, tuple[str, Path]]:
    return {
        "home": ("Home", home),
        "output": ("Output folder", output_root),
        "models": ("Models", data_dir / "models"),
        "loras": ("LoRAs", data_dir / "loras"),
    }


def host_drives() -> tuple[str, ...]:
    """Return the host's drive roots where the OS has them (Windows), else nothing."""
    list_drives = getattr(os, "listdrives", None)
    if list_drives is None:
        return ()
    try:
        return tuple(list_drives())
    except OSError:
        return ()


def list_directory(purpose: BrowsePurpose, path_text: str, *, default_dir: Path | None = None, show_hidden: bool = False, show_all: bool = False) -> DirectoryListing:
    """List the folder *path_text* names, keeping what *purpose* can choose.

    An empty path lists *default_dir* (home when it is missing or not given). Symbolic links are kept as
    written, so a linked file opens in the folder that holds the link. A file path lists its folder and names the file as ``focus``. Dotfiles are left out unless *show_hidden*;
    files the purpose can't choose are counted in ``hidden_count`` unless *show_all*, which lists them as
    not selectable.

    Raises:
        ValueError: If the path is relative or names something other than a folder or file.
        FileNotFoundError: If nothing exists at the path.
        PermissionError: If the folder can't be read.
    """
    directory, focus = _target_directory(path_text, default_dir)
    entries: list[BrowseEntry] = []
    hidden_count = 0
    with os.scandir(directory) as scan:
        for item in scan:
            if not show_hidden and item.name.startswith("."):
                continue
            entry = _entry(item, purpose)
            if entry is None:
                continue
            if entry.kind == "file" and not entry.selectable and not show_all:
                hidden_count += 1
                continue
            entries.append(entry)
    entries.sort(key=lambda entry: (entry.kind != "directory", entry.name.casefold(), entry.name))
    parent = directory.parent
    return DirectoryListing(
        path=str(directory),
        parent=str(parent) if parent != directory else None,
        focus=focus,
        entries=tuple(entries),
        hidden_count=hidden_count,
    )


def create_folder(purpose: BrowsePurpose, parent_text: str, name: str) -> Path:
    """Create the folder *name* inside *parent_text* for a folder purpose and return its path.

    Raises:
        ValueError: If the purpose chooses files, the name isn't one plain folder name, or the parent is relative.
        FileNotFoundError: If the parent doesn't exist.
        FileExistsError: If something already has that name.
    """
    if purpose.kind != "directory":
        raise ValueError(f"{purpose.label} chooses files; folders can't be created here.")
    clean = name.strip()
    if not clean or clean in {".", ".."} or "/" in clean or "\\" in clean or "\0" in clean:
        raise ValueError("Enter one folder name, without slashes.")
    parent = _absolute(parent_text)
    if not parent.is_dir():
        raise FileNotFoundError(f"No folder at {parent}.")
    target = parent / clean
    target.mkdir()
    return target


def _target_directory(path_text: str, default_dir: Path | None) -> tuple[Path, str | None]:
    if not path_text.strip():
        candidate = default_dir.expanduser() if default_dir is not None else None
        # A configured place may not exist yet (a fresh output folder); fall back to home.
        return Path(os.path.abspath(candidate if candidate is not None and candidate.is_dir() else Path.home())), None
    path = _absolute(path_text)
    if path.is_dir():
        return path, None
    if path.is_file():
        return path.parent, path.name
    if path.exists():
        raise ValueError(f"{path} is not a folder or a file.")
    raise FileNotFoundError(f"No folder at {path}.")


def _absolute(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Enter an absolute path, or one starting with ~.")
    # Normalise `..` without following links, so the browser shows the path the user chose.
    return Path(os.path.abspath(path))


def _entry(item: os.DirEntry[str], purpose: BrowsePurpose) -> BrowseEntry | None:
    try:
        is_dir = item.is_dir()
        stat = item.stat()
    except OSError:
        # A broken link or a file removed while listing.
        return None
    if is_dir:
        return BrowseEntry(name=item.name, path=item.path, kind="directory", size=None, modified=stat.st_mtime, selectable=purpose.kind == "directory")
    return BrowseEntry(name=item.name, path=item.path, kind="file", size=stat.st_size, modified=stat.st_mtime, selectable=_accepts_file(purpose, item.name))


def _accepts_file(purpose: BrowsePurpose, name: str) -> bool:
    if purpose.kind != "existing_file":
        return False
    return not purpose.accepted_extensions or Path(name).suffix.lower() in purpose.accepted_extensions
