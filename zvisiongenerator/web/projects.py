"""List and create projects: real folders directly under the output folder."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePath

PROJECT_FIELD = "project"
"""The form field a generate or plan request names its project in."""

NO_PROJECT_FILTER = "no_project"
"""The Gallery media filter that keeps only the files in no project."""

_NAME_RULE_MESSAGE = "A project name can't contain / \\ : * ? \" < > |, start with a dot or end with one."
_FORBIDDEN_CHARACTERS = frozenset('<>:"/\\|?*')
_RESERVED_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)), *(f"lpt{n}" for n in range(1, 10))})


@dataclass(frozen=True)
class Project:
    """Describe one project folder under the output folder."""

    name: str
    path: str


def list_projects(root: Path) -> tuple[Project, ...]:
    """Return the real folders directly under *root*, sorted without regard to case.

    Symlinks, files and dot-prefixed folders are skipped. *root* must exist.
    """
    with os.scandir(root) as entries:
        found = [Project(name=entry.name, path=entry.path) for entry in entries if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False)]
    return tuple(sorted(found, key=lambda project: project.name.casefold()))


def create_project(root: Path, name: str) -> Project:
    """Create the project folder *name* under *root* and return it.

    The name rules are the same on every platform, so a project made on one machine opens on any other.

    *root* must exist.

    Raises:
        ValueError: If the trimmed name is empty or breaks a name rule.
        FileExistsError: If a project with that name exists, ignoring case.
    """
    clean = name.strip()
    _validate_project_name(clean)
    if any(project.name.casefold() == clean.casefold() for project in list_projects(root)):
        raise FileExistsError(f"A project named {clean} already exists.")
    target = root / clean
    target.mkdir()
    return Project(name=clean, path=str(target))


def project_dir(root: Path, name: str | None) -> Path:
    """Return the folder for project *name* under *root*, or *root* itself for a blank name.

    Only the shape of the name is checked; the folder need not exist.

    Raises:
        ValueError: If the name isn't one plain folder name (separators, a leading dot, a NUL).
    """
    if name is None or not name.strip():
        return root
    if PurePath(name).name != name or name.startswith(".") or "\\" in name or "\0" in name:
        raise ValueError(f"Not a valid project name: {name!r}.")
    return root / name


def asset_project(asset_id: str) -> str | None:
    """Return the project an asset ID lives in: its first segment when the ID has more than one and it isn't a dot-folder."""
    first, separator, _rest = asset_id.partition("/")
    if not separator or not first or first.startswith("."):
        return None
    return first


def folder_project(root: Path, folder: str | Path) -> str | None:
    """Return the project a save into *folder* lands in: its first segment under *root*, or ``None`` for the top, a dot-folder or a folder outside *root*."""
    try:
        parts = Path(folder).resolve().relative_to(root.resolve()).parts
    except OSError, ValueError:
        return None
    return parts[0] if parts and not parts[0].startswith(".") else None


def _validate_project_name(name: str) -> None:
    if not name:
        raise ValueError("Enter a project name.")
    if name.startswith(".") or name.endswith(".") or any(char in _FORBIDDEN_CHARACTERS or ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError(_NAME_RULE_MESSAGE)
    if name.partition(".")[0].casefold() in _RESERVED_NAMES:
        raise ValueError(f"{name} is a reserved name on Windows. Choose another.")
