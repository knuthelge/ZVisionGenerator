"""Resolve model and LoRA paths from the ZIV data directory."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from zvisiongenerator.utils.platform import AliasMap, resolve_alias

_ziv_dirs_created: set[str] = set()
_LOCAL_PREFIXES = {"models", "checkpoints", "loras"}
_DISPLAY_SUFFIXES = (".safetensors", ".ckpt", ".bin", ".pt")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class HuggingFaceRepoReference:
    """Represent a conservative HuggingFace repository reference."""

    repo_id: str
    revision: str | None = None


def get_ziv_data_dir() -> Path:
    """Return the ZIV data directory (~/.ziv/ or ZIV_DATA_DIR).
    Creates the directory and models/ + loras/ subdirs if they don't exist.
    """
    env = os.environ.get("ZIV_DATA_DIR", "").strip()
    if env:
        data_dir = Path(env)
    else:
        data_dir = Path.home() / ".ziv"

    key = str(data_dir)
    if key not in _ziv_dirs_created:
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "loras").mkdir(parents=True, exist_ok=True)
        _ziv_dirs_created.add(key)
    return data_dir


def _is_url_like(value: str) -> bool:
    return bool(_URL_SCHEME_RE.match(value.strip()))


def _is_safe_hf_segment(segment: str) -> bool:
    if not segment or segment in {".", ".."}:
        return False
    if segment.startswith((".", "-")) or segment.endswith((".", "-")):
        return False
    if "--" in segment or ".." in segment:
        return False
    return bool(_SEGMENT_RE.fullmatch(segment))


def parse_huggingface_repo_reference(value: str) -> HuggingFaceRepoReference | None:
    """Parse a conservative HuggingFace repo reference without remote probing."""

    stripped = value.strip()
    if not stripped or _is_url_like(stripped):
        return None
    if stripped.startswith(("/", "./", "../", "~/")) or stripped.startswith(("//", "\\\\")):
        return None
    if _WINDOWS_DRIVE_RE.match(stripped) or "\\" in stripped:
        return None

    parts = stripped.split("/")
    if len(parts) != 2:
        return None

    owner, repo_part = parts
    if owner in _LOCAL_PREFIXES:
        return None
    if not _is_safe_hf_segment(owner):
        return None

    revision: str | None = None
    if "@" in repo_part:
        repo_name, revision = repo_part.split("@", 1)
        if not repo_name or not revision or "@" in revision or not _REVISION_RE.fullmatch(revision):
            return None
    else:
        repo_name = repo_part

    if not _is_safe_hf_segment(repo_name):
        return None

    return HuggingFaceRepoReference(repo_id=f"{owner}/{repo_name}", revision=revision)


def split_huggingface_revision(value: str) -> tuple[str, str | None]:
    """Split ``owner/repo@revision`` into the repo id and its revision.

    Any other value (a plain repo id, a local path) comes back unchanged with no revision.
    """

    reference = parse_huggingface_repo_reference(value)
    if reference is None or reference.revision is None:
        return value, None
    return reference.repo_id, reference.revision


def is_huggingface_repo_id(value: str) -> bool:
    """Return whether value matches the conservative HuggingFace repo contract."""

    return parse_huggingface_repo_reference(value) is not None


def is_explicit_local_path(value: str) -> bool:
    """Classify deterministic local path syntax without filesystem probing."""

    stripped = value.strip()
    if not stripped or _is_url_like(stripped):
        return False
    if stripped.startswith(("/", "./", "../", "~/", "//", "\\\\")):
        return True
    if _WINDOWS_DRIVE_RE.match(stripped) or "\\" in stripped:
        return True

    first_segment = re.split(r"[/\\]+", stripped, maxsplit=1)[0]
    if first_segment in _LOCAL_PREFIXES and re.search(r"[/\\]", stripped):
        return True
    if re.search(r"[/\\]", stripped) and parse_huggingface_repo_reference(stripped) is None:
        return True
    return False


def is_remote_lora_reference(value: str) -> bool:
    """Return whether a LoRA token is a HuggingFace-shaped remote reference."""

    return parse_huggingface_repo_reference(value) is not None


def is_existing_file(path: str) -> bool:
    """Return whether *path* names an existing file, expanding ``~``; a path the OS cannot look up is not one."""
    try:
        return Path(path).expanduser().is_file()
    except OSError:
        return False


def collapse_home(path: str, home: str | None = None) -> str:
    """Replace a leading home directory in *path* with ``~``, leaving other values unchanged.

    Args:
        path: A path or other reference, such as a Hugging Face repo ID.
        home: The home directory to collapse; defaults to the current user's.

    Returns:
        ``~`` followed by the rest of the path when *path* is inside *home*, else *path* as given.
    """
    home_text = (home if home is not None else str(Path.home())).rstrip("/\\")
    if not home_text or not path.startswith(home_text):
        return path
    rest = path[len(home_text) :]
    if rest and rest[0] not in "/\\":
        return path
    return "~" + rest


def expand_home(path: str) -> str:
    """Expand a leading ``~`` written by :func:`collapse_home`, leaving other values unchanged."""
    if path == "~" or path.startswith(("~/", "~\\")):
        return str(Path.home()) + path[1:]
    return path


def display_basename(value: str) -> str:
    """Return the final path/repo component across POSIX and Windows separators."""

    stripped = value.strip().rstrip("/\\")
    if not stripped:
        return stripped
    parts = re.split(r"[/\\]+", stripped)
    return parts[-1] or stripped


def display_stem(value: str, suffixes: tuple[str, ...] = _DISPLAY_SUFFIXES) -> str:
    """Return a display basename with one recognized final suffix removed."""

    name = display_basename(value)
    lower_name = name.lower()
    for suffix in suffixes:
        if lower_name.endswith(suffix.lower()):
            return name[: -len(suffix)]
    return name


def resolve_model_path(
    name_or_path: str,
    *,
    aliases: AliasMap | None = None,
    platform_key: str | None = None,
) -> str:
    """Resolve a model name/path to a filesystem path.

    Resolution order:
    1. Explicit local paths and supported repo references are returned as-is.
    2. Bare names check ~/.ziv/models/<name>/ and use it if found.
    3. Bare aliases resolve when provided. An alias whose value is a bare name names a folder in
       ~/.ziv/models/, whether or not it exists yet (a model built on first use).
    4. Otherwise the original string is returned unchanged.
    """
    stripped = name_or_path.strip()
    if stripped.startswith("~/") and not _is_url_like(stripped):
        return str(Path(stripped).expanduser())
    if is_explicit_local_path(stripped) or is_huggingface_repo_id(stripped):
        return stripped

    models_dir = get_ziv_data_dir() / "models"
    candidate = models_dir / stripped
    if candidate.is_dir():
        return str(candidate)

    if aliases and stripped in aliases:
        alias_value = aliases[stripped]
        if isinstance(alias_value, str):
            target = alias_value
        elif platform_key is not None:
            target = resolve_alias(alias_value, platform_key)
        else:
            return stripped
        return str(models_dir / target) if _is_bare_name(target) else target

    return stripped


def canonical_model_name(
    name: str,
    *,
    aliases: AliasMap,
    platform_key: str,
) -> str:
    """Return the alias that *name* names on *platform_key*, or *name* unchanged.

    A Hugging Face reference whose repo id is, in any letter case, the resolved repo of exactly one alias, with no
    revision or that alias's revision, is that alias: the same model, reached by its repo id. Anything else (a local
    path, an alias name, a repo that no alias or several aliases resolve to, another revision) is returned as given.
    """
    reference = parse_huggingface_repo_reference(name)
    if reference is None:
        return name
    matches: list[str] = []
    for alias, value in aliases.items():
        try:
            target = parse_huggingface_repo_reference(resolve_alias(value, platform_key))
        except ValueError:
            continue
        if target is not None and target.repo_id.casefold() == reference.repo_id.casefold() and reference.revision in (None, target.revision):
            matches.append(alias)
    return matches[0] if len(matches) == 1 else name


def _is_bare_name(value: str) -> bool:
    """Return whether *value* is a plain folder name: no separators, home prefix or repo-id shape."""
    return bool(value) and value not in {".", ".."} and not any(char in value for char in "/\\:~")


def resolve_lora_path(name_or_path: str) -> str:
    """Resolve a LoRA name/path to a filesystem path.

    Resolution order:
    1. Explicit local paths and remote-shaped LoRA references are returned as-is.
    2. Bare names check ~/.ziv/loras/<name>.safetensors and ~/.ziv/loras/<name>.
    3. Otherwise the original string is returned unchanged.
    """
    stripped = name_or_path.strip()
    if stripped.startswith("~/") and not _is_url_like(stripped):
        return str(Path(stripped).expanduser())
    if is_explicit_local_path(stripped) or is_remote_lora_reference(stripped):
        return stripped

    data_dir = get_ziv_data_dir()

    candidate = data_dir / "loras" / f"{stripped}.safetensors"
    if candidate.is_file():
        return str(candidate)

    candidate = data_dir / "loras" / stripped
    if candidate.is_file():
        return str(candidate)

    return stripped
