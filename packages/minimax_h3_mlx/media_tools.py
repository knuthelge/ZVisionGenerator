"""Release-safe ffmpeg/ffprobe resolver utilities.

The package must never silently depend on an operator's development checkout or an
undeclared shell PATH.  This module centralizes media-tool selection for generation,
doctor, deployment runners, and clean-room smoke tests.  Callers can still pass an
explicit executable path/name, but automatic lookup records the route and prefers
package/python-environment-local tools before PATH fallback.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

FFMPEG_ENV_VAR = "MINIMAX_H3_FFMPEG"
FFPROBE_ENV_VAR = "MINIMAX_H3_FFPROBE"
MEDIA_TOOL_ENV_VARS = {"ffmpeg": FFMPEG_ENV_VAR, "ffprobe": FFPROBE_ENV_VAR}
MEDIA_TOOL_ALIASES = {
    "ffmpeg": ("ffmpeg", "static_ffmpeg"),
    "ffprobe": ("ffprobe", "static_ffprobe"),
}
PROJECT_LOCAL_MEDIA_TOOL_DIRS = (
    ".venv/bin",
    "bin",
    "tools/bin",
    "tools/ffmpeg",
    "tools/ffmpeg/darwin_arm64",
    "vendor/bin",
)

WhichFn = Callable[[str], str | None]
PathPredicate = Callable[[Path], bool]


@dataclass(frozen=True)
class MediaToolResolution:
    """Machine-readable result of resolving one media executable."""

    name: str
    requested: str
    source: str
    command: str
    path: str | None
    exists: bool
    executable: bool
    route: str
    searched: tuple[str, ...]
    env_var: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.exists and self.executable and self.path)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _default_which(name: str) -> str | None:
    return shutil.which(name)


def _default_exists(path: Path) -> bool:
    return path.exists()


def _default_executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)


def _aliases_for(name: str) -> tuple[str, ...]:
    return MEDIA_TOOL_ALIASES.get(name, (name,))


def default_media_tool_dirs(*, cwd: str | Path | None = None) -> tuple[Path, ...]:
    """Return release-local directories searched before PATH.

    The installed package case is covered by the active Python environment's bin
    directory and an optional package-local ``bin`` directory.  The repo/deployment
    case is covered by conventional project-local tool directories under ``cwd``.
    """

    base = Path.cwd() if cwd is None else Path(cwd)
    package_dir = Path(__file__).resolve().parent
    candidates = [
        Path(sys.executable).resolve().parent,
        Path(sys.prefix).resolve() / "bin",
        package_dir / "bin",
    ]
    candidates.extend(base / directory for directory in PROJECT_LOCAL_MEDIA_TOOL_DIRS)
    return _dedupe_paths(candidates)


def media_tool_candidate_paths(
    name: str,
    *,
    cwd: str | Path | None = None,
    local_bin_dirs: Sequence[str | Path] | None = None,
) -> list[Path]:
    """Expand directory+alias candidates for ``name`` without checking them."""

    dirs = default_media_tool_dirs(cwd=cwd) if local_bin_dirs is None else _dedupe_paths([Path(p) for p in local_bin_dirs])
    candidates: list[Path] = []
    base = Path.cwd() if cwd is None else Path(cwd)
    for directory in dirs:
        directory = Path(directory).expanduser()
        if not directory.is_absolute():
            directory = base / directory
        for alias in _aliases_for(name):
            candidates.append(directory / alias)
    return candidates


def _resolve_requested_path(
    name: str,
    requested: str,
    *,
    source: str,
    env_var: str | None,
    cwd: Path,
    path_exists: PathPredicate,
    path_executable: PathPredicate,
    searched: list[str],
) -> MediaToolResolution:
    path = Path(requested).expanduser()
    if not path.is_absolute():
        path = cwd / path
    exists = path_exists(path)
    executable = path_executable(path) if exists else False
    searched.append(str(path))
    error = None
    if not exists:
        error = f"{name} requested via {source} was not found at {path}"
    elif not executable:
        error = f"{name} requested via {source} exists but is not executable: {path}"
    return MediaToolResolution(
        name=name,
        requested=requested,
        source=source,
        command=str(path),
        path=str(path) if exists else None,
        exists=exists,
        executable=executable,
        route="explicit_path" if source == "argument" else "env_path",
        searched=tuple(searched),
        env_var=env_var,
        error=error,
    )


def _resolve_requested_name(
    name: str,
    requested: str,
    *,
    source: str,
    env_var: str | None,
    which: WhichFn,
    searched: list[str],
) -> MediaToolResolution:
    searched.append(f"PATH:{requested}")
    resolved = which(requested)
    error = None if resolved else f"{name} requested via {source} as {requested!r} was not found on PATH"
    return MediaToolResolution(
        name=name,
        requested=requested,
        source=source,
        command=resolved or requested,
        path=resolved,
        exists=resolved is not None,
        executable=resolved is not None,
        route="explicit_name_on_path" if source == "argument" else "env_name_on_path",
        searched=tuple(searched),
        env_var=env_var,
        error=error,
    )


def resolve_media_tool(
    name: str,
    requested: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    env_var: str | None = None,
    cwd: str | Path | None = None,
    local_bin_dirs: Sequence[str | Path] | None = None,
    allow_path: bool = True,
    which: WhichFn | None = None,
    path_exists: PathPredicate | None = None,
    path_executable: PathPredicate | None = None,
) -> MediaToolResolution:
    """Resolve ``ffmpeg``/``ffprobe`` and record the selected route.

    Priority is:
    1. explicit argument path/name;
    2. documented environment variable (``MINIMAX_H3_FFMPEG``/``MINIMAX_H3_FFPROBE``);
    3. release-local candidates in the active Python environment/package/project;
    4. PATH fallback, unless ``allow_path`` is false.
    """

    env_map = os.environ if env is None else env
    env_name = env_var if env_var is not None else MEDIA_TOOL_ENV_VARS.get(name)
    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    which_fn = _default_which if which is None else which
    exists_fn = _default_exists if path_exists is None else path_exists
    executable_fn = _default_executable if path_executable is None else path_executable
    searched: list[str] = []

    raw_requested = None if requested in {None, "", "auto"} else str(requested)
    source = "argument" if raw_requested is not None else None
    if raw_requested is None and env_name and env_map.get(env_name):
        raw_requested = env_map[env_name]
        source = f"env:{env_name}"

    if raw_requested is not None:
        if "/" in raw_requested or raw_requested.startswith("~"):
            return _resolve_requested_path(
                name,
                raw_requested,
                source=source or "argument",
                env_var=env_name,
                cwd=cwd_path,
                path_exists=exists_fn,
                path_executable=executable_fn,
                searched=searched,
            )
        return _resolve_requested_name(
            name,
            raw_requested,
            source=source or "argument",
            env_var=env_name,
            which=which_fn,
            searched=searched,
        )

    for candidate in media_tool_candidate_paths(name, cwd=cwd_path, local_bin_dirs=local_bin_dirs):
        searched.append(str(candidate))
        exists = exists_fn(candidate)
        executable = executable_fn(candidate) if exists else False
        if exists and executable:
            return MediaToolResolution(
                name=name,
                requested=name,
                source="release_local",
                command=str(candidate),
                path=str(candidate),
                exists=True,
                executable=True,
                route="local_candidate",
                searched=tuple(searched),
                env_var=env_name,
            )

    if allow_path:
        for alias in _aliases_for(name):
            searched.append(f"PATH:{alias}")
            resolved = which_fn(alias)
            if resolved:
                return MediaToolResolution(
                    name=name,
                    requested=name,
                    source="PATH",
                    command=resolved,
                    path=resolved,
                    exists=True,
                    executable=True,
                    route="path_fallback",
                    searched=tuple(searched),
                    env_var=env_name,
                )

    path_value = env_map.get("PATH", "")
    path_state = "disabled" if not allow_path else ("empty" if not path_value else "searched")
    error = (
        f"{name} is unresolved: provide --{name}, set {env_name or 'the media-tool env var'}, "
        "or install a release-local provider in the package/venv/project media-tool directories; "
        f"system PATH lookup was {path_state} (PATH={path_value!r})."
    )
    return MediaToolResolution(
        name=name,
        requested=name,
        source="unresolved",
        command=name,
        path=None,
        exists=False,
        executable=False,
        route="unresolved",
        searched=tuple(searched),
        env_var=env_name,
        error=error,
    )


def media_tool_error(resolution: MediaToolResolution) -> str:
    """Return a human-readable error from a failed resolution."""

    if resolution.error:
        return resolution.error
    return f"{resolution.name} resolution failed via {resolution.source}"
