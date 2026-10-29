"""Tell which LTX version (2.3 or 2.5) a video model is, without loading it.

The version comes from the cheapest source that knows it: the config's version blocks (built-in aliases, and converted
folders by their manifest), the table of known repos, then, for a local model folder, the folder's own files. The files
read are small JSON files; nothing here imports torch, diffusers or MLX.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.config import ltx_version
from zvisiongenerator.utils.ltx_converted import CONVERTED_LTX_MANIFEST, LTX_MODEL_TYPES, read_converted_ltx
from zvisiongenerator.utils.model_files import is_ltx25_diffusers_dir
from zvisiongenerator.utils.video_model_detect import known_ltx_repo_version

__all__ = ["DEFAULT_LTX_VERSION", "ltx_folder_version", "ltx_model_version"]

# The version of an LTX model nothing says more about: the family block of the config is LTX-2.3.
DEFAULT_LTX_VERSION = "2.3"
# The transformer configs of an MLX folder, in the order ``LTXModelConfig.from_checkpoint_dir`` reads them (the first found wins).
_CONFIG_FILES = ("embedded_config.json", "config.json")


def ltx_model_version(config: dict[str, Any], name: str, resolved_path: str) -> str:
    """Return the LTX version of the model *name* that loads from *resolved_path*: ``"2.3"`` or ``"2.5"``.

    Args:
        config: The app config (its ``video_model_presets.ltx.versions`` blocks list the aliases of each version).
        name: The model's name: an alias, or a folder in the models folder.
        resolved_path: The repo ID or folder the name resolves to.

    Returns:
        ``"2.5"`` for an alias of an LTX-2.5 block, a known LTX-2.5 repo, a converted LTX-2.5 folder, an MLX folder whose
        transformer config says ``ff_bias: false`` or a diffusers folder with a Gemma 4 text encoder; ``"2.3"`` otherwise,
        including a folder with no readable config.
    """
    block = ltx_version(config, name)
    if block is not None:
        kind = ((config.get("video_model_presets") or {}).get("ltx") or {}).get("versions", {}).get(block, {}).get("model_type")
        return LTX_MODEL_TYPES.get(kind, DEFAULT_LTX_VERSION)
    known = known_ltx_repo_version(resolved_path)
    if known is not None:
        return known
    folder = Path(resolved_path).expanduser()
    return ltx_folder_version(folder) if folder.is_dir() else DEFAULT_LTX_VERSION


def ltx_folder_version(folder: Path) -> str:
    """Return the LTX version of the model folder *folder*, cached until one of its small config files changes.

    A converted folder says it in its manifest; an MLX folder by its transformer config (``ff_bias: false`` is LTX-2.5,
    the signal the vendored loader uses); a diffusers folder by its text encoder class.
    """
    return _folder_version(folder, tuple(_mtime(folder / name) for name in (CONVERTED_LTX_MANIFEST, *_CONFIG_FILES, "model_index.json")))


def _mtime(path: Path) -> int | None:
    """Return the modification time of *path* in nanoseconds, or None when it is missing."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


@lru_cache(maxsize=256)
def _folder_version(folder: Path, _stamp: tuple[int | None, ...]) -> str:
    """Read the version of *folder*; *_stamp* (the modification times of the files read) keys the cache."""
    manifest = read_converted_ltx(folder)
    if manifest is not None:
        return LTX_MODEL_TYPES[manifest["kind"]]
    if _ff_bias_disabled(folder) or is_ltx25_diffusers_dir(folder):
        return "2.5"
    return DEFAULT_LTX_VERSION


def _ff_bias_disabled(folder: Path) -> bool:
    """Return whether the first transformer config of *folder* declares ``ff_bias: false``; a missing or unreadable config does not."""
    for name in _CONFIG_FILES:
        path = folder / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return False
        transformer = config.get("transformer", config) if isinstance(config, dict) else None
        return isinstance(transformer, dict) and transformer.get("ff_bias") is False
    return False
