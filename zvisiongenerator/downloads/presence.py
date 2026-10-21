"""Offline presence test and partial-file cleanup for download targets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zvisiongenerator.downloads.target import DownloadTarget, is_literal
from zvisiongenerator.utils.model_files import cached_snapshot_dirs, has_complete_weights, huggingface_cache_dir

_SAFETENSORS_SUFFIX = ".safetensors"


def target_present_offline(
    target: DownloadTarget,
    *,
    cached_file: Callable[..., object] | None = None,
    snapshots: Callable[[str, str | None], list[Path]] = cached_snapshot_dirs,
) -> bool:
    """Return whether *target* is fully on disk, judged without the network.

    Args:
        target: The download to test.
        cached_file: Looks up one cached file like ``huggingface_hub.try_to_load_from_cache``; defaults to it.
        snapshots: Lists the cache snapshots of a repo and revision, newest first.
    """
    if target.local_dir is not None:
        return _local_dir_present(target)
    patterns = target.allow_patterns
    if patterns is None:
        return any(has_complete_weights(snapshot) for snapshot in snapshots(target.repo_id, target.revision))
    if all(is_literal(pattern) for pattern in patterns):
        return _literal_files_cached(target, patterns, cached_file)
    covered = _covered_folders(patterns)
    literal_weights = [pattern for pattern in patterns if is_literal(pattern) and pattern.endswith(_SAFETENSORS_SUFFIX)]
    return any(has_complete_weights(snapshot, only=covered) and all((snapshot / pattern).is_file() for pattern in literal_weights) for snapshot in snapshots(target.repo_id, target.revision))


def remove_partial_files(target: DownloadTarget, *, cache_dir: Path | None = None) -> int:
    """Delete the ``*.incomplete`` files a killed download left behind and return how many were removed.

    Args:
        target: The download whose leftovers to remove.
        cache_dir: The Hub cache folder; defaults to the configured one.
    """
    if target.local_dir is not None:
        candidates = Path(target.local_dir, ".cache", "huggingface", "download").rglob("*.incomplete")
    else:
        repo_dir = (cache_dir or huggingface_cache_dir()) / f"models--{target.repo_id.replace('/', '--')}"
        candidates = (repo_dir / "blobs").glob("*.incomplete")
    removed = 0
    for path in candidates:
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


def _local_dir_present(target: DownloadTarget) -> bool:
    """Return whether every literal allow pattern exists as a file under the target's local folder."""
    patterns = target.allow_patterns
    if patterns is None or not all(is_literal(pattern) for pattern in patterns):
        return False
    local_dir = Path(target.local_dir or "")
    return all((local_dir / pattern).is_file() for pattern in patterns)


def _literal_files_cached(target: DownloadTarget, patterns: tuple[str, ...], cached_file: Callable[..., object] | None) -> bool:
    """Return whether the Hub cache holds every literal file of *patterns* for the target's revision."""
    if cached_file is None:
        from huggingface_hub import try_to_load_from_cache

        cached_file = try_to_load_from_cache
    for pattern in patterns:
        found = cached_file(target.repo_id, pattern, revision=target.revision)
        if not isinstance(found, str | Path) or not Path(found).is_file():
            return False
    return True


def _covered_folders(patterns: tuple[str, ...]) -> set[str]:
    """Return the component folders that *patterns* select with a glob (``vae/*.safetensors``, ``scheduler/*``)."""
    covered: set[str] = set()
    for pattern in patterns:
        folder, separator, rest = pattern.partition("/")
        if separator and is_literal(folder) and not is_literal(rest):
            covered.add(folder)
    return covered
