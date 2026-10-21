"""Dry-run check of what a download target still needs from the Hub."""

from __future__ import annotations

from collections.abc import Callable

from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck


def check_target(target: DownloadTarget, *, snapshot_download: Callable[..., object] | None = None) -> TargetCheck | None:
    """Return the files and bytes *target* still needs, or ``None`` when the Hub cannot be reached.

    Args:
        target: The download to check.
        snapshot_download: Runs ``huggingface_hub.snapshot_download``-style calls; defaults to it.

    Returns:
        The missing counts, or ``None`` when offline or unreachable.

    Raises:
        RuntimeError: The Hub refused the request (gated, missing repo or revision, server error).
    """
    from huggingface_hub.errors import DryRunError, HfHubHTTPError

    if snapshot_download is None:
        from huggingface_hub import snapshot_download
    try:
        infos = snapshot_download(
            target.repo_id,
            revision=target.revision,
            allow_patterns=None if target.allow_patterns is None else list(target.allow_patterns),
            ignore_patterns=None if target.ignore_patterns is None else list(target.ignore_patterns),
            local_dir=target.local_dir,
            dry_run=True,
        )
    except DryRunError as exc:
        cause = exc.__cause__
        if cause is None or _is_unreachable(cause):
            return None
        if isinstance(cause, HfHubHTTPError):
            raise RuntimeError(f"Could not download {target.repo_id}: {cause}") from cause
        raise
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Could not download {target.repo_id}: {exc}") from exc
    except Exception as exc:
        if _is_unreachable(exc):
            return None
        raise
    missing = [info for info in infos if info.will_download]
    sizes = [info.file_size for info in missing]
    return TargetCheck(missing_files=len(missing), missing_bytes=None if any(size is None for size in sizes) else sum(sizes))


def _is_unreachable(exc: BaseException) -> bool:
    """Return whether *exc* means offline mode or an unreachable Hub rather than a refusal."""
    import httpx
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    return isinstance(exc, OfflineModeIsEnabled | LocalEntryNotFoundError | httpx.ConnectError | httpx.TimeoutException)
