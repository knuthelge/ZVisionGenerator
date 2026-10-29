"""Value objects that describe one Hugging Face download and its progress."""

from __future__ import annotations

import json
from dataclasses import dataclass

_GLOB_CHARS = frozenset("*?[")


def is_literal(pattern: str) -> bool:
    """Return whether *pattern* names a single file (no ``*``, ``?`` or ``[``)."""
    return not any(char in _GLOB_CHARS for char in pattern)


@dataclass(frozen=True)
class DownloadTarget:
    """One ``snapshot_download`` call: a repo, a revision, the files to fetch and where they go.

    Attributes:
        repo_id: The Hugging Face repo, ``owner/name``.
        revision: Branch, tag or commit; ``None`` means ``main``.
        allow_patterns: Glob patterns to fetch; ``None`` fetches the whole repo.
        ignore_patterns: Glob patterns to skip.
        local_dir: A plain folder to download into instead of the Hub cache.
        anonymous: Talk to the Hub without the user's token, for a file the user did not name themselves.
    """

    repo_id: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] | None = None
    ignore_patterns: tuple[str, ...] | None = None
    local_dir: str | None = None
    anonymous: bool = False

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...]]:
        """Identify the destination: repo, revision, local folder (empty for the cache) and, for a local folder, the files wanted.

        A plain folder holds exactly the files its target asks for, so two targets that fetch different files of one
        repo into one folder are different destinations. A cache target keeps the three-part key: its files are shared
        blobs, which the Hub library locks itself.
        """
        files = tuple(sorted(self.allow_patterns or ())) if self.local_dir else ()
        return (self.repo_id, self.revision or "main", self.local_dir or "", files)

    def to_json(self) -> str:
        """Serialize the target for the download child process."""
        return json.dumps(
            {
                "repo_id": self.repo_id,
                "revision": self.revision,
                "allow_patterns": None if self.allow_patterns is None else list(self.allow_patterns),
                "ignore_patterns": None if self.ignore_patterns is None else list(self.ignore_patterns),
                "local_dir": self.local_dir,
                "anonymous": self.anonymous,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> DownloadTarget:
        """Rebuild a target from :meth:`to_json` output."""
        data = json.loads(text)
        allow = data.get("allow_patterns")
        ignore = data.get("ignore_patterns")
        return cls(
            repo_id=data["repo_id"],
            revision=data.get("revision"),
            allow_patterns=None if allow is None else tuple(allow),
            ignore_patterns=None if ignore is None else tuple(ignore),
            local_dir=data.get("local_dir"),
            anonymous=bool(data.get("anonymous", False)),
        )


@dataclass(frozen=True)
class DownloadProgress:
    """A snapshot of a running download; unknown values are ``None``."""

    bytes_done: int
    bytes_total: int | None
    bytes_per_sec: float | None
    eta_secs: float | None


@dataclass(frozen=True)
class TargetCheck:
    """What a dry run says is still missing for a target; ``missing_bytes`` is ``None`` when any size is unknown."""

    missing_files: int
    missing_bytes: int | None
