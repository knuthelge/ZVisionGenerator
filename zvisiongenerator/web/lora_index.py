"""Cache the detected model family of each LoRA file in the app database.

A lookup stats every LoRA and reads the cached rows of the folder. A LoRA that is new or whose file changed since
it was cached has its safetensors header read and its family detected on the spot, and the result is stored for the
next lookup. The database is only a cache: when it fails, every family is reported as unknown.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import warnings

from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.lora_family import LORA_FAMILY_VERSION, LoraFamily, detect_lora_family
from zvisiongenerator.utils.safetensors_stream import read_header_and_metadata

__all__ = ["LoraFamilyInfo", "lookup"]

_TABLE_PREFIX = "loras_entries_v"
_TABLE = f"{_TABLE_PREFIX}{LORA_FAMILY_VERSION}"

# Failures of the database, or of binding a value to it (a name with a lone surrogate, a timestamp beyond 64 bits).
_DB_ERRORS = (sqlite3.Error, OSError, UnicodeError, OverflowError)


@dataclass(frozen=True)
class LoraFamilyInfo:
    """The family of one LoRA, or ``None`` for both fields when it is unknown."""

    family: str | None
    size_variant: str | None


_UNKNOWN = LoraFamilyInfo(None, None)


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Create this version's table; when the version rises, drop lower-versioned tables (never higher)."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
        "root TEXT NOT NULL, name TEXT NOT NULL, mtime_ns INTEGER NOT NULL, ctime_ns INTEGER NOT NULL, "
        "size INTEGER NOT NULL, family TEXT NULL, size_variant TEXT NULL, source TEXT NOT NULL, "
        "PRIMARY KEY (root, name)) WITHOUT ROWID"
    )
    if from_version >= LORA_FAMILY_VERSION:
        return
    app_db.drop_lower_versioned_tables(conn, _TABLE_PREFIX, LORA_FAMILY_VERSION)


_COMPONENT = app_db.Component("loras", LORA_FAMILY_VERSION, _migrate)


def lookup(loras_dir: str | Path, names: Sequence[str]) -> dict[str, LoraFamilyInfo]:
    """Return the family of each LoRA in *names*, detecting and caching the ones that are new or changed.

    Args:
        loras_dir: Directory holding ``<name>.safetensors`` files.
        names: LoRA names, the file stems.

    Returns:
        One :class:`LoraFamilyInfo` per name; unknown for a missing file, an unreadable header, or when the
        database fails.
    """
    if not names:
        return {}
    root = str(Path(loras_dir).resolve())
    paths = {name: Path(loras_dir) / f"{name}.safetensors" for name in names}
    stats = {name: app_db.stat_key(path) for name, path in paths.items()}
    result = dict.fromkeys(names, _UNKNOWN)
    try:
        with closing(app_db.connect(components=(_COMPONENT,))) as conn:
            cached = _cached_rows(conn, root)
            fresh = []
            for name, key in stats.items():
                if key is None:
                    continue
                if name in cached and cached[name][0] == key:
                    result[name] = cached[name][1]
                    continue
                found = _detect(paths[name])
                result[name] = LoraFamilyInfo(found.family, found.size_variant)
                fresh.append((root, name, *key, found.family, found.size_variant, found.source))
            wanted = set(names)
            stale = [name for name in cached if name not in wanted]
            _store(conn, root, fresh, stale)
    except _DB_ERRORS as exc:
        warnings.warn(f"The LoRA index is unavailable ({exc}); LoRA families are unknown for now.", stacklevel=2)
        return dict.fromkeys(names, _UNKNOWN)
    return result


def _cached_rows(conn: sqlite3.Connection, root: str) -> dict[str, tuple[app_db.StatKey, LoraFamilyInfo]]:
    """Return ``{name: (stat_key, info)}`` for every row stored under *root*."""
    cursor = conn.execute(f"SELECT name, mtime_ns, ctime_ns, size, family, size_variant FROM {_TABLE} WHERE root = ?", (root,))
    return {name: ((mtime_ns, ctime_ns, size), LoraFamilyInfo(family, size_variant)) for name, mtime_ns, ctime_ns, size, family, size_variant in cursor}


def _detect(path: Path) -> LoraFamily:
    """Read *path*'s header and detect its family; an unreadable or non-safetensors file has no family."""
    try:
        header, metadata = read_header_and_metadata(path)
    except OSError, ValueError:
        return LoraFamily(None, None, "none")
    return detect_lora_family(header, metadata)


def _store(conn: sqlite3.Connection, root: str, rows: Sequence[tuple], stale: Sequence[str]) -> None:
    """Upsert *rows* and delete the *stale* names of *root* in one short transaction."""
    if not rows and not stale:
        return
    with app_db.transaction(conn):
        conn.executemany(
            f"INSERT INTO {_TABLE} (root, name, mtime_ns, ctime_ns, size, family, size_variant, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(root, name) DO UPDATE SET mtime_ns = excluded.mtime_ns, ctime_ns = excluded.ctime_ns, size = excluded.size, "
            "family = excluded.family, size_variant = excluded.size_variant, source = excluded.source",
            rows,
        )
        conn.executemany(f"DELETE FROM {_TABLE} WHERE root = ? AND name = ?", [(root, name) for name in stale])
