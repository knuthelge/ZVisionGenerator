"""Keep a persistent index of the facts read from gallery files in the app database, reconciled on each listing.

The index stores the raw facts a reader extracts from each media file, keyed by ``(root, rel_path)`` and
validated by the file's ``(mtime_ns, ctime_ns, size)``. A listing hands the index the files its stat-only
walk found; the index returns facts from its rows and calls the reader only for new or changed files.

Rules the index follows:

* The database is a cache. Every error in ``_DB_ERRORS`` (``sqlite3.Error``, ``OSError``, and the ``UnicodeError`` and
  ``OverflowError`` that SQLite raises when a value cannot be bound) is caught and the facts are read without it.
* A read is stored only when it is complete, settled and stable (see :func:`_should_persist`).
* Rows of files the walk did not see are deleted only when the caller says the walk was trustworthy.
* Corruption is handled by :func:`zvisiongenerator.utils.app_db.recover_if_corrupt`, nowhere else.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from zvisiongenerator.utils import app_db

GALLERY_COMPONENT_NAME = "gallery"
# Bump when the table's columns change or the gallery's fact reader reads something different.
GALLERY_INDEX_VERSION = 1

_TABLE_PREFIX = "gallery_assets_v"
_TABLE_NAME_PATTERN = re.compile(rf"{_TABLE_PREFIX}(\d+)")

type Facts = dict[str, Any]
type StatKey = tuple[int, int, int]
type _Upsert = tuple[str, int, int, int, str]
type _Delete = tuple[str, int, int, int]

# Failures of the database or of binding a value to it: a path or fact with a lone surrogate (a non-UTF-8 file
# name) raises UnicodeError, and a timestamp beyond 64 bits raises OverflowError. The reader's errors are not here.
_DB_ERRORS = (sqlite3.Error, OSError, UnicodeError, OverflowError)

__all__ = [
    "GALLERY_COMPONENT_NAME",
    "GALLERY_INDEX_VERSION",
    "FactsRead",
    "FactsReader",
    "GalleryIndex",
    "ScannedFile",
]


@dataclass(frozen=True)
class ScannedFile:
    """Describe one media file found by the walk: its asset ID, where it is, and its stat key.

    Attributes:
        rel_path: POSIX path relative to the output root; the asset ID.
        path: Path as walked; used to read and re-stat the file.
        mtime_ns: ``st_mtime_ns`` of the walk's stat.
        ctime_ns: ``st_ctime_ns`` of the walk's stat.
        mtime: ``st_mtime`` of the same stat.
        size: ``st_size`` of the same stat.
    """

    rel_path: str
    path: Path
    mtime_ns: int
    ctime_ns: int
    mtime: float
    size: int

    @property
    def stat_key(self) -> StatKey:
        """Return ``(mtime_ns, ctime_ns, size)``, the key a stored row must match."""
        return (self.mtime_ns, self.ctime_ns, self.size)


@dataclass(frozen=True)
class FactsRead:
    """Hold the JSON-serializable facts read from one file and whether every read succeeded.

    Attributes:
        facts: Raw facts read from the file.
        complete: ``False`` when any reader failed, so the facts may be missing something.
    """

    facts: Facts
    complete: bool


type FactsReader = Callable[[ScannedFile], FactsRead]


def _migrate_gallery(conn: sqlite3.Connection, from_version: int, *, version: int) -> None:
    """Create this version's table; when the highest version rises, drop lower-versioned gallery tables (never higher).

    Args:
        conn: Connection inside the ``BEGIN IMMEDIATE`` that ``app_db`` opened.
        from_version: Highest gallery version that has run on the file, 0 for a new file.
        version: Version of the table to create.
    """
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE_PREFIX}{version} ("
        "root TEXT NOT NULL, rel_path TEXT NOT NULL, mtime_ns INTEGER NOT NULL, ctime_ns INTEGER NOT NULL, "
        "size INTEGER NOT NULL, facts TEXT NOT NULL, PRIMARY KEY (root, rel_path)) WITHOUT ROWID"
    )
    if from_version >= version:
        return
    names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB ?", (f"{_TABLE_PREFIX}[0-9]*",))]
    for name in names:
        match = _TABLE_NAME_PATTERN.fullmatch(name)
        if match is not None and int(match.group(1)) < version:
            conn.execute(f"DROP TABLE {_TABLE_PREFIX}{int(match.group(1))}")


def _restat(path: Path) -> StatKey | None:
    """Return the current ``(mtime_ns, ctime_ns, size)`` of *path*, or ``None`` when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _should_persist(file: ScannedFile, result: FactsRead, *, after: StatKey | None, now: float, settle_seconds: float) -> bool:
    """Return whether a read is safe to store: complete, settled, and the file unchanged since the walk."""
    if not result.complete or now - file.mtime < settle_seconds:
        return False
    return after is not None and after == file.stat_key


def _serialize(facts: Facts) -> str | None:
    """Return *facts* as compact JSON, or ``None`` when they cannot be serialized."""
    try:
        return json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    except TypeError, ValueError:
        return None


def _parse_facts(text: str) -> Facts | None:
    """Return the JSON object in *text*, or ``None`` when it is not valid or not an object."""
    try:
        facts = json.loads(text)
    except ValueError:
        return None
    return facts if isinstance(facts, dict) else None


def _load_rows(conn: sqlite3.Connection, table: str, root_key: str) -> dict[str, tuple[StatKey, str]]:
    """Return ``{rel_path: (stat_key, facts_text)}`` for every row stored under *root_key*."""
    cursor = conn.execute(f"SELECT rel_path, mtime_ns, ctime_ns, size, facts FROM {table} WHERE root = ?", (root_key,))
    return {rel_path: ((mtime_ns, ctime_ns, size), facts) for rel_path, mtime_ns, ctime_ns, size, facts in cursor}


def _write(conn: sqlite3.Connection, table: str, root_key: str, upserts: Sequence[_Upsert], deletes: Sequence[_Delete]) -> None:
    """Apply *upserts* and conditional *deletes* in one ``BEGIN IMMEDIATE`` transaction.

    A delete removes a row only if its key still equals the one the caller loaded.

    Raises:
        sqlite3.Error: When the transaction fails; it is rolled back first. A value that cannot be bound
            raises ``UnicodeError`` or ``OverflowError`` instead.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.executemany(
            f"INSERT INTO {table} (root, rel_path, mtime_ns, ctime_ns, size, facts) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(root, rel_path) DO UPDATE SET mtime_ns = excluded.mtime_ns, ctime_ns = excluded.ctime_ns, "
            "size = excluded.size, facts = excluded.facts",
            [(root_key, *upsert) for upsert in upserts],
        )
        conn.executemany(
            f"DELETE FROM {table} WHERE root = ? AND rel_path = ? AND mtime_ns = ? AND ctime_ns = ? AND size = ?",
            [(root_key, *delete) for delete in deletes],
        )
        conn.execute("COMMIT")
    except BaseException:
        app_db.rollback(conn)
        raise


@dataclass
class _WriteBuffer:
    """Collect rows to store and flush them in batches; after a write error it drops everything and stops writing."""

    conn: sqlite3.Connection
    table: str
    root_key: str
    flush_every: int
    upserts: list[_Upsert] = field(default_factory=list)
    deletes: list[_Delete] = field(default_factory=list)
    enabled: bool = True

    def add_upsert(self, row: _Upsert) -> None:
        """Queue *row* and flush once *flush_every* rows are waiting."""
        if not self.enabled:
            return
        self.upserts.append(row)
        if len(self.upserts) >= self.flush_every:
            self.flush()

    def add_deletes(self, rows: Iterable[_Delete]) -> None:
        """Queue conditional deletes for the next flush."""
        if self.enabled:
            self.deletes.extend(rows)

    def flush(self) -> None:
        """Write the queued rows; on an error, warn once and stop writing for the rest of the call."""
        if not self.enabled or not (self.upserts or self.deletes):
            return
        try:
            _write(self.conn, self.table, self.root_key, self.upserts, self.deletes)
        except _DB_ERRORS as exc:
            self.enabled = False
            warnings.warn(f"The gallery index could not be updated ({exc}); results are not saved for this listing.", stacklevel=2)
        finally:
            self.upserts.clear()
            self.deletes.clear()


class GalleryIndex:
    """Cache file facts keyed by (root, rel_path, mtime_ns, ctime_ns, size) in the app database; never raises.

    One instance serializes its reconciliations with a lock so concurrent cold listings read each file once.
    Each operation opens its own short-lived connection.
    """

    def __init__(
        self,
        path_provider: Callable[[], Path],
        *,
        version: int = GALLERY_INDEX_VERSION,
        clock: Callable[[], float] = time.time,
        settle_seconds: float = 2.0,
        flush_every: int = 64,
    ) -> None:
        """Create an index on the database file returned by *path_provider*.

        Args:
            path_provider: Returns the database file; called on every operation.
            version: Table version; the table is named ``gallery_assets_v{version}``.
            clock: Returns the current time in seconds, compared against file mtimes.
            settle_seconds: Minimum age of a file before its facts are stored.
            flush_every: Number of stored reads after which a batch is committed.

        Raises:
            ValueError: If *version* or *flush_every* is below 1.
        """
        if version < 1 or flush_every < 1:
            raise ValueError("version and flush_every must be at least 1")
        self._path_provider = path_provider
        self._clock = clock
        self._settle_seconds = settle_seconds
        self._flush_every = flush_every
        self._table = f"{_TABLE_PREFIX}{version}"
        self._component = app_db.Component(GALLERY_COMPONENT_NAME, version, partial(_migrate_gallery, version=version))
        self._lock = threading.Lock()

    def facts_for(self, root_key: str, files: Sequence[ScannedFile], read: FactsReader, *, prune: bool) -> dict[str, Facts]:
        """Return the facts of every file in *files* keyed by rel_path, reading only new or changed ones.

        When the database cannot be used, every file is read and nothing is stored.

        Args:
            root_key: Resolved output root the files belong to.
            files: Files the walk found under that root.
            read: Reads the facts of one file; an exception it raises propagates.
            prune: Delete the root's rows for files absent from *files*; set it only after an error-free
                walk that found at least one file.

        Returns:
            The facts of each file, keyed by ``rel_path``.
        """
        with self._lock:
            conn, rows = self._open_and_load(root_key)
            if conn is None:
                return {f.rel_path: read(f).facts for f in files}
            with closing(conn):
                return self._reconcile(conn, rows, root_key, files, read, prune)

    def forget(self, root_key: str, rel_paths: Iterable[str]) -> None:
        """Remove the rows of *rel_paths* under *root_key*; errors only warn.

        Takes no scan lock and creates no database file.
        """
        names = [(root_key, rel_path) for rel_path in rel_paths]
        path = self._path_provider()
        if not names or not path.exists():
            return
        conn: app_db.AppConnection | None = None
        try:
            conn = app_db.connect(path, components=(self._component,))
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(f"DELETE FROM {self._table} WHERE root = ? AND rel_path = ?", names)
            conn.execute("COMMIT")
        except _DB_ERRORS as exc:
            self._handle_failure(conn, exc, "could not forget deleted files")
        finally:
            if conn is not None:
                conn.close()

    def _open_and_load(self, root_key: str) -> tuple[app_db.AppConnection | None, dict[str, tuple[StatKey, str]]]:
        """Open a connection and load the root's rows; return ``(None, {})`` when the database is unusable."""
        conn: app_db.AppConnection | None = None
        try:
            conn = app_db.connect(self._path_provider(), components=(self._component,))
            return conn, _load_rows(conn, self._table, root_key)
        except _DB_ERRORS as exc:
            self._handle_failure(conn, exc, "reading file metadata without the cache")
            if conn is not None:
                conn.close()
            return None, {}

    @staticmethod
    def _handle_failure(conn: app_db.AppConnection | None, exc: BaseException, action: str) -> None:
        """Roll back, let ``app_db`` quarantine corruption, and warn about anything else."""
        if conn is not None:
            app_db.rollback(conn)
            if app_db.recover_if_corrupt(conn, exc):
                return
        warnings.warn(f"The gallery index is unavailable ({exc}); {action}.", stacklevel=3)

    def _reconcile(
        self,
        conn: app_db.AppConnection,
        rows: dict[str, tuple[StatKey, str]],
        root_key: str,
        files: Sequence[ScannedFile],
        read: FactsReader,
        prune: bool,
    ) -> dict[str, Facts]:
        """Reuse matching rows, read and store the rest, and prune unseen rows when *prune* is set."""
        buffer = _WriteBuffer(conn, self._table, root_key, self._flush_every)
        result: dict[str, Facts] = {}
        for file in files:
            row = rows.get(file.rel_path)
            cached = _parse_facts(row[1]) if row is not None and row[0] == file.stat_key else None
            if cached is not None:
                result[file.rel_path] = cached
                continue
            fresh = read(file)
            result[file.rel_path] = fresh.facts
            text = _serialize(fresh.facts)
            if text is not None and _should_persist(file, fresh, after=_restat(file.path), now=self._clock(), settle_seconds=self._settle_seconds):
                buffer.add_upsert((file.rel_path, file.mtime_ns, file.ctime_ns, file.size, text))
        if prune:
            buffer.add_deletes((rel_path, *key) for rel_path, (key, _) in rows.items() if rel_path not in result)
        buffer.flush()
        return result
