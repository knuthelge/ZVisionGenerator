"""Keep the user's own per-file choices, such as a star, in the app database, one row per output file."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path

from zvisiongenerator.utils import app_db

_TABLE = "assets_entries"

__all__ = [
    "AssetRecords",
    "AssetRecordsError",
]


class AssetRecordsError(RuntimeError):
    """Report that the asset records could not be read or written."""


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Create the table; a stored version above this one is accepted, since later versions only add columns that this code ignores."""
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_TABLE} (asset_id TEXT PRIMARY KEY, starred INTEGER NOT NULL DEFAULT 0)")


_COMPONENT = app_db.Component("assets", 1, _migrate)


class AssetRecords:
    """Store the user's choices about each file in the ``assets`` component of the app database.

    A row is keyed by *asset_id* alone, the file's output-root-relative POSIX path, so records do not depend on where
    the output folder is. A row is removed only by :meth:`forget`, never by unstarring, so columns a newer version added
    survive. The database path is resolved on every call and each operation uses its own short-lived connection.
    Storage failures raise :class:`AssetRecordsError`.
    """

    def __init__(self, path_provider: Callable[[], Path]) -> None:
        """Create a store that finds its database through *path_provider*."""
        self._path_provider = path_provider

    def _run[T](self, operation: Callable[[app_db.AppConnection], T]) -> T:
        """Run *operation* on a new connection; quarantine a corrupt file and wrap storage errors."""
        return app_db.run(self._path_provider(), (_COMPONENT,), operation, error=AssetRecordsError)

    def starred_ids(self) -> frozenset[str]:
        """Return the IDs of the starred files; creates no database file, so none is nothing starred."""
        if not self._path_provider().exists():
            return frozenset()

        def read(conn: app_db.AppConnection) -> frozenset[str]:
            return frozenset(row[0] for row in conn.execute(f"SELECT asset_id FROM {_TABLE} WHERE starred = 1"))

        return self._run(read)

    def set_starred(self, asset_id: str, starred: bool) -> None:
        """Star or unstar *asset_id*; repeating either is harmless."""

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.execute(
                    f"INSERT INTO {_TABLE} (asset_id, starred) VALUES (?, ?) ON CONFLICT(asset_id) DO UPDATE SET starred = excluded.starred",
                    (asset_id, starred),
                )

        self._run(write)

    def rename(self, moves: Iterable[tuple[str, str]]) -> None:
        """Move the record of each renamed file from ``old_id`` to ``new_id``, replacing any record already at ``new_id``.

        An old ID without a record is skipped. Creates no database file.
        """
        pairs = [(new_id, old_id) for old_id, new_id in moves]
        if not pairs or not self._path_provider().exists():
            return

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.executemany(f"UPDATE OR REPLACE {_TABLE} SET asset_id = ? WHERE asset_id = ?", pairs)

        self._run(write)

    def forget(self, asset_ids: Iterable[str]) -> None:
        """Remove the records of *asset_ids*; creates no database file."""
        rows = [(asset_id,) for asset_id in asset_ids]
        if not rows or not self._path_provider().exists():
            return

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.executemany(f"DELETE FROM {_TABLE} WHERE asset_id = ?", rows)

        self._run(write)
