"""Open the ZIV app database with the standard settings and keep each component's tables at its schema version.

One SQLite file, ``ziv.sqlite3`` in the ZIV data directory, is shared by every feature that persists
data. Features follow these rules:

* Open one short-lived connection per operation with
  ``with closing(app_db.connect(components=(MY_COMPONENT,))) as conn:``.
* Never keep a module-global connection and never share a connection across threads.
* Keep ``BEGIN IMMEDIATE`` ... ``COMMIT`` writes short, with no slow I/O inside them.
* Locks belong to the feature that needs them; this module holds none.
* Name tables ``<component>_*`` and never read or write another component's tables.
* When an operation fails, pass the exception to :func:`recover_if_corrupt` before deciding what to do.

``schema_versions`` records, per component, the highest version that has run on the file. A component's
``migrate`` callable receives the stored version and decides its own upgrade policy, including what to do
when that version is higher than its own.
"""

from __future__ import annotations

import os
import sqlite3
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.paths import get_ziv_data_dir

APP_DB_FILENAME = "ziv.sqlite3"
BUSY_TIMEOUT_SECONDS = 5.0

_CORRUPTION_CODES = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})
_SIDECAR_SUFFIXES = ("-wal", "-shm")

type FileIdentity = tuple[int, int]

__all__ = [
    "APP_DB_FILENAME",
    "BUSY_TIMEOUT_SECONDS",
    "AppConnection",
    "Component",
    "app_db_path",
    "connect",
    "is_corruption",
    "quarantine",
    "recover_if_corrupt",
    "rollback",
]


class AppConnection(sqlite3.Connection):
    """Remember which file a connection opened, so a corrupt file is only moved aside if it is still that file."""

    db_path: Path
    db_identity: FileIdentity | None


@dataclass(frozen=True)
class Component:
    """Name a feature's tables, their schema version and the step that brings them to it.

    Attributes:
        name: Table prefix and ``schema_versions`` key, for example ``"gallery"`` for ``gallery_*``.
        version: Schema version of the component's tables, at least 1.
        migrate: Called as ``migrate(conn, from_version)`` inside ``BEGIN IMMEDIATE`` whenever the stored
            version differs from *version*. ``from_version`` is 0 for a new component and may be higher
            than *version*; raising rolls the migration back.
    """

    name: str
    version: int
    migrate: Callable[[sqlite3.Connection, int], None]


def app_db_path() -> Path:
    """Return the app database file inside the ZIV data directory, resolved at call time."""
    return get_ziv_data_dir() / APP_DB_FILENAME


def _identity(path: Path) -> FileIdentity | None:
    """Return ``(st_dev, st_ino)`` of the file at *path*, or ``None`` when it does not exist."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino)


def _open_connection(path: Path) -> AppConnection:
    """Open a connection to *path* in autocommit mode, tagged with the file identity seen beforehand."""
    identity = _identity(path)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None, factory=AppConnection)
    conn.db_path = path
    conn.db_identity = identity
    return conn


def _stored_version(conn: sqlite3.Connection, name: str) -> int:
    """Return the highest version recorded for component *name*, or 0 when it has no row."""
    row = conn.execute("SELECT version FROM schema_versions WHERE component = ?", (name,)).fetchone()
    return 0 if row is None else row[0]


def rollback(conn: sqlite3.Connection) -> None:
    """Roll back an open transaction, ignoring a failure to do so.

    Args:
        conn: Connection that may be inside a transaction.
    """
    if not conn.in_transaction:
        return
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _bring_to_version(conn: sqlite3.Connection, component: Component) -> None:
    """Run *component*'s migration when its stored version differs, then record the highest version seen."""
    if _stored_version(conn, component.name) == component.version:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        stored = _stored_version(conn, component.name)
        if stored != component.version:
            component.migrate(conn, stored)
            conn.execute(
                "INSERT INTO schema_versions (component, version) VALUES (?, ?) ON CONFLICT(component) DO UPDATE SET version = excluded.version",
                (component.name, max(stored, component.version)),
            )
        conn.execute("COMMIT")
    except BaseException:
        rollback(conn)
        raise


def _configure(conn: sqlite3.Connection, components: Sequence[Component]) -> None:
    """Apply the standard pragmas, create ``schema_versions`` and migrate each component."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE IF NOT EXISTS schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL) WITHOUT ROWID")
    for component in components:
        _bring_to_version(conn, component)


def _open_configured(path: Path, components: Sequence[Component]) -> AppConnection:
    """Open and configure a connection, closing it again if configuring fails."""
    conn = _open_connection(path)
    try:
        _configure(conn, components)
    except BaseException:
        conn.close()
        raise
    return conn


def connect(path: Path | None = None, *, components: Sequence[Component] = ()) -> AppConnection:
    """Open a new connection with the standard pragmas and bring *components* to their versions.

    A corrupt file is moved aside once through :func:`recover_if_corrupt` and the open is retried against a
    fresh file. Any other error closes the connection and propagates. The caller closes the connection.

    Args:
        path: Database file; defaults to :func:`app_db_path`.
        components: Components whose tables must be at their versions before the connection is returned.

    Returns:
        A connection in autocommit mode with WAL, ``synchronous=NORMAL``, foreign keys and a 5 s busy timeout.

    Raises:
        sqlite3.Error: When the file cannot be opened, is locked, or a migration fails.
    """
    path = path or app_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_connection(path)
    try:
        _configure(conn, components)
    except BaseException as exc:
        if not recover_if_corrupt(conn, exc):
            conn.close()
            raise
        return _open_configured(path, components)
    return conn


def is_corruption(exc: BaseException) -> bool:
    """Return whether *exc* is SQLite reporting a damaged or non-database file (SQLITE_CORRUPT, SQLITE_NOTADB)."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    return code is not None and (code & 0xFF) in _CORRUPTION_CODES


def _sidecar(path: Path, suffix: str) -> Path:
    """Return the sibling of *path* with *suffix* appended to its name."""
    return path.with_name(path.name + suffix)


def _free_quarantine_target(path: Path) -> Path:
    """Return a ``.corrupt-<stamp>[-n]`` name next to *path* that neither it nor its sidecars occupy."""
    base = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    target = base
    counter = 0
    while target.exists() or any(_sidecar(target, suffix).exists() for suffix in _SIDECAR_SUFFIXES):
        counter += 1
        target = base.with_name(f"{base.name}-{counter}")
    return target


def quarantine(path: Path, identity: FileIdentity | None) -> Path | None:
    """Move a damaged database aside if the file at *path* still has *identity*; return the new path, or None if nothing moved.

    The ``-wal`` and ``-shm`` files move first and the main file last, so a fresh database is never paired
    with an old write-ahead log. Nothing is ever deleted.

    Args:
        path: The database file.
        identity: ``(st_dev, st_ino)`` recorded before the connection opened, or ``None`` if the file did not exist.

    Returns:
        The new location of the main file, or ``None`` when the file is gone or was replaced by another process.

    Raises:
        OSError: When a rename fails.
    """
    if identity is None or _identity(path) != identity:
        return None
    target = _free_quarantine_target(path)
    for suffix in _SIDECAR_SUFFIXES:
        sidecar = _sidecar(path, suffix)
        if sidecar.exists():
            sidecar.replace(_sidecar(target, suffix))
    path.replace(target)
    return target


def recover_if_corrupt(conn: AppConnection, exc: BaseException) -> bool:
    """Close *conn* and quarantine its file when *exc* is corruption; return whether it was corruption."""
    if not is_corruption(exc):
        return False
    conn.close()
    try:
        moved = quarantine(conn.db_path, conn.db_identity)
    except OSError as rename_error:
        outcome = f"it could not be moved aside ({rename_error})"
    else:
        outcome = f"it was moved to {moved}" if moved is not None else "nothing was moved because the file had changed"
    warnings.warn(f"The app database {conn.db_path} is damaged ({exc}); {outcome}.", stacklevel=2)
    return True
