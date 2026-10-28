"""Remember where each downloaded LoRA came from and what its trigger is, in the app database.

The ``lora_sources`` component keeps one row per LoRA file in ``lora_sources_v1``: the repository, commit, file and
SHA-256 it was downloaded as, the curated-list ``id`` when it came from there, the trigger the source named and the
trigger the user typed. A file downloaded from Hugging Face also keeps the facts its card showed (author, download count,
licence) and a small preview image (``lora_images_v1``), so the Installed tab describes it without a connection. The
table is not a cache and has nothing to do with the LoRA family rules, so a rules version bump never touches it. A row
is trusted only while the file's size still equals the stored size, so a file that was replaced by hand is not described
by the old row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

from zvisiongenerator.utils import app_db

__all__ = ["SOURCES", "LoraSourceRow", "LoraSources", "LoraSourcesError"]

_VERSION = 1
_TABLE = f"lora_sources_v{_VERSION}"
_IMAGES = f"lora_images_v{_VERSION}"
_COLUMNS = ("name", "size_bytes", "curated_id", "repo", "revision", "file", "sha256", "auto_trigger", "trigger_source", "user_trigger", "downloaded_at", "author", "downloads", "licence", "has_image")


class LoraSourcesError(RuntimeError):
    """Report that the LoRA sources could not be read or written."""


@dataclass(frozen=True)
class LoraSourceRow:
    """What is known about one installed LoRA file.

    Attributes:
        name: The LoRA's local name (the file stem).
        size_bytes: The file's size when the row was written.
        curated_id: Id of the curated entry it was downloaded for, or None.
        repo: Hugging Face repository it came from, or None for a row that only holds a typed trigger.
        revision: Commit it was downloaded at, or None.
        file: Path of the file in the repository, or None.
        sha256: SHA-256 of the installed file, or None.
        auto_trigger: The trigger its source named (the curated entry's, or the model card's first phrase), or None.
        trigger_source: ``"curated"`` or ``"card"`` for *auto_trigger*, or None.
        user_trigger: The trigger the user typed, or None.
        downloaded_at: Epoch seconds the row was written.
        author: The Hugging Face user or organisation that published the repository, or None.
        downloads: The repository's download count when the file was downloaded, or None.
        licence: The repository's licence id when the file was downloaded, or None.
        has_image: Whether a preview image is stored for the file (see :meth:`LoraSources.image`).
    """

    name: str
    size_bytes: int
    curated_id: str | None = None
    repo: str | None = None
    revision: str | None = None
    file: str | None = None
    sha256: str | None = None
    auto_trigger: str | None = None
    trigger_source: str | None = None
    user_trigger: str | None = None
    downloaded_at: float = 0.0
    author: str | None = None
    downloads: int | None = None
    licence: str | None = None
    has_image: bool = False


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Create the tables, refusing a database written by a newer ziv."""
    if from_version > _VERSION:
        raise LoraSourcesError("it was written by a newer version of ziv; update ziv to read it")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
        "name TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, curated_id TEXT NULL, repo TEXT NULL, revision TEXT NULL, "
        "file TEXT NULL, sha256 TEXT NULL, auto_trigger TEXT NULL, trigger_source TEXT NULL, user_trigger TEXT NULL, "
        "downloaded_at REAL NOT NULL, author TEXT NULL, downloads INTEGER NULL, licence TEXT NULL, has_image INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID"
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_IMAGES} (name TEXT PRIMARY KEY, content_type TEXT NOT NULL, image BLOB NOT NULL) WITHOUT ROWID")


_COMPONENT = app_db.Component("lora_sources", _VERSION, _migrate)


def _row(values: Sequence[object]) -> LoraSourceRow:
    """Return the row of one ``SELECT`` of ``_COLUMNS``."""
    *plain, has_image = values
    return LoraSourceRow(*plain, has_image=bool(has_image))


class LoraSources:
    """Store LoRA source rows in the ``lora_sources`` component of the app database.

    The database path is resolved on every call and each operation uses its own short-lived connection. Storage
    failures, and a schema written by a newer ziv, raise :class:`LoraSourcesError`.
    """

    def __init__(self, path_provider: Callable[[], Path], *, clock: Callable[[], float] = time.time) -> None:
        """Create a store that finds its database through *path_provider*."""
        self._path_provider = path_provider
        self._clock = clock

    def _run[T](self, operation: Callable[[app_db.AppConnection], T]) -> T:
        """Run *operation* on a new connection; quarantine a corrupt file and wrap storage errors."""
        return app_db.run(self._path_provider(), (_COMPONENT,), operation, error=LoraSourcesError)

    def record(self, row: LoraSourceRow, image: tuple[str, bytes] | None = None) -> None:
        """Store *row* for its name, replacing the source facts of an existing row and keeping the trigger the user typed there.

        Args:
            row: What is known about the file.
            image: ``(content type, bytes)`` of its preview image, stored with it (and the row marked as having one), or None for no image.
        """
        values = (
            row.name,
            row.size_bytes,
            row.curated_id,
            row.repo,
            row.revision,
            row.file,
            row.sha256,
            row.auto_trigger,
            row.trigger_source,
            row.user_trigger,
            row.downloaded_at or self._clock(),
            row.author,
            row.downloads,
            row.licence,
            int(image is not None),
        )

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.execute(
                    f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))}) "
                    "ON CONFLICT(name) DO UPDATE SET size_bytes = excluded.size_bytes, curated_id = excluded.curated_id, repo = excluded.repo, "
                    "revision = excluded.revision, file = excluded.file, sha256 = excluded.sha256, auto_trigger = excluded.auto_trigger, "
                    f"trigger_source = excluded.trigger_source, downloaded_at = excluded.downloaded_at, user_trigger = COALESCE(excluded.user_trigger, {_TABLE}.user_trigger), "
                    "author = excluded.author, downloads = excluded.downloads, licence = excluded.licence, has_image = excluded.has_image",
                    values,
                )
                if image is None:
                    conn.execute(f"DELETE FROM {_IMAGES} WHERE name = ?", (row.name,))
                else:
                    conn.execute(f"INSERT OR REPLACE INTO {_IMAGES} (name, content_type, image) VALUES (?, ?, ?)", (row.name, image[0], image[1]))

        self._run(write)

    def set_image(self, name: str, size_bytes: int, image: tuple[str, bytes]) -> bool:
        """Store the preview *image* for the row of *name* if that row still describes a file of *size_bytes*; return whether it was stored.

        A row that was forgotten or rewritten for another file since the download is left alone.
        """

        def write(conn: app_db.AppConnection) -> bool:
            with app_db.transaction(conn):
                updated = conn.execute(f"UPDATE {_TABLE} SET has_image = 1 WHERE name = ? AND size_bytes = ?", (name, size_bytes)).rowcount
                if updated == 0:
                    return False
                conn.execute(f"INSERT OR REPLACE INTO {_IMAGES} (name, content_type, image) VALUES (?, ?, ?)", (name, image[0], image[1]))
                return True

        return self._run(write)

    def rows(self, names: Iterable[str]) -> dict[str, LoraSourceRow]:
        """Return the stored row of each of *names* that has one."""
        wanted = list(dict.fromkeys(names))
        if not wanted:
            return {}

        def read(conn: app_db.AppConnection) -> dict[str, LoraSourceRow]:
            found: dict[str, LoraSourceRow] = {}
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                marks = ", ".join("?" * len(chunk))
                for values in conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE name IN ({marks})", chunk):
                    row = _row(values)
                    found[row.name] = row
            return found

        return self._run(read)

    def for_repo(self, repo: str) -> list[LoraSourceRow]:
        """Return the rows of the LoRAs downloaded from Hugging Face repository *repo*."""

        def read(conn: app_db.AppConnection) -> list[LoraSourceRow]:
            return [_row(values) for values in conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE repo = ? ORDER BY name", (repo,))]

        return self._run(read)

    def image(self, name: str) -> tuple[str, bytes] | None:
        """Return ``(content type, bytes)`` of the preview image stored for *name*, or None."""

        def read(conn: app_db.AppConnection) -> tuple[str, bytes] | None:
            found = conn.execute(f"SELECT content_type, image FROM {_IMAGES} WHERE name = ?", (name,)).fetchone()
            return None if found is None else (found[0], bytes(found[1]))

        return self._run(read)

    def set_user_trigger(self, name: str, size_bytes: int, trigger: str | None) -> None:
        """Store the trigger the user typed for *name* (None clears it); a name with no row gets one that holds only the trigger.

        A row that holds nothing after the change is removed.
        """

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                updated = conn.execute(f"UPDATE {_TABLE} SET user_trigger = ?, size_bytes = ? WHERE name = ?", (trigger, size_bytes, name)).rowcount
                if updated == 0 and trigger is not None:
                    conn.execute(f"INSERT INTO {_TABLE} (name, size_bytes, user_trigger, downloaded_at) VALUES (?, ?, ?, ?)", (name, size_bytes, trigger, self._clock()))
                conn.execute(f"DELETE FROM {_TABLE} WHERE name = ? AND repo IS NULL AND user_trigger IS NULL", (name,))

        self._run(write)

    def forget(self, name: str) -> None:
        """Remove the row of *name*, if any."""

        def remove(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.execute(f"DELETE FROM {_TABLE} WHERE name = ?", (name,))
                conn.execute(f"DELETE FROM {_IMAGES} WHERE name = ?", (name,))

        self._run(remove)


# The store of this ZIV data folder, shared by the routes and the listings of installed LoRAs.
SOURCES = LoraSources(app_db.app_db_path)
