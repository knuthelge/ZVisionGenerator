"""Keep the prompts the Web UI accepted in the app database, newest first."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils import app_db

_LIMIT = 50

_VERSION = 1
_TABLE = f"recent_prompts_entries_v{_VERSION}"
_COLUMNS = ("format", "prompt", "scene", "negative_prompt", "mode", "used_at")

type RecentPrompt = dict[str, Any]

__all__ = [
    "RecentPrompts",
    "RecentPromptsError",
    "recent_prompt_entry",
]


def recent_prompt_entry(
    *,
    prompt_source: str | None,
    prompt: str | None,
    json_prompt: str | None,
    scene: str | None,
    negative_prompt: str | None,
    mode: str,
    used_at: float,
) -> RecentPrompt | None:
    """Build the entry for an accepted submit, or ``None`` when it does not belong in the recents.

    A prompt-file run and a submit whose text and scene are both blank are not remembered. A JSON caption
    (image mode only) is stored as sent with ``format`` ``"json"``; a plain submit keeps the scene for
    video and the negative prompt for image, and blanks the other.

    Args:
        prompt_source: ``"file"`` for a prompt-file run; anything else is an inline prompt.
        prompt: The submitted prompt (the action, for video).
        json_prompt: The submitted JSON caption text.
        scene: The submitted video scene.
        negative_prompt: The submitted negative prompt.
        mode: ``"image"`` or ``"video"``.
        used_at: Server time of the submit in epoch seconds.

    Returns:
        The entry with ``format``, ``prompt``, ``scene``, ``negative_prompt``, ``mode`` and ``used_at``.
    """
    if prompt_source == "file":
        return None
    negative = (negative_prompt or "").strip()
    caption = (json_prompt or "").strip()
    if mode == "image" and caption:
        return {"format": "json", "prompt": caption, "scene": "", "negative_prompt": negative, "mode": mode, "used_at": used_at}
    text = (prompt or "").strip()
    stored_scene = (scene or "").strip() if mode == "video" else ""
    if not text and not stored_scene:
        return None
    return {
        "format": "text",
        "prompt": text,
        "scene": stored_scene,
        "negative_prompt": negative if mode == "image" else "",
        "mode": mode,
        "used_at": used_at,
    }


def _matches_query(entry: Mapping[str, Any], query: str) -> bool:
    """Return whether *entry*'s prompt or scene contains *query*, ignoring case; a blank query matches everything."""
    needle = query.strip().casefold()
    if not needle:
        return True
    return needle in str(entry.get("prompt", "")).casefold() or needle in str(entry.get("scene", "")).casefold()


class RecentPromptsError(RuntimeError):
    """Report that the recent prompts could not be read or written."""


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Create the recent prompts table, refusing a database written by a newer ziv."""
    if from_version > _VERSION:
        raise RecentPromptsError("it was written by a newer version of ziv; update ziv to read it")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
        "seq INTEGER PRIMARY KEY, format TEXT NOT NULL, prompt TEXT NOT NULL, scene TEXT NOT NULL, "
        "negative_prompt TEXT NOT NULL, mode TEXT NOT NULL, used_at REAL NOT NULL, "
        "UNIQUE (mode, format, prompt, scene, negative_prompt))"
    )


_COMPONENT = app_db.Component("recent_prompts", _VERSION, _migrate)


class RecentPrompts:
    """Store accepted prompts as rows of the ``recent_prompts`` component of the app database.

    One row exists per distinct ``(mode, format, prompt, scene, negative_prompt)``; recording it again makes it
    the newest. Searching and clearing are scoped to one mode, so an image prompt and a video scene-and-action
    pair are never mixed. Nothing is capped or trimmed. The database path is resolved on every call and each operation uses
    its own short-lived connection, so servers sharing one data folder share the prompts. Storage failures, and
    a schema written by a newer ziv, raise :class:`RecentPromptsError`.
    """

    def __init__(self, path_provider: Callable[[], Path]) -> None:
        """Create a store that finds its database through *path_provider*."""
        self._path_provider = path_provider

    def _run[T](self, operation: Callable[[app_db.AppConnection], T]) -> T:
        """Run *operation* on a new connection; quarantine a corrupt file and wrap storage errors."""
        return app_db.run(self._path_provider(), (_COMPONENT,), operation, error=RecentPromptsError)

    def record(self, entry: Mapping[str, Any]) -> None:
        """Store *entry* as the newest row, replacing an existing row with the same mode, format, prompt, scene and negative prompt."""
        values = tuple(entry[column] for column in _COLUMNS)

        def write(conn: app_db.AppConnection) -> None:
            with app_db.transaction(conn):
                conn.execute(f"INSERT OR REPLACE INTO {_TABLE} ({', '.join(_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)", values)

        self._run(write)

    def search(self, query: str, mode: str) -> list[RecentPrompt]:
        """Return up to 50 newest entries of *mode* whose prompt or scene contains *query*, newest first."""

        def read(conn: app_db.AppConnection) -> list[RecentPrompt]:
            found: list[RecentPrompt] = []
            for row in conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE mode = ? ORDER BY seq DESC", (mode,)):
                entry = dict(zip(_COLUMNS, row, strict=True))
                if _matches_query(entry, query):
                    found.append(entry)
                    if len(found) >= _LIMIT:
                        break
            return found

        return self._run(read)

    def clear(self, mode: str) -> int:
        """Remove every entry of *mode* and return how many there were."""

        def remove(conn: app_db.AppConnection) -> int:
            with app_db.transaction(conn):
                return conn.execute(f"DELETE FROM {_TABLE} WHERE mode = ?", (mode,)).rowcount

        return self._run(remove)
