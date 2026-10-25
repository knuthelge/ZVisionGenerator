"""Tests for the shared app database module."""

from __future__ import annotations

import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.app_db import AppConnection, Component


def _component(name: str, version: int) -> tuple[Component, MagicMock]:
    """Return a component whose migrate records its calls and creates ``<name>_items``."""
    migrate = MagicMock()

    def _run(conn: sqlite3.Connection, from_version: int) -> None:
        migrate(conn, from_version)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {name}_items (value TEXT)")

    return Component(name, version, _run), migrate


def _stored(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT version FROM schema_versions WHERE component = ?", (name,)).fetchone()
    return None if row is None else row[0]


def _error(exc_type: type[sqlite3.DatabaseError], code: int) -> sqlite3.DatabaseError:
    exc = exc_type("failure")
    exc.sqlite_errorcode = code
    return exc


def _write_garbage(path: Path) -> bytes:
    data = b"this is not a sqlite database " * 20
    path.write_bytes(data)
    return data


def _corrupt_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if ".corrupt-" in p.name)


def _replace_file(path: Path, data: bytes) -> None:
    """Swap *path* for a new file with a different inode."""
    scratch = path.with_name("scratch.tmp")
    scratch.write_bytes(data)
    os.replace(scratch, path)


def test_a1_new_database_has_standard_settings(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    with closing(app_db.connect(path)) as conn:
        assert isinstance(conn, AppConnection)
        assert conn.db_path == path
        assert conn.db_identity is None
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'schema_versions'").fetchone() is not None
    st = path.stat()
    with closing(app_db.connect(path)) as conn:
        assert conn.db_identity == (st.st_dev, st.st_ino)


def test_simultaneous_first_connects_all_succeed_and_leave_wal(tmp_path: Path) -> None:
    workers = 20
    for round_number in range(10):
        path = tmp_path / f"ziv-{round_number}.sqlite3"
        barrier = threading.Barrier(workers)

        def open_and_close(_: int, path: Path = path, barrier: threading.Barrier = barrier) -> None:
            barrier.wait()
            with closing(app_db.connect(path)):
                pass

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(open_and_close, i) for i in range(workers)]:
                future.result()

        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def _locked_error() -> sqlite3.OperationalError:
    """Return an error shaped like SQLite's "database is locked"."""
    exc = sqlite3.OperationalError("database is locked")
    exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return exc


def _locked_on_wal_switch(monkeypatch: pytest.MonkeyPatch, failures: int | None) -> list[str]:
    """Make the WAL switch raise "database is locked" *failures* times (always when None); return the statements seen."""
    real_open = app_db._open_connection
    seen: list[str] = []

    def open_connection(path: Path) -> AppConnection:
        conn = real_open(path)
        real_execute = conn.execute

        def execute(sql: str, *params: object) -> sqlite3.Cursor:
            if sql == "PRAGMA journal_mode=WAL":
                seen.append(sql)
                if failures is None or len(seen) <= failures:
                    raise _locked_error()
            return real_execute(sql, *params)

        conn.execute = execute
        return conn

    monkeypatch.setattr(app_db, "_open_connection", open_connection)
    return seen


def test_wal_switch_is_retried_while_another_connection_switches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _locked_on_wal_switch(monkeypatch, failures=3)

    with closing(app_db.connect(tmp_path / "ziv.sqlite3")) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    assert len(seen) == 4


def test_wal_switch_gives_up_after_the_busy_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_db, "BUSY_TIMEOUT_SECONDS", 0.05)
    _locked_on_wal_switch(monkeypatch, failures=None)

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        app_db.connect(tmp_path / "ziv.sqlite3")


def test_wal_switch_that_leaves_another_mode_does_not_spin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = app_db._open_connection
    seen: list[str] = []

    def open_connection(path: Path) -> AppConnection:
        conn = real_open(path)
        real_execute = conn.execute

        def execute(sql: str, *params: object) -> sqlite3.Cursor:
            if sql == "PRAGMA journal_mode=WAL":
                seen.append(sql)
                return real_execute("PRAGMA journal_mode=DELETE")
            return real_execute(sql, *params)

        conn.execute = execute
        return conn

    monkeypatch.setattr(app_db, "_open_connection", open_connection)
    monkeypatch.setattr(app_db, "BUSY_TIMEOUT_SECONDS", 0.5)

    with closing(app_db.connect(tmp_path / "ziv.sqlite3")) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT 1").fetchone()[0] == 1

    assert len(seen) == 1


def test_wal_file_skips_the_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ziv.sqlite3"
    with closing(app_db.connect(path)):
        pass
    seen = _locked_on_wal_switch(monkeypatch, failures=None)

    with closing(app_db.connect(path)):
        pass

    assert seen == []


def test_a2_path_follows_patched_data_dir_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    monkeypatch.setattr(app_db, "get_ziv_data_dir", lambda: first)
    assert app_db.app_db_path() == first / app_db.APP_DB_FILENAME
    monkeypatch.setattr(app_db, "get_ziv_data_dir", lambda: second)
    assert app_db.app_db_path() == second / "ziv.sqlite3"
    with closing(app_db.connect()) as conn:
        assert conn.db_path == second / "ziv.sqlite3"
    assert (second / "ziv.sqlite3").exists()


def test_a3_first_connect_migrates_once(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    component, migrate = _component("alpha", 1)
    with closing(app_db.connect(path, components=(component,))) as conn:
        assert _stored(conn, "alpha") == 1
    migrate.assert_called_once()
    assert migrate.call_args.args[1] == 0
    with closing(app_db.connect(path, components=(component,))) as conn:
        assert _stored(conn, "alpha") == 1
    migrate.assert_called_once()


def test_a4_version_bump_migrates_from_stored(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    v1, _ = _component("alpha", 1)
    v2, migrate_v2 = _component("alpha", 2)
    app_db.connect(path, components=(v1,)).close()
    with closing(app_db.connect(path, components=(v2,))) as conn:
        assert _stored(conn, "alpha") == 2
    migrate_v2.assert_called_once()
    assert migrate_v2.call_args.args[1] == 1


def test_a5_components_are_independent(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    x1, _ = _component("x", 1)
    y1, migrate_y = _component("y", 1)
    with closing(app_db.connect(path, components=(x1, y1))) as conn:
        conn.execute("INSERT INTO y_items (value) VALUES ('keep')")
    x2, migrate_x2 = _component("x", 2)
    with closing(app_db.connect(path, components=(x2, y1))) as conn:
        assert _stored(conn, "x") == 2
        assert _stored(conn, "y") == 1
        assert conn.execute("SELECT value FROM y_items").fetchall() == [("keep",)]
    migrate_x2.assert_called_once()
    migrate_y.assert_called_once()


def test_a6_failed_migration_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    v1, _ = _component("alpha", 1)
    app_db.connect(path, components=(v1,)).close()

    def _fail(conn: sqlite3.Connection, from_version: int) -> None:
        conn.execute("CREATE TABLE alpha_extra (value TEXT)")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        app_db.connect(path, components=(Component("alpha", 2, _fail),))
    with closing(app_db.connect(path)) as conn:
        assert _stored(conn, "alpha") == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'alpha_extra'").fetchone() is None
        assert not conn.in_transaction


def test_a7_garbage_file_is_quarantined_and_replaced(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    data = _write_garbage(path)
    component, migrate = _component("alpha", 1)
    with pytest.warns(UserWarning, match="damaged"):
        conn = app_db.connect(path, components=(component,))
    with closing(conn):
        assert _stored(conn, "alpha") == 1
    migrate.assert_called_once()
    moved = _corrupt_files(tmp_path)
    assert len(moved) == 1
    assert moved[0].read_bytes() == data


def test_a7_sidecars_move_before_the_main_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ziv.sqlite3"
    data = _write_garbage(path)
    (tmp_path / "ziv.sqlite3-wal").write_bytes(b"wal")
    (tmp_path / "ziv.sqlite3-shm").write_bytes(b"shm")
    moved: list[str] = []
    original = Path.replace

    def _record(self: Path, target: str | os.PathLike[str]) -> Path:
        moved.append(self.name)
        return original(self, target)

    monkeypatch.setattr(Path, "replace", _record)
    st = os.stat(path)
    target = app_db.quarantine(path, (st.st_dev, st.st_ino))
    assert target is not None
    assert moved == ["ziv.sqlite3-wal", "ziv.sqlite3-shm", "ziv.sqlite3"]
    assert target.read_bytes() == data
    assert Path(f"{target}-wal").read_bytes() == b"wal"
    assert Path(f"{target}-shm").read_bytes() == b"shm"
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith("ziv.sqlite3.corrupt-")] == []


def test_a8_directory_path_is_not_renamed(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    path.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        app_db.connect(path)
    assert path.is_dir()
    assert [p.name for p in tmp_path.iterdir()] == ["ziv.sqlite3"]


def test_a9_is_corruption_classification() -> None:
    assert app_db.is_corruption(_error(sqlite3.DatabaseError, sqlite3.SQLITE_CORRUPT))
    assert app_db.is_corruption(_error(sqlite3.DatabaseError, sqlite3.SQLITE_NOTADB))
    assert app_db.is_corruption(_error(sqlite3.DatabaseError, sqlite3.SQLITE_CORRUPT_VTAB))
    assert not app_db.is_corruption(_error(sqlite3.OperationalError, sqlite3.SQLITE_BUSY))
    assert not app_db.is_corruption(_error(sqlite3.OperationalError, sqlite3.SQLITE_LOCKED))
    assert not app_db.is_corruption(_error(sqlite3.OperationalError, sqlite3.SQLITE_CANTOPEN))
    assert not app_db.is_corruption(_error(sqlite3.OperationalError, sqlite3.SQLITE_READONLY))
    assert not app_db.is_corruption(_error(sqlite3.OperationalError, sqlite3.SQLITE_IOERR))
    assert not app_db.is_corruption(sqlite3.DatabaseError("no code"))
    assert not app_db.is_corruption(OSError("disk"))


def test_a9_real_garbage_file_is_classified_as_corruption(tmp_path: Path) -> None:
    path = tmp_path / "bad.sqlite3"
    _write_garbage(path)
    with closing(sqlite3.connect(path)) as conn, pytest.raises(sqlite3.DatabaseError) as info:
        conn.execute("PRAGMA journal_mode=WAL")
    assert app_db.is_corruption(info.value)


def test_a10_same_second_quarantines_get_distinct_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_db.time, "strftime", lambda _fmt: "20260101-000000")
    path = tmp_path / "ziv.sqlite3"
    targets = []
    for index in range(3):
        path.write_bytes(f"file {index}".encode())
        (tmp_path / "ziv.sqlite3-wal").write_bytes(b"wal")
        st = os.stat(path)
        target = app_db.quarantine(path, (st.st_dev, st.st_ino))
        assert target is not None
        targets.append(target)
    assert len(set(targets)) == 3
    assert targets[0].name == "ziv.sqlite3.corrupt-20260101-000000"
    assert targets[1].name == "ziv.sqlite3.corrupt-20260101-000000-1"
    assert [t.read_bytes() for t in targets] == [b"file 0", b"file 1", b"file 2"]
    assert not path.exists()


def test_a11_replaced_file_is_not_moved(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    path.write_bytes(b"old")
    old = os.stat(path)
    _replace_file(path, b"new")
    assert app_db.quarantine(path, (old.st_dev, old.st_ino)) is None
    assert path.read_bytes() == b"new"
    assert _corrupt_files(tmp_path) == []


def test_a11_missing_file_or_unknown_identity_is_not_moved(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    assert app_db.quarantine(path, (1, 2)) is None
    path.write_bytes(b"data")
    assert app_db.quarantine(path, None) is None
    assert path.read_bytes() == b"data"


def test_a12_higher_stored_version_is_never_lowered(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    with closing(app_db.connect(path)) as conn:
        conn.execute("INSERT INTO schema_versions (component, version) VALUES ('alpha', 3)")
    component, migrate = _component("alpha", 2)
    with closing(app_db.connect(path, components=(component,))) as conn:
        assert _stored(conn, "alpha") == 3
    migrate.assert_called_once()
    assert migrate.call_args.args[1] == 3


def test_a13_recover_ignores_non_corruption(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    with closing(app_db.connect(path)) as conn:
        assert app_db.recover_if_corrupt(conn, _error(sqlite3.OperationalError, sqlite3.SQLITE_BUSY)) is False
        assert app_db.recover_if_corrupt(conn, OSError("disk")) is False
        assert conn.execute("SELECT 1").fetchone() == (1,)
    assert path.exists()
    assert _corrupt_files(tmp_path) == []


def test_a13_recover_quarantines_corruption(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    app_db.connect(path).close()
    conn = app_db.connect(path)
    with pytest.warns(UserWarning, match="damaged"):
        assert app_db.recover_if_corrupt(conn, _error(sqlite3.DatabaseError, sqlite3.SQLITE_CORRUPT)) is True
    assert not path.exists()
    assert len(_corrupt_files(tmp_path)) >= 1
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_a13_recover_warns_when_file_was_replaced(tmp_path: Path) -> None:
    path = tmp_path / "ziv.sqlite3"
    app_db.connect(path).close()
    conn = app_db.connect(path)
    _replace_file(path, b"new")
    with pytest.warns(UserWarning, match="nothing was moved"):
        assert app_db.recover_if_corrupt(conn, _error(sqlite3.DatabaseError, sqlite3.SQLITE_NOTADB)) is True
    assert path.read_bytes() == b"new"
    assert _corrupt_files(tmp_path) == []


def test_a13_recover_warns_when_rename_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ziv.sqlite3"
    app_db.connect(path).close()
    conn = app_db.connect(path)

    def _fail(self: Path, target: object) -> Path:
        raise PermissionError("busy")

    monkeypatch.setattr(Path, "replace", _fail)
    with pytest.warns(UserWarning, match="could not be moved"):
        assert app_db.recover_if_corrupt(conn, _error(sqlite3.DatabaseError, sqlite3.SQLITE_CORRUPT)) is True
    assert path.exists()


def test_rollback_ends_an_open_transaction_and_ignores_none(tmp_path: Path) -> None:
    with closing(app_db.connect(tmp_path / "ziv.sqlite3")) as conn:
        app_db.rollback(conn)
        conn.execute("BEGIN IMMEDIATE")
        assert conn.in_transaction
        app_db.rollback(conn)
        assert not conn.in_transaction


def test_transaction_commits_on_success_and_rolls_back_then_reraises_on_failure(tmp_path: Path) -> None:
    with closing(app_db.connect(tmp_path / "ziv.sqlite3")) as conn:
        conn.execute("CREATE TABLE items (value TEXT)")

        with app_db.transaction(conn):
            assert conn.in_transaction
            conn.execute("INSERT INTO items VALUES ('kept')")
        assert not conn.in_transaction

        with pytest.raises(RuntimeError, match="boom"), app_db.transaction(conn):
            conn.execute("INSERT INTO items VALUES ('lost')")
            raise RuntimeError("boom")

        assert not conn.in_transaction
        assert conn.execute("SELECT value FROM items").fetchall() == [("kept",)]
