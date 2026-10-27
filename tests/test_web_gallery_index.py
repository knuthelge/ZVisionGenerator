"""Tests for the persistent gallery index."""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import threading
import time
import warnings
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest

from zvisiongenerator.utils import app_db
from zvisiongenerator.web import gallery_index
from zvisiongenerator.web.gallery_index import GALLERY_INDEX_VERSION, FactsRead, GalleryIndex, ScannedFile

NOW = 1_000_000.0


class _Reader:
    """Count reads per file and return facts naming the file."""

    def __init__(self, *, complete: bool = True) -> None:
        self.calls: Counter[str] = Counter()
        self.complete = complete

    def __call__(self, file: ScannedFile) -> FactsRead:
        self.calls[file.rel_path] += 1
        return FactsRead({"config": {"name": file.rel_path}, "image": {}}, self.complete)

    @property
    def total(self) -> int:
        return sum(self.calls.values())


def _make_files(root: Path, names: list[str], *, age: float = 60.0) -> None:
    """Create files under *root* whose mtime is *age* seconds before ``NOW``."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = root / name
        path.write_bytes(name.encode())
        os.utime(path, (NOW - age, NOW - age))


def _scan(root: Path) -> list[ScannedFile]:
    """Stat every file under *root* the way the gallery walk does."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            files.append(ScannedFile(path.relative_to(root).as_posix(), path, st.st_mtime_ns, st.st_ctime_ns, st.st_mtime, st.st_size))
    return files


def _index(tmp_path: Path, **kwargs) -> GalleryIndex:
    kwargs.setdefault("clock", lambda: NOW)
    return GalleryIndex(lambda: tmp_path / "ziv.sqlite3", **kwargs)


def _rows(tmp_path: Path, version: int = GALLERY_INDEX_VERSION) -> dict[str, tuple[int, int, int]]:
    with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as conn:
        return {row[0]: row[1:] for row in conn.execute(f"SELECT rel_path, mtime_ns, ctime_ns, size FROM gallery_assets_v{version}")}


def _tables(tmp_path: Path) -> set[str]:
    with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _stored_version(tmp_path: Path, name: str) -> int | None:
    with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as conn:
        row = conn.execute("SELECT version FROM schema_versions WHERE component = ?", (name,)).fetchone()
    return None if row is None else row[0]


def _corruption_error() -> sqlite3.DatabaseError:
    exc = sqlite3.DatabaseError("database disk image is malformed")
    exc.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
    return exc


def _corrupt_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if ".corrupt-" in p.name)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "out"


def test_g1_second_listing_reads_nothing_and_rows_are_stored(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    index, reader = _index(tmp_path), _Reader()
    first = index.facts_for("root", _scan(root), reader, prune=True)
    second = index.facts_for("root", _scan(root), reader, prune=True)
    assert reader.calls == {"a.png": 1, "b.png": 1}
    assert second == first == {"a.png": {"config": {"name": "a.png"}, "image": {}}, "b.png": {"config": {"name": "b.png"}, "image": {}}}
    assert set(_rows(tmp_path)) == {"a.png", "b.png"}


def test_g2_new_index_on_same_database_reads_nothing(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    _index(tmp_path).facts_for("root", _scan(root), _Reader(), prune=True)
    reader = _Reader()
    facts = _index(tmp_path).facts_for("root", _scan(root), reader, prune=True)
    assert reader.total == 0
    assert facts["a.png"] == {"config": {"name": "a.png"}, "image": {}}


@pytest.mark.parametrize("change", ["mtime", "size"])
def test_g3_changed_file_is_read_once_and_row_updated(tmp_path: Path, root: Path, change: str) -> None:
    _make_files(root, ["a.png", "b.png"])
    index, reader = _index(tmp_path), _Reader()
    index.facts_for("root", _scan(root), reader, prune=True)
    before = _rows(tmp_path)["a.png"]
    if change == "mtime":
        os.utime(root / "a.png", (NOW - 30, NOW - 30))
    else:
        (root / "a.png").write_bytes(b"longer content")
        os.utime(root / "a.png", (NOW - 60, NOW - 60))
    index.facts_for("root", _scan(root), reader, prune=True)
    index.facts_for("root", _scan(root), reader, prune=True)
    assert reader.calls == {"a.png": 2, "b.png": 1}
    assert _rows(tmp_path)["a.png"] != before


def test_g4_prune_flag_controls_deletion_of_unseen_rows(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    index = _index(tmp_path)
    index.facts_for("root", _scan(root), _Reader(), prune=True)
    only_a = [f for f in _scan(root) if f.rel_path == "a.png"]
    index.facts_for("root", only_a, _Reader(), prune=False)
    assert set(_rows(tmp_path)) == {"a.png", "b.png"}
    index.facts_for("root", only_a, _Reader(), prune=True)
    assert set(_rows(tmp_path)) == {"a.png"}


def test_prune_is_scoped_to_its_root(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png"])
    other = tmp_path / "other"
    _make_files(other, ["z.png"])
    index = _index(tmp_path)
    index.facts_for("root", _scan(root), _Reader(), prune=True)
    index.facts_for("other", _scan(other), _Reader(), prune=True)
    index.facts_for("root", _scan(root), _Reader(), prune=True)
    reader = _Reader()
    index.facts_for("other", _scan(other), reader, prune=True)
    assert reader.total == 0


def test_g5_higher_version_drops_lower_table_and_keeps_other_components(tmp_path: Path, root: Path) -> None:
    def migrate_dummy(conn: sqlite3.Connection, from_version: int) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS dummy_items (value TEXT)")

    dummy = app_db.Component("dummy", 1, migrate_dummy)
    with closing(app_db.connect(tmp_path / "ziv.sqlite3", components=(dummy,))) as conn:
        conn.execute("INSERT INTO dummy_items VALUES ('keep')")
    _make_files(root, ["a.png"])
    _index(tmp_path, version=1).facts_for("root", _scan(root), _Reader(), prune=True)
    assert "gallery_assets_v1" in _tables(tmp_path)
    reader = _Reader()
    _index(tmp_path, version=2).facts_for("root", _scan(root), reader, prune=True)
    assert reader.total == 1
    tables = _tables(tmp_path)
    assert "gallery_assets_v2" in tables
    assert "gallery_assets_v1" not in tables
    with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as conn:
        assert conn.execute("SELECT value FROM dummy_items").fetchall() == [("keep",)]
    assert _stored_version(tmp_path, "dummy") == 1
    assert _stored_version(tmp_path, "gallery") == 2


def test_g6_incomplete_read_is_not_persisted(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png"])
    index, reader = _index(tmp_path), _Reader(complete=False)
    index.facts_for("root", _scan(root), reader, prune=True)
    index.facts_for("root", _scan(root), reader, prune=True)
    assert reader.calls["a.png"] == 2
    assert _rows(tmp_path) == {}


def test_g7_recent_file_is_persisted_only_after_it_settles(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png"], age=0.5)
    clock = [NOW]
    index, reader = _index(tmp_path, clock=lambda: clock[0]), _Reader()
    index.facts_for("root", _scan(root), reader, prune=True)
    assert _rows(tmp_path) == {}
    clock[0] = NOW + 5
    index.facts_for("root", _scan(root), reader, prune=True)
    assert set(_rows(tmp_path)) == {"a.png"}
    index.facts_for("root", _scan(root), reader, prune=True)
    assert reader.calls["a.png"] == 2


def test_g8_file_changed_during_read_is_not_persisted(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png"])
    inner = _Reader()

    def reader(file: ScannedFile) -> FactsRead:
        os.utime(file.path, (NOW - 30, NOW - 30))
        return inner(file)

    result = _index(tmp_path).facts_for("root", _scan(root), reader, prune=True)
    assert result["a.png"] == {"config": {"name": "a.png"}, "image": {}}
    assert _rows(tmp_path) == {}


def test_g8_restat_failure_is_not_persisted_and_does_not_raise(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png"])
    inner = _Reader()

    def reader(file: ScannedFile) -> FactsRead:
        file.path.unlink()
        return inner(file)

    result = _index(tmp_path).facts_for("root", _scan(root), reader, prune=True)
    assert set(result) == {"a.png"}
    assert _rows(tmp_path) == {}


def test_g9_batches_flushed_before_a_reader_error_stay_committed(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png", "c.png"])
    inner = _Reader()

    def reader(file: ScannedFile) -> FactsRead:
        if file.rel_path == "c.png":
            raise RuntimeError("boom")
        return inner(file)

    with pytest.raises(RuntimeError, match="boom"):
        _index(tmp_path, flush_every=2).facts_for("root", _scan(root), reader, prune=True)
    assert set(_rows(tmp_path)) == {"a.png", "b.png"}


def test_g10_corrupt_database_is_moved_aside_and_rows_persist_in_a_fresh_file(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    garbage = b"this is not a sqlite database " * 20
    (tmp_path / "ziv.sqlite3").write_bytes(garbage)
    reader = _Reader()
    with pytest.warns(UserWarning):
        facts = _index(tmp_path).facts_for("root", _scan(root), reader, prune=True)
    assert set(facts) == {"a.png", "b.png"}
    moved = _corrupt_files(tmp_path)
    assert len(moved) == 1
    assert moved[0].read_bytes() == garbage
    again = _Reader()
    _index(tmp_path).facts_for("root", _scan(root), again, prune=True)
    assert again.total == 0


def test_g11_corruption_in_the_load_query_quarantines_and_reads_uncached(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_files(root, ["a.png"])
    index = _index(tmp_path)
    index.facts_for("root", _scan(root), _Reader(), prune=True)

    def failing_load(*args, **kwargs):
        raise _corruption_error()

    monkeypatch.setattr(gallery_index, "_load_rows", failing_load)
    reader = _Reader()
    with pytest.warns(UserWarning, match="damaged"):
        facts = index.facts_for("root", _scan(root), reader, prune=True)
    assert reader.calls["a.png"] == 1
    assert facts["a.png"] == {"config": {"name": "a.png"}, "image": {}}
    assert len(_corrupt_files(tmp_path)) == 1
    assert not (tmp_path / "ziv.sqlite3").exists()


def test_g12_directory_as_database_reads_uncached_and_is_not_renamed(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    db_dir = tmp_path / "ziv.sqlite3"
    db_dir.mkdir()
    reader = _Reader()
    with pytest.warns(UserWarning):
        facts = GalleryIndex(lambda: db_dir, clock=lambda: NOW).facts_for("root", _scan(root), reader, prune=True)
    assert facts == {rel: {"config": {"name": rel}, "image": {}} for rel in ("a.png", "b.png")}
    assert db_dir.is_dir()
    assert _corrupt_files(tmp_path) == []


def test_g13_write_failure_keeps_the_full_result_and_warns(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_files(root, ["a.png", "b.png", "c.png"])

    def failing_write(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(gallery_index, "_write", failing_write)
    with pytest.warns(UserWarning, match="could not be updated") as caught:
        facts = _index(tmp_path, flush_every=2).facts_for("root", _scan(root), _Reader(), prune=True)
    assert set(facts) == {"a.png", "b.png", "c.png"}
    assert len([w for w in caught if "could not be updated" in str(w.message)]) == 1
    assert _rows(tmp_path) == {}


def test_g14_concurrent_cold_listings_read_each_file_once(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png", "c.png"])
    index, inner = _index(tmp_path), _Reader()

    def slow_reader(file: ScannedFile) -> FactsRead:
        time.sleep(0.02)
        return inner(file)

    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(index.facts_for("root", _scan(root), slow_reader, prune=True))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert inner.calls == {"a.png": 1, "b.png": 1, "c.png": 1}
    assert len(results) == 2
    assert results[0] == results[1]


def test_g15_prune_keeps_a_row_changed_since_it_was_loaded(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_files(root, ["a.png", "gone.png"])
    index = _index(tmp_path)
    index.facts_for("root", _scan(root), _Reader(), prune=True)
    remaining = [f for f in _scan(root) if f.rel_path == "a.png"]
    real_load = gallery_index._load_rows

    def load_then_change(conn: sqlite3.Connection, table: str, root_key: str):
        rows = real_load(conn, table, root_key)
        with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as other:
            other.execute(f"UPDATE {table} SET size = size + 1 WHERE rel_path = 'gone.png'")
            other.commit()
        return rows

    monkeypatch.setattr(gallery_index, "_load_rows", load_then_change)
    index.facts_for("root", remaining, _Reader(), prune=True)
    monkeypatch.setattr(gallery_index, "_load_rows", real_load)
    assert set(_rows(tmp_path)) == {"a.png", "gone.png"}
    index.facts_for("root", remaining, _Reader(), prune=True)
    assert set(_rows(tmp_path)) == {"a.png"}


def test_g18_forget_does_not_wait_for_a_running_scan(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    index = _index(tmp_path)
    files = _scan(root)
    index.facts_for("root", [f for f in files if f.rel_path == "a.png"], _Reader(), prune=False)
    entered, release = threading.Event(), threading.Event()

    def blocking_reader(file: ScannedFile) -> FactsRead:
        entered.set()
        assert release.wait(timeout=10)
        return FactsRead({"config": None, "image": {}}, True)

    scan = threading.Thread(target=lambda: index.facts_for("root", files, blocking_reader, prune=False))
    scan.start()
    try:
        assert entered.wait(timeout=10)
        forgetter = threading.Thread(target=lambda: index.forget("root", ["a.png"]))
        forgetter.start()
        forgetter.join(timeout=5)
        assert not forgetter.is_alive()
        assert "a.png" not in _rows(tmp_path)
    finally:
        release.set()
        scan.join(timeout=10)


def test_g19_forget_removes_only_the_named_rows(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png", "c.png"])
    index = _index(tmp_path)
    index.facts_for("root", _scan(root), _Reader(), prune=True)
    index.facts_for("other", _scan(root), _Reader(), prune=True)
    index.forget("root", ["a.png", "missing.png"])
    with closing(sqlite3.connect(tmp_path / "ziv.sqlite3")) as conn:
        remaining = set(conn.execute(f"SELECT root, rel_path FROM gallery_assets_v{GALLERY_INDEX_VERSION}"))
    assert remaining == {("root", "b.png"), ("root", "c.png"), ("other", "a.png"), ("other", "b.png"), ("other", "c.png")}


def test_g19_forget_without_a_database_creates_none(tmp_path: Path) -> None:
    path = tmp_path / "data" / "ziv.sqlite3"
    GalleryIndex(lambda: path).forget("root", ["a.png"])
    assert not path.exists()
    assert not path.parent.exists()


def test_forget_with_unusable_database_only_warns(tmp_path: Path) -> None:
    db_dir = tmp_path / "ziv.sqlite3"
    db_dir.mkdir()
    with pytest.warns(UserWarning):
        GalleryIndex(lambda: db_dir).forget("root", ["a.png"])
    assert db_dir.is_dir()


def _surrogate_scan(root: Path, bad_name: str) -> list[ScannedFile]:
    """Scan *root* and rename one file's asset ID to *bad_name*, as a non-UTF-8 file name appears after the walk."""
    return [dataclasses.replace(f, rel_path=bad_name) if f.rel_path == "b.png" else f for f in _scan(root)]


def _facts_without_warnings(index: GalleryIndex, root_key: str, files: list[ScannedFile], reader) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return index.facts_for(root_key, files, reader, prune=True)


def test_a_lone_surrogate_in_a_path_returns_every_fact_and_warns(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png", "c.png"])
    reader = _Reader()
    with pytest.warns(UserWarning, match="could not be updated"):
        facts = _index(tmp_path).facts_for("root", _surrogate_scan(root, "bad\udcff.png"), reader, prune=True)
    assert set(facts) == {"a.png", "bad\udcff.png", "c.png"}
    assert reader.total == 3
    assert _rows(tmp_path) == {}


def test_a_lone_surrogate_in_the_root_reads_without_the_cache(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["a.png", "b.png"])
    reader = _Reader()
    with pytest.warns(UserWarning, match="unavailable"):
        facts = _index(tmp_path).facts_for("bad\udcffroot", _scan(root), reader, prune=True)
    assert set(facts) == {"a.png", "b.png"}
    assert reader.total == 2


def test_non_ascii_facts_and_paths_are_stored(tmp_path: Path, root: Path) -> None:
    _make_files(root, ["café 東京 🎨.png", "b.png"])

    def reader(file: ScannedFile) -> FactsRead:
        return FactsRead({"config": {"prompt": "café 東京 🎨"}, "image": {}}, True)

    index = _index(tmp_path)
    _facts_without_warnings(index, "root", _scan(root), reader)
    assert set(_rows(tmp_path)) == {"café 東京 🎨.png", "b.png"}
    counting = _Reader()
    facts = _facts_without_warnings(index, "root", _scan(root), counting)
    assert counting.total == 0
    assert facts["b.png"]["config"]["prompt"] == "café 東京 🎨"
