"""Tests for the per-file asset records store."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from zvisiongenerator.web.asset_records import AssetRecords, AssetRecordsError


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "ziv.sqlite3"


@pytest.fixture()
def records(db) -> AssetRecords:
    return AssetRecords(lambda: db)


def _rows(db) -> list[tuple]:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT * FROM assets_entries ORDER BY asset_id").fetchall()


class TestStarring:
    def test_nothing_is_starred_at_first(self, records):
        assert records.starred_ids() == frozenset()

    def test_starring_and_unstarring_a_file(self, records):
        records.set_starred("a.png", True)
        records.set_starred("p/b.png", True)
        records.set_starred("a.png", False)

        assert records.starred_ids() == {"p/b.png"}

    def test_starring_twice_and_unstarring_an_unstarred_file_are_harmless(self, records):
        records.set_starred("a.png", True)
        records.set_starred("a.png", True)
        records.set_starred("other.png", False)

        assert records.starred_ids() == {"a.png"}


class TestRename:
    def test_a_starred_file_keeps_its_star_under_the_new_id(self, records):
        records.set_starred("a.png", True)
        records.set_starred("b.png", True)

        records.rename([("a.png", "proj/a.png")])

        assert records.starred_ids() == {"proj/a.png", "b.png"}

    def test_an_unstarred_file_gains_no_star(self, records):
        records.set_starred("b.png", True)

        records.rename([("a.png", "proj/a.png")])

        assert records.starred_ids() == {"b.png"}

    def test_a_record_already_at_the_new_id_is_replaced(self, records, db):
        records.set_starred("a.png", True)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("INSERT INTO assets_entries (asset_id, starred) VALUES ('proj/a.png', 0)")
            conn.commit()

        records.rename([("a.png", "proj/a.png")])

        assert _rows(db) == [("proj/a.png", 1)]


class TestForget:
    def test_forgetting_removes_only_the_named_records(self, records):
        for name in ("a.png", "b.png", "c.png"):
            records.set_starred(name, True)

        records.forget(["a.png", "c.png", "never-starred.png"])

        assert records.starred_ids() == {"b.png"}

    def test_reading_renaming_and_forgetting_with_no_database_create_none(self, records, db):
        assert records.starred_ids() == frozenset()
        records.rename([("a.png", "b.png")])
        records.forget(["a.png"])

        assert not db.exists()


class TestNewerVersion:
    @pytest.fixture()
    def newer_db(self, db):
        """A database a later ziv wrote: the component at version 2 and an extra column with a default."""
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL) WITHOUT ROWID")
            conn.execute("INSERT INTO schema_versions VALUES ('assets', 2)")
            conn.execute("CREATE TABLE assets_entries (asset_id TEXT PRIMARY KEY, starred INTEGER NOT NULL DEFAULT 0, rating INTEGER NOT NULL DEFAULT 0)")
            conn.execute("INSERT INTO assets_entries VALUES ('a.png', 1, 5)")
            conn.commit()
        return db

    def test_it_reads_and_writes_and_leaves_the_extra_column_alone(self, newer_db):
        records = AssetRecords(lambda: newer_db)

        assert records.starred_ids() == {"a.png"}
        records.set_starred("a.png", True)
        records.set_starred("b.png", True)
        records.rename([("a.png", "p/a.png")])
        records.forget(["b.png"])

        assert records.starred_ids() == {"p/a.png"}
        with closing(sqlite3.connect(newer_db)) as conn:
            assert conn.execute("SELECT version FROM schema_versions WHERE component = 'assets'").fetchone() == (2,)
            assert conn.execute("SELECT rating FROM assets_entries WHERE asset_id = 'p/a.png'").fetchone() == (5,)

    def test_unstarring_and_renaming_keep_a_newer_column(self, newer_db):
        records = AssetRecords(lambda: newer_db)

        records.set_starred("a.png", False)
        records.rename([("a.png", "p/a.png")])

        with closing(sqlite3.connect(newer_db)) as conn:
            assert conn.execute("SELECT starred, rating FROM assets_entries WHERE asset_id = 'p/a.png'").fetchall() == [(0, 5)]


class TestStorageFailure:
    def test_a_storage_failure_is_reported_as_a_records_error(self, tmp_path):
        records = AssetRecords(lambda: tmp_path)

        with pytest.raises(AssetRecordsError):
            records.set_starred("a.png", True)
