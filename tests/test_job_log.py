"""Tests for the persisted job log: entry builder, filters, paging and the app-database store."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

from zvisiongenerator.utils import app_db
from zvisiongenerator.web.job_log import (
    JOB_LOG_STATUS_FILTERS,
    JobLog,
    JobLogError,
    build_job_log_entry,
    is_loggable,
    matches_status,
    page_entries,
)

ENTRY_KEYS = {
    "job_id",
    "job_type",
    "workflow",
    "model",
    "prompt",
    "settings",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "duration_secs",
    "outputs",
    "error",
    "failed_runs",
}


def _entry(job_id: str = "j1", *, status: str = "completed", failed_runs: int = 0, **overrides) -> dict:
    entry = {"job_id": job_id, "status": status, "failed_runs": failed_runs}
    entry.update(overrides)
    return entry


def _build(**overrides) -> dict:
    values = dict(
        job_id="abc",
        job_type="image",
        context={"prompt": "a lake", "model": "m1", "workflow": "txt2img", "settings": {"steps": 4}},
        status="completed",
        created_at=100.0,
        started_at=110.0,
        completed_at=112.5,
        outputs=[{"id": "a/b.png"}, {"id": 7}, {"no_id": 1}, "junk"],
        error=None,
        failed_runs=0,
    )
    values.update(overrides)
    return build_job_log_entry(**values)


class TestBuildEntry:
    def test_entry_has_exactly_the_documented_keys(self):
        entry = _build()

        assert set(entry) == ENTRY_KEYS
        assert entry["job_id"] == "abc"
        assert entry["workflow"] == "txt2img"
        assert entry["model"] == "m1"
        assert entry["settings"] == {"steps": 4}
        assert entry["outputs"] == ["a/b.png", "7"]
        assert entry["duration_secs"] == 2.5
        assert (entry["created_at"], entry["started_at"], entry["completed_at"]) == (100.0, 110.0, 112.5)
        assert entry["error"] is None
        assert json.loads(json.dumps(entry)) == entry

    def test_prompt_is_cut_and_workflow_falls_back_to_the_job_type(self):
        entry = _build(context={"prompt": "x" * 900}, job_type="upscale")

        assert len(entry["prompt"]) == 500
        assert entry["workflow"] == "upscale"
        assert entry["model"] == ""
        assert entry["settings"] == {}

    def test_settings_are_copied_not_shared(self):
        settings = {"steps": 4}
        entry = _build(context={"settings": settings})
        settings["steps"] = 99

        assert entry["settings"] == {"steps": 4}

    def test_duration_is_never_negative(self):
        assert _build(started_at=50.0, completed_at=40.0)["duration_secs"] == 0.0

    def test_error_and_failed_runs_are_normalised(self):
        entry = _build(error=RuntimeError("boom"), failed_runs=3)

        assert entry["error"] == "boom"
        assert entry["failed_runs"] == 3


class TestHelpers:
    @pytest.mark.parametrize(
        ("job_type", "started_at", "expected"),
        [("image", 1.0, True), ("video", 0.0, True), ("upscale", 2.0, True), ("dummy", 1.0, False), ("test", 1.0, False), ("image", None, False)],
    )
    def test_is_loggable(self, job_type, started_at, expected):
        assert is_loggable(job_type, started_at) is expected

    def test_status_filters_constant_lists_the_route_values(self):
        assert JOB_LOG_STATUS_FILTERS == ("all", "failed", "completed", "cancelled")

    def test_matches_status(self):
        failed = _entry(status="failed")
        partial_done = _entry(status="completed", failed_runs=2)
        partial_cancelled = _entry(status="cancelled", failed_runs=1)
        clean = _entry(status="completed")

        assert [matches_status(e, "failed") for e in (failed, partial_done, partial_cancelled, clean)] == [True, True, True, False]
        assert [matches_status(e, "completed") for e in (failed, partial_done, partial_cancelled, clean)] == [False, True, False, True]
        assert [matches_status(e, "cancelled") for e in (failed, partial_done, partial_cancelled, clean)] == [False, False, True, False]
        assert all(matches_status(e, "all") for e in (failed, partial_done, partial_cancelled, clean))

    def test_page_entries_is_newest_first_and_clamps_the_page(self):
        entries = [_entry(f"j{i}") for i in range(5)]

        first = page_entries(entries, status="all", page=1, page_size=2)
        last = page_entries(entries, status="all", page=99, page_size=2)

        assert [e["job_id"] for e in first["entries"]] == ["j4", "j3"]
        assert set(first) == {"entries", "page", "total_pages", "total_count", "page_size"}
        assert (first["page"], first["total_pages"], first["total_count"], first["page_size"]) == (1, 3, 5, 2)
        assert [e["job_id"] for e in last["entries"]] == ["j0"]
        assert last["page"] == 3

    def test_page_entries_counts_after_the_filter_and_has_one_empty_page(self):
        entries = [_entry("a", status="failed"), _entry("b"), _entry("c", status="cancelled", failed_runs=1)]

        failed = page_entries(entries, status="failed", page=1, page_size=50)
        empty = page_entries([], status="all", page=5, page_size=50)

        assert [e["job_id"] for e in failed["entries"]] == ["c", "a"]
        assert failed["total_count"] == 2
        assert empty == {"entries": [], "page": 1, "total_pages": 1, "total_count": 0, "page_size": 50}


def _ids(log: JobLog) -> list[str]:
    """Return the ids in *log*, newest first."""
    return [e["job_id"] for e in log.page("all", 1, 10_000)["entries"]]


def _seed_newer_schema(db: Path) -> None:
    """Create a database whose recorded jobs schema is newer than this ziv's."""
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL) WITHOUT ROWID")
        conn.execute("INSERT INTO schema_versions VALUES ('jobs', 2)")
        conn.commit()


def _tables(db: Path) -> set[str]:
    with closing(sqlite3.connect(db)) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


class TestJobLogStore:
    @pytest.fixture()
    def db(self, tmp_path) -> Path:
        return tmp_path / "ziv.sqlite3"

    @pytest.fixture()
    def log(self, db) -> JobLog:
        return JobLog(lambda: db)

    def test_empty_database_reads_as_empty(self, log):
        assert log.page("all", 1, 50)["total_count"] == 0
        assert log.delete("x") is False
        assert log.clear() == 0

    def test_first_use_creates_the_table_and_records_the_version(self, log, db):
        log.page("all", 1)

        assert "jobs_entries_v1" in _tables(db)
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT component, version FROM schema_versions WHERE component = 'jobs'").fetchall() == [("jobs", 1)]

    def test_entries_round_trip_in_written_form(self, log):
        entry = _build(context={"prompt": "blå øyne", "settings": {"path": Path("/x")}})
        log.append(entry)

        (stored,) = log.page("all", 1, 50)["entries"]

        assert list(stored) == list(entry)
        assert stored["created_at"] == 100.0
        assert isinstance(stored["created_at"], float)
        assert stored["prompt"] == "blå øyne"
        assert stored["settings"] == {"path": str(Path("/x"))}

    @pytest.mark.parametrize("entry", [{"job_id": 5}, {}])
    def test_entry_without_a_string_job_id_is_refused(self, log, entry):
        with pytest.raises(ValueError):
            log.append(entry)

        assert log.page("all", 1, 50)["total_count"] == 0

    def test_delete_removes_every_entry_of_the_named_id_only(self, log):
        log.append(_entry("a"))
        log.append(_entry("a", status="failed"))
        log.append(_entry("b"))
        log.append(_entry("c"))

        assert log.delete("a") is True
        assert log.delete("a") is False

        assert _ids(log) == ["c", "b"]

    def test_clear_returns_the_count_and_empties_the_log(self, log):
        for job_id in ("a", "b"):
            log.append(_entry(job_id))

        assert log.clear() == 2
        assert log.clear() == 0

        assert log.page("all", 1, 50)["total_count"] == 0

    def test_page_uses_the_filter(self, log):
        log.append(_entry("ok"))
        log.append(_entry("bad", status="failed"))

        assert [e["job_id"] for e in log.page("failed", 1, 50)["entries"]] == ["bad"]
        assert [e["job_id"] for e in log.page("completed", 1, 50)["entries"]] == ["ok"]

    def test_the_path_is_resolved_on_every_call(self, tmp_path):
        current = {"path": tmp_path / "one.sqlite3"}
        log = JobLog(lambda: current["path"])

        log.append(_entry("a"))
        current["path"] = tmp_path / "two.sqlite3"

        assert log.page("all", 1, 50)["total_count"] == 0
        assert (tmp_path / "one.sqlite3").exists()

    def test_concurrent_appends_lose_nothing(self, log):
        threads = [threading.Thread(target=log.append, args=(_entry(f"j{i}"),)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert log.page("all", 1, 50)["total_count"] == 20
        assert set(_ids(log)) == {f"j{i}" for i in range(20)}


class TestJobLogErrors:
    def test_a_directory_at_the_path_raises_job_log_error_and_is_left_alone(self, tmp_path):
        log = JobLog(lambda: tmp_path)

        for call in (lambda: log.append(_entry("a")), lambda: log.page("all", 1, 50), lambda: log.delete("a"), log.clear):
            with pytest.raises(JobLogError) as caught:
                call()
            assert isinstance(caught.value.__cause__, sqlite3.Error)

        assert tmp_path.is_dir()
        assert list(tmp_path.iterdir()) == []

    def test_a_newer_schema_is_refused_and_left_untouched(self, tmp_path):
        db = tmp_path / "ziv.sqlite3"
        _seed_newer_schema(db)
        log = JobLog(lambda: db)

        for call in (lambda: log.append(_entry("a")), lambda: log.page("all", 1, 50), lambda: log.delete("a"), log.clear):
            with pytest.raises(JobLogError) as caught:
                call()
            assert caught.value.__cause__ is None

        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT version FROM schema_versions WHERE component = 'jobs'").fetchone() == (2,)
        assert "jobs_entries_v1" not in _tables(db)
        assert [p.name for p in tmp_path.iterdir() if ".corrupt-" in p.name] == []

    def test_corruption_mid_operation_quarantines_the_file_and_the_log_restarts_empty(self, tmp_path, monkeypatch):
        db = tmp_path / "ziv.sqlite3"
        log = JobLog(lambda: db)
        log.append(_entry("a"))
        real_connect = app_db.connect

        def failing_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            real_execute = conn.execute

            def execute(sql, *params):
                if sql.startswith("SELECT") and "jobs_entries_v1" in sql:
                    error = sqlite3.DatabaseError("database disk image is malformed")
                    error.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
                    raise error
                return real_execute(sql, *params)

            conn.execute = execute
            return conn

        monkeypatch.setattr(app_db, "connect", failing_connect)
        with pytest.warns(UserWarning), pytest.raises(JobLogError):
            log.page("all", 1, 50)
        monkeypatch.setattr(app_db, "connect", real_connect)

        assert [p.name for p in tmp_path.iterdir() if p.name.startswith("ziv.sqlite3.corrupt-")]
        assert log.page("all", 1, 50)["total_count"] == 0
