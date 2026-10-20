"""Tests for the persisted job log: entry builder, filters, paging and the file-backed store."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from zvisiongenerator.web import job_log as job_log_module
from zvisiongenerator.web.job_log import (
    JOB_LOG_FILENAME,
    JOB_LOG_KEEP_LINES,
    JOB_LOG_MAX_LINES,
    JOB_LOG_STATUS_FILTERS,
    JobLog,
    build_job_log_entry,
    is_loggable,
    job_log_path,
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

    def test_builder_is_total_on_junk_input(self):
        class Unprintable:
            def __str__(self) -> str:
                raise RuntimeError("no text")

        class BrokenMapping(dict):
            def get(self, *_args, **_kwargs):
                raise RuntimeError("no get")

        junk_contexts = [
            {"prompt": Unprintable(), "model": Unprintable(), "workflow": Unprintable(), "settings": Unprintable()},
            {"settings": ["not", "a", "mapping"], "prompt": None, "model": None},
            BrokenMapping(),
            None,
            "text",
        ]
        for context in junk_contexts:
            entry = _build(context=context, outputs=None, error=Unprintable(), failed_runs="x", started_at=None, completed_at="soon")
            assert set(entry) == ENTRY_KEYS
            assert entry["failed_runs"] == 0
            assert entry["duration_secs"] == 0.0
            assert entry["outputs"] == []

    def test_unserialisable_settings_still_build(self):
        assert _build(context={"settings": {"path": Path("/x")}})["settings"] == {"path": Path("/x")}


class TestHelpers:
    @pytest.mark.parametrize(
        ("job_type", "started_at", "expected"),
        [("image", 1.0, True), ("video", 0.0, True), ("upscale", 2.0, True), ("dummy", 1.0, False), ("test", 1.0, False), ("image", None, False)],
    )
    def test_is_loggable(self, job_type, started_at, expected):
        assert is_loggable(job_type, started_at) is expected

    def test_default_path_is_resolved_at_call_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(job_log_module, "get_ziv_data_dir", lambda: tmp_path / "one")
        first = job_log_path()
        monkeypatch.setattr(job_log_module, "get_ziv_data_dir", lambda: tmp_path / "two")

        assert first == tmp_path / "one" / JOB_LOG_FILENAME
        assert job_log_path() == tmp_path / "two" / JOB_LOG_FILENAME

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


class TestJobLogStore:
    @pytest.fixture()
    def path(self, tmp_path) -> Path:
        return tmp_path / "jobs.jsonl"

    @pytest.fixture()
    def log(self, path) -> JobLog:
        return JobLog(lambda: path)

    @staticmethod
    def _lines(path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines()

    def test_missing_file_reads_as_empty(self, log, path):
        assert log.page("all", 1, 50)["total_count"] == 0
        assert log.delete("x") is False
        assert log.clear() == 0
        assert not path.exists()

    def test_append_writes_one_json_line_per_entry_keeping_unicode(self, log, path):
        log.append(_entry("a", prompt="blå øyne"))
        log.append(_entry("b"))

        lines = self._lines(path)
        assert [json.loads(line)["job_id"] for line in lines] == ["a", "b"]
        assert "blå øyne" in lines[0]

    def test_values_that_are_not_json_are_written_as_text(self, log, path):
        log.append(_entry("a", settings={"path": Path("/x")}))

        assert json.loads(self._lines(path)[0])["settings"] == {"path": str(Path("/x"))}

    def test_appending_past_the_cap_keeps_the_newest_lines(self, log, path):
        path.write_text("".join(json.dumps(_entry(f"j{i}")) + "\n" for i in range(JOB_LOG_MAX_LINES)), encoding="utf-8")

        log.append(_entry("newest"))

        ids = [json.loads(line)["job_id"] for line in self._lines(path)]
        assert len(ids) == JOB_LOG_KEEP_LINES
        assert ids[-1] == "newest"
        assert ids[0] == f"j{JOB_LOG_MAX_LINES - JOB_LOG_KEEP_LINES + 1}"

    def test_exactly_at_the_cap_nothing_is_trimmed(self, log, path):
        path.write_text("".join(json.dumps(_entry(f"j{i}")) + "\n" for i in range(JOB_LOG_MAX_LINES - 1)), encoding="utf-8")

        log.append(_entry("last"))

        assert len(self._lines(path)) == JOB_LOG_MAX_LINES

    def test_corrupt_file_still_reads_and_the_next_rewrite_drops_bad_lines(self, log, path):
        good_a = json.dumps(_entry("a")).encode()
        good_b = json.dumps(_entry("b", prompt="ok")).encode()
        path.write_bytes(
            b"\n".join(
                [
                    good_a,
                    b"\xff\xfe not utf-8 \xc3(",
                    b"",
                    b"[1, 2, 3]",
                    b'"just a string"',
                    json.dumps({"no_id": 1}).encode(),
                    json.dumps({"job_id": 5}).encode(),
                    good_b,
                    b'{"job_id": "cut", "status": "comp',
                ]
            )
        )

        assert [e["job_id"] for e in log.page("all", 1, 50)["entries"]] == ["b", "a"]

        log.append(_entry("c"))

        assert [json.loads(line)["job_id"] for line in self._lines(path)] == ["a", "b", "c"]

    def test_delete_removes_only_the_named_entry(self, log, path):
        for job_id in ("a", "b", "c"):
            log.append(_entry(job_id))

        assert log.delete("b") is True
        assert log.delete("b") is False

        assert [json.loads(line)["job_id"] for line in self._lines(path)] == ["a", "c"]

    def test_clear_returns_the_count_and_empties_the_file(self, log, path):
        for job_id in ("a", "b"):
            log.append(_entry(job_id))

        assert log.clear() == 2

        assert path.read_text(encoding="utf-8") == ""
        assert log.page("all", 1, 50)["total_count"] == 0

    def test_page_uses_the_filter(self, log):
        log.append(_entry("ok"))
        log.append(_entry("bad", status="failed"))

        assert [e["job_id"] for e in log.page("failed", 1, 50)["entries"]] == ["bad"]
        assert [e["job_id"] for e in log.page("completed", 1, 50)["entries"]] == ["ok"]

    def test_the_path_is_resolved_on_every_call(self, tmp_path):
        current = {"path": tmp_path / "one.jsonl"}
        log = JobLog(lambda: current["path"])

        log.append(_entry("a"))
        current["path"] = tmp_path / "two.jsonl"

        assert log.page("all", 1, 50)["total_count"] == 0
        assert (tmp_path / "one.jsonl").exists()

    def test_os_errors_propagate(self, tmp_path):
        log = JobLog(lambda: tmp_path)  # a directory, not a file

        for call in (lambda: log.append(_entry("a")), lambda: log.page("all", 1, 50), lambda: log.delete("a"), log.clear):
            with pytest.raises(OSError):
                call()

    def test_concurrent_appends_lose_nothing(self, log, path):
        threads = [threading.Thread(target=log.append, args=(_entry(f"j{i}"),)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(json.loads(line)["job_id"] for line in self._lines(path)) == sorted(f"j{i}" for i in range(20))
