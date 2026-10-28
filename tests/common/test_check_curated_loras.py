"""Tests for the curated-list Hub check, with a fake Hub."""

from __future__ import annotations

import dataclasses
import json

from scripts.check_curated_loras import check_entries, compare_entry, refresh_downloads
from zvisiongenerator.utils.curated_loras import load_curated_loras

_ENTRY = load_curated_loras()[0]


def _sibling(**overrides):
    item = {"rfilename": _ENTRY.file, "size": _ENTRY.size_bytes, "lfs": {"sha256": _ENTRY.sha256}}
    item.update(overrides)
    return item


class TestCompareEntry:
    def test_matching_file_has_no_problems(self):
        assert compare_entry(_ENTRY, [_sibling()]) == []

    def test_missing_file_is_reported(self):
        assert compare_entry(_ENTRY, [_sibling(rfilename="other.safetensors")])

    def test_size_and_hash_changes_are_reported_separately(self):
        assert len(compare_entry(_ENTRY, [_sibling(size=_ENTRY.size_bytes + 1)])) == 1
        assert len(compare_entry(_ENTRY, [_sibling(lfs={"sha256": "0" * 64})])) == 1
        assert len(compare_entry(_ENTRY, [_sibling(lfs=None)])) == 1


class TestCheckEntries:
    def test_collects_problems_by_name_and_reports_an_unreachable_repo(self):
        good, changed, gone = (dataclasses.replace(_ENTRY, id=name, name=name) for name in ("good", "changed", "gone"))

        def fetch(entry):
            if entry.name == "gone":
                raise RuntimeError("the Hub answered 404")
            return {"siblings": [_sibling(size=1)] if entry.name == "changed" else [_sibling()], "downloads": 7}

        waits: list[float] = []
        failures, downloads = check_entries([good, changed, gone], fetch=fetch, delay=0.5, sleep=waits.append)
        assert set(failures) == {"changed", "gone"}
        assert waits == [0.5, 0.5]
        # A repository that answered has its count, whether or not its file still matches; one that did not has none.
        assert downloads == {"good": 7, "changed": 7}

    def test_ignores_a_download_count_that_is_not_a_number(self):
        _, downloads = check_entries([_ENTRY], fetch=lambda _entry: {"siblings": [_sibling()], "downloads": "many"}, delay=0)
        assert downloads == {}


class TestRefreshDownloads:
    def test_rewrites_only_the_counts_that_changed_and_keeps_the_layout(self, tmp_path):
        path = tmp_path / "curated.json"
        data = {"schema": 1, "entries": [{"id": "a", "downloads": 1}, {"id": "b", "downloads": 2}, {"id": "c", "downloads": 3}]}
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        assert refresh_downloads(path, {"a": 1, "b": 20}) == 1
        written = json.loads(path.read_text(encoding="utf-8"))
        assert [entry["downloads"] for entry in written["entries"]] == [1, 20, 3]
        assert path.read_text(encoding="utf-8") == json.dumps(written, indent=1, ensure_ascii=False) + "\n"

    def test_leaves_the_file_alone_when_nothing_changed(self, tmp_path):
        path = tmp_path / "curated.json"
        path.write_text('{"schema": 1, "entries": [{"id": "a", "downloads": 1}]}', encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        assert refresh_downloads(path, {"a": 1}) == 0
        assert path.read_text(encoding="utf-8") == before
