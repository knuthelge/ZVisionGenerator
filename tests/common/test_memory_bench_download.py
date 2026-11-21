"""Tests for the memory bench's guard against a row that fetched or built a model inside the measured process."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench.memory.__main__ as bench_main
from bench.memory import video
from bench.memory.measure import GIB, run_case
from bench.memory.storage import DOWNLOAD_THRESHOLD_BYTES, storage_bytes


def test_storage_bytes_counts_each_file_once_through_links(tmp_path: Path) -> None:
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots").mkdir()
    elsewhere.mkdir()
    (root / "blobs" / "abc").write_bytes(b"x" * 100)
    (root / "snapshots" / "weights.safetensors").symlink_to(root / "blobs" / "abc")
    (elsewhere / "file").write_bytes(b"y" * 50)
    (root / "linked").symlink_to(elsewhere, target_is_directory=True)

    assert storage_bytes([root]) == 150
    assert storage_bytes([root / "missing"]) == 0


def test_storage_bytes_survives_a_link_back_to_a_parent(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file").write_bytes(b"x" * 10)
    (tmp_path / "a" / "loop").symlink_to(tmp_path, target_is_directory=True)

    assert storage_bytes([tmp_path]) == 10


class _ListedWithoutFileId:
    """A directory entry as Windows lists it: its ``stat()`` carries the size but no device or file id."""

    def __init__(self, entry: os.DirEntry[str]) -> None:
        self._entry = entry
        self.path = entry.path

    def is_dir(self) -> bool:
        return self._entry.is_dir()

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_dev=0, st_ino=0, st_size=self._entry.stat().st_size)


def test_storage_bytes_reads_the_file_ids_from_the_path_when_the_listing_has_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "blobs").mkdir()
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "blobs" / "abc").write_bytes(b"x" * 100)
    (tmp_path / "blobs" / "other").write_bytes(b"y" * 30)
    (tmp_path / "snapshots" / "weights.safetensors").symlink_to(tmp_path / "blobs" / "abc")
    os.link(tmp_path / "blobs" / "other", tmp_path / "snapshots" / "other-copy")
    real_scandir = os.scandir

    @contextlib.contextmanager
    def listing(path: str):
        with real_scandir(path) as entries:
            yield [_ListedWithoutFileId(entry) for entry in entries]

    monkeypatch.setattr("bench.memory.storage.os.scandir", listing)

    assert storage_bytes([tmp_path]) == 130


@pytest.fixture
def quiet_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.memory.measure.platform_peak", lambda: {"platform_peak_bytes": 9 * GIB, "gpu_total_bytes": 10 * GIB})
    monkeypatch.setattr("bench.memory.measure.smi_memory", lambda: (GIB, 10 * GIB))
    monkeypatch.setattr("bench.memory.measure._driver_memory", lambda: None)


def _run_with_storage(tmp_path: Path, sizes: list[int]) -> dict:
    readings = iter(sizes)

    def cli(_argv: list[str]) -> None:
        (tmp_path / "clip.mp4").write_bytes(b"x")

    case = {"id": "c", "family": "ltx", "model": "ltx-4", "steps": 2, "seed": 1}
    return run_case(case, [], cli, tmp_path, estimate=lambda _case: None, outputs=video.OUTPUT_SUFFIXES, storage=lambda: next(readings))


def test_a_case_that_grew_the_model_folders_is_flagged_as_downloaded(tmp_path: Path, quiet_machine: None) -> None:
    assert _run_with_storage(tmp_path, [100, 100 + DOWNLOAD_THRESHOLD_BYTES + 1])["downloaded"] is True


def test_a_case_that_left_the_model_folders_alone_is_not_flagged(tmp_path: Path, quiet_machine: None) -> None:
    assert _run_with_storage(tmp_path, [100, 100 + 4096])["downloaded"] is False


def _fake_spawn(rows: list[dict], spawned: list[dict]):
    def spawn(case: dict, run_dir: Path, timeout: int) -> dict:
        spawned.append(case)
        return {"id": case["id"], "status": "ok", "seconds": 1} | rows[len(spawned) - 1]

    return spawn


def test_a_downloaded_row_is_measured_again_and_the_second_row_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench_main, "_spawn", _fake_spawn([{"downloaded": True, "platform_peak_bytes": 9}, {"downloaded": False, "platform_peak_bytes": 7}], spawned))

    assert bench_main.main(["run", "--media", "video", "--case", "ltx4-s", "--label", "t"]) == 0

    rows = bench_main.read_rows(tmp_path / "video" / "t" / "results.jsonl")
    assert len(spawned) == 2 and [row["downloaded"] for row in rows] == [True, False]
    assert [row["platform_peak_bytes"] for row in bench_main._latest_rows(rows)] == [7]


def test_a_failed_download_row_is_not_measured_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench_main, "_spawn", _fake_spawn([{"downloaded": True, "status": "failed"}], spawned))

    bench_main.main(["run", "--media", "video", "--case", "ltx4-s", "--label", "t"])

    assert len(spawned) == 1


def test_a_case_whose_only_row_downloaded_is_run_again_next_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results = tmp_path / "video" / "t" / "results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"id": "ltx4-s", "status": "ok", "downloaded": True}) + "\n", encoding="utf-8")
    spawned: list[dict] = []
    monkeypatch.setattr(bench_main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(bench_main, "_spawn", _fake_spawn([{"downloaded": False}], spawned))

    bench_main.main(["run", "--media", "video", "--case", "ltx4-s", "--label", "t"])

    assert [case["id"] for case in spawned] == ["ltx4-s"]
