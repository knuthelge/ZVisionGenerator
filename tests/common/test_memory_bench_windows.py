"""Tests for the Windows process-memory sampler of the memory bench, with the reader faked (no Windows needed)."""

from __future__ import annotations

import ctypes
import sys
import time
from types import SimpleNamespace

from bench.memory import measure, windows
from bench.memory.windows import WindowsSampler

GIB = 1024**3


def _poll(sampler: WindowsSampler, rounds: int) -> dict:
    """Start the sampler, take *rounds* - 1 more polling rounds without waiting on the interval, and stop it (which reads once more)."""
    sampler.start()
    for _ in range(rounds - 1):
        sampler._sample()  # noqa: SLF001
    return sampler.stop()


def _reading(private: int, working: int, peak_working: int, commit: int) -> dict[str, int]:
    return {"private": private, "working_set": working, "peak_working_set": peak_working, "peak_commit": commit}


def test_the_sampler_keeps_the_peaks_of_the_private_and_the_full_working_set_and_the_commit_charge() -> None:
    readings = iter(
        [
            _reading(2 * GIB, 3 * GIB, 3 * GIB, 5 * GIB),
            _reading(6 * GIB, 9 * GIB, 9 * GIB, 20 * GIB),
            _reading(4 * GIB, 7 * GIB, 9 * GIB, 20 * GIB),
            _reading(3 * GIB, 6 * GIB, 12 * GIB, 22 * GIB),  # the stop reading: the OS's own peaks include a spike the polls missed
        ]
    )

    peaks = _poll(WindowsSampler(interval=60, memory=lambda: next(readings), commit=lambda: None, physical=lambda: 64 * GIB), 3)

    assert peaks == {
        "private_working_set_peak_bytes": 6 * GIB,
        "working_set_sampled_peak_bytes": 9 * GIB,
        "working_set_peak_bytes": 12 * GIB,
        "private_commit_peak_bytes": 22 * GIB,
        "system_commit_rise_bytes": None,
        "physical_ram_bytes": 64 * GIB,
    }


def test_the_sampler_records_how_far_the_systems_commit_rose_over_its_first_reading() -> None:
    commits = iter([10 * GIB, 40 * GIB, 55 * GIB, 50 * GIB])

    peaks = _poll(WindowsSampler(interval=60, memory=lambda: None, commit=lambda: next(commits)), 3)

    assert peaks["system_commit_rise_bytes"] == 45 * GIB
    assert peaks["private_working_set_peak_bytes"] is None


def test_the_sampler_polls_every_tenth_of_a_second_like_the_linux_counters() -> None:
    assert WindowsSampler()._interval == 0.1  # noqa: SLF001


def test_a_round_slower_than_the_interval_never_overlaps_the_next_one() -> None:
    active, overlaps, rounds = [0], [], []

    def slow_round() -> dict[str, int]:
        overlaps.append(active[0])
        active[0] += 1
        time.sleep(0.03)
        active[0] -= 1
        rounds.append(1)
        return _reading(GIB, GIB, GIB, GIB)

    sampler = WindowsSampler(interval=0.01, memory=slow_round, commit=lambda: None)
    sampler.start()
    time.sleep(0.15)
    peaks = sampler.stop()

    assert len(rounds) >= 3
    assert set(overlaps) == {0}
    assert peaks["private_working_set_peak_bytes"] == GIB


def test_a_figure_that_was_never_read_is_none() -> None:
    sampler = WindowsSampler(interval=60, memory=lambda: None, commit=lambda: None, physical=lambda: None)

    sampler.start()

    assert set(sampler.stop().values()) == {None}


def test_the_readers_give_none_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    assert windows.process_memory() is None
    assert windows.system_commit() is None
    assert windows.physical_ram() is None


def test_the_process_reader_takes_every_figure_from_one_psutil_call(monkeypatch) -> None:
    info = SimpleNamespace(uss=2 * GIB, wset=5 * GIB, peak_wset=6 * GIB, peak_pagefile=30 * GIB)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: SimpleNamespace(memory_full_info=lambda: info)))

    assert windows.process_memory() == {"private": 2 * GIB, "working_set": 5 * GIB, "peak_working_set": 6 * GIB, "peak_commit": 30 * GIB}


def test_the_process_reader_gives_none_when_psutil_refuses(monkeypatch) -> None:
    def refuse() -> None:
        raise OSError("access denied")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=refuse))

    assert windows.process_memory() is None


def test_the_system_commit_is_the_committed_pages_times_the_page_size(monkeypatch) -> None:
    def get_performance_info(reference, size: int) -> int:
        reference._obj.CommitTotal, reference._obj.PageSize = 1000, 4096  # noqa: SLF001
        assert size == ctypes.sizeof(reference._obj)  # noqa: SLF001
        return 1

    fake = SimpleNamespace(psapi=SimpleNamespace(GetPerformanceInfo=get_performance_info))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", fake, raising=False)

    assert windows.system_commit() == 1000 * 4096


def test_the_physical_ram_is_the_physical_pages_times_the_page_size(monkeypatch) -> None:
    def get_performance_info(reference, size: int) -> int:
        reference._obj.PhysicalTotal, reference._obj.PageSize = 16 * 1024 * 1024, 4096  # noqa: SLF001
        return 1

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(psapi=SimpleNamespace(GetPerformanceInfo=get_performance_info)), raising=False)

    assert windows.physical_ram() == 64 * GIB


def test_the_physical_ram_is_none_when_the_call_fails(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(psapi=SimpleNamespace(GetPerformanceInfo=lambda reference, size: 0)), raising=False)

    assert windows.physical_ram() is None


def test_a_windows_case_records_the_os_and_the_samplers_peak(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(measure.sys, "platform", "win32")
    monkeypatch.setattr(
        measure,
        "WindowsSampler",
        lambda: SimpleNamespace(start=lambda: None, stop=lambda: {"private_working_set_peak_bytes": 5 * GIB, "working_set_peak_bytes": 9 * GIB, "private_commit_peak_bytes": 20 * GIB}),
    )
    monkeypatch.setattr(measure, "smi_memory", lambda: None)

    def cli(_argv: list[str]) -> None:
        (tmp_path / "out.png").write_bytes(b"x")

    row = measure.run_case({"id": "w", "prompt": "p"}, [], cli, tmp_path, estimate=lambda _case: None, outputs=frozenset({".png"}), storage=lambda: 0)

    assert row["os"] == "win32" and row["status"] == "ok"
    assert (row["private_working_set_peak_bytes"], row["working_set_peak_bytes"], row["private_commit_peak_bytes"]) == (5 * GIB, 9 * GIB, 20 * GIB)
