"""Tests for the Windows process-memory sampler of the memory bench, with the reader faked (no Windows needed)."""

from __future__ import annotations

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


def test_the_sampler_keeps_the_peak_of_the_private_working_set() -> None:
    working = iter([2 * GIB, 6 * GIB, 4 * GIB, 3 * GIB])

    peaks = _poll(WindowsSampler(interval=60, working_set=lambda: next(working)), 3)

    assert peaks == {"private_working_set_peak_bytes": 6 * GIB}


def test_the_sampler_polls_every_tenth_of_a_second_like_the_linux_counters() -> None:
    assert WindowsSampler()._interval == 0.1  # noqa: SLF001


def test_a_round_slower_than_the_interval_never_overlaps_the_next_one() -> None:
    active, overlaps, rounds = [0], [], []

    def slow_round() -> int:
        overlaps.append(active[0])
        active[0] += 1
        time.sleep(0.03)
        active[0] -= 1
        rounds.append(1)
        return GIB

    sampler = WindowsSampler(interval=0.01, working_set=slow_round)
    sampler.start()
    time.sleep(0.15)
    peaks = sampler.stop()

    assert len(rounds) >= 3
    assert set(overlaps) == {0}
    assert peaks["private_working_set_peak_bytes"] == GIB


def test_a_figure_that_was_never_read_is_none() -> None:
    sampler = WindowsSampler(interval=60, working_set=lambda: None)

    sampler.start()

    assert sampler.stop() == {"private_working_set_peak_bytes": None}


def test_the_reader_gives_none_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    assert windows.private_working_set() is None


def test_a_windows_case_records_the_os_and_the_samplers_peak(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(measure.sys, "platform", "win32")
    monkeypatch.setattr(measure, "WindowsSampler", lambda: SimpleNamespace(start=lambda: None, stop=lambda: {"private_working_set_peak_bytes": 5 * GIB}))
    monkeypatch.setattr(measure, "smi_memory", lambda: None)

    def cli(_argv: list[str]) -> None:
        (tmp_path / "out.png").write_bytes(b"x")

    row = measure.run_case({"id": "w", "prompt": "p"}, [], cli, tmp_path, estimate=lambda _case: None, outputs=frozenset({".png"}), storage=lambda: 0)

    assert row["os"] == "win32" and row["status"] == "ok"
    assert row["private_working_set_peak_bytes"] == 5 * GIB
