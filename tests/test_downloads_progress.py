"""Tests for the download byte sink, counters and speed meter."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from zvisiongenerator.downloads.progress import ByteCounts, SpeedMeter, make_byte_sink, progress_from_counts

_ROOT = Path(__file__).resolve().parent.parent

_BYTES_DESC = "Downloading (incomplete total...)"
_WRITTEN_DESC = "Reconstructing (incomplete total...)"


def _bars(counts: ByteCounts):
    sink = make_byte_sink(counts)
    return (
        sink(total=3, desc="Fetching 3 files", unit="it"),
        sink(total=1000, desc=_BYTES_DESC, unit="B", unit_scale=True),
        sink(total=1000, desc=_WRITTEN_DESC, unit="B", unit_scale=True),
    )


def test_byte_counts_start_at_zero() -> None:
    assert ByteCounts().snapshot() == (0, 0)


def test_xet_sequence_counts_each_bar_separately() -> None:
    counts = ByteCounts()
    fetching, wire, written = _bars(counts)
    done = []
    for step in (200, 300, 435):  # wire bytes are ~93.5 % of the file bytes
        wire.update(step)
        done.append(progress_from_counts(*counts.snapshot(), 1000, None).bytes_done)
    for file_bytes in (400, 600):
        written.update(file_bytes)
        fetching.update(1)
        done.append(progress_from_counts(*counts.snapshot(), 1000, None).bytes_done)
    assert counts.snapshot() == (935, 1000)
    assert done == sorted(done)
    assert max(done) <= 1000
    assert done[-1] == 1000


def test_plain_http_sequence_does_not_double_count() -> None:
    counts = ByteCounts()
    _, wire, written = _bars(counts)
    for chunk in (100, 250, 150):
        wire.update(chunk)
        written.update(chunk)
        wire_n, written_n = counts.snapshot()
        assert progress_from_counts(wire_n, written_n, 1000, None).bytes_done == wire_n
    assert counts.snapshot() == (500, 500)
    assert progress_from_counts(*counts.snapshot(), 1000, None).bytes_done == 500


def test_other_bars_are_ignored() -> None:
    counts = ByteCounts()
    sink = make_byte_sink(counts)
    sink(desc="Fetching 3 files", unit="it").update(3)
    sink(desc="model.safetensors: downloading", unit="it").update(7)
    sink(desc="something else", unit="B").update(9)
    sink(unit="B").update(11)
    assert counts.snapshot() == (0, 0)


def test_positional_desc_is_read() -> None:
    counts = ByteCounts()
    make_byte_sink(counts)(None, "Downloading bytes", unit="B").update(5)
    assert counts.snapshot() == (5, 0)


def test_per_file_bars_count_by_desc() -> None:
    counts = ByteCounts()
    sink = make_byte_sink(counts)
    sink(desc="a.bin: downloading bytes", unit="B").update(10)
    sink(desc="a.bin: reconstructing file", unit="B").update(4)
    assert counts.snapshot() == (10, 4)


@pytest.mark.parametrize("n", [None, 0, -3])
def test_empty_or_negative_updates_are_safe(n) -> None:
    counts = ByteCounts()
    _, wire, _ = _bars(counts)
    wire.update(n)
    assert counts.snapshot() == (0, 0)


def test_default_update_step_counts_one() -> None:
    counts = ByteCounts()
    _, wire, _ = _bars(counts)
    wire.update()
    assert counts.snapshot() == (1, 0)


def test_sink_counts_with_tqdm_disabled_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TQDM_DISABLE", "1")
    counts = ByteCounts()
    _, wire, written = _bars(counts)
    wire.update(30)
    written.update(20)
    assert counts.snapshot() == (30, 20)


def test_update_returns_without_raising_when_counting_fails() -> None:
    class Broken(ByteCounts):
        def add_wire(self, n: int) -> None:
            raise RuntimeError("boom")

    _, wire, _ = _bars(Broken())
    wire.update(5)


def test_threads_sum_exactly() -> None:
    counts = ByteCounts()
    _, wire, written = _bars(counts)

    def work() -> None:
        for _ in range(1000):
            wire.update(1)
            written.update(2)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counts.snapshot() == (8000, 16000)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_meter_is_none_until_two_samples_two_seconds_apart() -> None:
    clock = _Clock()
    meter = SpeedMeter(clock=clock)
    assert meter.add(0) is None
    clock.now += 1.5
    assert meter.add(1500) is None
    clock.now += 0.5
    assert meter.add(2000) == pytest.approx(1000.0)


def test_meter_averages_over_the_window() -> None:
    clock = _Clock()
    meter = SpeedMeter(window_secs=10.0, clock=clock)
    meter.add(0)
    for _ in range(10):  # 100 B/s for 10 s
        clock.now += 1.0
        meter.add(int((clock.now - 100.0) * 100))
    for _ in range(10):  # then 1000 B/s for 10 s
        clock.now += 1.0
        speed = meter.add(1000 + int((clock.now - 110.0) * 1000))
    assert speed == pytest.approx(1000.0)


def test_meter_reports_zero_when_nothing_moves() -> None:
    clock = _Clock()
    meter = SpeedMeter(clock=clock)
    meter.add(500)
    clock.now += 3.0
    assert meter.add(500) == 0.0


def test_progress_caps_done_at_total_and_takes_the_larger_count() -> None:
    progress = progress_from_counts(900, 1200, 1000, 100.0)
    assert progress.bytes_done == 1000
    assert progress.bytes_total == 1000
    assert progress.eta_secs == 0.0
    assert progress_from_counts(600, 400, 1000, None).bytes_done == 600


def test_progress_eta_needs_total_and_speed() -> None:
    assert progress_from_counts(250, 0, 1000, 50.0).eta_secs == pytest.approx(15.0)
    assert progress_from_counts(250, 0, None, 50.0).eta_secs is None
    assert progress_from_counts(250, 0, 1000, None).eta_secs is None
    assert progress_from_counts(250, 0, 1000, 0.0).eta_secs is None


def test_progress_without_total_does_not_clamp() -> None:
    progress = progress_from_counts(5000, 0, None, 10.0)
    assert (progress.bytes_done, progress.bytes_total) == (5000, None)


def test_importing_progress_does_not_import_tqdm() -> None:
    code = "import sys; import zvisiongenerator.downloads.progress; sys.exit(1 if 'tqdm' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, check=False)
    assert result.returncode == 0
