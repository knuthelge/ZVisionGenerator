"""Tests for the ensure step: skip, own, wait, stop, detach and throttle."""

from __future__ import annotations

import importlib
import subprocess
import sys
import threading
from collections.abc import Callable

import pytest

import zvisiongenerator.downloads as downloads
from zvisiongenerator.downloads.child import ChildResult
from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, DownloadReport, ensure_downloaded
from zvisiongenerator.downloads.flights import DownloadFlights, Flight
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck

TIMEOUT = 10.0
MISSING = TargetCheck(missing_files=2, missing_bytes=1000)


class _SpyFlights(DownloadFlights):
    """Record every finish so tests can assert each flight ends exactly once."""

    def __init__(self) -> None:
        super().__init__()
        self.finished: list[tuple[str, str, str | None]] = []
        self._spy_lock = threading.Lock()

    def finish(self, flight: Flight, status: str, message: str | None = None) -> None:
        with self._spy_lock:
            self.finished.append((flight.target.repo_id, status, message))
        super().finish(flight, status, message)


class _Clock:
    """A manual clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _target(name: str = "org/a") -> DownloadTarget:
    return DownloadTarget(name)


def _never() -> bool:
    return False


def _run_counts(steps: list[tuple[int, int]], clock: _Clock | None = None, dt: float = 0.0, status: str = "done", message: str | None = None):
    """Build a fake ``run`` that feeds *steps* to ``on_counts`` and returns *status*."""

    def run(target: DownloadTarget, *, on_counts: Callable[[int, int], None], cancelled: Callable[[], bool]) -> ChildResult:
        for wire, written in steps:
            if clock is not None:
                clock.now += dt
            on_counts(wire, written)
        return ChildResult(status, message)  # type: ignore[arg-type]

    return run


def _ensure(targets, flights, *, reports=None, **overrides):
    reports = [] if reports is None else reports
    kwargs = {
        "flights": flights,
        "report": reports.append,
        "cancelled": _never,
        "present": lambda t: False,
        "check": lambda t: MISSING,
        "run": _run_counts([]),
        "wait_secs": 0.01,
    }
    kwargs.update(overrides)
    return ensure_downloaded(targets, **kwargs), reports


def test_all_present_is_ready_without_check_or_reports() -> None:
    flights = _SpyFlights()
    calls: list[str] = []
    result, reports = _ensure(
        [_target("org/a"), _target("org/b")],
        flights,
        present=lambda t: True,
        check=lambda t: calls.append(t.repo_id),
    )
    assert result == READY
    assert calls == []
    assert reports == []
    assert flights.finished == []


def test_unchecked_stops_the_plan_and_finishes_flight_stopped() -> None:
    flights = _SpyFlights()
    ran: list[str] = []

    def run(target, *, on_counts, cancelled):
        ran.append(target.repo_id)
        return ChildResult("done")

    result, reports = _ensure([_target("org/a"), _target("org/b")], flights, check=lambda t: None, run=run)
    assert result == UNCHECKED
    assert ran == []
    assert reports == []
    assert flights.finished == [("org/a", "stopped", None)]
    assert flights.active() == ()


def test_nothing_missing_finishes_done_without_download() -> None:
    flights = _SpyFlights()
    ran: list[str] = []

    def run(target, *, on_counts, cancelled):
        ran.append(target.repo_id)
        return ChildResult("done")

    result, reports = _ensure([_target()], flights, check=lambda t: TargetCheck(0, 0), run=run)
    assert result == READY
    assert ran == []
    assert reports == []
    assert flights.finished == [("org/a", "done", None)]


def test_multi_part_reports_only_the_missing_part_with_snapped_final() -> None:
    flights = _SpyFlights()
    first, second = _target("org/a"), _target("org/b")
    clock = _Clock()
    result, reports = _ensure(
        [first, second],
        flights,
        present=lambda t: t is first,
        clock=clock,
        run=_run_counts([(100, 0), (300, 0), (900, 0)], clock, dt=2.0),
    )
    assert result == READY
    assert {r.repo for r in reports} == {"org/b"}
    assert {(r.part, r.parts) for r in reports} == {(2, 2)}
    assert reports[0].progress.bytes_done == 0
    assert reports[0].progress.bytes_total == 1000
    assert reports[-1].progress.bytes_done == 1000
    assert reports[-1].progress.eta_secs == 0.0
    assert not any(r.shared for r in reports)
    assert flights.finished == [("org/b", "done", None)]
    assert flights.active() == ()


def test_final_report_passes_even_inside_the_throttle_interval() -> None:
    flights = _SpyFlights()
    clock = _Clock()
    _, reports = _ensure([_target()], flights, clock=clock, run=_run_counts([(10, 0), (20, 0)], clock, dt=0.01))
    assert reports[0].progress.bytes_done == 0
    assert reports[-1].progress.bytes_done == 1000
    assert len(reports) == 2


def test_reports_are_throttled_to_one_per_second() -> None:
    flights = _SpyFlights()
    clock = _Clock()
    stamps: list[float] = []

    def report(r: DownloadReport) -> None:
        stamps.append(clock.now)

    steps = [(i * 10, 0) for i in range(1, 31)]  # 10 counts/s for 3 s
    ensure_downloaded(
        [_target()],
        flights=flights,
        report=report,
        cancelled=_never,
        present=lambda t: False,
        check=lambda t: MISSING,
        run=_run_counts(steps, clock, dt=0.1),
        clock=clock,
    )
    interior = stamps[:-1]  # the last report is the owned part's final one
    assert 3 <= len(interior) <= 5
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(interior, interior[1:], strict=False))


def test_counts_use_written_when_no_wire_bytes_and_cap_at_total() -> None:
    flights = _SpyFlights()
    clock = _Clock()
    _, reports = _ensure(
        [_target()],
        flights,
        clock=clock,
        interval_secs=0.0,
        run=_run_counts([(0, 100), (0, 300), (0, 500), (5000, 0)], clock, dt=1.0, status="stopped"),
    )
    mid = [r.progress for r in reports[1:]]
    assert mid[2].bytes_done == 500
    assert mid[2].bytes_per_sec == pytest.approx(200.0)
    assert mid[3].bytes_done == 1000  # capped at the total


def test_run_failure_names_repo_and_progress_and_finishes_failed() -> None:
    flights = _SpyFlights()
    with pytest.raises(RuntimeError, match="boom") as excinfo:
        _ensure([_target("org/a")], flights, run=_run_counts([(400, 0)], status="failed", message="boom"))
    assert "org/a" in str(excinfo.value)
    assert flights.finished == [("org/a", "failed", str(excinfo.value))]
    assert flights.active() == ()


def test_check_error_propagates_and_finishes_failed() -> None:
    flights = _SpyFlights()

    def check(target):
        raise RuntimeError("Could not download org/a: gated")

    with pytest.raises(RuntimeError, match="gated"):
        _ensure([_target("org/a")], flights, check=check)
    assert flights.finished == [("org/a", "failed", "Could not download org/a: gated")]
    assert flights.active() == ()


def test_owner_stop_returns_stopped_and_leaves_no_flight() -> None:
    flights = _SpyFlights()
    stop = threading.Event()

    def run(target, *, on_counts, cancelled):
        on_counts(100, 0)
        stop.set()
        assert cancelled()
        return ChildResult("stopped")

    result, reports = _ensure([_target()], flights, run=run, cancelled=stop.is_set)
    assert result == STOPPED
    assert flights.finished == [("org/a", "stopped", None)]
    assert flights.active() == ()
    assert reports[-1].progress.bytes_done < 1000


def test_cancel_before_the_child_starts_never_runs_it() -> None:
    flights = _SpyFlights()
    ran: list[str] = []

    def run(target, *, on_counts, cancelled):
        ran.append("x")
        return ChildResult("done")

    result, _ = _ensure([_target()], flights, run=run, cancelled=lambda: True)
    assert result == STOPPED
    assert ran == []
    assert flights.active() == ()


def test_on_counts_never_raises_even_when_report_does() -> None:
    flights = _SpyFlights()
    calls: list[int] = []

    def report(r: DownloadReport) -> None:
        calls.append(r.progress.bytes_done)
        if len(calls) > 1:
            raise ValueError("ui gone")

    def run(target, *, on_counts, cancelled):
        on_counts(100, 0)
        on_counts(200, 0)
        return ChildResult("stopped")

    result = ensure_downloaded(
        [_target()],
        flights=flights,
        report=report,
        cancelled=_never,
        present=lambda t: False,
        check=lambda t: MISSING,
        run=run,
        interval_secs=0.0,
    )
    assert result == STOPPED
    assert flights.active() == ()


class _Scenario:
    """Shared state for the thread tests: a fake disk and a gated fake child."""

    def __init__(self) -> None:
        self.on_disk = False
        self.started = threading.Event()
        self.counts = threading.Event()
        self.finish = threading.Event()
        self.runs: list[str] = []

    def present(self, target: DownloadTarget) -> bool:
        return self.on_disk

    def run(self, target, *, on_counts, cancelled) -> ChildResult:
        self.runs.append(target.repo_id)
        self.started.set()
        assert self.counts.wait(TIMEOUT)
        on_counts(400, 0)
        assert self.finish.wait(TIMEOUT)
        self.on_disk = True
        return ChildResult("done")


def _start(name: str, **kwargs):
    """Run ``ensure_downloaded`` in a thread; return ``(thread, outcome dict, reports)``."""
    outcome: dict[str, object] = {}
    reports: list[DownloadReport] = []

    def work() -> None:
        try:
            outcome["result"] = ensure_downloaded([_target()], report=reports.append, wait_secs=0.01, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=work, name=name, daemon=True)
    thread.start()
    return thread, outcome, reports


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = threading.Event()
    for _ in range(int(TIMEOUT / 0.01)):
        if predicate():
            return
        deadline.wait(0.01)
    raise AssertionError("condition not reached")


def test_waiter_mirrors_owner_progress_shared_and_throttled() -> None:
    flights = _SpyFlights()
    scenario = _Scenario()
    owner, owner_out, owner_reports = _start("owner", flights=flights, cancelled=_never, present=scenario.present, check=lambda t: MISSING, run=scenario.run)
    assert scenario.started.wait(TIMEOUT)
    clock = _Clock()
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return False

    waiter, waiter_out, waiter_reports = _start("waiter", flights=flights, cancelled=waiter_cancelled, present=scenario.present, check=lambda t: MISSING, run=scenario.run, clock=clock)
    assert polls.wait(TIMEOUT)
    threading.Event().wait(0.1)
    assert waiter_reports == []  # the owner has not reported bytes yet
    scenario.counts.set()
    _wait_until(lambda: len(waiter_reports) >= 1)
    first = waiter_reports[0]
    assert first.shared is True
    assert (first.part, first.parts, first.repo) == (1, 1, "org/a")
    assert first.progress.bytes_done == 400
    threading.Event().wait(0.1)
    assert len(waiter_reports) == 1  # frozen clock: throttled, and the same snapshot is not repeated
    scenario.finish.set()
    owner.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert owner_out["result"] == READY
    assert waiter_out["result"] == READY
    assert scenario.runs == ["org/a"]
    assert not any(r.shared for r in owner_reports)
    assert flights.active() == ()
    assert [f[1] for f in flights.finished] == ["done"]


def test_waiter_stop_detaches_and_the_owner_continues() -> None:
    flights = _SpyFlights()
    scenario = _Scenario()
    owner, owner_out, _ = _start("owner", flights=flights, cancelled=_never, present=scenario.present, check=lambda t: MISSING, run=scenario.run)
    assert scenario.started.wait(TIMEOUT)
    stop = threading.Event()
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return stop.is_set()

    waiter, waiter_out, _ = _start("waiter", flights=flights, cancelled=waiter_cancelled, present=scenario.present, check=lambda t: MISSING, run=scenario.run)
    assert polls.wait(TIMEOUT)
    stop.set()
    waiter.join(TIMEOUT)
    assert waiter_out["result"] == DETACHED
    assert flights.active() == (_target(),)
    assert flights.finished == []
    scenario.counts.set()
    scenario.finish.set()
    owner.join(TIMEOUT)
    assert owner_out["result"] == READY
    assert flights.active() == ()


def test_waiter_takes_over_after_the_owner_stops() -> None:
    flights = _SpyFlights()
    state = {"disk": False, "calls": 0}
    release_owner = threading.Event()
    started = threading.Event()

    def run(target, *, on_counts, cancelled):
        state["calls"] += 1
        if state["calls"] == 1:
            started.set()
            assert release_owner.wait(TIMEOUT)
            return ChildResult("stopped")
        state["disk"] = True
        return ChildResult("done")

    common = {"flights": flights, "present": lambda t: state["disk"], "check": lambda t: MISSING, "run": run}
    owner, owner_out, _ = _start("owner", cancelled=_never, **common)
    assert started.wait(TIMEOUT)
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return False

    waiter, waiter_out, waiter_reports = _start("waiter", cancelled=waiter_cancelled, **common)
    assert polls.wait(TIMEOUT)
    release_owner.set()
    owner.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert owner_out["result"] == STOPPED
    assert waiter_out["result"] == READY
    assert state["calls"] == 2
    assert [f[1] for f in flights.finished] == ["stopped", "done"]
    assert waiter_reports[-1].progress.bytes_done == 1000
    assert flights.active() == ()


def test_waiter_takes_over_after_the_owner_was_unchecked() -> None:
    flights = _SpyFlights()
    state = {"checks": 0, "disk": False}
    release_owner = threading.Event()
    in_check = threading.Event()

    def check(target):
        state["checks"] += 1
        if state["checks"] == 1:
            in_check.set()
            assert release_owner.wait(TIMEOUT)
            return None
        return TargetCheck(0, 0)

    owner, owner_out, _ = _start("owner", flights=flights, cancelled=_never, present=lambda t: state["disk"], check=check)
    assert in_check.wait(TIMEOUT)
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return False

    waiter, waiter_out, _ = _start("waiter", flights=flights, cancelled=waiter_cancelled, present=lambda t: state["disk"], check=check)
    assert polls.wait(TIMEOUT)
    release_owner.set()
    owner.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert owner_out["result"] == UNCHECKED
    assert waiter_out["result"] == READY
    assert [f[1] for f in flights.finished] == ["stopped", "done"]


def test_waiter_raises_the_owner_failure_message() -> None:
    flights = _SpyFlights()
    release_owner = threading.Event()
    started = threading.Event()

    def run(target, *, on_counts, cancelled):
        started.set()
        assert release_owner.wait(TIMEOUT)
        return ChildResult("failed", "disk full")

    common = {"flights": flights, "present": lambda t: False, "check": lambda t: MISSING, "run": run}
    owner, owner_out, _ = _start("owner", cancelled=_never, **common)
    assert started.wait(TIMEOUT)
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return False

    waiter, waiter_out, _ = _start("waiter", cancelled=waiter_cancelled, **common)
    assert polls.wait(TIMEOUT)
    release_owner.set()
    owner.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert isinstance(owner_out["error"], RuntimeError)
    assert isinstance(waiter_out["error"], RuntimeError)
    assert "disk full" in str(waiter_out["error"])
    assert "org/a" in str(waiter_out["error"])
    assert str(waiter_out["error"]) == str(owner_out["error"])
    assert [f[1] for f in flights.finished] == ["failed"]
    assert flights.active() == ()


def test_every_name_in_all_imports_and_the_package_stays_light() -> None:
    for name in downloads.__all__:
        assert hasattr(downloads, name), name
    module = importlib.import_module("zvisiongenerator.downloads")
    assert set(downloads.__all__) <= set(vars(module))
    code = (
        "import sys, zvisiongenerator.downloads as d\n"
        "heavy = [m for m in ('huggingface_hub', 'tqdm', 'torch', 'mflux') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "for name in d.__all__:\n"
        "    getattr(d, name)\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)


def test_report_error_on_the_first_report_finishes_the_flight_once() -> None:
    flights = _SpyFlights()
    runs: list[str] = []

    def report(r: DownloadReport) -> None:
        raise ValueError("ui gone")

    def run(target, *, on_counts, cancelled):
        runs.append(target.repo_id)
        return ChildResult("done")

    with pytest.raises(ValueError, match="ui gone"):
        ensure_downloaded([_target()], flights=flights, report=report, cancelled=_never, present=lambda t: False, check=lambda t: MISSING, run=run)
    assert runs == []
    assert [f[1] for f in flights.finished] == ["failed"]
    assert flights.active() == ()


def test_report_error_on_the_final_report_finishes_the_flight_once() -> None:
    flights = _SpyFlights()
    seen: list[int] = []

    def report(r: DownloadReport) -> None:
        seen.append(r.progress.bytes_done)
        if len(seen) > 1:
            raise ValueError("ui gone")

    with pytest.raises(ValueError, match="ui gone"):
        ensure_downloaded(
            [_target()],
            flights=flights,
            report=report,
            cancelled=_never,
            present=lambda t: False,
            check=lambda t: MISSING,
            run=_run_counts([]),
        )
    assert len(flights.finished) == 1
    assert flights.active() == ()


def test_base_exception_from_the_child_run_finishes_the_flight_stopped() -> None:
    flights = _SpyFlights()

    def run(target, *, on_counts, cancelled):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ensure_downloaded(
            [_target()],
            flights=flights,
            report=lambda r: None,
            cancelled=_never,
            present=lambda t: False,
            check=lambda t: MISSING,
            run=run,
        )
    assert [f[1] for f in flights.finished] == ["stopped"]
    assert flights.active() == ()


def test_waiter_rechecks_presence_and_downloads_when_still_absent_after_done() -> None:
    flights = _SpyFlights()
    state = {"checks": 0}
    release_owner = threading.Event()
    in_check = threading.Event()
    runs: list[str] = []

    def check(target):
        state["checks"] += 1
        if state["checks"] == 1:
            in_check.set()
            assert release_owner.wait(TIMEOUT)
            return TargetCheck(0, 0)
        return MISSING

    def run(target, *, on_counts, cancelled):
        runs.append("run")
        state["disk"] = True
        return ChildResult("done")

    state["disk"] = False
    owner, owner_out, _ = _start("owner", flights=flights, cancelled=_never, present=lambda t: state["disk"], check=check, run=run)
    assert in_check.wait(TIMEOUT)
    polls = threading.Event()

    def waiter_cancelled() -> bool:
        polls.set()
        return False

    waiter, waiter_out, _ = _start("waiter", flights=flights, cancelled=waiter_cancelled, present=lambda t: state["disk"], check=check, run=run)
    assert polls.wait(TIMEOUT)
    release_owner.set()
    owner.join(TIMEOUT)
    waiter.join(TIMEOUT)
    assert owner_out["result"] == READY
    assert waiter_out["result"] == READY
    assert runs == ["run"]
    assert [f[1] for f in flights.finished] == ["done", "done"]
    assert flights.active() == ()
