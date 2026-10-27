"""Tests for the Models-page download lane: ordering, dedupe, cancel, outcomes, retention and shutdown."""

from __future__ import annotations

import threading

import pytest

from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, DownloadReport
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.web_runner import WebRunner

_TARGETS = (DownloadTarget("owner/repo"),)
_WAIT = 5.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeEnsure:
    """A scripted ``ensure_downloaded``: each call records itself and waits for the test to release it."""

    def __init__(self, outcome: str = READY) -> None:
        self.outcome = outcome
        self.calls: list[tuple[DownloadTarget, ...]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.started = threading.Semaphore(0)
        self.release = threading.Event()
        self.reports: list[DownloadReport] = []
        self.cancelled_seen = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, targets, *, flights, report, cancelled, **_):
        with self._lock:
            self.calls.append(tuple(targets))
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.release()
        try:
            for item in self.reports:
                report(item)
            while not self.release.wait(0.01):
                if cancelled():
                    self.cancelled_seen.set()
                    return STOPPED
            return self.outcome
        finally:
            with self._lock:
                self.concurrent -= 1


def _targets_for(name: str):
    return lambda: (DownloadTarget(f"owner/{name}"),)


def _lane(ensure, **kwargs) -> DownloadLane:
    return DownloadLane(flights=DownloadFlights(), ensure=ensure, **kwargs)


def _wait_state(lane: DownloadLane, item_id: str, state: str) -> dict:
    deadline = threading.Event()
    for _ in range(int(_WAIT / 0.01)):
        item = next((entry for entry in lane.snapshot()["items"] if entry["id"] == item_id), None)
        if item is not None and item["state"] == state:
            return item
        deadline.wait(0.01)
    raise AssertionError(f"item {item_id} never reached {state}: {lane.snapshot()}")


def _submit(lane: DownloadLane, name: str, *, kind: str = "model", plan=None) -> dict:
    return lane.submit(kind=kind, name=name, label=name.upper(), plan=plan or _targets_for(name))


@pytest.fixture
def ensure():
    fake = _FakeEnsure()
    yield fake
    fake.release.set()


class TestOrderAndQueue:
    def test_items_run_one_at_a_time_in_submit_order(self, ensure):
        lane = _lane(ensure)
        first = _submit(lane, "a")
        second = _submit(lane, "b")
        third = _submit(lane, "c", kind="enhancer")

        assert ensure.started.acquire(timeout=_WAIT)
        assert [item["id"] for item in lane.snapshot()["items"]] == [first["id"], second["id"], third["id"]]
        ensure.release.set()
        _wait_state(lane, third["id"], "done")

        assert [call[0].repo_id for call in ensure.calls] == ["owner/a", "owner/b", "owner/c"]
        assert ensure.max_concurrent == 1

    def test_queued_items_report_their_position_and_the_running_one_does_not(self, ensure):
        lane = _lane(ensure)
        first = _submit(lane, "a")
        second = _submit(lane, "b")
        third = _submit(lane, "c")
        assert ensure.started.acquire(timeout=_WAIT)

        items = {item["id"]: item for item in lane.snapshot()["items"]}

        assert items[first["id"]]["state"] == "running"
        assert "queue_position" not in items[first["id"]]
        assert (items[second["id"]]["state"], items[second["id"]]["queue_position"]) == ("queued", 1)
        assert items[third["id"]]["queue_position"] == 2

    def test_submitting_a_row_that_is_queued_or_running_returns_the_same_item(self, ensure):
        lane = _lane(ensure)
        running = _submit(lane, "a")
        queued = _submit(lane, "b")
        assert ensure.started.acquire(timeout=_WAIT)

        assert _submit(lane, "a")["id"] == running["id"]
        assert _submit(lane, "b")["id"] == queued["id"]
        assert _submit(lane, "a", kind="enhancer")["id"] not in (running["id"], queued["id"])
        assert len(lane.snapshot()["items"]) == 3

    def test_a_finished_row_can_be_submitted_again_as_a_new_item(self, ensure):
        ensure.release.set()
        lane = _lane(ensure)
        first = _submit(lane, "a")
        _wait_state(lane, first["id"], "done")

        second = _submit(lane, "a")

        assert second["id"] != first["id"]


class TestCancel:
    def test_cancelling_a_queued_item_cancels_it_at_once_and_never_runs_it(self, ensure):
        lane = _lane(ensure)
        _submit(lane, "a")
        queued = _submit(lane, "b")
        assert ensure.started.acquire(timeout=_WAIT)

        cancelled = lane.cancel(queued["id"])
        ensure.release.set()

        assert cancelled["state"] == "cancelled"
        assert "queue_position" not in cancelled
        _wait_state(lane, queued["id"], "cancelled")
        assert [call[0].repo_id for call in ensure.calls] == ["owner/a"]

    def test_cancelling_the_running_item_stops_it_and_the_next_one_starts(self, ensure):
        lane = _lane(ensure)
        running = _submit(lane, "a")
        waiting = _submit(lane, "b")
        assert ensure.started.acquire(timeout=_WAIT)

        lane.cancel(running["id"])

        assert ensure.cancelled_seen.wait(_WAIT)
        _wait_state(lane, running["id"], "cancelled")
        assert ensure.started.acquire(timeout=_WAIT)
        ensure.release.set()
        _wait_state(lane, waiting["id"], "done")

    def test_cancelling_a_finished_item_leaves_it_unchanged(self, ensure):
        ensure.release.set()
        lane = _lane(ensure)
        item = _submit(lane, "a")
        _wait_state(lane, item["id"], "done")

        assert lane.cancel(item["id"])["state"] == "done"

    def test_cancelling_an_unknown_item_raises_key_error(self, ensure):
        with pytest.raises(KeyError):
            _lane(ensure).cancel("dl-999")


class TestOutcomes:
    @pytest.mark.parametrize(("outcome", "state"), [(READY, "done"), (STOPPED, "cancelled"), (DETACHED, "cancelled"), (UNCHECKED, "failed")])
    def test_ensure_outcomes_map_to_states(self, outcome, state):
        fake = _FakeEnsure(outcome)
        fake.release.set()
        lane = _lane(fake)
        item = _submit(lane, "a")

        final = _wait_state(lane, item["id"], state)

        if state == "failed":
            assert final["message"] is not None
        else:
            assert final["message"] is None

    def test_a_plan_that_cannot_be_made_fails_without_running_ensure(self, ensure):
        lane = _lane(ensure)
        item = _submit(lane, "a", plan=lambda: None)

        final = _wait_state(lane, item["id"], "failed")

        assert final["message"] is not None
        assert ensure.calls == []

    def test_a_runtime_error_fails_the_item_with_its_message_and_the_lane_goes_on(self):
        calls = []

        def _ensure(targets, **_):
            calls.append(targets)
            if len(calls) == 1:
                raise RuntimeError("Download of owner/a failed after 1 KB of 2 KB: disk full")
            return READY

        lane = _lane(_ensure)
        failing = _submit(lane, "a")
        following = _submit(lane, "b")

        assert _wait_state(lane, failing["id"], "failed")["message"] == "Download of owner/a failed after 1 KB of 2 KB: disk full"
        _wait_state(lane, following["id"], "done")

    def test_a_plan_that_raises_fails_the_item(self, ensure):
        def _plan():
            raise ValueError("bad config")

        lane = _lane(ensure)
        item = _submit(lane, "a", plan=_plan)

        assert _wait_state(lane, item["id"], "failed")["message"] == "bad config"


class TestProgress:
    def test_reports_show_up_as_progress_fields(self, ensure):
        ensure.reports = [DownloadReport("owner/a", 2, 3, DownloadProgress(10, 100, 5.0, 18.0), shared=True)]
        lane = _lane(ensure)
        item = _submit(lane, "a")
        assert "repo" not in item
        assert ensure.started.acquire(timeout=_WAIT)

        (running,) = lane.snapshot()["items"]

        assert {key: running[key] for key in ("repo", "part", "parts", "bytes_done", "bytes_total", "bytes_per_sec", "download_eta_secs", "shared")} == {
            "repo": "owner/a",
            "part": 2,
            "parts": 3,
            "bytes_done": 10,
            "bytes_total": 100,
            "bytes_per_sec": 5.0,
            "download_eta_secs": 18.0,
            "shared": True,
        }

    def test_an_item_keeps_its_last_progress_after_it_finishes(self, ensure):
        ensure.reports = [DownloadReport("owner/a", 1, 1, DownloadProgress(100, 100, None, 0.0))]
        ensure.release.set()
        lane = _lane(ensure)
        item = _submit(lane, "a")

        final = _wait_state(lane, item["id"], "done")

        assert (final["bytes_done"], final["bytes_total"], final["shared"]) == (100, 100, False)


class TestRetention:
    def test_finished_items_stay_for_the_retention_and_are_then_dropped(self, ensure):
        ensure.release.set()
        clock = _Clock()
        lane = _lane(ensure, retention_secs=60.0, clock=clock)
        item = _submit(lane, "a")
        _wait_state(lane, item["id"], "done")

        clock.now = 59.0
        assert [entry["id"] for entry in lane.snapshot()["items"]] == [item["id"]]
        clock.now = 61.0

        assert lane.snapshot() == {"items": []}
        with pytest.raises(KeyError):
            lane.cancel(item["id"])

    def test_only_the_newest_finished_items_are_kept(self, ensure):
        ensure.release.set()
        clock = _Clock()
        lane = _lane(ensure, max_finished=2, clock=clock)
        ids = []
        for index, name in enumerate("abc"):
            clock.now = float(index)
            item = _submit(lane, name)
            _wait_state(lane, item["id"], "done")
            ids.append(item["id"])

        assert [entry["id"] for entry in lane.snapshot()["items"]] == [ids[2], ids[1]]


class TestShutdown:
    def test_shutdown_cancels_queued_items_signals_the_running_one_and_refuses_new_submits(self, ensure):
        lane = _lane(ensure)
        running = _submit(lane, "a")
        queued = _submit(lane, "b")
        assert ensure.started.acquire(timeout=_WAIT)

        lane.shutdown(join_timeout=_WAIT)

        states = {item["id"]: item["state"] for item in lane.snapshot()["items"]}
        assert states == {running["id"]: "cancelled", queued["id"]: "cancelled"}
        assert [call[0].repo_id for call in ensure.calls] == ["owner/a"]
        with pytest.raises(RuntimeError):
            _submit(lane, "c")

    def test_shutdown_of_an_idle_lane_is_harmless(self, ensure):
        lane = _lane(ensure)
        lane.shutdown()

        with pytest.raises(RuntimeError):
            _submit(lane, "a")


def test_a_running_lane_item_leaves_generation_jobs_admitted(ensure):
    flights = DownloadFlights()
    lane = DownloadLane(flights=flights, ensure=ensure)
    runner = WebRunner(download_flights=flights)
    try:
        item = _submit(lane, "a")
        assert ensure.started.acquire(timeout=_WAIT)

        assert runner.has_active_jobs() is False
        assert runner.get_active_exclusive_job_snapshot() is None
        job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.0)

        assert job_id
        assert lane.snapshot()["items"][0]["id"] == item["id"]
        assert lane.snapshot()["items"][0]["state"] == "running"
    finally:
        ensure.release.set()
        runner.shutdown()
        lane.shutdown()


class TestSharedFlightAndConcurrency:
    def test_an_item_waiting_on_a_jobs_flight_mirrors_it_and_cancel_detaches_it(self):
        """The real ensure: the lane item waits on the flight a job owns, mirrors its progress and detaches on cancel."""
        flights = DownloadFlights()
        target = DownloadTarget("lane-test/shared-repo")
        flight, owner = flights.claim(target)
        assert owner
        flight.update(DownloadProgress(bytes_done=10, bytes_total=100, bytes_per_sec=5.0, eta_secs=18.0))
        lane = DownloadLane(flights=flights)
        try:
            item = lane.submit(kind="model", name="m", label="M", plan=lambda: (target,))
            mirrored = None
            for _ in range(int(_WAIT / 0.01)):
                mirrored = next((entry for entry in lane.snapshot()["items"] if entry["id"] == item["id"] and "shared" in entry), None)
                if mirrored is not None:
                    break
                threading.Event().wait(0.01)
            assert mirrored is not None and mirrored["shared"] is True and mirrored["bytes_done"] == 10
            assert mirrored["state"] == "running"

            lane.cancel(item["id"])

            finished = _wait_state(lane, item["id"], "cancelled")
            assert finished["message"] is None
            assert flights.active() == (target,)  # the job's flight is untouched
        finally:
            flights.finish(flight, "stopped")
            lane.shutdown(join_timeout=_WAIT)

    def test_concurrent_submits_cancels_and_snapshots_keep_one_running_item_and_consistent_positions(self, ensure):
        lane = _lane(ensure)
        errors: list[BaseException] = []

        def _hammer(prefix: str) -> None:
            try:
                for index in range(15):
                    item = _submit(lane, f"{prefix}{index}")
                    for entry in lane.snapshot()["items"]:
                        if entry["state"] == "queued":
                            assert entry["queue_position"] >= 1
                    if index % 2:
                        lane.cancel(item["id"])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_hammer, args=(f"t{n}-",)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(_WAIT)
        ensure.release.set()
        lane.shutdown(join_timeout=_WAIT)

        assert errors == []
        assert ensure.max_concurrent == 1
        queued = [entry for entry in lane.snapshot()["items"] if entry["state"] in ("queued", "running")]
        assert queued == []
