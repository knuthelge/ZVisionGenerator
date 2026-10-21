"""Tests for the single-flight download registry."""

from __future__ import annotations

import threading

from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget

_TIMEOUT = 5.0


def _target(repo: str = "org/model", **kwargs) -> DownloadTarget:
    return DownloadTarget(repo, **kwargs)


def test_first_claim_owns_second_joins():
    flights = DownloadFlights()
    first, first_owner = flights.claim(_target())
    second, second_owner = flights.claim(_target(allow_patterns=("*.json",)))
    assert first_owner is True
    assert second_owner is False
    assert second is first
    assert first.status == "running"


def test_different_keys_are_independent_flights():
    flights = DownloadFlights()
    a, a_owner = flights.claim(_target())
    b, b_owner = flights.claim(_target(revision="dev"))
    c, c_owner = flights.claim(_target(local_dir="/tmp/x"))
    assert a_owner and b_owner and c_owner
    assert len({id(a), id(b), id(c)}) == 3
    assert len(flights.active()) == 3


def test_main_revision_equals_none_revision():
    flights = DownloadFlights()
    flights.claim(_target())
    _, owner = flights.claim(_target(revision="main"))
    assert owner is False


def test_two_threads_claiming_one_key_yield_one_owner():
    flights = DownloadFlights()
    barrier = threading.Barrier(8)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait(_TIMEOUT)
        _, owner = flights.claim(_target())
        with lock:
            results.append(owner)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_TIMEOUT)
    assert sorted(results) == [False] * 7 + [True]
    assert len(flights.active()) == 1


def test_wait_times_out_while_running_and_returns_after_finish():
    flights = DownloadFlights()
    flight, _ = flights.claim(_target())
    assert flight.wait(0.01) is False
    flights.finish(flight, "done")
    assert flight.wait(0) is True
    assert flight.status == "done"


def test_waiter_thread_wakes_on_finish_and_sees_message():
    flights = DownloadFlights()
    flight, _ = flights.claim(_target())
    seen: list[tuple[bool, str, str | None]] = []

    def waiter() -> None:
        woke = flight.wait(_TIMEOUT)
        seen.append((woke, flight.status, flight.message))

    thread = threading.Thread(target=waiter)
    thread.start()
    flights.finish(flight, "failed", "boom")
    thread.join(_TIMEOUT)
    assert seen == [(True, "failed", "boom")]


def test_progress_is_none_then_visible_to_waiters():
    flights = DownloadFlights()
    owner_flight, _ = flights.claim(_target())
    waiter_flight, _ = flights.claim(_target())
    assert waiter_flight.progress is None
    progress = DownloadProgress(10, 100, 5.0, 18.0)
    owner_flight.update(progress)
    assert waiter_flight.progress is progress
    newer = DownloadProgress(50, 100, 5.0, 10.0)
    owner_flight.update(newer)
    assert waiter_flight.progress is newer


def test_finish_unregisters_and_next_claim_is_new_owner():
    flights = DownloadFlights()
    flight, _ = flights.claim(_target())
    flights.finish(flight, "stopped")
    assert flights.active() == ()
    again, owner = flights.claim(_target())
    assert owner is True
    assert again is not flight
    assert again.status == "running"
    assert flight.status == "stopped"


def test_finish_of_stale_flight_keeps_newer_registration():
    flights = DownloadFlights()
    old, _ = flights.claim(_target())
    flights.finish(old, "done")
    new, _ = flights.claim(_target())
    flights.finish(old, "done")
    assert flights.active() == (new.target,)
    _, owner = flights.claim(_target())
    assert owner is False


def test_waiter_with_different_patterns_becomes_owner_without_spinning():
    flights = DownloadFlights()
    owner_flight, owner = flights.claim(_target(allow_patterns=("a/*",)))
    assert owner is True
    waiter_target = _target(allow_patterns=("b/*",))
    claims: list[bool] = []
    claim_calls = 0
    first_claim_done = threading.Event()
    result: dict[str, object] = {}

    def waiter() -> None:
        nonlocal claim_calls
        while True:
            flight, is_owner = flights.claim(waiter_target)
            claim_calls += 1
            claims.append(is_owner)
            first_claim_done.set()
            if is_owner:
                result["flight"] = flight
                return
            assert flight.wait(_TIMEOUT)

    thread = threading.Thread(target=waiter)
    thread.start()
    assert first_claim_done.wait(_TIMEOUT)
    # The waiter is now blocked in wait(); it must not have claimed again.
    thread.join(0.05)
    assert thread.is_alive()
    assert claim_calls == 1
    flights.finish(owner_flight, "done")
    thread.join(_TIMEOUT)
    assert not thread.is_alive()
    assert claims == [False, True]
    assert claim_calls == 2
    assert result["flight"] is not owner_flight
    assert flights.active() == (waiter_target,)


def test_active_lists_running_targets_only():
    flights = DownloadFlights()
    a, _ = flights.claim(_target("org/a"))
    b, _ = flights.claim(_target("org/b"))
    assert set(flights.active()) == {a.target, b.target}
    flights.finish(a, "done")
    assert flights.active() == (b.target,)


def test_conflict_none_without_flights_or_matches():
    flights = DownloadFlights()
    assert flights.conflict(repo_ids=["org/model"], paths=["/x"], any_cache=True) is None
    flights.claim(_target("org/other", local_dir="/data/other"))
    assert flights.conflict(repo_ids=["org/model"], paths=["/data/else"]) is None
    assert flights.conflict() is None
    assert flights.conflict(any_cache=True) is None


def test_conflict_by_repo_id_case_insensitive():
    flights = DownloadFlights()
    flights.claim(_target("Org/Model"))
    assert flights.conflict(repo_ids=["org/model"]) == "Org/Model"
    assert flights.conflict(repo_ids=("org/model", "x/y")) == "Org/Model"
    assert flights.conflict(repo_ids=["org/model-2"]) is None


def test_conflict_by_local_dir_equal_or_inside():
    flights = DownloadFlights()
    flights.claim(_target("org/m", local_dir="/data/models/m/.m.download"))
    assert flights.conflict(paths=["/data/models/m/.m.download"]) == "org/m"
    assert flights.conflict(paths=["/data/models/m"]) == "org/m"
    assert flights.conflict(paths=["/data/models"]) == "org/m"
    assert flights.conflict(paths=["/data/models/m/.m.download/sub"]) is None
    assert flights.conflict(paths=["/data/models/m2"]) is None
    assert flights.conflict(paths=["/data/model"]) is None


def test_conflict_path_accepts_pathlike_and_trailing_separator():
    from pathlib import Path

    flights = DownloadFlights()
    flights.claim(_target("org/m", local_dir="/data/models/m/"))
    assert flights.conflict(paths=[Path("/data/models")]) == "org/m"
    assert flights.conflict(paths=["/data/models/"]) == "org/m"


def test_conflict_any_cache_ignores_local_dir_flights():
    flights = DownloadFlights()
    flights.claim(_target("org/local", local_dir="/data/x"))
    assert flights.conflict(any_cache=True) is None
    flights.claim(_target("org/cached"))
    assert flights.conflict(any_cache=True) == "org/cached"
    assert flights.conflict(any_cache=False) is None


def test_conflict_clears_after_finish():
    flights = DownloadFlights()
    flight, _ = flights.claim(_target("org/m"))
    assert flights.conflict(repo_ids=["org/m"]) == "org/m"
    flights.finish(flight, "done")
    assert flights.conflict(repo_ids=["org/m"]) is None
