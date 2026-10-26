"""Tests for web jobs running the download ensure step and publishing ``model_downloading``."""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import _make_args, _make_video_args
from tests.test_downloads_child import _FakeProcess, _Launcher
from tests.test_web_runner import _sse_payload, _stub_request_job, _wait_for_status
from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.downloads import DETACHED, READY, STOPPED, UNCHECKED, DownloadFlights, DownloadProgress, DownloadReport, DownloadTarget, TargetCheck, ensure_downloaded
from zvisiongenerator.downloads.child import ChildResult, run_in_child
from zvisiongenerator.preflight import run_preflight as _real_run_preflight
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_contract import DOWNLOADING_EVENT

_REAL_ENSURE = ensure_downloaded
_TARGET = DownloadTarget("owner/repo", allow_patterns=("*.safetensors",))
_PROMPTS = {"default": [("prompt", None)]}


@pytest.fixture(autouse=True)
def _no_real_downloads(monkeypatch):
    """Never start a download; conftest plans none, and tests install the planners and the ensure step they need."""
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
    monkeypatch.setattr(web_runner_module, "ensure_downloaded", MagicMock(side_effect=AssertionError("unexpected ensure step")))


def _fake_ensure(monkeypatch, *, check=lambda _t: TargetCheck(1, 1000), run, present=lambda _t: False) -> None:
    """Run the real ensure step with a fake dry-run, child runner and presence test, reporting without throttling."""
    monkeypatch.setattr(web_runner_module, "ensure_downloaded", functools.partial(_REAL_ENSURE, present=present, check=check, run=run, interval_secs=0.0, wait_secs=0.01))


def _child_that_reports(steps: int):
    """Return a child runner that reports *steps* byte counts of a 1000-byte download and finishes."""

    def run(_target, *, on_counts, cancelled):
        for step in range(1, steps + 1):
            on_counts(step * 1000 // steps, 0)
        return ChildResult("done")

    return run


def _runner(**kwargs) -> web_runner_module.WebRunner:
    return web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, **kwargs)


def _image_request(**overrides) -> ImageGenerationRequest:
    return ImageGenerationRequest(backend=None, model=None, prompt="prompt", model_family="zimage", **overrides)


def _submit_image(monkeypatch, runner, *, request=None, config=None, run_batch=None) -> str:
    _stub_request_job(monkeypatch)
    monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(name="model"), MagicMock(name="info")))
    monkeypatch.setattr(web_runner_module, "run_batch", run_batch or _batch_that_completes)
    return runner.submit_image_request_job(request=request or _image_request(model_name="atlas"), prompts_data=_PROMPTS, config=config or {}, args=_make_args(), model_ref="owner/atlas")


def _batch_that_completes(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
    progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})


def _history(runner, job_id) -> list[dict]:
    with runner._jobs_lock:
        record = runner._jobs[job_id]
    with record.lock:
        return [dict(event) for event in record.history]


def _types(runner, job_id) -> list[str]:
    return [event["type"] for event in _history(runner, job_id)]


def _wait_for_event(runner, job_id, event_type, *, timeout=2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event_type in _types(runner, job_id):
            return
        time.sleep(0.01)
    raise AssertionError(f"{event_type} was not published")


def _downloading(index: int) -> dict:
    return {
        "type": DOWNLOADING_EVENT,
        "mode": "image",
        "role": "model",
        "model": "m",
        "repo": "o/r",
        "part": 1,
        "parts": 1,
        "bytes_done": index,
        "bytes_total": 100,
        "bytes_per_sec": None,
        "download_eta_secs": 9.0,
        "shared": False,
    }


class TestJobFlow:
    def test_an_image_job_downloads_before_it_loads_and_history_keeps_the_latest_progress(self, monkeypatch):
        """SC-2: model_downloading comes before model_loading, and the history keeps only the latest, in event_id order."""
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        _fake_ensure(monkeypatch, run=_child_that_reports(3))
        runner = _runner()
        try:
            job_id = _submit_image(monkeypatch, runner)
            assert _wait_for_status(runner, job_id, "completed")["status"] == "completed"
            history = _history(runner, job_id)
        finally:
            runner.shutdown()

        types = [event["type"] for event in history]
        assert types.count(DOWNLOADING_EVENT) == 1
        assert types.index(DOWNLOADING_EVENT) < types.index("model_loading")
        ids = [event["event_id"] for event in history]
        assert ids == sorted(ids)
        latest = history[types.index(DOWNLOADING_EVENT)]
        assert latest["bytes_done"] == latest["bytes_total"] == 1000  # the owned part's final report is the one kept

    def test_the_download_events_carry_the_contract_fields_and_no_generation_eta(self, monkeypatch):
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        seen: list[dict] = []
        _fake_ensure(monkeypatch, run=_child_that_reports(2))
        runner = _runner()
        original = runner._publish_event
        runner._publish_event = lambda job_id, event: (seen.append(event) if event["type"] == DOWNLOADING_EVENT else None, original(job_id, event))[1]
        try:
            job_id = _submit_image(monkeypatch, runner)
            _wait_for_status(runner, job_id, "completed")
            kept = [event for event in _history(runner, job_id) if event["type"] == DOWNLOADING_EVENT]
        finally:
            runner.shutdown()

        assert len(seen) >= 2
        assert set(seen[0]) == {"type", "mode", "role", "model", "repo", "part", "parts", "bytes_done", "bytes_total", "bytes_per_sec", "download_eta_secs", "shared"}
        assert seen[0]["mode"] == "image"
        assert seen[0]["role"] == "model"
        assert seen[0]["model"] == "atlas"
        assert seen[0]["repo"] == "owner/repo"
        assert (seen[0]["part"], seen[0]["parts"]) == (1, 1)
        assert seen[0]["shared"] is False
        assert "eta_secs" not in kept[0]

    def test_the_download_eta_never_becomes_the_generation_eta(self):
        runner = _runner()
        try:
            job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.0)
            _wait_for_status(runner, job_id, "completed")
            runner._publish_event(job_id, _downloading(10))
            runner._publish_event(job_id, {"type": "model_loading", "mode": "image", "model": "m"})
            events = _history(runner, job_id)
        finally:
            runner.shutdown()

        downloading = next(event for event in events if event["type"] == DOWNLOADING_EVENT)
        loading = next(event for event in events if event["type"] == "model_loading")
        assert downloading["download_eta_secs"] == 9.0
        assert "eta_secs" not in downloading
        assert "eta_secs" not in loading

    def test_a_video_job_downloads_before_model_loading(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(web_runner_module, "video_plan", lambda *a, **kw: (calls.append((a, kw)), (_TARGET,))[1])
        _fake_ensure(monkeypatch, run=_child_that_reports(1))
        _stub_request_job(monkeypatch)
        monkeypatch.setattr(
            web_runner_module, "run_video_batch", lambda *, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})
        )
        runner = _runner()
        try:
            request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", model_name="ltx-q4", upscale="x2")
            job_id = runner.submit_video_request_job(request=request, prompts_data=_PROMPTS, config={"k": 1}, args=_make_video_args(), model_ref="owner/ltx")
            _wait_for_status(runner, job_id, "completed")
            types = _types(runner, job_id)
            first = next(event for event in _history(runner, job_id) if event["type"] == DOWNLOADING_EVENT)
        finally:
            runner.shutdown()

        assert types.index(DOWNLOADING_EVENT) < types.index("model_loading")
        assert first["mode"] == "video"
        assert first["model"] == "ltx-q4"
        assert calls == [(("owner/ltx",), {"family": "ltx", "config": {"k": 1}, "model_name": "ltx-q4", "upscale": True})]

    def test_an_unchecked_download_loads_the_model_without_download_events(self, monkeypatch):
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        _fake_ensure(monkeypatch, check=lambda _t: None, run=MagicMock(side_effect=AssertionError("no download")))
        runner = _runner()
        try:
            job_id = _submit_image(monkeypatch, runner)
            assert _wait_for_status(runner, job_id, "completed")["status"] == "completed"
            types = _types(runner, job_id)
        finally:
            runner.shutdown()

        assert DOWNLOADING_EVENT not in types
        assert "model_loading" in types

    def test_a_plan_that_cannot_be_made_skips_the_ensure_step(self, monkeypatch):
        """``None`` means the plan is unknown (offline): the ensure step is not called and the model loads as before."""
        runner = _runner()
        try:
            job_id = _submit_image(monkeypatch, runner)  # the autouse ensure raises when called
            assert _wait_for_status(runner, job_id, "completed")["status"] == "completed"
            types = _types(runner, job_id)
        finally:
            runner.shutdown()

        assert DOWNLOADING_EVENT not in types
        assert "model_loading" in types

    def test_a_failed_download_fails_the_job_with_the_repo_in_the_message(self, monkeypatch):
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        _fake_ensure(monkeypatch, run=lambda _target, **_kw: ChildResult("failed", "disk full"))
        runner = _runner()
        try:
            job_id = _submit_image(monkeypatch, runner)
            snapshot = _wait_for_status(runner, job_id, "failed")
            events = _history(runner, job_id)
        finally:
            runner.shutdown()

        assert snapshot["status"] == "failed"
        failure = events[-1]
        assert failure["type"] == "job_failed"
        assert "owner/repo" in failure["message"]
        assert "model_loading" not in [event["type"] for event in events]


_ENHANCER_TARGET = DownloadTarget("a/enhancer", allow_patterns=("*.safetensors",))
_ENHANCER_CONFIG = {"generation": {}, "prompt_enhancer": {"model": {"darwin": "a/enhancer", "win32": "a/enhancer", "linux": "a/enhancer"}}}


class _EnhancerHarness:
    """Run the real preflight with a fake enhancer session whose cache state the download changes."""

    def __init__(self, monkeypatch, *, cached: bool = False):
        self.cached = cached
        self.loaded: list[tuple[str, str | None]] = []
        self.planned: list[tuple[str, str | None]] = []
        enhancer = MagicMock(name="enhancer")
        enhancer.generate.side_effect = lambda *_a, **_kw: iter(["A vivid rewrite with soft light."])
        session = PromptEnhancerSession(
            lambda repo, revision: (self.loaded.append((repo, revision)), enhancer)[1],
            scheduler=lambda delay, callback: None,
            downloaded=lambda repo, revision: self.cached,
        )
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)
        monkeypatch.setattr(web_runner_module, "enhancer_plan", lambda repo, revision: (self.planned.append((repo, revision)), (_ENHANCER_TARGET,))[1])
        monkeypatch.setattr(web_runner_module, "run_preflight", lambda *a, **kw: _real_run_preflight(*a, session=session, release_memory=lambda: None, **kw))

    def child(self, steps: int = 2):
        """Return a child runner that finishes the download and fills the cache."""
        report = _child_that_reports(steps)

        def run(target, **kwargs):
            result = report(target, **kwargs)
            self.cached = True
            return result

        return run


def _submit_enhanced_image(monkeypatch, runner) -> str:
    _stub_request_job(monkeypatch, run_preflight=web_runner_module.run_preflight)
    monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(name="model"), MagicMock(name="info")))
    monkeypatch.setattr(web_runner_module, "run_batch", _batch_that_completes)
    request = _image_request(model_name="atlas")
    return runner.submit_image_request_job(
        request=request, prompts_data=_PROMPTS, config=_ENHANCER_CONFIG, args=_make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None), model_ref="owner/atlas"
    )


class TestEnhancerDownload:
    def test_the_enhancer_downloads_in_preflight_before_it_loads_and_then_reports_loading(self, monkeypatch):
        harness = _EnhancerHarness(monkeypatch)
        _fake_ensure(monkeypatch, run=harness.child(3))
        runner = _runner()
        try:
            job_id = _submit_enhanced_image(monkeypatch, runner)
            assert _wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
            history = _history(runner, job_id)
        finally:
            runner.shutdown()

        types = [event["type"] for event in history]
        download = history[types.index(DOWNLOADING_EVENT)]
        assert download["role"] == "enhancer"
        assert download["model"] == "a/enhancer"
        assert download["repo"] == "a/enhancer"
        assert download["bytes_done"] == download["bytes_total"] == 1000
        assert types.count(DOWNLOADING_EVENT) == 1
        # The card clears the enhancer download on enhancer_loading, which therefore follows it.
        assert types.index("preflight_started") < types.index(DOWNLOADING_EVENT) < types.index("enhancer_loading") < types.index("prompts_enhancing") < types.index("preflight_finished")
        assert [event["phase"] for event in history if event["type"] == "enhancer_loading"] == ["loading"]
        assert harness.planned == [("a/enhancer", None)]
        assert harness.loaded == [("a/enhancer", None)]

    def test_stopping_the_enhancer_download_ends_the_job_once_without_loading_anything(self, monkeypatch):
        harness = _EnhancerHarness(monkeypatch)
        process = _FakeProcess(running_polls=None, wait_timeouts=1)
        cleanup = MagicMock()
        _fake_ensure(monkeypatch, run=functools.partial(run_in_child, popen=_Launcher(process), poll_secs=0.01, grace_secs=0.01, cleanup=cleanup))
        flights = DownloadFlights()
        runner = _runner(download_flights=flights)
        try:
            job_id = _submit_enhanced_image(monkeypatch, runner)
            _wait_for_event(runner, job_id, DOWNLOADING_EVENT)
            runner.queue_job_control(job_id, "quit")
            snapshot = _wait_for_status(runner, job_id, "cancelled", timeout=3.0)
            events = _history(runner, job_id)
        finally:
            runner.shutdown()

        types = [event["type"] for event in events]
        assert snapshot["status"] == "cancelled"
        assert types.count("job_cancelled") == 1
        assert events[-1]["download"] == "stopped"
        assert events[-1]["total_iterations"] == 1
        assert events[next(i for i, t in enumerate(types) if t == DOWNLOADING_EVENT)]["role"] == "enhancer"
        cleanup.assert_called_once_with(_ENHANCER_TARGET)
        assert harness.loaded == []
        assert not {"enhancer_loading", "prompts_enhancing", "preflight_finished", "model_loading"} & set(types)

    def test_a_job_waiting_on_a_lane_download_of_the_enhancer_mirrors_it_and_detaches_on_stop(self, monkeypatch):
        harness = _EnhancerHarness(monkeypatch)
        _fake_ensure(monkeypatch, run=MagicMock(side_effect=AssertionError("the waiter must not download")))
        flights = DownloadFlights()
        owner_flight, owner = flights.claim(_ENHANCER_TARGET)
        assert owner
        owner_flight.update(DownloadProgress(bytes_done=300, bytes_total=1000, bytes_per_sec=50.0, eta_secs=14.0))
        runner = _runner(download_flights=flights)
        try:
            job_id = _submit_enhanced_image(monkeypatch, runner)
            _wait_for_event(runner, job_id, DOWNLOADING_EVENT)
            mirrored = next(event for event in _history(runner, job_id) if event["type"] == DOWNLOADING_EVENT)
            runner.queue_job_control(job_id, "quit")
            _wait_for_status(runner, job_id, "cancelled", timeout=3.0)
            events = _history(runner, job_id)
            still_active = flights.active()
        finally:
            flights.finish(owner_flight, "stopped")
            runner.shutdown()

        assert (mirrored["role"], mirrored["shared"], mirrored["bytes_done"]) == ("enhancer", True, 300)
        assert events[-1]["type"] == "job_cancelled"
        assert events[-1]["download"] == "detached"
        assert still_active == (_ENHANCER_TARGET,)
        assert harness.loaded == []

    def test_an_unchecked_download_keeps_the_text_downloading_phase(self, monkeypatch):
        harness = _EnhancerHarness(monkeypatch)
        _fake_ensure(monkeypatch, check=lambda _t: None, run=MagicMock(side_effect=AssertionError("no download")))
        runner = _runner()
        try:
            job_id = _submit_enhanced_image(monkeypatch, runner)
            assert _wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
            history = _history(runner, job_id)
        finally:
            runner.shutdown()

        assert DOWNLOADING_EVENT not in [event["type"] for event in history]
        assert [event["phase"] for event in history if event["type"] == "enhancer_loading"] == ["downloading"]
        assert harness.loaded == [("a/enhancer", None)]

    def test_the_video_hook_labels_the_events_with_the_mode_and_counts_the_job_iterations(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(web_runner_module, "enhancer_plan", lambda repo, revision: (_ENHANCER_TARGET,))
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, **_kw: STOPPED)
        runner = _runner()
        try:
            hook = runner._enhancer_download_hook(
                mode="video", prompts_data={"a": [("x", None), ("y", None)], "b": [("z", None)]}, args=_make_video_args(runs=2), progress_callback=seen.append, control_signal=SkipSignal()
            )
            result = hook("a/enhancer", None)
        finally:
            runner.shutdown()

        assert result is False
        assert seen == [{"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 6, "download": "stopped"}]

    def test_a_job_without_enhancement_never_plans_an_enhancer_download(self, monkeypatch):
        harness = _EnhancerHarness(monkeypatch)
        runner = _runner()
        try:
            _stub_request_job(monkeypatch, run_preflight=web_runner_module.run_preflight)
            monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
            monkeypatch.setattr(web_runner_module, "run_batch", _batch_that_completes)
            job_id = runner.submit_image_request_job(
                request=_image_request(model_name="atlas"), prompts_data=_PROMPTS, config=_ENHANCER_CONFIG, args=_make_args(enhance=None, no_enhance=True, enhance_model=None), model_ref="owner/atlas"
            )
            assert _wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
        finally:
            runner.shutdown()

        assert harness.planned == []


class TestStop:
    def test_stopping_the_owning_job_terminates_then_kills_the_child_cleans_up_and_cancels(self, monkeypatch):
        """SC-3: the owner's Stop ends the child (kill after the grace period), cleans up and reports download "stopped"."""
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        process = _FakeProcess(running_polls=None, wait_timeouts=1)
        cleanup = MagicMock()
        run = functools.partial(run_in_child, popen=_Launcher(process), poll_secs=0.01, grace_secs=0.01, cleanup=cleanup)
        _fake_ensure(monkeypatch, run=run)
        flights = DownloadFlights()
        runner = _runner(download_flights=flights)
        try:
            job_id = _submit_image(monkeypatch, runner)
            _wait_for_event(runner, job_id, DOWNLOADING_EVENT)
            runner.queue_job_control(job_id, "quit")
            snapshot = _wait_for_status(runner, job_id, "cancelled")
            events = _history(runner, job_id)
        finally:
            runner.shutdown()

        assert snapshot["status"] == "cancelled"
        assert events[-1]["type"] == "job_cancelled"
        assert events[-1]["download"] == "stopped"
        assert events[-1]["completed_iterations"] == 0
        assert "terminate" in process.calls and process.calls.index("terminate") < process.calls.index("kill")
        cleanup.assert_called_once_with(_TARGET)
        assert flights.active() == ()
        assert "model_loading" not in [event["type"] for event in events]

    def test_stopping_a_job_that_waits_on_another_download_detaches_and_leaves_that_download_running(self, monkeypatch):
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        _fake_ensure(monkeypatch, run=MagicMock(side_effect=AssertionError("the waiter must not download")))
        flights = DownloadFlights()
        owner_flight, owner = flights.claim(_TARGET)
        assert owner
        owner_flight.update(DownloadProgress(bytes_done=400, bytes_total=1000, bytes_per_sec=50.0, eta_secs=12.0))
        runner = _runner(download_flights=flights)
        try:
            job_id = _submit_image(monkeypatch, runner)
            _wait_for_event(runner, job_id, DOWNLOADING_EVENT)
            mirrored = next(event for event in _history(runner, job_id) if event["type"] == DOWNLOADING_EVENT)
            runner.queue_job_control(job_id, "quit")
            _wait_for_status(runner, job_id, "cancelled")
            events = _history(runner, job_id)
            still_active = flights.active()
        finally:
            flights.finish(owner_flight, "stopped")
            runner.shutdown()

        assert mirrored["shared"] is True
        assert (mirrored["bytes_done"], mirrored["bytes_total"], mirrored["download_eta_secs"]) == (400, 1000, 12.0)
        assert events[-1]["type"] == "job_cancelled"
        assert events[-1]["download"] == "detached"
        assert still_active == (_TARGET,)
        assert owner_flight.status == "stopped"  # only the test's own finish ended it

    def test_the_quit_is_polled_without_being_consumed(self, monkeypatch):
        runner = _runner()
        signal = SkipSignal()
        polled: list[bool] = []

        def ensure(targets, *, flights, report, cancelled):
            signal.queue_action("quit")
            polled.append(cancelled())
            polled.append(cancelled())
            return STOPPED

        events: list[dict] = []
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", ensure)
        try:
            result = runner._ensure_downloads((_TARGET,), mode="image", role="model", model="m", total_iterations=3, progress_callback=events.append, control_signal=signal)
        finally:
            runner.shutdown()

        assert result is False
        assert polled == [True, True]
        assert signal.pending() == "quit"
        assert events == [{"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 3, "download": "stopped"}]


class TestEnsureDownloads:
    def test_no_targets_skip_the_ensure_step(self):
        runner = _runner()
        try:
            for targets in (None, ()):
                assert runner._ensure_downloads(targets, mode="image", role="model", model="m", total_iterations=1, progress_callback=MagicMock(), control_signal=SkipSignal()) is True
        finally:
            runner.shutdown()

    @pytest.mark.parametrize(("outcome", "ok", "download"), [(READY, True, None), (UNCHECKED, True, None), (STOPPED, False, "stopped"), (DETACHED, False, "detached")])
    def test_the_outcome_decides_whether_the_job_goes_on(self, monkeypatch, outcome, ok, download):
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, **_kw: outcome)
        events: list[dict] = []
        runner = _runner()
        try:
            result = runner._ensure_downloads((_TARGET,), mode="video", role="model", model="m", total_iterations=4, progress_callback=events.append, control_signal=SkipSignal())
        finally:
            runner.shutdown()

        assert result is ok
        assert [event.get("download") for event in events] == ([] if ok else [download])
        if not ok:
            assert events[0]["type"] == "batch_cancelled"
            assert events[0]["mode"] == "video"
            assert events[0]["total_iterations"] == 4

    def test_a_report_becomes_a_model_downloading_event(self, monkeypatch):
        def ensure(targets, *, flights, report, cancelled):
            report(DownloadReport("o/r", 2, 3, DownloadProgress(10, 100, 5.0, 18.0), shared=True))
            return READY

        monkeypatch.setattr(web_runner_module, "ensure_downloaded", ensure)
        events: list[dict] = []
        runner = _runner()
        try:
            runner._ensure_downloads((_TARGET,), mode="video", role="enhancer", model="label", total_iterations=1, progress_callback=events.append, control_signal=SkipSignal())
        finally:
            runner.shutdown()

        assert events == [
            {
                "type": DOWNLOADING_EVENT,
                "mode": "video",
                "role": "enhancer",
                "model": "label",
                "repo": "o/r",
                "part": 2,
                "parts": 3,
                "bytes_done": 10,
                "bytes_total": 100,
                "bytes_per_sec": 5.0,
                "download_eta_secs": 18.0,
                "shared": True,
            }
        ]

    def test_the_runner_shares_the_flights_it_is_given(self, monkeypatch):
        flights = DownloadFlights()
        seen: list = []
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, flights, **_kw: (seen.append(flights), READY)[1])
        runner = _runner(download_flights=flights)
        try:
            runner._ensure_downloads((_TARGET,), mode="image", role="model", model="m", total_iterations=1, progress_callback=MagicMock(), control_signal=SkipSignal())
            assert runner.download_flights is flights
            assert isinstance(_runner().download_flights, DownloadFlights)
        finally:
            runner.shutdown()

        assert seen == [flights]


class TestPlannerArguments:
    def test_the_image_job_plans_with_its_family_config_and_platform(self, monkeypatch):
        plan = MagicMock(return_value=None)
        monkeypatch.setattr(web_runner_module, "image_plan", plan)
        runner = _runner()
        try:
            job_id = _submit_image(monkeypatch, runner, config={"generation": {}})
            _wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        plan.assert_called_once_with("owner/atlas", family="zimage", config={"generation": {}}, platform_key=sys.platform)

    def test_the_creative_upscale_plans_the_image_model_with_the_app_config(self, monkeypatch, tmp_path):
        plan = MagicMock(return_value=None)
        monkeypatch.setattr(web_runner_module, "image_plan", plan)
        _stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
        monkeypatch.setattr(
            web_runner_module, "run_upscale", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
        )
        runner = _runner()
        try:
            request = ImageGenerationRequest(backend=None, model=None, prompt="p", model_family="flux2_klein", model_name="atlas", upscale_factor=2, upscale_source="/out/a.png")
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="/models/atlas", config={"generation": {}}, context={"output_dir": str(tmp_path)})
            _wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        plan.assert_called_once_with("/models/atlas", family="flux2_klein", config={"generation": {}}, platform_key=sys.platform)

    def test_the_creative_upscale_builds_the_model_before_loading_it(self, monkeypatch, tmp_path):
        """A model built on first use (MOSS-Image, Ideogram 4 Instant) is built from the app config on the upscale path too."""
        calls: list = []
        config = {"moss_image": {}}
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: None)
        monkeypatch.setattr(web_runner_module, "ensure_built_model", lambda model_ref, cfg, **_kw: calls.append(("built", model_ref, cfg)) or True)
        _stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: calls.append(("load",)) or (MagicMock(), MagicMock()))
        monkeypatch.setattr(
            web_runner_module, "run_upscale", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
        )
        runner = _runner()
        try:
            request = ImageGenerationRequest(backend=None, model=None, prompt="p", model_family="zimage", model_name="moss", upscale_factor=2, upscale_source="/out/a.png")
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="/models/moss", config=config, context={"output_dir": str(tmp_path)})
            _wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        assert calls == [("built", "/models/moss", config), ("load",)]

    def test_the_seedvr2_upscale_plans_its_tier_files_and_labels_the_download_with_the_tier(self, monkeypatch, tmp_path):
        upscaler = SimpleNamespace(weights=FP8_WEIGHTS)
        plan = MagicMock(return_value=(_TARGET,))
        monkeypatch.setattr(web_runner_module, "seedvr2_plan", plan)
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: upscaler)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        _fake_ensure(monkeypatch, run=_child_that_reports(1))
        _stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_seedvr2", lambda _engine, _name, *, on_phase, **_kw: (on_phase("loading"), MagicMock())[1])
        monkeypatch.setattr(
            web_runner_module, "run_upscale", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
        )
        runner = _runner()
        try:
            request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_factor=2, upscale_source="/out/a.png", upscale_method="seedvr2", upscale_model="seedvr2-7b")
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="numz/SeedVR2_comfyUI", config={}, context={"output_dir": str(tmp_path)})
            _wait_for_status(runner, job_id, "completed")
            events = _history(runner, job_id)
        finally:
            runner.shutdown()

        plan.assert_called_once_with("seedvr2-7b", models_dir=tmp_path / "models", weights=FP8_WEIGHTS)
        downloading = next(event for event in events if event["type"] == DOWNLOADING_EVENT)
        assert downloading["model"] == "SeedVR2 7B fp8"
        assert [event["type"] for event in events].index(DOWNLOADING_EVENT) < [event["type"] for event in events].index("model_loading")

    def test_an_upscale_stopped_while_downloading_reports_one_iteration(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_runner_module, "image_plan", lambda *_a, **_kw: (_TARGET,))
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", lambda *_a, **_kw: STOPPED)
        _stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_image_model", MagicMock(side_effect=AssertionError("no load after a stopped download")))
        runner = _runner()
        try:
            request = ImageGenerationRequest(backend=None, model=None, prompt="p", model_family="zimage", upscale_factor=2, upscale_source="/out/a.png")
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="/models/atlas", config={}, context={"output_dir": str(tmp_path)})
            _wait_for_status(runner, job_id, "cancelled")
            last = _history(runner, job_id)[-1]
        finally:
            runner.shutdown()

        assert last["type"] == "job_cancelled"
        assert (last["total_iterations"], last["download"]) == (1, "stopped")


class TestHistoryReplacement:
    def test_a_new_progress_event_replaces_the_previous_one_in_event_id_order(self):
        runner = _runner()
        try:
            job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.0)
            _wait_for_status(runner, job_id, "completed")
            runner._publish_event(job_id, _downloading(10))
            runner._publish_event(job_id, {"type": "model_loading", "mode": "image", "model": "m"})
            runner._publish_event(job_id, _downloading(20))
            history = _history(runner, job_id)
        finally:
            runner.shutdown()

        kept = [event for event in history if event["type"] == DOWNLOADING_EVENT]
        assert [event["bytes_done"] for event in kept] == [20]
        ids = [event["event_id"] for event in history]
        assert ids == sorted(ids)
        assert history[-1]["event_id"] == kept[0]["event_id"]

    def test_replay_after_a_cursor_holds_only_the_latest_progress_event(self):
        runner = _runner()
        try:
            job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.0)
            _wait_for_status(runner, job_id, "completed")
            first = runner._get_job(job_id).next_event_id
            runner._publish_event(job_id, _downloading(10))
            runner._publish_event(job_id, _downloading(20))
            runner._publish_event(job_id, _downloading(30))

            async def replay() -> list[dict]:
                return [_sse_payload(frame) async for frame in runner.stream_job_events(job_id, after_event_id=first)]

            replayed = asyncio.run(replay())
        finally:
            runner.shutdown()

        assert [(event["type"], event["bytes_done"]) for event in replayed] == [(DOWNLOADING_EVENT, 30)]

    def test_download_progress_does_not_clear_a_review_or_preview(self):
        runner = _runner()
        try:
            job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.0)
            _wait_for_status(runner, job_id, "completed")
            record = runner._get_job(job_id)
            record.review = {"iteration": 1}
            record.preview_jpeg = b"jpeg"
            runner._publish_event(job_id, _downloading(10))
            kept = (record.review, record.preview_jpeg)
        finally:
            runner.shutdown()

        assert kept == ({"iteration": 1}, b"jpeg")

    @pytest.mark.parametrize("mode", ["image", "video"])
    def test_the_hook_counts_every_planned_iteration_across_sets_and_runs(self, monkeypatch, mode):
        monkeypatch.setattr(web_runner_module, "enhancer_plan", lambda repo, revision: (DownloadTarget(repo, revision),))
        runner = _runner()
        captured: dict[str, object] = {}
        monkeypatch.setattr(runner, "_ensure_downloads", lambda targets, **kwargs: (captured.update(kwargs, targets=targets), False)[1])
        try:
            prompts_data = {"a": [("one", None), ("two", None)], "b": [("three", None)]}
            hook = runner._enhancer_download_hook(mode=mode, prompts_data=prompts_data, args=_make_args(runs=4), progress_callback=MagicMock(), control_signal=SkipSignal())
            assert hook("a/enhancer", "rev") is False
        finally:
            runner.shutdown()

        assert captured["total_iterations"] == 12
        assert (captured["mode"], captured["role"], captured["model"]) == (mode, "enhancer", "a/enhancer")
        assert captured["targets"] == (DownloadTarget("a/enhancer", "rev"),)


class TestMacH3VideoJob:
    """A MiniMax H3 job on macOS plans the Mac downloads and loads with options the MLX backend accepts."""

    _REF = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"

    @pytest.fixture
    def run_job(self, monkeypatch):
        from zvisiongenerator.backends.mac import h3_video
        from zvisiongenerator.utils.config import load_config

        _stub_request_job(monkeypatch)
        monkeypatch.setattr("sys.platform", "darwin")
        planned: list = []
        from zvisiongenerator.downloads.plans import video_plan as real_plan

        monkeypatch.setattr(web_runner_module, "video_plan", lambda *a, **kw: (planned.append(real_plan(*a, **kw)), None)[1])
        backend = h3_video.H3MlxVideoBackend()
        loaded: list = []
        monkeypatch.setattr(h3_video, "resolve_mlx_files", lambda *_a, **_kw: SimpleNamespace(lora_file="turbo.safetensors"))
        monkeypatch.setattr(h3_video, "_turbo_rank", lambda _path: 4)
        monkeypatch.setattr(h3_video, "ensure_text_encoder_copy", lambda models_dir, name, *_a: (loaded.append(name), models_dir / name)[1])
        monkeypatch.setattr(web_runner_module, "get_video_backend", lambda _family: backend)
        monkeypatch.setattr(
            web_runner_module,
            "run_video_batch",
            lambda *, model, progress_callback, **_kw: (loaded.append(model), progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1}))[1],
        )

        def run(*, image_path=None, extend=False, upscale=None) -> tuple[list, list]:
            runner = _runner()
            try:
                request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="h3", model_name="h3", image_path=image_path, upscale=upscale)
                args = _make_video_args(extend_source=object() if extend else None)
                job_id = runner.submit_video_request_job(request=request, prompts_data=_PROMPTS, config=load_config(), args=args, model_ref=self._REF)
                snapshot = _wait_for_status(runner, job_id, "completed")
                assert snapshot["status"] == "completed", snapshot
            finally:
                runner.shutdown()
            return planned, loaded

        return run

    @pytest.mark.parametrize(("kwargs", "mode"), [({}, "t2v"), ({"image_path": "frame.png"}, "i2v"), ({"extend": True}, "extend")])
    def test_each_mode_loads_quietly_with_what_the_mlx_backend_needs(self, run_job, kwargs, mode):
        planned, loaded = run_job(**kwargs)

        assert [target.repo_id for target in planned[0]] == ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "MiniMaxAI/MiniMax-H3", "lightx2v/Minimax-h3-Turbo"]
        assert loaded[0] == "h3"
        model = loaded[1]
        assert (model.mode, model.verbose, model.turbo_lora_alpha) == (mode, False, 0.0625 * 4)
