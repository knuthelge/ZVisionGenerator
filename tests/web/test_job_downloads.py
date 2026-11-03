"""Tests for the job download planner: its parts match what the runner downloads, event names included."""

from __future__ import annotations

import functools
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._factories import _make_args, _make_video_args, make_extend_source
from tests.web._helpers import stub_request_job, wait_for_status

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import READY, DownloadTarget
from zvisiongenerator.preflight import run_preflight as real_run_preflight
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL
from zvisiongenerator.utils.matte_models import MATTE_LABEL
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_downloads import DownloadPlanners, JobDownloadPart, drop_in_flight, job_download_parts

_PROMPTS = {"default": [("prompt", None)]}
_CONFIG = {"generation": {}, "prompt_enhancer": {"model": {"darwin": "a/enhancer", "win32": "a/enhancer", "linux": "a/enhancer"}}}
_PLATFORM = sys.platform


def _image_target(model_ref: str, family: str) -> tuple[DownloadTarget, ...]:
    return (DownloadTarget(f"image/{model_ref}", allow_patterns=(family,)),)


def _fake_planners(tmp_path) -> DownloadPlanners:
    """Return planners whose targets name the inputs they were given, so a part built from other inputs differs."""
    return DownloadPlanners(
        image=lambda model_ref, *, family, config, platform_key: _image_target(model_ref, f"{family}:{platform_key}"),
        video=lambda model_ref, *, family, config, model_name, upscale: (DownloadTarget(f"video/{model_ref}", allow_patterns=(f"{family}:{model_name}:{upscale}",)),),
        matte=lambda: (DownloadTarget("matte/pinned"),),
        seedvr2=lambda model, *, models_dir, weights: (DownloadTarget("seedvr2/tier", allow_patterns=(model, str(models_dir)), local_dir=models_dir),),
        enhancer=lambda repo, revision: (DownloadTarget(repo, revision),),
        seedvr2_upscaler=lambda: SimpleNamespace(weights=FP8_WEIGHTS),
        data_dir=lambda: tmp_path,
    )


@pytest.fixture
def runner_planners(monkeypatch, tmp_path):
    """Bind the fake planners where the runner looks them up, and record what it hands the ensure step."""
    planners = _fake_planners(tmp_path)
    monkeypatch.setattr(web_runner_module, "image_plan", planners.image)
    monkeypatch.setattr(web_runner_module, "video_plan", planners.video)
    monkeypatch.setattr(web_runner_module, "matte_plan", planners.matte)
    monkeypatch.setattr(web_runner_module, "seedvr2_plan", planners.seedvr2)
    monkeypatch.setattr(web_runner_module, "enhancer_plan", planners.enhancer)
    monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", planners.seedvr2_upscaler)
    monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", planners.data_dir)
    monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)
    ensure = MagicMock(return_value=READY)
    monkeypatch.setattr(web_runner_module, "ensure_downloaded", ensure)
    steps: list[tuple[str, str]] = []
    original = web_runner_module.WebRunner._ensure_downloads

    def spy(self, targets, *, role, model, **kwargs):
        steps.append((role, model))
        return original(self, targets, role=role, model=model, **kwargs)

    monkeypatch.setattr(web_runner_module.WebRunner, "_ensure_downloads", spy)
    return SimpleNamespace(planners=planners, ensure=ensure, steps=steps)


def _enhancing_preflight(monkeypatch):
    """Run the real preflight with a fake enhancer, so the runner's enhancer step comes from the same planning as the parts."""
    enhancer = MagicMock(name="enhancer")
    enhancer.generate.side_effect = lambda *_a, **_kw: iter(["A vivid rewrite with soft light."])
    session = PromptEnhancerSession(lambda repo, revision: enhancer, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
    return functools.partial(real_run_preflight, session=session, release_memory=lambda: None)


def _downloaded(fixture) -> list[tuple[str, str, tuple[DownloadTarget, ...]]]:
    """Return what the runner downloaded as ``(role, event model, targets)``, in the order it ran."""
    targets = [call.args[0] for call in fixture.ensure.call_args_list]
    return [(role, model, planned) for (role, model), planned in zip(fixture.steps, targets, strict=True)]


def _from_parts(parts) -> list[tuple[str, str, tuple[DownloadTarget, ...]]]:
    return [(part.role, part.event_model, part.targets) for part in parts]


def _same_downloads(parts, fixture) -> None:
    """Assert the parts list the downloads the runner made, whatever the order the two keep."""
    assert sorted(_from_parts(parts), key=repr) == sorted(_downloaded(fixture), key=repr)


def _image_request(**overrides) -> ImageGenerationRequest:
    return ImageGenerationRequest(backend=None, model=None, prompt="prompt", model_family="zimage", model_name="atlas", **overrides)


def _upscale_request(**overrides) -> ImageGenerationRequest:
    return _image_request(upscale_factor=2, upscale_source="/out/a.png", **overrides)


def _run_upscale(monkeypatch, runner_planners, request, tmp_path) -> None:
    stub_request_job(monkeypatch)
    monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(web_runner_module, "load_seedvr2", lambda _engine, _name, *, on_phase, **_kw: (on_phase("loading"), MagicMock())[1])
    monkeypatch.setattr(
        web_runner_module, "run_upscale", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
    )
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
    try:
        job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="owner/atlas", config=_CONFIG, context={"output_dir": str(tmp_path)})
        assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
    finally:
        runner.shutdown()


class TestPartsMatchTheRunner:
    def test_an_image_job_with_enhance_and_transparent_downloads_its_model_enhancer_and_matte(self, monkeypatch, runner_planners):
        stub_request_job(monkeypatch, run_preflight=_enhancing_preflight(monkeypatch))
        monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
        monkeypatch.setattr(
            web_runner_module, "run_batch", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
        )
        request = _image_request(transparent=True)
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_image_request_job(request=request, prompts_data=_PROMPTS, config=_CONFIG, args=args, model_ref="owner/atlas")
            assert wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
        finally:
            runner.shutdown()

        parts = job_download_parts(request, args, _PROMPTS, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=runner_planners.planners)

        assert [part.role for part in parts] == ["enhancer", "model", "matte"]
        assert [part.event_model for part in parts] == ["a/enhancer", "atlas", MATTE_LABEL]
        assert parts[2].model_id == MATTE_LICENCE_MODEL
        _same_downloads(parts, runner_planners)

    def test_an_image_job_has_no_seedvr2_part_and_no_enhancer_unless_it_enhances(self, runner_planners):
        request = _image_request(upscale_method="creative", upscale_model="seedvr2-7b")

        parts = job_download_parts(request, _make_args(), _PROMPTS, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=runner_planners.planners)

        assert [(part.role, part.event_model) for part in parts] == [("model", "atlas")]

    def test_a_video_job_with_upscale_downloads_the_upscale_files_and_the_enhancer(self, monkeypatch, runner_planners):
        stub_request_job(monkeypatch, run_preflight=_enhancing_preflight(monkeypatch))
        monkeypatch.setattr(web_runner_module, "video_load_options", lambda *_a, **_kw: {})
        monkeypatch.setattr(
            web_runner_module, "run_video_batch", lambda *, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})
        )
        request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", model_name="ltx-8", upscale=2)
        args = _make_video_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None, scene=None)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_video_request_job(request=request, prompts_data=_PROMPTS, config=_CONFIG, args=args, model_ref="owner/ltx")
            assert wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
        finally:
            runner.shutdown()

        parts = job_download_parts(request, args, _PROMPTS, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/ltx", planners=runner_planners.planners)

        assert [part.role for part in parts] == ["enhancer", "model"]
        assert parts[1].targets[0].allow_patterns == ("ltx:ltx-8:True",)  # the upscale files are planned
        _same_downloads(parts, runner_planners)

    def test_an_image_upscale_with_seedvr2_downloads_the_tier_labelled_with_its_name(self, monkeypatch, runner_planners, tmp_path):
        request = _upscale_request(upscale_method="seedvr2", upscale_model="seedvr2-7b")
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, None, None, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=runner_planners.planners)

        assert [(part.role, part.event_model, part.label) for part in parts] == [("model", "SeedVR2 7B fp8", "SeedVR2 7B fp8")]
        _same_downloads(parts, runner_planners)

    def test_a_creative_upscale_downloads_the_refinement_model(self, monkeypatch, runner_planners, tmp_path):
        request = _upscale_request()
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, None, None, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=runner_planners.planners)

        assert [(part.role, part.event_model) for part in parts] == [("model", "atlas")]
        _same_downloads(parts, runner_planners)

    @pytest.mark.parametrize("extra", [{}, {"upscale_method": "seedvr2", "upscale_model": "seedvr2-3b"}], ids=["creative", "seedvr2"])
    def test_the_upscale_of_a_transparent_image_adds_the_matte_models_last(self, monkeypatch, runner_planners, tmp_path, extra):
        request = _upscale_request(transparent=True, **extra)
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, None, None, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=runner_planners.planners)

        assert [part.role for part in parts] == ["model", "matte"]
        _same_downloads(parts, runner_planners)

    def test_a_video_file_upscale_downloads_the_seedvr2_tier(self, monkeypatch, runner_planners, tmp_path):
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_seedvr2", lambda _engine, _name, *, on_phase, **_kw: (on_phase("loading"), MagicMock())[1])
        monkeypatch.setattr(
            web_runner_module,
            "run_video_file_upscale",
            lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1}),
        )
        request = VideoFileUpscaleRequest(
            source=make_extend_source(path=tmp_path / "clip.mp4"), recorded_config=None, output_dir=str(tmp_path), width=1536, height=896, upscale_model="seedvr2-7b", softness=0.0, batch_size=25
        )
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_video_file_upscale_job(request=request, context={"output_dir": str(tmp_path), "media_type": "video"})
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
        finally:
            runner.shutdown()

        parts = job_download_parts(request, None, None, None, _CONFIG, platform_key=_PLATFORM, planners=runner_planners.planners)

        assert [(part.role, part.event_model) for part in parts] == [("model", "SeedVR2 7B fp8")]
        _same_downloads(parts, runner_planners)


class TestUncheckedParts:
    def test_an_enhancer_missing_while_offline_is_an_unchecked_part(self, monkeypatch):
        def offline(repo, revision):
            raise RuntimeError(f"Enhancer model {repo} is not downloaded and Hugging Face is offline.")

        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", offline)
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)

        parts = job_download_parts(_image_request(), args, _PROMPTS, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=DownloadPlanners(image=lambda *_a, **_kw: ()))

        enhancer = parts[0]
        assert (enhancer.role, enhancer.event_model, enhancer.model_id, enhancer.targets) == ("enhancer", "a/enhancer", "a/enhancer", None)
        assert parts[1].targets == ()

    def test_a_planner_that_cannot_plan_gives_an_unchecked_model_part(self, runner_planners):
        planners = DownloadPlanners(image=lambda *_a, **_kw: None, matte=lambda: None)

        parts = job_download_parts(_image_request(transparent=True), _make_args(), _PROMPTS, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=planners)

        assert [(part.role, part.targets) for part in parts] == [("model", None), ("matte", None)]

    def test_a_platform_without_a_seedvr2_engine_gives_an_unchecked_part(self):
        request = _upscale_request(upscale_method="seedvr2", upscale_model="seedvr2-7b")

        parts = job_download_parts(request, None, None, None, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", planners=DownloadPlanners(seedvr2_upscaler=lambda: None))

        assert [(part.event_model, part.targets) for part in parts] == [("seedvr2-7b", None)]


def test_a_job_that_needs_a_model_ref_refuses_without_one():
    with pytest.raises(ValueError, match="model_ref"):
        job_download_parts(_image_request(), _make_args(), _PROMPTS, None, _CONFIG, platform_key=_PLATFORM)


@pytest.fixture
def recorded(runner_planners, monkeypatch):
    """Wrap each planner the runner calls in a recorder, so a test pins the exact arguments it receives."""
    recorders = SimpleNamespace(
        image=MagicMock(wraps=runner_planners.planners.image),
        video=MagicMock(wraps=runner_planners.planners.video),
        matte=MagicMock(wraps=runner_planners.planners.matte),
        seedvr2=MagicMock(wraps=runner_planners.planners.seedvr2),
        enhancer=MagicMock(wraps=runner_planners.planners.enhancer),
    )
    for name in ("image", "video", "matte", "seedvr2", "enhancer"):
        monkeypatch.setattr(web_runner_module, f"{name}_plan", getattr(recorders, name))
    return recorders


def _run_image_job(monkeypatch, request, args) -> None:
    stub_request_job(monkeypatch, run_preflight=_enhancing_preflight(monkeypatch))
    monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr(
        web_runner_module, "run_batch", lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})
    )
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
    try:
        job_id = runner.submit_image_request_job(request=request, prompts_data=_PROMPTS, config=_CONFIG, args=args, model_ref="owner/atlas")
        assert wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
    finally:
        runner.shutdown()


class TestRunnerHandsThePlannersTheirArguments:
    """Pin the arguments and order of every planner call, so a builder that drifts from the runner's former calls fails."""

    def test_an_image_job_plans_its_model_enhancer_and_matte_with_the_request_values(self, monkeypatch, recorded, runner_planners):
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)

        _run_image_job(monkeypatch, _image_request(transparent=True), args)

        recorded.image.assert_called_once_with("owner/atlas", family="zimage", config=_CONFIG, platform_key=sys.platform)
        recorded.enhancer.assert_called_once_with("a/enhancer", None)
        recorded.matte.assert_called_once_with()
        recorded.video.assert_not_called()
        recorded.seedvr2.assert_not_called()
        assert runner_planners.steps == [("enhancer", "a/enhancer"), ("model", "atlas"), ("matte", MATTE_LABEL)]  # the order the runner downloads them in

    def test_a_video_job_plans_its_model_with_the_upscale_flag_and_name(self, monkeypatch, recorded, runner_planners):
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "video_load_options", lambda *_a, **_kw: {})
        monkeypatch.setattr(
            web_runner_module, "run_video_batch", lambda *, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})
        )
        request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", model_name="ltx-8", upscale=2)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_video_request_job(request=request, prompts_data=_PROMPTS, config=_CONFIG, args=_make_video_args(no_enhance=True, scene=None), model_ref="owner/ltx")
            assert wait_for_status(runner, job_id, "completed", timeout=3.0)["status"] == "completed"
        finally:
            runner.shutdown()

        recorded.video.assert_called_once_with("owner/ltx", family="ltx", config=_CONFIG, model_name="ltx-8", upscale=True)
        assert runner_planners.steps == [("model", "ltx-8")]

    def test_a_seedvr2_upscale_plans_the_tier_in_the_data_dir_with_the_engine_weights(self, monkeypatch, recorded, runner_planners, tmp_path):
        _run_upscale(monkeypatch, runner_planners, _upscale_request(upscale_method="seedvr2", upscale_model="seedvr2-7b"), tmp_path)

        recorded.seedvr2.assert_called_once_with("seedvr2-7b", models_dir=tmp_path / "models", weights=FP8_WEIGHTS)
        recorded.image.assert_not_called()
        assert runner_planners.steps == [("model", "SeedVR2 7B fp8")]

    def test_a_creative_upscale_plans_the_refinement_model_for_this_platform(self, monkeypatch, recorded, runner_planners, tmp_path):
        _run_upscale(monkeypatch, runner_planners, _upscale_request(transparent=True), tmp_path)

        recorded.image.assert_called_once_with("owner/atlas", family="zimage", config=_CONFIG, platform_key=sys.platform)
        recorded.seedvr2.assert_not_called()
        assert runner_planners.steps == [("model", "atlas"), ("matte", MATTE_LABEL)]

    def test_a_video_file_upscale_plans_the_tier_it_was_asked_for(self, monkeypatch, recorded, runner_planners, tmp_path):
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "load_seedvr2", lambda _engine, _name, *, on_phase, **_kw: (on_phase("loading"), MagicMock())[1])
        monkeypatch.setattr(
            web_runner_module,
            "run_video_file_upscale",
            lambda *_a, progress_callback, **_kw: progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1}),
        )
        request = VideoFileUpscaleRequest(
            source=make_extend_source(path=tmp_path / "clip.mp4"), recorded_config=None, output_dir=str(tmp_path), width=1536, height=896, upscale_model="seedvr2-3b", softness=0.0, batch_size=25
        )
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_video_file_upscale_job(request=request, context={"output_dir": str(tmp_path), "media_type": "video"})
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
        finally:
            runner.shutdown()

        recorded.seedvr2.assert_called_once_with("seedvr2-3b", models_dir=tmp_path / "models", weights=FP8_WEIGHTS)

    def test_the_plan_lists_the_same_planner_calls_as_the_runner_made(self, monkeypatch, recorded, runner_planners):
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)
        _run_image_job(monkeypatch, _image_request(transparent=True), args)
        run_calls = {name: list(getattr(recorded, name).call_args_list) for name in ("image", "matte", "enhancer")}
        for name in run_calls:
            getattr(recorded, name).reset_mock()

        job_download_parts(
            _image_request(transparent=True),
            args,
            _PROMPTS,
            None,
            _CONFIG,
            platform_key=sys.platform,
            model_ref="owner/atlas",
            planners=DownloadPlanners(**vars(recorded) | {"seedvr2_upscaler": runner_planners.planners.seedvr2_upscaler}),
        )

        assert {name: list(getattr(recorded, name).call_args_list) for name in run_calls} == run_calls


class TestOfflineEnhancerInARealJob:
    def test_a_missing_offline_enhancer_still_fails_the_job_and_downloads_nothing(self, monkeypatch):
        def offline(repo, revision):
            raise RuntimeError(f"Enhancer model {repo} is not downloaded and Hugging Face is offline.")

        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", offline)
        ensure = MagicMock(return_value=READY)
        monkeypatch.setattr(web_runner_module, "ensure_downloaded", ensure)
        stub_request_job(monkeypatch, run_preflight=_enhancing_preflight(monkeypatch))
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_image_request_job(request=_image_request(), prompts_data=_PROMPTS, config=_CONFIG, args=args, model_ref="owner/atlas")
            snapshot = wait_for_status(runner, job_id, "failed", timeout=3.0)
        finally:
            runner.shutdown()

        assert snapshot["terminal_event"] == "job_failed"
        assert "offline" in snapshot["last_event"]["message"]
        ensure.assert_not_called()


class TestDropInFlight:
    @staticmethod
    def _part(role, *targets):
        return JobDownloadPart(role, role, role, role, targets if targets != (None,) else None)

    def test_a_target_that_a_running_download_fetches_is_dropped_from_its_part(self):
        files, other = DownloadTarget("a/model", allow_patterns=("*.bin",)), DownloadTarget("a/other")
        part = self._part("model", files, other)

        (kept,) = drop_in_flight((part,), [files])

        assert kept.targets == (other,)

    def test_a_part_with_every_target_running_is_dropped(self):
        target = DownloadTarget("a/model")

        assert drop_in_flight((self._part("model", target),), [target]) == ()

    def test_two_files_of_one_repo_in_one_folder_are_two_flights(self):
        wanted = DownloadTarget("a/vae", allow_patterns=("vae.safetensors",), local_dir="/models")
        running = DownloadTarget("a/vae", allow_patterns=("other.safetensors",), local_dir="/models")

        (kept,) = drop_in_flight((self._part("model", wanted),), [running])

        assert kept.targets == (wanted,)

    def test_an_unchecked_part_stays_whatever_runs(self):
        unchecked = self._part("enhancer", None)

        assert drop_in_flight((unchecked,), [DownloadTarget("a/model")]) == (unchecked,)
