"""Tests for the job download planner: its parts match what the runner downloads, event names included."""

from __future__ import annotations

import functools
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._audio_presets import audio_config, higgs3_preset, sa3_preset
from tests._factories import _make_args, _make_video_args, make_extend_source
from tests.web._helpers import stub_request_job, wait_for_status

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import READY, DownloadTarget
from zvisiongenerator.preflight import run_preflight as real_run_preflight
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL
from zvisiongenerator.utils.matte_models import MATTE_LABEL
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web import job_downloads as job_downloads_module
from zvisiongenerator.web.job_downloads import JobDownloadPart, audio_model_part, drop_in_flight, enhancer_part, job_download_parts

_PROMPTS = {"default": [("prompt", None)]}
_CONFIG = {"generation": {}, "prompt_enhancer": {"model": {"darwin": "a/enhancer", "win32": "a/enhancer", "linux": "a/enhancer"}}}
_PLATFORM = sys.platform


def _image_target(model_ref: str, family: str) -> tuple[DownloadTarget, ...]:
    return (DownloadTarget(f"image/{model_ref}", allow_patterns=(family,)),)


def _fake_planners(tmp_path) -> SimpleNamespace:
    """Return planners whose targets name the inputs they were given, so a part built from other inputs differs."""
    return SimpleNamespace(
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
    monkeypatch.setattr(job_downloads_module, "image_plan", planners.image)
    monkeypatch.setattr(job_downloads_module, "video_plan", planners.video)
    monkeypatch.setattr(job_downloads_module, "matte_plan", planners.matte)
    monkeypatch.setattr(job_downloads_module, "seedvr2_plan", planners.seedvr2)
    monkeypatch.setattr(job_downloads_module, "enhancer_plan", planners.enhancer)
    for module in (job_downloads_module, web_runner_module):
        monkeypatch.setattr(module, "get_seedvr2_upscaler", planners.seedvr2_upscaler)
        monkeypatch.setattr(module, "get_ziv_data_dir", planners.data_dir)
    monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)
    ensure = MagicMock(return_value=READY)
    monkeypatch.setattr(web_runner_module, "ensure_downloaded", ensure)
    steps: list[tuple[str, str]] = []
    original = web_runner_module.WebRunner._ensure_downloads

    def spy(self, part, **kwargs):
        steps.append((part.role, part.event_model))
        return original(self, part, **kwargs)

    monkeypatch.setattr(web_runner_module.WebRunner, "_ensure_downloads", spy)
    return SimpleNamespace(ensure=ensure, steps=steps)


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

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=args)

        assert [part.role for part in parts] == ["enhancer", "model", "matte"]
        assert [part.event_model for part in parts] == ["Fast enhancer model", "atlas", MATTE_LABEL]
        assert parts[2].model_id == MATTE_LICENCE_MODEL
        _same_downloads(parts, runner_planners)

    def test_an_image_job_has_no_seedvr2_part_and_no_enhancer_unless_it_enhances(self, runner_planners):
        request = _image_request(upscale_method="creative", upscale_model="seedvr2-7b")

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=_make_args())

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

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/ltx", args=args)

        assert [part.role for part in parts] == ["enhancer", "model"]
        assert parts[1].targets[0].allow_patterns == ("ltx:ltx-8:True",)  # the upscale files are planned
        _same_downloads(parts, runner_planners)

    def test_an_image_upscale_with_seedvr2_downloads_the_tier_labelled_with_its_name(self, monkeypatch, runner_planners, tmp_path):
        request = _upscale_request(upscale_method="seedvr2", upscale_model="seedvr2-7b")
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas")

        assert [(part.role, part.event_model, part.label) for part in parts] == [("model", "SeedVR2 7B fp8", "SeedVR2 7B fp8")]
        assert [part.model_id for part in parts] == ["seedvr2"]
        _same_downloads(parts, runner_planners)

    def test_a_creative_upscale_downloads_the_refinement_model(self, monkeypatch, runner_planners, tmp_path):
        request = _upscale_request()
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas")

        assert [(part.role, part.event_model) for part in parts] == [("model", "atlas")]
        _same_downloads(parts, runner_planners)

    @pytest.mark.parametrize("extra", [{}, {"upscale_method": "seedvr2", "upscale_model": "seedvr2-3b"}], ids=["creative", "seedvr2"])
    def test_the_upscale_of_a_transparent_image_adds_the_matte_models_last(self, monkeypatch, runner_planners, tmp_path, extra):
        request = _upscale_request(transparent=True, **extra)
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas")

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

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM)

        assert [(part.role, part.event_model) for part in parts] == [("model", "SeedVR2 7B fp8")]
        _same_downloads(parts, runner_planners)


class TestAudioParts:
    """An audio job downloads its one model, planned from its preset for the platform it runs on."""

    @staticmethod
    def _preset(model_id: str, config: dict):
        return next(preset for preset in parse_audio_presets(config) if preset.id == model_id)

    def test_an_audio_job_has_one_model_part_named_by_its_preset(self, monkeypatch):
        audio = MagicMock(side_effect=lambda preset, key: (DownloadTarget(f"audio/{preset.id}", allow_patterns=(key,)),))
        monkeypatch.setattr(job_downloads_module, "audio_plan", audio)
        config = audio_config(higgs3=higgs3_preset(), sa3=sa3_preset())

        part = audio_model_part(self._preset("higgs3", config), "mps")

        assert part == JobDownloadPart("Higgs TTS 3 4B", "higgs3", "higgs3", "model", (DownloadTarget("audio/higgs3", allow_patterns=("mps",)),))

    def test_the_real_planner_downloads_the_preset_s_files(self):
        part = audio_model_part(self._preset("sa3", audio_config(sa3=sa3_preset())), "cuda")

        assert [target.repo_id for target in part.targets] == ["Comfy-Org/stable-audio-3", "cocktailpeanut/stable-audio-3-medium", "mlx-community/stable-audio-3-small-sfx"]


class TestPartLabels:
    def test_model_parts_are_labelled_with_the_configured_display_name_and_keep_the_alias_as_their_id(self, runner_planners):
        config = {**_CONFIG, "model_labels": {"atlas": {"name": "Atlas Turbo"}, "ltx-8": {"name": "LTX 8-bit"}}}
        image = job_download_parts(_image_request(), config, platform_key=_PLATFORM, model_ref="owner/atlas", args=_make_args())
        video_request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", model_name="ltx-8")
        video = job_download_parts(video_request, config, platform_key=_PLATFORM, model_ref="owner/ltx", args=_make_args(no_enhance=True))

        assert [(part.label, part.event_model, part.model_id) for part in (*image, *video)] == [("Atlas Turbo", "Atlas Turbo", "atlas"), ("LTX 8-bit", "LTX 8-bit", "ltx-8")]

    def test_an_enhancer_part_is_labelled_by_its_tier_or_else_by_its_reference(self, monkeypatch, runner_planners):
        config = {"prompt_enhancer": {"model": "a/fast", "smart": {"model": "a/smart", "revision": "r1"}, "user_model": "a/mine"}}
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)

        labels = [enhancer_part(repo, revision, config, platform_key=_PLATFORM).label for repo, revision in (("a/fast", None), ("a/smart", "r1"), ("a/mine", None), ("a/other", None))]

        assert labels == ["Fast enhancer model", "Smart enhancer model", "Custom enhancer model", "a/other"]

    def test_an_enhancer_part_looks_up_its_licence_by_tier_and_a_custom_or_unnamed_model_by_its_reference(self, monkeypatch, runner_planners):
        config = {"prompt_enhancer": {"model": "a/fast", "smart": {"model": "a/smart", "revision": "r1"}, "user_model": "a/mine"}}
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)

        ids = [enhancer_part(repo, revision, config, platform_key=_PLATFORM).model_id for repo, revision in (("a/fast", None), ("a/smart", "r1"), ("a/mine", None), ("a/other", None))]

        assert ids == ["enhancer-fast", "enhancer-smart", "a/mine", "a/other"]


class TestUncheckedParts:
    def test_an_enhancer_missing_while_offline_is_an_unchecked_part(self, monkeypatch):
        def offline(repo, revision):
            raise RuntimeError(f"Enhancer model {repo} is not downloaded and Hugging Face is offline.")

        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", offline)
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, enhance_model=None)

        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *_a, **_kw: ())
        parts = job_download_parts(_image_request(), _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=args)

        enhancer = parts[0]
        assert (enhancer.role, enhancer.event_model, enhancer.model_id, enhancer.targets) == ("enhancer", "Fast enhancer model", "enhancer-fast", None)
        assert parts[1].targets == ()

    def test_a_planner_that_cannot_plan_gives_an_unchecked_model_part(self, monkeypatch, runner_planners):
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *_a, **_kw: None)
        monkeypatch.setattr(job_downloads_module, "matte_plan", lambda: None)

        parts = job_download_parts(_image_request(transparent=True), _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=_make_args())

        assert [(part.role, part.targets) for part in parts] == [("model", None), ("matte", None)]

    def test_a_platform_without_a_seedvr2_engine_gives_an_unchecked_part(self, monkeypatch):
        request = _upscale_request(upscale_method="seedvr2", upscale_model="seedvr2-7b")
        monkeypatch.setattr(job_downloads_module, "get_seedvr2_upscaler", lambda: None)

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas")

        assert [(part.event_model, part.model_id, part.targets) for part in parts] == [("seedvr2-7b", "seedvr2", None)]


def test_a_job_that_needs_a_model_ref_refuses_without_one():
    with pytest.raises(ValueError, match="model_ref"):
        job_download_parts(_image_request(), _CONFIG, platform_key=_PLATFORM, args=_make_args())


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
