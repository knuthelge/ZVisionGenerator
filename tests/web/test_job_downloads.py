"""Tests for the job download planner: its parts match what the runner downloads, event names included."""

from __future__ import annotations

import functools
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._audio_presets import audio_config, qwen3_tts_preset, sa3_preset
from tests._factories import _make_args, _make_video_args, make_extend_source, use_enhancer_models, video_config
from tests.web._helpers import stub_request_job, wait_for_status

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import READY, DownloadTarget
from zvisiongenerator.preflight import run_preflight as real_run_preflight
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web import job_downloads as job_downloads_module
from zvisiongenerator.web.job_downloads import JobDownloadPart, audio_design_part, audio_model_part, enhancer_part, is_committed, job_download_parts, missing_targets, planned_targets

_PROMPTS = {"default": [("prompt", None)]}
_CONFIG = {"generation": {}, **video_config()}
MATTE_LABEL = "Background removal"
_PLATFORM = sys.platform


def _image_target(model_ref: str, family: str) -> tuple[DownloadTarget, ...]:
    return (DownloadTarget(f"image/{model_ref}", allow_patterns=(family,)),)


def _fake_planners(tmp_path) -> SimpleNamespace:
    """Return planners whose targets name the inputs they were given, so a part built from other inputs differs."""
    return SimpleNamespace(
        image=lambda model_ref, *, family, platform_key: _image_target(model_ref, f"{family}:{platform_key}"),
        video=lambda model_ref, *, family, config, model_name, platform_key, upscale: (DownloadTarget(f"video/{model_ref}", allow_patterns=(f"{family}:{model_name}:{upscale}",)),),
        matte=lambda: (DownloadTarget("matte/pinned"),),
        seedvr2=lambda model, *, platform_key, models_dir, weights: (DownloadTarget("seedvr2/tier", allow_patterns=(model, str(models_dir)), local_dir=models_dir),),
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
        steps.append((part.role, part.label))
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
    return [(part.role, part.label, part.targets) for part in parts]


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
        assert [part.label for part in parts] == ["Fast enhancer model", "atlas", MATTE_LABEL]
        assert parts[2].model_id == MATTE_LICENCE_MODEL
        _same_downloads(parts, runner_planners)

    def test_an_image_job_has_no_seedvr2_part_and_no_enhancer_unless_it_enhances(self, runner_planners):
        request = _image_request(upscale_method="creative", upscale_model="seedvr2-7b")

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=_make_args())

        assert [(part.role, part.label) for part in parts] == [("model", "atlas")]

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

        assert [(part.role, part.label) for part in parts] == [("model", "SeedVR2 7B fp8")]
        assert [part.model_id for part in parts] == ["seedvr2-7b"]
        _same_downloads(parts, runner_planners)

    def test_a_creative_upscale_downloads_the_refinement_model(self, monkeypatch, runner_planners, tmp_path):
        request = _upscale_request()
        _run_upscale(monkeypatch, runner_planners, request, tmp_path)

        parts = job_download_parts(request, _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas")

        assert [(part.role, part.label) for part in parts] == [("model", "atlas")]
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

        assert [(part.role, part.label) for part in parts] == [("model", "SeedVR2 7B fp8")]
        _same_downloads(parts, runner_planners)


class TestAudioParts:
    """An audio job downloads its one model, planned from its preset."""

    @staticmethod
    def _preset(model_id: str, config: dict):
        return next(preset for preset in parse_audio_presets(config) if preset.id == model_id)

    def test_an_audio_job_has_one_model_part_named_by_its_preset(self, monkeypatch):
        audio = MagicMock(side_effect=lambda preset, voice=False: (DownloadTarget(f"audio/{preset.id}", allow_patterns=("files",)),))
        monkeypatch.setattr(job_downloads_module, "audio_plan", audio)
        config = audio_config(qwen3_tts=qwen3_tts_preset(), sa3=sa3_preset())

        part = audio_model_part(self._preset("qwen3_tts", config))

        assert part == JobDownloadPart("Qwen3-TTS 1.7B", "qwen3_tts", "model", (DownloadTarget("audio/qwen3_tts", allow_patterns=("files",)),))

    def test_the_part_of_a_take_with_a_recording_holds_the_cloning_files_too(self):
        preset = self._preset("qwen3_tts", audio_config(qwen3_tts=qwen3_tts_preset()))

        assert [target.repo_id for target in audio_model_part(preset).targets] == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
        assert [target.repo_id for target in audio_model_part(preset, voice=True).targets] == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]

    def test_the_design_part_is_the_voicedesign_checkpoint_alone(self):
        preset = self._preset("qwen3_tts", audio_config(qwen3_tts=qwen3_tts_preset()))

        part = audio_design_part(preset)

        assert (part.label, part.model_id, part.role) == ("Qwen3-TTS 1.7B VoiceDesign", "qwen3_tts", "model")
        assert [target.repo_id for target in part.targets] == ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]

    def test_the_real_planner_downloads_the_preset_s_files(self):
        part = audio_model_part(self._preset("sa3", audio_config(sa3=sa3_preset())))

        assert [target.repo_id for target in part.targets] == ["Comfy-Org/stable-audio-3", "cocktailpeanut/stable-audio-3-medium", "mlx-community/stable-audio-3-small-sfx"]


class TestPartLabels:
    def test_model_parts_are_labelled_with_the_metadata_display_name_and_keep_the_alias_as_their_id(self, runner_planners):
        config = _CONFIG
        image_request = ImageGenerationRequest(backend=None, model=None, prompt="prompt", model_family="zimage", model_name="zit")
        image = job_download_parts(image_request, config, platform_key=_PLATFORM, model_ref="owner/zit", args=_make_args())
        video_request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", model_name="ltx-8")
        video = job_download_parts(video_request, config, platform_key=_PLATFORM, model_ref="owner/ltx", args=_make_args(no_enhance=True))

        assert [(part.label, part.model_id) for part in (*image, *video)] == [("Z-Image Turbo", "zit"), ("LTX-2.3 8-bit", "ltx-8")]

    def test_a_model_without_an_entry_is_labelled_by_its_name(self, runner_planners):
        image = job_download_parts(_image_request(), _CONFIG, platform_key=_PLATFORM, model_ref="owner/atlas", args=_make_args())

        assert [(part.label, part.model_id) for part in image] == [("atlas", "atlas")]

    def test_an_enhancer_part_is_labelled_by_its_tier_or_else_by_its_reference(self, monkeypatch, tmp_path, runner_planners):
        use_enhancer_models(monkeypatch, tmp_path, fast="a/fast", smart="a/smart@r1")
        config = {"prompt_enhancer": {"user_model": "a/mine"}}
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)

        labels = [enhancer_part(repo, revision, config, platform_key=_PLATFORM).label for repo, revision in (("a/fast", None), ("a/smart", "r1"), ("a/mine", None), ("a/other", None))]

        assert labels == ["Fast enhancer model", "Smart enhancer model", "Custom enhancer model", "a/other"]

    def test_an_enhancer_part_looks_up_its_licence_by_tier_and_a_custom_or_unnamed_model_by_its_reference(self, monkeypatch, tmp_path, runner_planners):
        use_enhancer_models(monkeypatch, tmp_path, fast="a/fast", smart="a/smart@r1")
        config = {"prompt_enhancer": {"user_model": "a/mine"}}
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
        assert (enhancer.role, enhancer.label, enhancer.model_id, enhancer.targets) == ("enhancer", "Fast enhancer model", "enhancer-fast", None)
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

        assert [(part.label, part.model_id, part.targets) for part in parts] == [("seedvr2-7b", "seedvr2-7b", None)]


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


class TestIsCommitted:
    @staticmethod
    def _part(role, *targets):
        return JobDownloadPart(role, role, role, targets if targets != (None,) else None)

    @staticmethod
    def _absent(_target):
        return False

    def _committed(self, part, committed, present):
        return is_committed(missing_targets(part, present), committed)

    def test_a_part_whose_every_missing_target_is_on_its_way_is_dropped(self):
        files, other = DownloadTarget("a/model", allow_patterns=("*.bin",)), DownloadTarget("a/other")

        assert self._committed(self._part("model", files, other), [files, other], self._absent)

    def test_the_same_model_queued_twice_is_asked_about_once(self):
        target = DownloadTarget("a/model")

        assert self._committed(self._part("model", target), [target], self._absent)

    def test_a_part_with_its_own_file_missing_stays_when_it_only_shares_one_that_is_on_its_way(self):
        encoder, own = DownloadTarget("a/encoder"), DownloadTarget("a/ltx-8")
        part = self._part("model", own, encoder)

        assert not self._committed(part, [encoder], self._absent)

    def test_a_shared_file_already_on_disk_does_not_hide_the_part_missing_its_own(self):
        encoder, own = DownloadTarget("a/encoder"), DownloadTarget("a/ltx-8")
        part = self._part("model", own, encoder)

        assert not self._committed(part, [encoder], lambda target: target == encoder)

    def test_a_part_wholly_on_disk_is_not_dropped_for_being_on_its_way(self):
        target = DownloadTarget("a/model")
        part = self._part("model", target)

        assert not self._committed(part, [target], lambda _target: True)

    def test_a_part_nothing_is_fetching_stays_unchanged(self):
        part = self._part("model", DownloadTarget("a/model"))

        assert not self._committed(part, [DownloadTarget("a/elsewhere")], self._absent)

    def test_two_files_of_one_repo_in_one_folder_are_two_destinations(self):
        wanted = DownloadTarget("a/vae", allow_patterns=("vae.safetensors",), local_dir="/models")
        committed = DownloadTarget("a/vae", allow_patterns=("other.safetensors",), local_dir="/models")
        part = self._part("model", wanted)

        assert not self._committed(part, [committed], self._absent)

    def test_other_files_of_one_cached_repo_are_not_covered_by_a_download_of_part_of_it(self):
        committed = DownloadTarget("a/repo", allow_patterns=("one.bin",))
        wanted = DownloadTarget("a/repo", allow_patterns=("one.bin", "two.bin"))
        part = self._part("model", wanted)

        assert not self._committed(part, [committed], self._absent)
        assert self._committed(part, [DownloadTarget("a/repo")], self._absent)

    def test_an_unchecked_part_stays_whatever_is_on_its_way(self):
        unchecked = self._part("enhancer", None)

        assert not self._committed(unchecked, [DownloadTarget("a/model")], self._absent)

    def test_the_planned_targets_of_all_parts_in_order_leave_out_an_unchecked_part(self):
        first, second = DownloadTarget("a/one"), DownloadTarget("a/two")

        assert planned_targets((self._part("enhancer", None), self._part("model", first, second))) == (first, second)
