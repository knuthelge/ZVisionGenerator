"""Tests for the ``enhance`` record of an asset: building it, validating it, and writing and reading it in provenance."""

from __future__ import annotations

import warnings

import pytest

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.job_plan import EnhancerUsed, EnhanceStatus, IterationPlan
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.enhance_record import build_enhance_record, enhancer_tier, parse_enhance_record, planned_enhance_record
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.provenance import build_image_config_payload, build_video_config_payload, recorded_settings

_CONFIG = {
    "prompt_enhancer": {
        "model": {"linux": "org/fast"},
        "revision": {"linux": "r1"},
        "smart": {"model": {"linux": "org/smart"}},
    }
}


class TestTier:
    @pytest.mark.parametrize(("model", "tier"), [("org/fast@r1", "fast"), ("org/smart", "smart"), ("someone/else", "custom"), ("org/fast@other", "custom")])
    def test_tier_comes_from_the_configured_models(self, model, tier):
        assert enhancer_tier(_CONFIG, platform_key="linux", model=model) == tier

    @pytest.mark.parametrize("configured", ["~/models/fast", "{home}/models/fast"])
    @pytest.mark.parametrize("recorded", ["~/models/fast", "{home}/models/fast"])
    def test_a_local_tier_folder_matches_however_home_is_written(self, tmp_path, monkeypatch, configured, recorded):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "models" / "fast").mkdir(parents=True)
        config = {"prompt_enhancer": {"model": {"linux": configured.format(home=tmp_path)}}}
        assert enhancer_tier(config, platform_key="linux", model=recorded.format(home=tmp_path)) == "fast"


class TestBuild:
    def test_a_local_model_folder_is_recorded_under_home_as_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        folder = tmp_path / "models" / "enhancer"
        folder.mkdir(parents=True)
        record = build_enhance_record(EnhanceSettings(), mode="image", repo=str(folder), revision=None, tier="custom", fmt="text")
        assert record["model"] == "~/models/enhancer"
        config = {"prompt_enhancer": {"user_model": str(folder)}}
        parsed = parse_enhance_record(record, mode="image", config=config, platform_key="linux")
        assert (parsed["model"], parsed["tier"]) == ("~/models/enhancer", "custom")

    def test_image_record_leaves_out_video_options_and_an_empty_direction(self):
        record = build_enhance_record(EnhanceSettings(style="photo"), mode="image", repo="org/fast", revision="r1", tier="fast", fmt="text")
        assert record["model"] == "org/fast@r1" and record["tier"] == "fast" and record["format"] == "text"
        assert record["settings"]["style"] == "photo"
        assert not {"camera", "pace", "motion", "sound", "parts", "note"} & set(record["settings"])

    def test_video_record_keeps_motion_sound_and_parts(self):
        settings = EnhanceSettings(motion=("action",), sound=("ambient",), parts="action", note="at dusk")
        record = build_enhance_record(settings, mode="video", repo="org/fast", revision=None, tier="fast", fmt="text")
        assert record["settings"]["motion"] == ["action"] and record["settings"]["sound"] == ["ambient"]
        assert record["settings"]["parts"] == "action" and record["settings"]["note"] == "at dusk"
        assert record["model"] == "org/fast"

    def test_planned_record_only_for_an_enhanced_iteration(self):
        used = EnhancerUsed("org/smart", None, tier="smart", format="json")
        base = dict(run_index=0, set_name="a", prompt_index=0, prompt="a fox", seed=12345678, resolved_prompt="a fox", enhance=EnhanceSettings())
        enhanced = IterationPlan(**base, enhanced_prompt="A fox.", enhance_status=EnhanceStatus.ENHANCED)
        failed = IterationPlan(**base, enhance_status=EnhanceStatus.FAILED)
        assert planned_enhance_record(enhanced, used, mode="image")["format"] == "json"
        assert planned_enhance_record(failed, used, mode="image") is None
        assert planned_enhance_record(enhanced, None, mode="image") is None


class TestReadBack:
    def test_an_image_record_reads_back_without_warnings(self):
        record = build_enhance_record(EnhanceSettings(style="film"), mode="image", repo="org/fast", revision=None, tier="fast", fmt="text")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert parse_enhance_record(record, mode="image")["settings"]["style"] == "film"


class TestParse:
    def test_tier_is_worked_out_not_trusted(self):
        record = parse_enhance_record({"settings": {"style": "film"}, "model": "org/smart", "tier": "fast"}, mode="image", config=_CONFIG, platform_key="linux")
        assert record["tier"] == "smart"

    def test_absent_record_is_none(self):
        assert parse_enhance_record(None, mode="image") is None

    @pytest.mark.parametrize(
        "raw",
        [
            "not a mapping",
            {"settings": {}},
            {"settings": {"style": "nope"}, "model": "org/fast"},
            {"settings": {}, "model": "org/fast", "format": "yaml"},
            {"settings": {}, "model": "org/fast", "tier": "turbo"},
        ],
    )
    def test_malformed_record_is_rejected(self, raw):
        with pytest.raises(ValueError):
            parse_enhance_record(raw, mode="image")


def _image_request(**overrides):
    values = dict(backend=None, model=None, prompt="a fox in snow", model_name="zit", model_family="zimage", width=512, height=512, seed=12345678, steps=8, guidance=1.0)
    values.update(overrides)
    return ImageGenerationRequest(**values)


_RECORD = {"settings": {"style": "photo", "mood": "keep", "details": [], "length": "same"}, "model": "org/fast@r1", "tier": "fast", "format": "text"}


class TestProvenance:
    def test_job_rewrite_records_the_template_and_the_record(self):
        artifacts = ImageWorkingArtifacts(metadata={"enhanced_prompt": "A red fox curls up in deep snow."})
        payload = build_image_config_payload(_image_request(enhance_record=_RECORD), artifacts)
        assert payload["prompt"] == "A red fox curls up in deep snow."
        assert payload["prompt_input"] == "a fox in snow"
        assert payload["enhance"] == _RECORD

    def test_web_rewrite_records_the_users_own_text(self):
        request = _image_request(prompt="A red fox curls up in deep snow.", prompt_input="a fox in snow", enhance_record=_RECORD)
        payload = build_image_config_payload(request, ImageWorkingArtifacts())
        assert payload["prompt"] == "A red fox curls up in deep snow." and payload["prompt_input"] == "a fox in snow"

    def test_plain_image_records_neither(self):
        payload = build_image_config_payload(_image_request(), ImageWorkingArtifacts())
        assert "prompt_input" not in payload and "enhance" not in payload

    def test_video_records_the_record(self):
        record = {**_RECORD, "settings": {**_RECORD["settings"], "motion": ["action"], "sound": []}}
        request = VideoGenerationRequest(backend=None, model=None, prompt="a fox runs", enhance_record=record)
        payload = build_video_config_payload(request, VideoWorkingArtifacts(metadata={"enhanced_prompt": "A fox sprints."}))
        assert payload["enhance"] == record

    def test_record_reads_back(self):
        payload = build_image_config_payload(_image_request(enhance_record=_RECORD), ImageWorkingArtifacts(metadata={"enhanced_prompt": "A fox."}))
        recorded = recorded_settings(payload)
        assert recorded.enhance["settings"]["style"] == "photo" and recorded.enhance["tier"] == "fast"
        assert recorded.prompt_input == "a fox in snow"

    def test_malformed_recorded_block_reads_as_none(self):
        assert recorded_settings({"prompt": "x", "enhance": {"model": "", "settings": {}}}).enhance is None
        assert recorded_settings({"prompt": "x"}).enhance is None
