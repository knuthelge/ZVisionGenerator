"""Tests for the MiniMax H3 preset value types."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.core.video_family import FrameRule
from zvisiongenerator.utils.config import load_config, video_family, video_load_options
from zvisiongenerator.downloads.target import HubFiles
from zvisiongenerator.utils.h3_preset import H3MlxPreset, H3Preset, H3Schedule, h3_load_options, h3_mlx_load_options, h3_mlx_preset, h3_turbo_lora
from zvisiongenerator.utils.h3_preset import h3_preset as parse_h3_preset
from zvisiongenerator.utils.model_metadata import builtin_aliases, model_display_name
from zvisiongenerator.web.model_delete import freed_repo_ids


def _hub(name: str = "a/b") -> HubFiles:
    return HubFiles(name, "rev", ("x.json",))


def _preset(**overrides) -> H3Preset:
    fields = dict(
        repo="o/r",
        revision="rev",
        repo_allow=("a/*",),
        repo_ignore=("b/*",),
        text_encoder=_hub("te/r"),
        text_encoder_layers=50,
        lora=_hub("lo/ra"),
        lora_scale=0.0625,
        latent_upsampler=_hub("up/sampler"),
        schedules={8: H3Schedule(8, 6.0, 3.0), 4: H3Schedule(4, 6.0, 3.0)},
    )
    fields.update(overrides)
    return H3Preset(**fields)


@pytest.mark.parametrize("forwards", [4, 8])
def test_num_inference_steps_is_forwards_plus_the_terminal_zero(forwards):
    assert H3Schedule(forwards, 6.0, 3.0).num_inference_steps == forwards + 1


@pytest.fixture
def config() -> dict:
    return load_config()


class TestShippedPreset:
    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_the_packaged_preset_is_built_from_the_declared_files(self, config, platform):
        preset = parse_h3_preset(config, platform)
        assert preset.repo == "MiniMaxAI/MiniMax-H3"
        assert preset.text_encoder.repo_id == "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit"
        assert len(preset.text_encoder.files) == 6
        assert preset.text_encoder_layers == 50
        assert (preset.lora.repo_id, preset.lora.files) == ("lightx2v/Minimax-h3-Turbo", ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",))
        assert preset.lora_scale == 0.0625
        assert preset.latent_upsampler.repo_id == "LBH-123-AI/Minimax_h3_latent_Upscaler"
        assert len(preset.latent_upsampler.files) == 1
        assert preset.schedules == {8: H3Schedule(8, 6.0, 3.0), 4: H3Schedule(4, 6.0, 3.0)}
        assert preset.ffn_chunk_tokens == 4096
        assert "text_encoder/*" in preset.repo_ignore
        assert "transformer/*" in preset.repo_allow

    def test_every_pinned_revision_is_a_full_commit_hash(self, config):
        preset = parse_h3_preset(config, "linux")
        mlx = h3_mlx_preset()
        revisions = [preset.revision, preset.text_encoder.revision, preset.lora.revision, preset.latent_upsampler.revision, mlx.transformer.revision, mlx.components.revision]

        assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions), revisions

    def test_the_family_is_described_for_the_shared_rules(self, config):
        family = video_family(config, "h3")
        assert family is not None
        assert (family.label, family.default_steps, family.max_steps, family.step_options, family.default_frames, family.extend_frame_step) == ("MiniMax H3", 8, 8, (4, 8), 124, 17)
        assert (family.frame_rule.alignment, family.frame_rule.offset, family.frame_rule.round) == (17, 5, "up")
        assert family.capabilities.end_frame is True
        assert family.platform_capabilities == {"darwin": {"upscale": False, "upscale_review": False}}

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_the_cuda_platforms_offer_every_capability(self, config, monkeypatch, platform):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", platform)
        capabilities = backends.video_capabilities(config, "h3")

        assert capabilities is not None
        assert (capabilities.i2v, capabilities.end_frame, capabilities.extend, capabilities.upscale, capabilities.upscale_review, capabilities.lora, capabilities.audio) == (True,) * 7
        assert capabilities.low_memory is False

    def test_macos_offers_everything_but_upscale(self, config, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        capabilities = backends.video_capabilities(config, "h3")

        assert capabilities is not None
        assert (capabilities.i2v, capabilities.end_frame, capabilities.extend, capabilities.lora, capabilities.audio) == (True,) * 5
        assert (capabilities.upscale, capabilities.upscale_review, capabilities.low_memory) == (False, False, False)

    def test_the_alias_and_label(self, config):
        preset = parse_h3_preset(config, "linux")
        assert builtin_aliases()["h3"] == f"{preset.repo}@{preset.revision}"
        assert model_display_name("h3") == "MiniMax H3"


class TestPresetErrors:
    def test_a_missing_ffn_chunk_setting_turns_chunking_off(self):
        assert parse_h3_preset({}, "linux").ffn_chunk_tokens is None

    def test_a_valid_ffn_chunk_setting_is_used(self):
        config = {"video_model_presets": {"h3": {"diffusers": {"ffn_chunk_tokens": 1024}}}}

        assert parse_h3_preset(config, "linux").ffn_chunk_tokens == 1024

    @pytest.mark.parametrize("value", [0, -1, True, "4096"])
    def test_a_malformed_ffn_chunk_setting_names_its_key_path(self, value):
        config = {"video_model_presets": {"h3": {"diffusers": {"ffn_chunk_tokens": value}}}}

        with pytest.raises(ValueError, match=r"^video_model_presets\.h3\.diffusers\.ffn_chunk_tokens must be an integer of at least 1\.$"):
            parse_h3_preset(config, "linux")

    def test_macos_declares_no_cuda_files(self, config):
        with pytest.raises(ValueError, match=r"declares no 'text_encoder' download on 'darwin'"):
            parse_h3_preset(config, "darwin")


class TestLoadOptionsAndCompanions:
    def test_the_load_options_carry_the_model_name_and_preset(self, config):
        options = h3_load_options(config, "h3", "linux")
        assert options["model_name"] == "h3"
        assert options["preset"] == parse_h3_preset(config, "linux")

    def test_the_family_dispatch_adds_upscale_only_when_asked(self, config):
        assert "upscale" not in video_load_options("h3", config, "h3", platform_key="linux", upscale=False)
        assert video_load_options("h3", config, "h3", platform_key="linux", upscale=True)["upscale"] is True

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_only_the_keys_the_h3_backend_takes_are_given(self, config, platform):
        assert set(video_load_options("h3", config, "h3", platform_key=platform, upscale=True)) <= {"model_name", "preset", "upscale"}

    @staticmethod
    def _entry(resolved_path: str, name: str = "h3"):
        return SimpleNamespace(name=name, family="h3", resolved_path=resolved_path)

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_the_companions_are_the_text_encoder_the_lora_and_the_latent_upsampler_repos(self, platform, tmp_path):
        companions = freed_repo_ids(self._entry(builtin_aliases()["h3"]), (), models_dir=tmp_path, platform_key=platform)

        assert companions == ("unsloth/Qwen3-VL-32B-Instruct-bnb-4bit", "lightx2v/Minimax-h3-Turbo", "LBH-123-AI/Minimax_h3_latent_Upscaler")

    def test_seedvr2_repos_are_not_companions(self, tmp_path):
        assert not any("seedvr2" in repo.lower() for repo in freed_repo_ids(self._entry(builtin_aliases()["h3"]), (), models_dir=tmp_path, platform_key="linux"))

    def test_a_model_the_metadata_does_not_declare_has_none(self, tmp_path):
        assert freed_repo_ids(self._entry("some/other-repo", name="custom"), (), models_dir=tmp_path, platform_key="linux") == ()


class TestMlxPreset:
    def test_the_shipped_entry_parses_into_the_int8_transformer_and_the_fl2va_components(self, config):
        preset = h3_mlx_preset()

        assert preset.transformer.repo_id == "water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8"
        assert preset.transformer.files[:3] == ("config.json", "quant_config.json", "model.safetensors.index.json")
        assert len(preset.transformer.files) == 11
        cuda = parse_h3_preset(config, "linux")
        assert (preset.components.repo_id, preset.components.revision) == (cuda.repo, cuda.revision)
        assert (preset.text_encoder_bits, preset.text_encoder_group_size, preset.text_encoder_layers) == (4, 64, 50)

    def test_the_components_are_literal_fl2va_files_with_the_text_encoder_shards_one_to_eleven_and_fourteen(self):
        files = h3_mlx_preset().components.files
        shards = [name for name in files if name.endswith(".safetensors") and "/text_encoder/" in name]

        assert len(files) == len(set(files)) == 33
        assert all(name.startswith("FL2VA/") and not any(char in name for char in "*?[") for name in files)
        assert shards == [f"FL2VA/text_encoder/model-{n:05d}-of-00014.safetensors" for n in [*range(1, 12), 14]]
        assert {"FL2VA/model_index.json", "FL2VA/video_vae/source/model.safetensors", "FL2VA/audio_vae/model.safetensors"} <= set(files)

    @pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
    def test_the_turbo_lora_is_the_one_every_platform_declares(self, config, platform):
        lora, scale = h3_turbo_lora(platform)

        assert lora == parse_h3_preset(config, "linux").lora
        assert scale == 0.0625

    def test_the_frame_rule_and_schedules_are_the_ones_the_cuda_preset_and_the_family_use(self, config):
        preset = h3_mlx_preset()

        assert preset.frame_rule == video_family(config, "h3").frame_rule == FrameRule(17, 5, "up")
        assert preset.schedules == parse_h3_preset(config, "linux").schedules
        assert preset.schedules == {4: H3Schedule(4, 6.0, 3.0), 8: H3Schedule(8, 6.0, 3.0)}

    def test_a_changed_shift_or_frame_rule_in_the_metadata_reaches_both_presets(self, config, monkeypatch, tmp_path):
        data = metadata_copy()
        data["video_families"]["h3"]["schedules"][4]["video_shift"] = 5.0
        data["video_families"]["h3"]["frame_rule"] = {"alignment": 17, "offset": 5, "round": "nearest"}
        use_model_metadata(monkeypatch, tmp_path, data)

        preset = h3_mlx_preset()

        assert preset.schedules[4].video_shift == 5.0
        assert parse_h3_preset(config, "linux").schedules[4].video_shift == 5.0
        assert preset.frame_rule == FrameRule(17, 5, "nearest")


class TestMlxLoadOptions:
    def test_the_options_carry_what_the_macos_backend_requires(self):
        options = h3_mlx_load_options("h3")

        assert options == {"preset": h3_mlx_preset(), "lora": h3_turbo_lora("darwin")[0], "lora_scale": 0.0625, "model_name": "h3"}

    @pytest.mark.parametrize("name", ["MiniMaxAI/MiniMax-H3", "/models/h3", "C:\\h3", "~/h3", "..", None, ""])
    def test_a_name_that_is_not_a_folder_name_is_left_out(self, name):
        assert "model_name" not in h3_mlx_load_options(name)

    @pytest.mark.parametrize("upscale", [False, True])
    def test_the_family_dispatch_gives_the_macos_options_on_macos(self, config, upscale):
        options = video_load_options("h3", config, "h3", platform_key="darwin", upscale=upscale)

        assert isinstance(options["preset"], H3MlxPreset)
        assert set(options) == {"preset", "lora", "lora_scale", "model_name", *(["upscale"] if upscale else [])}

    def test_a_quiet_run_turns_the_forks_progress_off_on_macos_only(self, config):
        assert video_load_options("h3", config, "h3", platform_key="darwin", upscale=False, quiet=True)["verbose"] is False
        assert "verbose" not in video_load_options("h3", config, "h3", platform_key="darwin", upscale=False)
        assert "verbose" not in video_load_options("h3", config, "h3", platform_key="linux", upscale=False, quiet=True)

    def test_the_companions_on_macos_are_the_int8_transformer_and_the_lora_repos(self, tmp_path):
        entry = SimpleNamespace(name="h3", family="h3", resolved_path=builtin_aliases()["h3"])

        assert freed_repo_ids(entry, (), models_dir=tmp_path, platform_key="darwin") == ("water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "lightx2v/Minimax-h3-Turbo")
