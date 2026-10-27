"""Tests for the MiniMax H3 preset value types."""

from __future__ import annotations

import copy
import re

import pytest

from zvisiongenerator.core.video_family import FrameRule
from zvisiongenerator.utils.config import h3_companion_repos, load_config, video_companion_repos, video_family, video_load_options
from zvisiongenerator.utils.h3_preset import H3MlxPreset, H3Preset, H3Schedule, HubFiles, h3_load_options, h3_mlx_load_options, h3_mlx_preset, h3_turbo_lora
from zvisiongenerator.utils.h3_preset import h3_preset as parse_h3_preset


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
    def test_the_packaged_preset_parses(self, config):
        preset = parse_h3_preset(config)
        assert preset.repo == "MiniMaxAI/MiniMax-H3"
        assert preset.text_encoder.repo_id == "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit"
        assert len(preset.text_encoder.files) == 6
        assert preset.text_encoder_layers == 50
        assert (preset.lora.repo_id, preset.lora.files) == ("lightx2v/Minimax-h3-Turbo", ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",))
        assert preset.lora_scale == 0.0625
        assert preset.latent_upsampler.repo_id == "LBH-123-AI/Minimax_h3_latent_Upscaler"
        assert len(preset.latent_upsampler.files) == 1
        assert preset.schedules == {8: H3Schedule(8, 6.0, 3.0), 4: H3Schedule(4, 6.0, 3.0)}
        assert preset.attention == "sdpa"
        assert preset.ffn_chunk_tokens == 4096
        assert "text_encoder/*" in preset.repo_ignore
        assert "transformer/*" in preset.repo_allow

    def test_every_pinned_revision_is_a_full_commit_hash(self, config):
        preset = parse_h3_preset(config)
        mlx = h3_mlx_preset(config)
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
        preset = parse_h3_preset(config)
        assert config["model_aliases"]["h3"] == f"{preset.repo}@{preset.revision}"
        assert config["model_labels"]["h3"]["name"] == "MiniMax H3"


class TestPresetErrors:
    @staticmethod
    def _broken(config: dict, path: str, value) -> dict:
        broken = copy.deepcopy(config)
        node = broken["video_model_presets"]["h3"]
        *parents, last = path.split(".")
        for key in parents:
            node = node[int(key)] if key.isdigit() else node[key]
        node[int(last) if last.isdigit() else last] = value
        return broken

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            ("repo_ref", "MiniMaxAI/MiniMax-H3"),  # no revision
            ("repo_ref", 5),  # not a string
            ("diffusers", None),  # not a mapping
            ("diffusers.repo_allow", []),  # empty list
            ("diffusers.text_encoder.layers", 0),  # integer below the minimum
            ("diffusers.lora.scale", -1),  # number not above 0
            ("diffusers.lora.filename", ""),  # blank string
            ("diffusers.attention", "spark"),  # not one of the choices
        ],
    )
    def test_a_malformed_entry_names_its_key_path(self, config, path, value):
        with pytest.raises(ValueError, match=re.escape(f"video_model_presets.h3.{path} ")):
            parse_h3_preset(self._broken(config, path, value))

    def test_a_missing_entry_is_refused(self):
        with pytest.raises(ValueError, match=r"video_model_presets\.h3 must be a mapping"):
            parse_h3_preset({})

    def test_attention_defaults_to_sdpa(self, config):
        del config["video_model_presets"]["h3"]["diffusers"]["attention"]
        assert parse_h3_preset(config).attention == "sdpa"

    def test_ffn_chunking_is_off_without_the_key(self, config):
        del config["video_model_presets"]["h3"]["diffusers"]["ffn_chunk_tokens"]
        assert parse_h3_preset(config).ffn_chunk_tokens is None


class TestLoadOptionsAndCompanions:
    def test_the_load_options_carry_the_model_name_and_preset(self, config):
        options = h3_load_options(config, "h3")
        assert options["model_name"] == "h3"
        assert options["preset"] == parse_h3_preset(config)

    def test_the_family_dispatch_adds_upscale_only_when_asked(self, config):
        assert "upscale" not in video_load_options("h3", config, "h3", upscale=False)
        assert video_load_options("h3", config, "h3", upscale=True)["upscale"] is True

    def test_only_the_keys_the_h3_backend_takes_are_given(self, config, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "linux")
        assert set(video_load_options("h3", config, "h3", upscale=True)) <= {"model_name", "preset", "upscale"}

    @pytest.mark.parametrize("suffix", [None, "", "@other"])
    def test_the_companions_are_the_text_encoder_the_lora_and_the_latent_upsampler_repos(self, config, monkeypatch, suffix):
        import zvisiongenerator.backends as backends

        repo_id = config["model_aliases"]["h3"] if suffix is None else f"MiniMaxAI/MiniMax-H3{suffix}"

        monkeypatch.setattr(backends.sys, "platform", "linux")
        assert video_companion_repos("h3", config, repo_id) == ("unsloth/Qwen3-VL-32B-Instruct-bnb-4bit", "lightx2v/Minimax-h3-Turbo", "LBH-123-AI/Minimax_h3_latent_Upscaler")

    def test_seedvr2_repos_are_not_companions(self, config):
        assert not any("seedvr2" in repo.lower() for repo in h3_companion_repos(config, config["model_aliases"]["h3"]))

    @pytest.mark.parametrize("repo_id", ["some/other-repo", "/models/h3-local"])
    def test_another_repository_has_none(self, config, repo_id):
        assert video_companion_repos("h3", config, repo_id) == ()


class TestMlxPreset:
    def test_the_shipped_entry_parses_into_the_int8_transformer_and_the_fl2va_components(self, config):
        preset = h3_mlx_preset(config)

        assert preset.transformer.repo_id == "water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8"
        assert preset.transformer.files[:3] == ("config.json", "quant_config.json", "model.safetensors.index.json")
        assert len(preset.transformer.files) == 11
        cuda = parse_h3_preset(config)
        assert (preset.components.repo_id, preset.components.revision) == (cuda.repo, cuda.revision)
        assert (preset.text_encoder_bits, preset.text_encoder_group_size, preset.text_encoder_layers) == (4, 64, 50)

    def test_the_components_are_literal_fl2va_files_with_the_text_encoder_shards_one_to_eleven_and_fourteen(self, config):
        files = h3_mlx_preset(config).components.files
        shards = [name for name in files if name.endswith(".safetensors") and "/text_encoder/" in name]

        assert len(files) == len(set(files)) == 33
        assert all(name.startswith("FL2VA/") and not any(char in name for char in "*?[") for name in files)
        assert shards == [f"FL2VA/text_encoder/model-{n:05d}-of-00014.safetensors" for n in [*range(1, 12), 14]]
        assert {"FL2VA/model_index.json", "FL2VA/video_vae/source/model.safetensors", "FL2VA/audio_vae/model.safetensors"} <= set(files)

    def test_the_turbo_lora_is_the_one_the_diffusers_block_names(self, config):
        lora, scale = h3_turbo_lora(config)

        assert lora == parse_h3_preset(config).lora
        assert scale == 0.0625

    @staticmethod
    def _broken(config: dict, path: str, value) -> dict:
        broken = copy.deepcopy(config)
        node = broken["video_model_presets"]["h3"]
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
        return broken

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            ("mlx", None),  # not a mapping
            ("mlx.transformer.repo", "no/revision"),  # no revision
            ("mlx.text_encoder.bits", 3),  # not one of the choices
        ],
    )
    def test_a_malformed_entry_names_its_key_path(self, config, path, value):
        with pytest.raises(ValueError, match=re.escape(f"video_model_presets.h3.{path} ")):
            h3_mlx_preset(self._broken(config, path, value))

    def test_a_missing_h3_entry_is_refused(self):
        with pytest.raises(ValueError, match=r"video_model_presets\.h3 must be a mapping"):
            h3_mlx_preset({})
        with pytest.raises(ValueError, match=r"video_model_presets\.h3 must be a mapping"):
            h3_turbo_lora({})

    def test_a_malformed_lora_names_its_key_path(self, config):
        with pytest.raises(ValueError, match=r"video_model_presets\.h3\.diffusers\.lora\.scale must be a number above 0"):
            h3_turbo_lora(self._broken(config, "diffusers.lora.scale", 0))

    def test_the_macos_preset_does_not_need_the_cuda_only_entries(self, config):
        del config["video_model_presets"]["h3"]["diffusers"]["text_encoder"]

        assert h3_mlx_load_options(config, "h3")["preset"] == h3_mlx_preset(config)

    def test_the_frame_rule_and_schedules_are_the_ones_the_cuda_preset_and_the_family_use(self, config):
        preset = h3_mlx_preset(config)

        assert preset.frame_rule == video_family(config, "h3").frame_rule == FrameRule(17, 5, "up")
        assert preset.schedules == parse_h3_preset(config).schedules
        assert preset.schedules == {4: H3Schedule(4, 6.0, 3.0), 8: H3Schedule(8, 6.0, 3.0)}

    def test_a_changed_shift_or_frame_rule_in_the_config_reaches_the_macos_preset(self, config):
        config["video_model_presets"]["h3"]["diffusers"]["schedules"][4]["video_shift"] = 5.0
        config["video_model_presets"]["h3"]["frame_rule"] = {"alignment": 17, "offset": 5, "round": "nearest"}

        preset = h3_mlx_preset(config)

        assert preset.schedules[4].video_shift == 5.0
        assert preset.frame_rule == FrameRule(17, 5, "nearest")

    def test_a_missing_schedules_entry_names_its_key_path(self, config):
        with pytest.raises(ValueError, match=r"video_model_presets\.h3\.diffusers\.schedules must be a mapping"):
            h3_mlx_preset(self._broken(config, "diffusers.schedules", None))


class TestMlxLoadOptions:
    def test_the_options_carry_what_the_macos_backend_requires(self, config):
        options = h3_mlx_load_options(config, "h3")

        assert options == {"preset": h3_mlx_preset(config), "lora": h3_turbo_lora(config)[0], "lora_scale": 0.0625, "model_name": "h3"}

    @pytest.mark.parametrize("name", ["MiniMaxAI/MiniMax-H3", "/models/h3", "C:\\h3", "~/h3", "..", None, ""])
    def test_a_name_that_is_not_a_folder_name_is_left_out(self, config, name):
        assert "model_name" not in h3_mlx_load_options(config, name)

    @pytest.mark.parametrize("upscale", [False, True])
    def test_the_family_dispatch_gives_the_macos_options_on_macos(self, config, monkeypatch, upscale):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        options = video_load_options("h3", config, "h3", upscale=upscale)

        assert isinstance(options["preset"], H3MlxPreset)
        assert set(options) == {"preset", "lora", "lora_scale", "model_name", *(["upscale"] if upscale else [])}

    def test_a_quiet_run_turns_the_forks_progress_off_on_macos_only(self, config, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        assert video_load_options("h3", config, "h3", upscale=False, quiet=True)["verbose"] is False
        assert "verbose" not in video_load_options("h3", config, "h3", upscale=False)

        monkeypatch.setattr(backends.sys, "platform", "linux")
        assert "verbose" not in video_load_options("h3", config, "h3", upscale=False, quiet=True)

    def test_the_companions_on_macos_are_the_int8_transformer_and_the_lora_repos(self, config, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")

        assert video_companion_repos("h3", config, config["model_aliases"]["h3"]) == ("water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "lightx2v/Minimax-h3-Turbo")
        assert video_companion_repos("h3", config, "some/other-repo") == ()
