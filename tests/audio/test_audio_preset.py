"""Tests for the audio preset parser: the three presets, each refusal, the runnable list and the platform settings merge."""

from __future__ import annotations

import dataclasses
import re
import textwrap

import pytest
import yaml

from tests._audio_presets import HIGGS_CODEC_REPO, INSTRUMENTAL_SECTIONS, audio_config, higgs3_preset, sa3_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._paths import REPO_ROOT

from zvisiongenerator.utils import config as config_module
from zvisiongenerator.utils.audio_preset import AudioCapabilities, AudioLimits, AudioMemory, parse_audio_presets, runnable_audio_presets
from zvisiongenerator.utils.licences import parse_licences


def _parse_one(**changes):
    """Parse the yue2 preset after applying *changes* (top-level keys; ``None`` deletes the key)."""
    preset = yue2_preset()
    for key, value in changes.items():
        if value is None:
            del preset[key]
        else:
            preset[key] = value
    return parse_audio_presets({"audio_model_presets": {"yue2": preset}})


class TestParse:
    def test_a_config_without_the_section_has_no_presets(self):
        assert parse_audio_presets({}) == ()

    def test_the_presets_come_in_config_order_keyed_by_model_id(self):
        presets = parse_audio_presets(audio_config())

        assert [(preset.id, preset.kind, preset.label) for preset in presets] == [
            ("yue2", "music", "YuE2 3B"),
            ("higgs3", "speech", "Higgs TTS 3 4B"),
            ("sa3", "sound", "Stable Audio 3 Medium"),
        ]

    def test_yue2_values(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert yue2.capabilities == AudioCapabilities(lyrics=True, duration=False, voice_clone=False)
        assert yue2.limits == AudioLimits(max_seconds=360)
        assert yue2.settings["instrumental_sections"] == INSTRUMENTAL_SECTIONS
        assert (yue2.settings["cot_song"], yue2.settings["cot_instrumental"], yue2.settings["abc_max_tokens"]) == ("full", "melody", 4096)
        assert (yue2.settings["cfg"], yue2.settings["ode_steps"], yue2.settings["memory_budget_gib"]) == (1.0, 32, 24)
        assert (yue2.settings["vae_core_frames"], yue2.settings["fade_seconds"]) == (512, 2)
        assert yue2.memory == AudioMemory(cuda_gb=6.7, mps_gb=None)
        assert set(yue2.platforms) == {"cuda", "mps"}
        assert [(files.repo_id, files.revision) for files in yue2.platforms["cuda"].files] == [
            ("m-a-p/YuE2-3B", "8f24312f187763b9854adeec87e437e2b03bbff1"),
            ("m-a-p/YuE2-Vae", "152733a19ad43aa67e367f9b5503ef8075bb5126"),
        ]

    def test_higgs3_values(self):
        (higgs3,) = parse_audio_presets(audio_config(higgs3=higgs3_preset()))

        assert higgs3.capabilities == AudioCapabilities(lyrics=False, duration=False, voice_clone=True)
        assert higgs3.limits == AudioLimits(max_new_tokens=1500, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)
        assert dict(higgs3.settings) == {"temperature": 0.7, "top_p": 0.95, "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}
        assert [files.repo_id for files in higgs3.platforms["cuda"].files][-1] == "bosonai/higgs-audio-v2-tokenizer"

    def test_sa3_values(self):
        (sa3,) = parse_audio_presets(audio_config(sa3=sa3_preset()))

        assert sa3.capabilities == AudioCapabilities(lyrics=False, duration=True, voice_clone=False)
        assert sa3.limits == AudioLimits(max_seconds=380)
        assert dict(sa3.settings) == {"steps": 8, "cfg_scale": 1.0, "sample_size": 16777216}
        assert {platform.family for platform in sa3.platforms.values()} == {"stable_audio_3"}

    def test_the_values_are_frozen_and_do_not_alias_the_config(self):
        config = audio_config(yue2=yue2_preset())
        (yue2,) = parse_audio_presets(config)

        with pytest.raises(dataclasses.FrozenInstanceError):
            yue2.label = "other"  # type: ignore[misc]
        with pytest.raises(TypeError):
            yue2.settings["cfg"] = 2.0  # type: ignore[index]
        config["audio_model_presets"]["yue2"]["settings"]["cfg"] = 9.0
        assert yue2.settings["cfg"] == 1.0

    def test_settings_are_an_open_mapping(self):
        preset = yue2_preset()
        preset["settings"] = {"anything": {"nested": [1, 2]}}

        (parsed,) = parse_audio_presets({"audio_model_presets": {"yue2": preset}})

        assert dict(parsed.settings) == {"anything": {"nested": [1, 2]}}

    def test_settings_may_be_left_out(self):
        preset = yue2_preset()
        del preset["settings"]
        del preset["platforms"]["mps"]["settings"]

        (parsed,) = parse_audio_presets({"audio_model_presets": {"yue2": preset}})

        assert dict(parsed.settings) == {} and dict(parsed.platforms["mps"].settings) == {}


class TestPlatformSettings:
    def test_a_platform_overrides_the_shared_settings(self):
        preset = higgs3_preset()
        preset["settings"]["codec_device"] = "cuda"
        (higgs3,) = parse_audio_presets({"audio_model_presets": {"higgs3": preset}})

        assert higgs3.settings_for("cuda")["codec_device"] == "cpu"
        assert higgs3.settings_for("mps")["codec_device"] == "mps"
        assert higgs3.settings_for("cuda")["temperature"] == 0.7

    def test_each_platform_adds_its_own_keys(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert yue2.settings_for("cuda")["weight_swap"] is True
        assert yue2.settings_for("mps")["weight_swap"] is False
        assert "weight_swap" not in yue2.settings

    def test_the_merged_settings_are_a_copy(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        yue2.settings_for("cuda")["cfg"] = 5.0

        assert yue2.settings_for("cuda")["cfg"] == 1.0

    def test_a_platform_the_preset_does_not_run_on_is_refused(self):
        preset = yue2_preset()
        del preset["platforms"]["mps"]
        (yue2,) = parse_audio_presets({"audio_model_presets": {"yue2": preset}})

        with pytest.raises(ValueError, match="does not run on 'mps'"):
            yue2.settings_for("mps")


class TestRunnable:
    def test_a_preset_without_an_entry_for_the_platform_is_not_runnable(self):
        sa3 = sa3_preset()
        del sa3["platforms"]["mps"]
        config = audio_config(yue2=yue2_preset(), sa3=sa3, higgs3=higgs3_preset())

        assert [preset.id for preset in runnable_audio_presets(config, "mps")] == ["yue2", "higgs3"]
        assert [preset.id for preset in runnable_audio_presets(config, "cuda")] == ["yue2", "sa3", "higgs3"]

    def test_the_order_is_the_config_order(self):
        config = audio_config(sa3=sa3_preset(), yue2=yue2_preset())

        assert [preset.id for preset in runnable_audio_presets(config, "cuda")] == ["sa3", "yue2"]

    def test_an_unknown_platform_runs_nothing(self):
        assert runnable_audio_presets(audio_config(), "rocm") == ()


class TestRefusals:
    def test_a_section_that_is_not_a_mapping(self):
        with pytest.raises(ValueError, match="'audio_model_presets' must be a mapping"):
            parse_audio_presets({"audio_model_presets": ["yue2"]})

    def test_a_preset_that_is_not_a_mapping(self):
        with pytest.raises(ValueError, match="audio_model_presets.yue2. must be a mapping"):
            parse_audio_presets({"audio_model_presets": {"yue2": "x"}})

    @pytest.mark.parametrize("key", ["label", "kind", "capabilities", "limits", "memory", "platforms"])
    def test_a_missing_required_key(self, key):
        with pytest.raises(ValueError, match=f"missing the required key '{key}'"):
            _parse_one(**{key: None})

    def test_an_unknown_kind(self):
        with pytest.raises(ValueError, match="kind must be one of music, speech, sound, got 'video'"):
            _parse_one(kind="video")

    @pytest.mark.parametrize("name", ["lyrics", "duration", "voice_clone"])
    def test_a_missing_capability(self, name):
        caps = yue2_preset()["capabilities"]
        del caps[name]

        with pytest.raises(ValueError, match=f"capabilities.{name} is required"):
            _parse_one(capabilities=caps)

    def test_a_capability_that_is_not_a_boolean(self):
        with pytest.raises(ValueError, match="capabilities.lyrics must be true or false"):
            _parse_one(capabilities={"lyrics": "yes", "duration": False, "voice_clone": False})

    def test_an_unknown_capability(self):
        with pytest.raises(ValueError, match="capabilities.remix is not a known key"):
            _parse_one(capabilities={"lyrics": True, "duration": False, "voice_clone": False, "remix": True})

    def test_a_music_model_without_max_seconds(self):
        with pytest.raises(ValueError, match="limits.max_seconds is required for a music model"):
            _parse_one(limits={})

    @pytest.mark.parametrize("name", ["max_new_tokens", "part_chars", "max_text_chars", "min_voice_seconds", "max_voice_seconds"])
    def test_a_speech_model_without_one_of_its_limits(self, name):
        preset = higgs3_preset()
        del preset["limits"][name]

        with pytest.raises(ValueError, match=f"limits.{name} is required for a speech model"):
            parse_audio_presets({"audio_model_presets": {"higgs3": preset}})

    def test_a_limit_that_belongs_to_another_kind(self):
        with pytest.raises(ValueError, match="limits.part_chars is not a known key"):
            _parse_one(limits={"max_seconds": 360, "part_chars": 500})

    @pytest.mark.parametrize("value", [0, -1, "360", True, None])
    def test_a_limit_that_is_not_a_positive_number(self, value):
        with pytest.raises(ValueError, match="limits.max_seconds must be a number above 0"):
            _parse_one(limits={"max_seconds": value})

    def test_a_token_limit_must_be_an_integer(self):
        preset = higgs3_preset()
        preset["limits"]["max_new_tokens"] = 1500.5

        with pytest.raises(ValueError, match="limits.max_new_tokens must be an integer above 0"):
            parse_audio_presets({"audio_model_presets": {"higgs3": preset}})

    def test_memory_needs_both_values_but_mps_may_be_null(self):
        with pytest.raises(ValueError, match="memory.mps_gb is required"):
            _parse_one(memory={"cuda_gb": 8.8})
        with pytest.raises(ValueError, match="memory.cuda_gb must be a number above 0"):
            _parse_one(memory={"cuda_gb": None, "mps_gb": None})
        assert _parse_one(memory={"cuda_gb": 8.8, "mps_gb": 12})[0].memory == AudioMemory(8.8, 12.0)

    def test_platforms_must_name_one_known_platform(self):
        with pytest.raises(ValueError, match="platforms must name at least one platform"):
            _parse_one(platforms={})
        with pytest.raises(ValueError, match="platforms.rocm is not a known key"):
            _parse_one(platforms={"rocm": yue2_preset()["platforms"]["cuda"]})

    def test_a_platform_without_a_family(self):
        platform = yue2_preset()["platforms"]["cuda"]
        del platform["family"]

        with pytest.raises(ValueError, match="platforms.cuda is missing the required key 'family'"):
            _parse_one(platforms={"cuda": platform})

    def test_a_platform_without_files(self):
        platform = yue2_preset()["platforms"]["cuda"]
        del platform["files"]

        with pytest.raises(ValueError, match="platforms.cuda is missing the required key 'files'"):
            _parse_one(platforms={"cuda": platform})

    def test_a_platform_with_an_empty_file_list(self):
        platform = yue2_preset()["platforms"]["cuda"]
        platform["files"] = []

        with pytest.raises(ValueError, match="platforms.cuda.files must be a non-empty list"):
            _parse_one(platforms={"cuda": platform})

    def test_a_platform_with_a_blank_family(self):
        platform = yue2_preset()["platforms"]["cuda"]
        platform["family"] = " "

        with pytest.raises(ValueError, match="platforms.cuda.family must be a non-empty string"):
            _parse_one(platforms={"cuda": platform})

    @pytest.mark.parametrize("repo", ["m-a-p/YuE2-3B", "m-a-p/YuE2-3B@main", "m-a-p/YuE2-3B@8f24312f"])
    def test_a_repo_that_is_not_pinned_to_a_commit(self, repo):
        platform = yue2_preset()["platforms"]["cuda"]
        platform["files"] = [{"repo": repo, "files": ["model.safetensors"]}]

        with pytest.raises(ValueError, match="must be pinned to a 40-character commit"):
            _parse_one(platforms={"cuda": platform})

    @pytest.mark.parametrize("files", [None, [], [""], "model.safetensors"])
    def test_a_repo_without_file_names(self, files):
        platform = yue2_preset()["platforms"]["cuda"]
        platform["files"] = [{"repo": HIGGS_CODEC_REPO, "files": files}]

        with pytest.raises(ValueError, match="files must be a non-empty list of file names or patterns"):
            _parse_one(platforms={"cuda": platform})

    def test_the_error_names_the_preset(self):
        with pytest.raises(ValueError, match="config 'audio_model_presets.yue2'"):
            _parse_one(kind="video")


class TestLoadConfig:
    """``load_config`` refuses a malformed section, so the app does not start with it."""

    def _load_with_user_config(self, monkeypatch, tmp_path, text: str):
        (tmp_path / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        return config_module.load_config()

    def test_the_packaged_section_holds_the_three_models(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))

        assert list(config_module.load_config()["audio_model_presets"]) == ["yue2", "higgs3", "sa3"]

    def test_a_section_that_is_not_a_mapping_is_refused(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="'audio_model_presets' must be a mapping"):
            self._load_with_user_config(monkeypatch, tmp_path, "audio_model_presets: [yue2]\n")

    def test_a_preset_with_an_unknown_kind_is_refused(self, monkeypatch, tmp_path):
        text = """
        audio_model_presets:
          demo:
            label: Demo
            kind: video
            capabilities: {lyrics: false, duration: false, voice_clone: false}
            limits: {max_seconds: 10}
            memory: {cuda_gb: 1, mps_gb: null}
            platforms: {}
        """
        with pytest.raises(ValueError, match="kind must be one of"):
            self._load_with_user_config(monkeypatch, tmp_path, text)

    def test_a_valid_preset_loads_and_a_licence_may_cover_it(self, monkeypatch, tmp_path):
        text = f"""
        audio_model_presets:
          demo:
            label: Demo
            kind: sound
            capabilities: {{lyrics: false, duration: true, voice_clone: false}}
            limits: {{max_seconds: 10}}
            memory: {{cuda_gb: 1, mps_gb: null}}
            platforms:
              cuda: {{family: stable_audio_3, files: [{{repo: "{HIGGS_CODEC_REPO}", files: [config.json]}}]}}
        licences:
          demo-licence:
            name: Demo Licence
            supplier: Demo
            summary: Free
            url: https://huggingface.co/demo/demo/blob/{"a" * 40}/LICENSE
            models: [demo]
        """
        config = self._load_with_user_config(monkeypatch, tmp_path, text)

        assert [preset.id for preset in runnable_audio_presets(config, "cuda")] == ["yue2", "higgs3", "sa3", "demo"]
        assert [record.id for record in parse_licences(config) if "demo" in record.models] == ["demo-licence"]


class TestConfigurationDocExample:
    """The example in the configuration docs is the three presets, so it is parsed and checked like the config."""

    def _example(self) -> dict:
        text = (REPO_ROOT / "docs" / "concepts" / "configuration.md").read_text(encoding="utf-8")
        section = text.split("## `audio_model_presets`", 1)[1].split("\n## ", 1)[0]
        block = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
        assert block, "the audio_model_presets section needs a yaml example"
        return yaml.safe_load(block.group(1))

    def test_the_example_parses(self):
        presets = parse_audio_presets(self._example())

        assert [preset.id for preset in presets] == ["yue2", "higgs3", "sa3"]

    def test_the_example_holds_the_shipped_values(self):
        example = self._example()["audio_model_presets"]
        expected = audio_config()["audio_model_presets"]

        for model_id, preset in expected.items():
            documented = example[model_id]
            for key in ("label", "kind", "capabilities", "limits", "settings", "memory"):
                assert documented[key] == preset[key], f"{model_id}.{key}"
            assert documented["platforms"]["cuda"] == preset["platforms"]["cuda"], f"{model_id}.platforms.cuda"

    def test_the_example_has_the_yue2_mps_platform(self):
        assert self._example()["audio_model_presets"]["yue2"]["platforms"]["mps"] == yue2_preset()["platforms"]["mps"]


class TestShippedPresets:
    """The packaged ``config.yaml`` ships the three presets the tests use as fixtures."""

    def test_the_packaged_config_holds_the_fixture_presets(self):
        shipped = load_packaged_config()["audio_model_presets"]

        assert list(shipped) == ["yue2", "higgs3", "sa3"]
        for model_id, preset in audio_config()["audio_model_presets"].items():
            assert shipped[model_id] == preset, model_id

    @pytest.mark.parametrize("platform", ["cuda", "mps"])
    def test_every_platform_runs_all_three(self, platform):
        presets = runnable_audio_presets(load_packaged_config(), platform)

        assert [preset.id for preset in presets] == ["yue2", "higgs3", "sa3"]
