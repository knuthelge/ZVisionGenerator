"""Tests for the audio preset parser: the three presets, each refusal and the platform settings and memory."""

from __future__ import annotations

import dataclasses
import re
import textwrap

import pytest
import yaml

from tests._audio_presets import HIGGS_CODEC_REPO, audio_config, higgs3_preset, sa3_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._paths import REPO_ROOT

from zvisiongenerator.utils import config as config_module
from zvisiongenerator.utils.audio_preset import AudioLimits, parse_audio_presets


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

    def test_higgs3_values(self):
        (higgs3,) = parse_audio_presets(audio_config(higgs3=higgs3_preset()))

        assert (higgs3.takes_lyrics, higgs3.sets_length, higgs3.follows_voice) == (False, False, True)
        assert higgs3.limits == AudioLimits(max_new_tokens=1500, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)
        assert dict(higgs3.settings) == {"temperature": 0.7, "top_p": 0.95, "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}
        assert higgs3.family == "higgs3"
        assert [files.repo_id for files in higgs3.files][-1] == "bosonai/higgs-audio-v2-tokenizer"

    def test_the_values_are_frozen_and_do_not_alias_the_config(self):
        config = audio_config(yue2=yue2_preset())
        (yue2,) = parse_audio_presets(config)

        with pytest.raises(dataclasses.FrozenInstanceError):
            yue2.label = "other"  # type: ignore[misc]
        with pytest.raises(TypeError):
            yue2.settings["cfg"] = 2.0  # type: ignore[index]
        config["audio_model_presets"]["yue2"]["settings"]["cfg"] = 9.0
        assert yue2.settings["cfg"] == 1.0


class TestKinds:
    def test_what_a_model_can_do_follows_from_its_kind(self):
        presets = {preset.id: preset for preset in parse_audio_presets(audio_config())}

        assert [(p.takes_lyrics, p.sets_length, p.follows_voice) for p in (presets["yue2"], presets["sa3"], presets["higgs3"])] == [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]


class TestPlatformSettings:
    def test_a_platform_overrides_the_shared_settings(self):
        preset = higgs3_preset()
        preset["platforms"] = {"cuda": {"memory_gb": 8.2, "settings": {"temperature": 0.5}}}
        (higgs3,) = parse_audio_presets({"audio_model_presets": {"higgs3": preset}})

        assert higgs3.settings_for("cuda")["temperature"] == 0.5
        assert higgs3.settings_for("mps")["temperature"] == 0.7
        assert higgs3.settings_for("cuda")["top_p"] == 0.95

    def test_a_platform_adds_a_key_the_shared_settings_lack(self):
        preset = yue2_preset()
        preset["platforms"] = {"cuda": {"memory_gb": 6.7, "settings": {"weight_swap": True}}}
        (yue2,) = parse_audio_presets(audio_config(yue2=preset))

        assert yue2.settings_for("cuda")["weight_swap"] is True
        assert "weight_swap" not in yue2.settings_for("mps")

    def test_the_packaged_presets_name_no_device_in_their_shared_settings(self):
        presets = parse_audio_presets(load_packaged_config())

        assert not [preset.id for preset in presets if {"weight_swap", "codec_device"} & set(preset.settings)]

    def test_the_memory_peak_belongs_to_the_platform_it_was_measured_on(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert yue2.memory_gb_for("cuda") == 6.7
        assert yue2.memory_gb_for("mps") is None

    def test_a_preset_without_platforms_uses_the_shared_values_everywhere(self):
        preset = sa3_preset()
        del preset["platforms"]
        (sa3,) = parse_audio_presets(audio_config(sa3=preset))

        assert sa3.settings_for("mps") == dict(sa3.settings)
        assert sa3.memory_gb_for("cuda") is None


class TestParseOnce:
    def test_the_same_section_is_parsed_once_and_a_changed_one_again(self):
        config = audio_config(yue2=yue2_preset())

        first = parse_audio_presets(config)
        assert parse_audio_presets(config) is first
        config["audio_model_presets"]["yue2"]["label"] = "Other"
        assert parse_audio_presets(config)[0].label == "Other"


class TestRefusals:
    @pytest.mark.parametrize("key", ["label", "kind", "family", "files", "limits"])
    def test_a_missing_required_key(self, key):
        with pytest.raises(ValueError, match=f"missing the required key '{key}'"):
            _parse_one(**{key: None})

    def test_an_unknown_kind(self):
        with pytest.raises(ValueError, match="kind must be one of music, speech, sound, got 'video'"):
            _parse_one(kind="video")

    def test_a_key_that_is_not_a_preset_key(self):
        with pytest.raises(ValueError, match="capabilities is not a known key"):
            _parse_one(capabilities={"lyrics": True})

    def test_a_platform_that_is_not_cuda_or_mps(self):
        with pytest.raises(ValueError, match="platforms.rocm is not a known key"):
            _parse_one(platforms={"rocm": {}})

    def test_a_platform_key_that_is_not_an_override(self):
        with pytest.raises(ValueError, match="platforms.cuda.files is not a known key"):
            _parse_one(platforms={"cuda": {"files": []}})

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

    @pytest.mark.parametrize("repo", ["m-a-p/YuE2-3B", "m-a-p/YuE2-3B@main", "m-a-p/YuE2-3B@8f24312f"])
    def test_a_repo_that_is_not_pinned_to_a_commit(self, repo):
        with pytest.raises(ValueError, match="must be pinned to a 40-character commit"):
            _parse_one(files=[{"repo": repo, "files": ["model.safetensors"]}])

    def test_files_that_are_not_a_list(self):
        with pytest.raises(ValueError, match="files must be a non-empty list"):
            _parse_one(files=[])


class TestLoadConfig:
    """``load_config`` refuses a malformed section, so the app does not start with it."""

    def _load_with_user_config(self, monkeypatch, tmp_path, text: str):
        (tmp_path / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        return config_module.load_config()

    def test_a_preset_with_an_unknown_kind_is_refused(self, monkeypatch, tmp_path):
        text = f"""
        audio_model_presets:
          demo:
            label: Demo
            kind: video
            family: stable_audio_3
            files: [{{repo: "{HIGGS_CODEC_REPO}", files: [config.json]}}]
            limits: {{max_seconds: 10}}
        """
        with pytest.raises(ValueError, match="kind must be one of"):
            self._load_with_user_config(monkeypatch, tmp_path, text)

    def test_a_valid_preset_loads_and_its_id_is_a_known_model(self, monkeypatch, tmp_path):
        text = f"""
        audio_model_presets:
          demo:
            label: Demo
            kind: sound
            family: stable_audio_3
            files: [{{repo: "{HIGGS_CODEC_REPO}", files: [config.json]}}]
            limits: {{max_seconds: 10}}
            platforms:
              cuda: {{memory_gb: 1}}
        """
        config = self._load_with_user_config(monkeypatch, tmp_path, text)

        assert [preset.id for preset in parse_audio_presets(config)] == ["yue2", "higgs3", "sa3", "demo"]


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
            for key in ("label", "kind", "family", "files", "limits", "settings", "platforms"):
                assert documented[key] == preset[key], f"{model_id}.{key}"


class TestShippedPresets:
    """The packaged ``config.yaml`` ships the three presets the tests use as fixtures."""

    def test_the_packaged_config_holds_the_fixture_presets(self):
        shipped = load_packaged_config()["audio_model_presets"]

        assert list(shipped) == ["yue2", "higgs3", "sa3"]
        for model_id, preset in audio_config()["audio_model_presets"].items():
            assert shipped[model_id] == preset, model_id
