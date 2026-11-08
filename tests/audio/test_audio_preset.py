"""Tests for the audio preset parser: the three presets, each refusal and the platform settings and memory."""

from __future__ import annotations

import copy
import dataclasses
import re
import textwrap
from typing import Any

import pytest
import yaml

from tests._audio_presets import QWEN_BASE_REPO, QWEN_CUSTOM_REPO, YUE2_REPO, YUE2_VAE_REPO, audio_config, qwen3_tts_preset, sa3_two_kind_preset, yue2_preset
from tests._factories import load_packaged_config, packaged_metadata_copy, use_model_metadata
from tests._paths import REPO_ROOT

from zvisiongenerator.utils import config as config_module
from zvisiongenerator.utils.audio_preset import AudioLimits, AudioVoice, parse_audio_presets
from zvisiongenerator.utils.model_metadata import load_model_metadata, model_ids


def _packaged_entry(model_id: str) -> dict[str, Any]:
    """Return a private copy of the packaged metadata entry of *model_id*."""
    return packaged_metadata_copy()["models"][model_id]


def _serve(monkeypatch, tmp_path, model_id: str, entry: dict[str, Any]) -> None:
    """Serve the packaged metadata with the ``models`` entry *model_id* replaced by *entry*."""
    data = packaged_metadata_copy()
    data["models"][model_id] = entry
    use_model_metadata(monkeypatch, tmp_path, data)


class TestParse:
    def test_the_presets_come_in_metadata_order_keyed_by_model_id(self):
        assert [preset.id for preset in parse_audio_presets(audio_config())] == list(model_ids("audio"))

    def test_the_packaged_presets_name_their_kinds_and_labels(self):
        presets = {preset.id: preset for preset in parse_audio_presets(audio_config())}

        assert [(presets[model_id].kinds, presets[model_id].label) for model_id in ("yue2", "qwen3_tts", "sa3")] == [
            (("music",), "YuE2 3B"),
            (("speech",), "Qwen3-TTS 1.7B"),
            (("sound", "music"), "Stable Audio 3 Medium"),
        ]

    def test_qwen3_tts_values(self):
        qwen = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "qwen3_tts")

        assert (qwen.takes_lyrics("speech"), qwen.sets_length("speech"), qwen.follows_voice("speech")) == (False, False, True)
        assert qwen.limits == AudioLimits(max_new_tokens=1200, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)
        assert dict(qwen.settings) == {"language": "auto", "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}
        assert qwen.family == "qwen3_tts"
        assert [files.repo_id for files in qwen.files] == [QWEN_CUSTOM_REPO.split("@")[0]]

    def test_the_files_are_the_declared_targets_and_the_voice_files_the_voice_role(self):
        presets = {preset.id: preset for preset in parse_audio_presets(audio_config())}

        assert [(files.repo_id, files.revision) for files in presets["yue2"].files] == [("m-a-p/YuE2-3B", YUE2_REPO.split("@")[1]), ("m-a-p/YuE2-Vae", YUE2_VAE_REPO.split("@")[1])]
        assert presets["yue2"].voice_files == ()
        assert [(files.repo_id, files.revision) for files in presets["qwen3_tts"].voice_files] == [("Qwen/Qwen3-TTS-12Hz-1.7B-Base", QWEN_BASE_REPO.split("@")[1])]
        assert "speech_tokenizer/model.safetensors" in presets["qwen3_tts"].voice_files[0].patterns

    def test_a_model_without_a_config_block_has_no_settings(self):
        sa3 = next(preset for preset in parse_audio_presets({"audio_model_presets": {"yue2": audio_config()["audio_model_presets"]["yue2"]}}) if preset.id == "sa3")

        assert dict(sa3.settings) == {}

    def test_a_config_id_with_no_entry_is_ignored(self):
        config = audio_config()
        config["audio_model_presets"]["demo"] = {"settings": {"cfg": 1.0}}

        assert [preset.id for preset in parse_audio_presets(config)] == list(model_ids("audio"))

    def test_the_other_keys_of_a_config_entry_are_ignored(self):
        config = audio_config()
        config["audio_model_presets"]["sa3"].update({"label": "Other", "kinds": ["speech"], "files": []})

        sa3 = next(preset for preset in parse_audio_presets(config) if preset.id == "sa3")

        assert (sa3.label, sa3.kinds) == ("Stable Audio 3 Medium", ("sound", "music"))

    def test_the_values_are_frozen_and_do_not_alias_the_config(self):
        config = audio_config()
        yue2 = parse_audio_presets(config)[0]

        with pytest.raises(dataclasses.FrozenInstanceError):
            yue2.label = "other"  # type: ignore[misc]
        with pytest.raises(TypeError):
            yue2.settings["cfg"] = 2.0  # type: ignore[index]
        config["audio_model_presets"]["yue2"]["settings"]["cfg"] = 9.0
        assert yue2.settings["cfg"] == 1.0

    def test_parsing_changes_neither_the_config_nor_the_metadata(self):
        config = audio_config()
        config_before = copy.deepcopy(config)
        metadata_before = copy.deepcopy(load_model_metadata())

        parse_audio_presets(config)

        assert config == config_before
        assert load_model_metadata() == metadata_before

    def test_a_changed_config_is_read_again(self):
        config = audio_config()
        parse_audio_presets(config)
        config["audio_model_presets"]["yue2"]["settings"]["cfg"] = 3.0

        assert parse_audio_presets(config)[0].settings["cfg"] == 3.0


class TestKinds:
    def test_what_a_model_can_do_follows_from_its_kind_and_its_flags(self):
        presets = {preset.id: preset for preset in parse_audio_presets(audio_config())}

        assert [(p.takes_lyrics(k), p.sets_length(k), p.follows_voice(k)) for p, k in ((presets["yue2"], "music"), (presets["sa3"], "sound"), (presets["qwen3_tts"], "speech"))] == [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]

    def test_a_model_decides_per_kind(self):
        sa3 = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "sa3")

        assert sa3.kinds == ("sound", "music")
        assert sa3.default_kind == "sound"
        assert [(sa3.has_kind(k), sa3.takes_lyrics(k), sa3.sets_length(k), sa3.follows_voice(k)) for k in ("sound", "music", "speech")] == [
            (True, False, True, False),
            (True, False, True, False),
            (False, False, False, False),
        ]

    def test_the_first_kind_is_the_default(self, monkeypatch, tmp_path):
        entry = _packaged_entry("sa3")
        entry["kinds"] = ["music", "sound"]
        _serve(monkeypatch, tmp_path, "sa3", entry)

        sa3 = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "sa3")

        assert sa3.default_kind == "music"

    def test_a_model_with_lyrics_follows_them_for_its_length(self):
        yue2 = parse_audio_presets(audio_config())[0]

        assert yue2.lyrics is True
        assert (yue2.takes_lyrics("music"), yue2.sets_length("music")) == (True, False)
        assert (yue2.takes_lyrics("sound"), yue2.sets_length("sound")) == (False, False)

    def test_the_limits_are_the_entrys_limits(self, monkeypatch, tmp_path):
        entry = _packaged_entry("sa3")
        entry["kinds"] = ["sound", "speech"]
        entry["limits"] = {**qwen3_tts_preset()["limits"], "max_seconds": 380}
        _serve(monkeypatch, tmp_path, "sa3", entry)

        model = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "sa3")

        assert model.limits == AudioLimits(max_seconds=380, max_new_tokens=1200, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)


class TestPlatformSettings:
    def test_a_platform_overrides_the_shared_settings(self):
        config = audio_config()
        config["audio_model_presets"]["qwen3_tts"]["platforms"] = {"cuda": {"settings": {"language": "English"}}}
        qwen = next(preset for preset in parse_audio_presets(config) if preset.id == "qwen3_tts")

        assert qwen.settings_for("cuda")["language"] == "English"
        assert qwen.settings_for("mps")["language"] == "auto"
        assert qwen.settings_for("cuda")["join_pause_ms"] == 300

    def test_a_platform_adds_a_key_the_shared_settings_lack(self):
        config = audio_config()
        config["audio_model_presets"]["yue2"]["platforms"] = {"cuda": {"settings": {"weight_swap": True}}}
        yue2 = parse_audio_presets(config)[0]

        assert yue2.settings_for("cuda")["weight_swap"] is True
        assert "weight_swap" not in yue2.settings_for("mps")

    def test_the_packaged_presets_name_no_device_in_their_shared_settings(self):
        presets = parse_audio_presets(load_packaged_config())

        assert not [preset.id for preset in presets if {"weight_swap", "codec_device"} & set(preset.settings)]

    def test_the_memory_peak_belongs_to_the_platform_it_was_measured_on(self, monkeypatch, tmp_path):
        entry = _packaged_entry("yue2")
        del entry["platforms"]["mps"]
        _serve(monkeypatch, tmp_path, "yue2", entry)

        yue2 = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "yue2")

        assert (yue2.memory_gb_for("cuda"), yue2.memory_gb_for("mps")) == (6.7, None)

    def test_a_preset_without_platforms_uses_the_shared_values_everywhere(self, monkeypatch, tmp_path):
        entry = _packaged_entry("sa3")
        del entry["platforms"]
        _serve(monkeypatch, tmp_path, "sa3", entry)
        sa3 = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "sa3")

        assert sa3.settings_for("mps") == dict(sa3.settings)
        assert sa3.memory_gb_for("cuda") is None


class TestPlatformBackend:
    def test_a_platform_runs_its_own_family_and_keeps_the_presets_files(self):
        qwen = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "qwen3_tts")

        mac = qwen.on_platform("mps")

        assert (mac.family, mac.files, mac.voice_files) == ("qwen3_tts_mlx", qwen.files, qwen.voice_files)
        assert qwen.on_platform("cuda") == qwen


class TestRefusals:
    def test_a_model_with_lyrics_needs_the_section_list_of_an_empty_lyrics_box(self):
        config = audio_config()
        del config["audio_model_presets"]["yue2"]["settings"]["instrumental_sections"]

        with pytest.raises(ValueError, match="instrumental_sections is required for a model that sings lyrics"):
            parse_audio_presets(config)

    def test_a_model_without_lyrics_needs_no_section_list(self, monkeypatch, tmp_path):
        entry = _packaged_entry("yue2")
        del entry["lyrics"]
        _serve(monkeypatch, tmp_path, "yue2", entry)
        config = audio_config()
        config["audio_model_presets"]["yue2"]["settings"] = {}

        assert parse_audio_presets(config)[0].lyrics is False

    @pytest.mark.parametrize("section", [[], "yue2"], ids=["list", "text"])
    def test_the_section_must_be_a_mapping(self, section):
        with pytest.raises(ValueError, match="audio_model_presets' must be a mapping"):
            parse_audio_presets({"audio_model_presets": section})

    def test_an_entry_must_be_a_mapping(self):
        config = audio_config()
        config["audio_model_presets"]["sa3"] = ["steps"]

        with pytest.raises(ValueError, match=r"audio_model_presets\.sa3' must be a mapping"):
            parse_audio_presets(config)

    def test_settings_must_be_a_mapping(self):
        config = audio_config()
        config["audio_model_presets"]["sa3"]["settings"] = 8

        with pytest.raises(ValueError, match=r"audio_model_presets\.sa3'\.settings must be a mapping"):
            parse_audio_presets(config)

    def test_platform_settings_must_be_a_mapping(self):
        config = audio_config()
        config["audio_model_presets"]["sa3"]["platforms"] = {"cuda": {"settings": 8}}

        with pytest.raises(ValueError, match=r"platforms\.cuda\.settings must be a mapping"):
            parse_audio_presets(config)


class TestLoadConfig:
    """``load_config`` takes the audio settings as they are: the audio models are the metadata's, so a user entry cannot add one."""

    def _load_with_user_config(self, monkeypatch, tmp_path, text: str):
        (tmp_path / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        return config_module.load_config()

    def test_a_user_entry_that_names_no_model_adds_none(self, monkeypatch, tmp_path):
        text = """
        audio_model_presets:
          demo:
            settings: {cfg: 1.0}
        """
        config = self._load_with_user_config(monkeypatch, tmp_path, text)

        assert [preset.id for preset in parse_audio_presets(config)] == list(model_ids("audio"))

    def test_a_user_setting_is_merged_over_the_shipped_one(self, monkeypatch, tmp_path):
        text = """
        audio_model_presets:
          sa3:
            settings: {steps: 12}
        """
        config = self._load_with_user_config(monkeypatch, tmp_path, text)

        sa3 = next(preset for preset in parse_audio_presets(config) if preset.id == "sa3")
        assert (sa3.settings["steps"], sa3.settings["cfg_scale"]) == (12, 1.0)


class TestConfigurationDocExample:
    """The example in the configuration docs is the three presets' settings, so it is parsed and checked like the config."""

    def _example(self) -> dict:
        text = (REPO_ROOT / "docs" / "concepts" / "configuration.md").read_text(encoding="utf-8")
        section = text.split("## `audio_model_presets`", 1)[1].split("\n## ", 1)[0]
        block = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
        assert block, "the audio_model_presets section needs a yaml example"
        return yaml.safe_load(block.group(1))

    def test_the_example_parses(self):
        presets = parse_audio_presets(self._example())

        assert [preset.id for preset in presets] == list(model_ids("audio"))

    def test_the_example_holds_the_shipped_values(self):
        example = self._example()["audio_model_presets"]
        expected = _shipped_presets()

        for model_id, preset in expected.items():
            documented = example[model_id]
            assert documented["settings"] == preset["settings"], model_id


def _shipped_presets() -> dict[str, Any]:
    """Return the three fixture presets: facts and settings in one block, Stable Audio making sound and music."""
    return {"yue2": yue2_preset(), "qwen3_tts": qwen3_tts_preset(), "sa3": sa3_two_kind_preset()}


class TestShippedPresets:
    """The packaged ``config.yaml`` holds the settings of the three presets the tests use as fixtures, and nothing else."""

    def test_the_packaged_config_holds_the_fixture_settings(self):
        shipped = load_packaged_config()["audio_model_presets"]

        assert list(shipped) == list(model_ids("audio"))
        for model_id, preset in _shipped_presets().items():
            assert shipped[model_id] == {"settings": preset["settings"]}, model_id


class TestVoices:
    """A speech model lists its built-in voices, the first of them the default."""

    def test_the_voices_keep_their_order_and_the_first_is_the_default(self):
        preset = next(item for item in parse_audio_presets(audio_config()) if item.id == "qwen3_tts")

        assert [voice.id for voice in preset.voices][:3] == ["ryan", "aiden", "vivian"]
        assert preset.voices[0] == AudioVoice("ryan", "Ryan", "English")
        assert preset.default_voice == "ryan"
        assert len(preset.voices) == 9

    def test_a_model_without_voices_has_no_default_voice(self):
        yue2 = parse_audio_presets(audio_config())[0]

        assert (yue2.voices, yue2.default_voice) == ((), None)


class TestVoiceFiles:
    """The cloning files are the targets of the ``voice`` role."""

    def test_voice_files_are_the_voice_role_targets(self):
        preset = next(item for item in parse_audio_presets(audio_config()) if item.id == "qwen3_tts")

        assert [(files.repo_id, files.revision) for files in preset.voice_files] == [("Qwen/Qwen3-TTS-12Hz-1.7B-Base", QWEN_BASE_REPO.split("@")[1])]
        assert "speech_tokenizer/model.safetensors" in preset.voice_files[0].patterns
        assert [files.repo_id for files in preset.files] == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]

    def test_a_model_without_voice_targets_has_no_voice_files(self):
        yue2 = parse_audio_presets(audio_config())[0]

        assert yue2.voice_files == ()
