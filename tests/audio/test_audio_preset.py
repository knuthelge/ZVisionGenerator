"""Tests for the audio preset parser: the three presets, each refusal and the platform settings and memory."""

from __future__ import annotations

import dataclasses
import re
import textwrap
from typing import Any

import pytest
import yaml

from tests._audio_presets import QWEN_BASE_REPO, QWEN_CUSTOM_REPO, audio_config, qwen3_tts_preset, sa3_preset, sa3_two_kind_preset, yue2_preset
from tests._factories import load_packaged_config
from tests._paths import REPO_ROOT

from zvisiongenerator.utils import config as config_module
from zvisiongenerator.utils.audio_preset import AudioLimits, AudioVoice, parse_audio_presets


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

        assert [(preset.id, preset.kinds, preset.label) for preset in presets] == [
            ("yue2", ("music",), "YuE2 3B"),
            ("qwen3_tts", ("speech",), "Qwen3-TTS 1.7B"),
            ("sa3", ("sound",), "Stable Audio 3 Medium"),
        ]

    def test_qwen3_tts_values(self):
        (qwen,) = parse_audio_presets(audio_config(qwen3_tts=qwen3_tts_preset()))

        assert (qwen.takes_lyrics("speech"), qwen.sets_length("speech"), qwen.follows_voice("speech")) == (False, False, True)
        assert qwen.limits == AudioLimits(max_new_tokens=1200, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)
        assert dict(qwen.settings) == {"language": "auto", "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}
        assert qwen.family == "qwen3_tts"
        assert [files.repo_id for files in qwen.files] == [QWEN_CUSTOM_REPO.split("@")[0]]

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
    def test_what_a_model_can_do_follows_from_its_kind_and_its_flags(self):
        presets = {preset.id: preset for preset in parse_audio_presets(audio_config())}

        assert [(p.takes_lyrics(k), p.sets_length(k), p.follows_voice(k)) for p, k in ((presets["yue2"], "music"), (presets["sa3"], "sound"), (presets["qwen3_tts"], "speech"))] == [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]

    def test_a_model_decides_per_kind(self):
        (sa3,) = parse_audio_presets(audio_config(sa3=sa3_two_kind_preset()))

        assert sa3.kinds == ("sound", "music")
        assert sa3.default_kind == "sound"
        assert [(sa3.has_kind(k), sa3.takes_lyrics(k), sa3.sets_length(k), sa3.follows_voice(k)) for k in ("sound", "music", "speech")] == [
            (True, False, True, False),
            (True, False, True, False),
            (False, False, False, False),
        ]

    def test_the_first_kind_is_the_default(self):
        preset = sa3_two_kind_preset()
        preset["kinds"] = ["music", "sound"]
        (sa3,) = parse_audio_presets(audio_config(sa3=preset))

        assert sa3.default_kind == "music"

    def test_a_model_with_lyrics_follows_them_for_its_length(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert yue2.lyrics is True
        assert (yue2.takes_lyrics("music"), yue2.sets_length("music")) == (True, False)
        assert (yue2.takes_lyrics("sound"), yue2.sets_length("sound")) == (False, False)

    def test_the_limits_are_the_union_over_the_kinds(self):
        preset = sa3_two_kind_preset()
        preset["kinds"] = ["sound", "speech"]
        preset["limits"] = {**qwen3_tts_preset()["limits"], "max_seconds": 380}
        (model,) = parse_audio_presets(audio_config(sa3=preset))

        assert model.limits == AudioLimits(max_seconds=380, max_new_tokens=1200, part_chars=500, max_text_chars=10000, min_voice_seconds=3, max_voice_seconds=30)

    def test_a_voice_model_may_carry_voices_next_to_another_kind(self):
        preset = qwen3_tts_preset()
        preset["kinds"] = ["speech", "sound"]
        preset["limits"]["max_seconds"] = 30
        (model,) = parse_audio_presets(audio_config(qwen3_tts=preset))

        assert model.default_voice == "ryan"
        assert model.limits.max_seconds == 30


class TestPlatformSettings:
    def test_a_platform_overrides_the_shared_settings(self):
        preset = qwen3_tts_preset()
        preset["platforms"] = {"cuda": {"memory_gb": 5.8, "settings": {"language": "English"}}}
        (qwen,) = parse_audio_presets({"audio_model_presets": {"qwen3_tts": preset}})

        assert qwen.settings_for("cuda")["language"] == "English"
        assert qwen.settings_for("mps")["language"] == "auto"
        assert qwen.settings_for("cuda")["join_pause_ms"] == 300

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
        preset = yue2_preset()
        del preset["platforms"]["mps"]
        (yue2,) = parse_audio_presets(audio_config(yue2=preset))

        assert yue2.memory_gb_for("cuda") == 6.7
        assert yue2.memory_gb_for("mps") is None

    def test_a_preset_without_platforms_uses_the_shared_values_everywhere(self):
        preset = sa3_preset()
        del preset["platforms"]
        (sa3,) = parse_audio_presets(audio_config(sa3=preset))

        assert sa3.settings_for("mps") == dict(sa3.settings)
        assert sa3.memory_gb_for("cuda") is None


class TestPlatformBackend:
    def test_a_platform_runs_its_own_family_and_keeps_the_presets_files(self):
        (yue2,) = _parse_one(platforms={"mps": {"family": "yue2_mlx"}})

        mac = yue2.on_platform("mps")

        assert (mac.family, mac.files, mac.voice_files) == ("yue2_mlx", yue2.files, yue2.voice_files)
        assert yue2.on_platform("cuda") == yue2

    @pytest.mark.parametrize("key", ["files", "voice_files"])
    def test_a_platform_does_not_take_its_own_files(self, key):
        with pytest.raises(ValueError, match=f"platforms.mps.{key} is not a known key"):
            _parse_one(platforms={"mps": {key: [{"repo": "m-a-p/YuE2-3B@" + "a" * 40, "files": ["model.safetensors"]}]}})


class TestParseOnce:
    def test_the_same_section_is_parsed_once_and_a_changed_one_again(self):
        config = audio_config(yue2=yue2_preset())

        first = parse_audio_presets(config)
        assert parse_audio_presets(config) is first
        config["audio_model_presets"]["yue2"]["label"] = "Other"
        assert parse_audio_presets(config)[0].label == "Other"


class TestRefusals:
    @pytest.mark.parametrize("key", ["label", "kinds", "family", "files", "limits"])
    def test_a_missing_required_key(self, key):
        with pytest.raises(ValueError, match=f"missing the required key '{key}'"):
            _parse_one(**{key: None})

    @pytest.mark.parametrize("kinds", [[], "music"], ids=["empty", "not a list"])
    def test_kinds_must_be_a_non_empty_list(self, kinds):
        with pytest.raises(ValueError, match="kinds must be a non-empty list of kinds"):
            _parse_one(kinds=kinds)

    def test_an_unknown_kind(self):
        with pytest.raises(ValueError, match="kinds must only list music, speech, sound, got 'video'"):
            _parse_one(kinds=["music", "video"])

    def test_a_repeated_kind(self):
        with pytest.raises(ValueError, match="kinds must not repeat a kind"):
            _parse_one(kinds=["music", "music"])

    def test_the_single_kind_key_is_not_a_preset_key(self):
        with pytest.raises(ValueError, match="kind is not a known key"):
            _parse_one(kind="music")

    def test_lyrics_need_a_model_that_makes_music(self):
        with pytest.raises(ValueError, match="lyrics is only for a model that makes music"):
            _parse_one(kinds=["sound"], lyrics=True)

    def test_lyrics_need_the_section_list_of_an_empty_lyrics_box(self):
        settings = {key: value for key, value in yue2_preset()["settings"].items() if key != "instrumental_sections"}

        with pytest.raises(ValueError, match="instrumental_sections is required when lyrics is true"):
            _parse_one(settings=settings)

    def test_lyrics_must_be_a_boolean(self):
        with pytest.raises(ValueError, match="lyrics must be true or false"):
            _parse_one(lyrics="yes")

    def test_a_music_model_without_lyrics_needs_no_section_list(self):
        (model,) = _parse_one(lyrics=None, settings={})

        assert model.lyrics is False

    def test_voices_need_a_model_that_makes_speech(self):
        with pytest.raises(ValueError, match="voices is only for a model that makes speech"):
            _parse_one(voices=qwen3_tts_preset()["voices"])

    def test_a_key_that_is_not_a_preset_key(self):
        with pytest.raises(ValueError, match="capabilities is not a known key"):
            _parse_one(capabilities={"lyrics": True})

    def test_a_platform_that_is_not_cuda_or_mps(self):
        with pytest.raises(ValueError, match="platforms.rocm is not a known key"):
            _parse_one(platforms={"rocm": {}})

    def test_a_platform_key_that_is_not_an_override(self):
        with pytest.raises(ValueError, match="platforms.cuda.label is not a known key"):
            _parse_one(platforms={"cuda": {"label": "YuE2 on CUDA"}})

    def test_a_music_model_without_max_seconds(self):
        with pytest.raises(ValueError, match="limits.max_seconds is required for a model that makes music"):
            _parse_one(limits={})

    @pytest.mark.parametrize("name", ["max_new_tokens", "part_chars", "max_text_chars", "min_voice_seconds", "max_voice_seconds"])
    def test_a_speech_model_without_one_of_its_limits(self, name):
        preset = qwen3_tts_preset()
        del preset["limits"][name]

        with pytest.raises(ValueError, match=f"limits.{name} is required for a model that makes speech"):
            parse_audio_presets({"audio_model_presets": {"qwen3_tts": preset}})

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
            kinds: [video]
            family: stable_audio_3
            files: [{{repo: "{QWEN_BASE_REPO}", files: [config.json]}}]
            limits: {{max_seconds: 10}}
        """
        with pytest.raises(ValueError, match="kinds must only list"):
            self._load_with_user_config(monkeypatch, tmp_path, text)

    def test_a_valid_preset_loads_and_its_id_is_a_known_model(self, monkeypatch, tmp_path):
        text = f"""
        audio_model_presets:
          demo:
            label: Demo
            kinds: [sound, music]
            family: stable_audio_3
            files: [{{repo: "{QWEN_BASE_REPO}", files: [config.json]}}]
            limits: {{max_seconds: 10}}
            platforms:
              cuda: {{memory_gb: 1}}
        """
        config = self._load_with_user_config(monkeypatch, tmp_path, text)

        assert [preset.id for preset in parse_audio_presets(config)] == ["yue2", "qwen3_tts", "sa3", "demo"]


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

        assert [preset.id for preset in presets] == ["yue2", "qwen3_tts", "sa3"]

    def test_the_example_holds_the_shipped_values(self):
        example = self._example()["audio_model_presets"]
        expected = _shipped_presets()

        for model_id, preset in expected.items():
            documented = example[model_id]
            for key in ("label", "kinds", "family", "files", "limits", "settings", "platforms"):
                assert documented[key] == preset[key], f"{model_id}.{key}"
            assert documented.get("lyrics", False) == preset.get("lyrics", False), f"{model_id}.lyrics"


def _shipped_presets() -> dict[str, Any]:
    """Return the three presets as the packaged config ships them: Stable Audio makes sound and music."""
    return audio_config(yue2=yue2_preset(), qwen3_tts=qwen3_tts_preset(), sa3=sa3_two_kind_preset())["audio_model_presets"]


class TestShippedPresets:
    """The packaged ``config.yaml`` ships the three presets the tests use as fixtures."""

    def test_the_packaged_config_holds_the_fixture_presets(self):
        shipped = load_packaged_config()["audio_model_presets"]

        assert list(shipped) == ["yue2", "qwen3_tts", "sa3"]
        for model_id, preset in _shipped_presets().items():
            assert shipped[model_id] == preset, model_id
        assert shipped["qwen3_tts"] == qwen3_tts_preset()


class TestVoices:
    """A speech preset lists its built-in voices, the first of them the default."""

    def _parse(self, **changes):
        preset = qwen3_tts_preset()
        preset.update(changes)
        return parse_audio_presets({"audio_model_presets": {"qwen3_tts": preset}})[0]

    def test_the_voices_keep_their_order_and_the_first_is_the_default(self):
        preset = self._parse()

        assert [voice.id for voice in preset.voices][:3] == ["ryan", "aiden", "vivian"]
        assert preset.voices[0] == AudioVoice("ryan", "Ryan", "English")
        assert preset.default_voice == "ryan"
        assert len(preset.voices) == 9

    def test_a_preset_without_voices_has_no_default_voice(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert (yue2.voices, yue2.default_voice) == ((), None)

    @pytest.mark.parametrize(
        "voices",
        [[], "ryan", [{"id": "ryan", "label": "Ryan"}], [{"id": "ryan", "label": "Ryan", "language": "English", "x": 1}], [{"id": "a", "label": "A", "language": "x"}] * 2],
        ids=["empty", "not a list", "missing language", "unknown key", "repeated id"],
    )
    def test_malformed_voices_are_refused(self, voices):
        with pytest.raises(ValueError, match="voices"):
            self._parse(voices=voices)

    @pytest.mark.parametrize("key", ["voices", "voice_files"])
    def test_only_a_speech_model_has_voices_or_voice_files(self, key):
        preset = yue2_preset()
        preset[key] = qwen3_tts_preset()[key]

        with pytest.raises(ValueError, match=f"{key} is only for a model that makes speech"):
            parse_audio_presets({"audio_model_presets": {"yue2": preset}})


class TestVoiceFiles:
    """The cloning files are a second file list, parsed like ``files``."""

    def test_voice_files_are_parsed_like_files(self):
        (preset,) = parse_audio_presets(audio_config(qwen3_tts=qwen3_tts_preset()))

        assert [(files.repo_id, files.revision) for files in preset.voice_files] == [("Qwen/Qwen3-TTS-12Hz-1.7B-Base", QWEN_BASE_REPO.split("@")[1])]
        assert "speech_tokenizer/model.safetensors" in preset.voice_files[0].patterns

    def test_a_preset_without_voice_files_has_none(self):
        (yue2,) = parse_audio_presets(audio_config(yue2=yue2_preset()))

        assert yue2.voice_files == ()

    def test_voice_files_must_be_pinned_to_a_commit(self):
        preset = qwen3_tts_preset()
        preset["voice_files"] = [{"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "files": ["config.json"]}]

        with pytest.raises(ValueError, match=r"voice_files\[0\]\.repo must be pinned"):
            parse_audio_presets({"audio_model_presets": {"qwen3_tts": preset}})
