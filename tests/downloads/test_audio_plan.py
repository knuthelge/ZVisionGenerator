"""Tests for ``audio_plan``: the targets of each audio preset and the cloning checkpoint a recording adds."""

from __future__ import annotations

from tests._audio_presets import audio_config, qwen3_tts_preset, sa3_preset, yue2_preset

from zvisiongenerator.downloads.plans import audio_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import parse_audio_presets


def _preset(name: str, **presets):
    return next(preset for preset in parse_audio_presets(audio_config(**presets)) if preset.id == name)


class TestTargets:
    def test_yue2_downloads_the_model_and_the_vae(self):
        targets = audio_plan(_preset("yue2", yue2=yue2_preset()))

        assert [(target.repo_id, target.revision) for target in targets] == [
            ("m-a-p/YuE2-3B", "8f24312f187763b9854adeec87e437e2b03bbff1"),
            ("m-a-p/YuE2-Vae", "152733a19ad43aa67e367f9b5503ef8075bb5126"),
        ]
        assert "model.safetensors" in targets[0].allow_patterns
        assert targets[1].allow_patterns == ("config.json", "model.safetensors", "modeling_vae.py", "weights_manifest.json")

    def test_sa3_downloads_the_checkpoint_the_config_and_the_text_encoder(self):
        targets = audio_plan(_preset("sa3", sa3=sa3_preset()))

        assert targets == (
            DownloadTarget("Comfy-Org/stable-audio-3", "75aad8836271cb5a5123fb58304fae5d3bb4b0f9", ("checkpoints/stable_audio_3_medium.safetensors",)),
            DownloadTarget("cocktailpeanut/stable-audio-3-medium", "a30034d70bd58f6e6f967ef28ecaa6c52c185d6c", ("model_config.json",)),
            DownloadTarget(
                "mlx-community/stable-audio-3-small-sfx",
                "ca6c38718d27cd2b446a4661c8851f1f3d0a3870",
                (
                    "t5gemma-b-b-ul2/config.json",
                    "t5gemma-b-b-ul2/generation_config.json",
                    "t5gemma-b-b-ul2/model.safetensors",
                    "t5gemma-b-b-ul2/special_tokens_map.json",
                    "t5gemma-b-b-ul2/tokenizer.json",
                    "t5gemma-b-b-ul2/tokenizer_config.json",
                ),
            ),
        )

    def test_every_target_is_pinned_and_downloads_into_the_cache(self):
        for name, builder in (("yue2", yue2_preset), ("qwen3_tts", qwen3_tts_preset), ("sa3", sa3_preset)):
            for target in audio_plan(_preset(name, **{name: builder()})):
                assert target.revision is not None and len(target.revision) == 40
                assert target.allow_patterns and target.local_dir is None and target.ignore_patterns is None and not target.anonymous


class TestQwenCloningFiles:
    """Qwen3-TTS reads in built-in voices with CustomVoice and follows a recording with Base, which only a take with a recording downloads."""

    def _preset(self):
        return _preset("qwen3_tts", qwen3_tts=qwen3_tts_preset())

    def test_a_built_in_voice_downloads_customvoice_alone(self):
        targets = audio_plan(self._preset())

        assert [(target.repo_id, target.revision) for target in targets] == [("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe")]
        assert "speech_tokenizer/model.safetensors" in targets[0].allow_patterns

    def test_a_recording_adds_the_cloning_checkpoint_after_the_main_files(self):
        targets = audio_plan(self._preset(), voice=True)

        assert [(target.repo_id, target.revision) for target in targets] == [
            ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe"),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3"),
        ]

    def test_a_model_without_cloning_files_plans_the_same_either_way(self):
        preset = _preset("yue2", yue2=yue2_preset())

        assert audio_plan(preset, voice=True) == audio_plan(preset)
