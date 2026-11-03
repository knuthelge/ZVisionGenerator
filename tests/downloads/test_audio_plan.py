"""Tests for ``audio_plan``: the targets of each audio preset on each platform, and the pinned Higgs codec."""

from __future__ import annotations

import pytest

from tests._audio_presets import audio_config, higgs3_preset, sa3_preset, yue2_preset

from zvisiongenerator.downloads import audio_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import parse_audio_presets


def _preset(name: str, **presets):
    return next(preset for preset in parse_audio_presets(audio_config(**presets)) if preset.id == name)


class TestTargets:
    @pytest.mark.parametrize("platform_key", ["cuda", "mps"])
    def test_yue2_downloads_the_model_and_the_vae(self, platform_key):
        targets = audio_plan(_preset("yue2", yue2=yue2_preset()), platform_key)

        assert [(target.repo_id, target.revision) for target in targets] == [
            ("m-a-p/YuE2-3B", "8f24312f187763b9854adeec87e437e2b03bbff1"),
            ("m-a-p/YuE2-Vae", "152733a19ad43aa67e367f9b5503ef8075bb5126"),
        ]
        assert "model.safetensors" in targets[0].allow_patterns
        assert targets[1].allow_patterns == ("config.json", "model.safetensors", "modeling_vae.py", "weights_manifest.json")

    @pytest.mark.parametrize("platform_key", ["cuda", "mps"])
    def test_higgs_downloads_the_weights_and_the_pinned_codec(self, platform_key):
        targets = audio_plan(_preset("higgs3", higgs3=higgs3_preset()), platform_key)

        assert [(target.repo_id, target.revision) for target in targets] == [
            ("bosonai/higgs-tts-3-4b", "239f63fb7b02b1aa085f98d9efae5e35cc5523e8"),
            ("bosonai/higgs-audio-v2-tokenizer", "815a03f1b57dcb5ae918db415f22cd0c8a6f0f54"),
        ]

    @pytest.mark.parametrize("platform_key", ["cuda", "mps"])
    def test_sa3_downloads_the_checkpoint_the_config_and_the_text_encoder(self, platform_key):
        targets = audio_plan(_preset("sa3", sa3=sa3_preset()), platform_key)

        assert targets == (
            DownloadTarget("Comfy-Org/stable-audio-3", "75aad8836271cb5a5123fb58304fae5d3bb4b0f9", ("checkpoints/stable_audio_3_medium.safetensors",)),
            DownloadTarget("cocktailpeanut/stable-audio-3-medium", "a30034d70bd58f6e6f967ef28ecaa6c52c185d6c", ("model_config.json",)),
            DownloadTarget("mlx-community/stable-audio-3-small-sfx", "ca6c38718d27cd2b446a4661c8851f1f3d0a3870", ("t5gemma-b-b-ul2/*",)),
        )

    def test_every_target_is_pinned_and_downloads_into_the_cache(self):
        for name, builder in (("yue2", yue2_preset), ("higgs3", higgs3_preset), ("sa3", sa3_preset)):
            for target in audio_plan(_preset(name, **{name: builder()}), "cuda"):
                assert target.revision is not None and len(target.revision) == 40
                assert target.allow_patterns and target.local_dir is None and target.ignore_patterns is None and not target.anonymous


class TestPlatforms:
    def test_a_platform_has_its_own_files(self):
        preset = yue2_preset()
        preset["platforms"]["mps"]["files"] = [{"repo": "m-a-p/YuE2-3B@" + "9" * 40, "files": ["model.safetensors"]}]

        cuda = audio_plan(_preset("yue2", yue2=preset), "cuda")
        mps = audio_plan(_preset("yue2", yue2=preset), "mps")

        assert len(cuda) == 2
        assert mps == (DownloadTarget("m-a-p/YuE2-3B", "9" * 40, ("model.safetensors",)),)
