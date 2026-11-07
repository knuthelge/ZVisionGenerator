"""The audio side of the backend registry: the family lookup, and the one place that maps the platform to cuda or mps."""

from __future__ import annotations


import pytest

import zvisiongenerator.backends as backends_module
from zvisiongenerator.backends import audio_platform_key, get_audio_backend


class TestFamilies:
    @pytest.mark.parametrize("family", ["ace_step", ""])
    def test_a_family_without_a_backend_is_refused_with_the_known_ones(self, family):
        with pytest.raises(RuntimeError, match=r"No audio backend for model family '.*'\. Known: \['yue2', 'qwen3_tts', 'qwen3_tts_mlx', 'stable_audio_3'\]"):
            get_audio_backend(family)

    def test_the_qwen_family_has_its_backend_and_it_is_an_audio_backend(self):
        from zvisiongenerator.core.audio_backend import AudioBackend

        backend = get_audio_backend("qwen3_tts")

        assert type(backend).__name__ == "Qwen3TtsBackend" and isinstance(backend, AudioBackend)


class TestQwenPerPlatform:
    """The one ``qwen3_tts`` preset runs its torch engine on CUDA and its MLX engine on macOS."""

    @pytest.mark.parametrize(
        ("platform", "family", "backend"), [("linux", "qwen3_tts", "Qwen3TtsBackend"), ("win32", "qwen3_tts", "Qwen3TtsBackend"), ("darwin", "qwen3_tts_mlx", "Qwen3TtsMlxBackend")]
    )
    def test_the_platform_picks_the_family_and_its_backend(self, monkeypatch, platform, family, backend):
        from tests._speech_fakes import qwen_preset

        monkeypatch.setattr(backends_module.sys, "platform", platform)

        preset = qwen_preset().on_platform(audio_platform_key())

        assert (preset.family, type(get_audio_backend(preset.family)).__name__) == (family, backend)

    def test_the_mlx_engine_needs_mlx_audio_only_when_it_loads_its_checkpoint(self, monkeypatch, tmp_path):
        import sys

        from tests._speech_fakes import qwen_preset
        from zvisiongenerator.backends.audio import qwen3_tts_mlx

        monkeypatch.setitem(sys.modules, "mlx_audio", None)  # not installed, as on Windows and Linux
        monkeypatch.setattr(qwen3_tts_mlx, "snapshot_dir", lambda files: tmp_path)
        backend = get_audio_backend("qwen3_tts_mlx")  # nothing is imported yet

        with pytest.raises(ImportError):
            backend.load(qwen_preset().on_platform("mps"), "mps")


class TestPlatform:
    @pytest.mark.parametrize(("platform", "expected"), [("linux", "cuda"), ("win32", "cuda"), ("darwin", "mps")])
    def test_the_platform_key_follows_the_platform(self, monkeypatch, platform, expected):
        monkeypatch.setattr(backends_module.sys, "platform", platform)

        assert audio_platform_key() == expected

    def test_a_platform_with_neither_is_refused(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd14")

        with pytest.raises(RuntimeError, match="needs CUDA or Apple MPS; platform 'freebsd14'"):
            audio_platform_key()
