"""The audio side of the backend registry: the family lookup, and the one place that maps the platform to cuda or mps."""

from __future__ import annotations


import pytest

import zvisiongenerator.backends as backends_module
from zvisiongenerator.backends import audio_platform_key, get_audio_backend


class TestFamilies:
    @pytest.mark.parametrize("family", ["ace_step", ""])
    def test_a_family_without_a_backend_is_refused_with_the_known_ones(self, family):
        with pytest.raises(RuntimeError, match=r"No audio backend for model family '.*'\. Known: \['yue2', 'higgs3', 'stable_audio_3'\]"):
            get_audio_backend(family)


class TestPlatform:
    @pytest.mark.parametrize(("platform", "expected"), [("linux", "cuda"), ("win32", "cuda"), ("darwin", "mps")])
    def test_the_platform_key_follows_the_platform(self, monkeypatch, platform, expected):
        monkeypatch.setattr(backends_module.sys, "platform", platform)

        assert audio_platform_key() == expected

    def test_a_platform_with_neither_is_refused(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd14")

        with pytest.raises(RuntimeError, match="needs CUDA or Apple MPS; platform 'freebsd14'"):
            audio_platform_key()
