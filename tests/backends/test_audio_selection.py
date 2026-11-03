"""The audio side of the backend registry: the family lookup, and the one place that maps the platform to cuda or mps."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import zvisiongenerator.backends as backends_module
from zvisiongenerator.backends import AUDIO_FAMILIES, audio_device, audio_platform_key, get_audio_backend
from zvisiongenerator.core.audio_backend import AudioBackend


class TestFamilies:
    def test_the_known_families(self):
        assert AUDIO_FAMILIES == ("yue2", "higgs3", "stable_audio_3")

    @pytest.mark.parametrize("family", ["ace_step", ""])
    def test_a_family_without_a_backend_is_refused_with_the_known_ones(self, family):
        with pytest.raises(RuntimeError, match=r"No audio backend for model family '.*'\. Known: \['yue2', 'higgs3', 'stable_audio_3'\]"):
            get_audio_backend(family)

    def test_every_family_has_a_factory(self):
        assert tuple(backends_module._AUDIO_BACKEND_FACTORIES) == AUDIO_FAMILIES

    def test_a_registered_family_gets_a_new_backend_per_call(self, monkeypatch):
        factory = MagicMock(side_effect=[MagicMock(), MagicMock()])
        monkeypatch.setitem(backends_module._AUDIO_BACKEND_FACTORIES, "yue2", factory)

        first, second = get_audio_backend("yue2"), get_audio_backend("yue2")

        assert first is not second
        assert factory.call_count == 2

    def test_the_factory_is_not_called_until_the_family_is_asked_for(self, monkeypatch):
        factory = MagicMock()
        monkeypatch.setitem(backends_module._AUDIO_BACKEND_FACTORIES, "higgs3", factory)

        with pytest.raises(RuntimeError):
            get_audio_backend("ace_step")

        factory.assert_not_called()


class TestProtocol:
    def test_an_object_with_load_generate_and_close_is_a_backend(self):
        class Fake:
            def load(self, preset, platform_entry, device): ...
            def generate(self, request, on_progress, cancelled): ...
            def close(self): ...

        assert isinstance(Fake(), AudioBackend)

    def test_an_object_without_close_is_not(self):
        class Fake:
            def load(self, preset, platform_entry, device): ...
            def generate(self, request, on_progress, cancelled): ...

        assert not isinstance(Fake(), AudioBackend)


class TestPlatform:
    @pytest.mark.parametrize(("platform", "expected"), [("linux", "cuda"), ("win32", "cuda"), ("darwin", "mps")])
    def test_the_device_and_the_platform_key_follow_the_platform(self, monkeypatch, platform, expected):
        monkeypatch.setattr(backends_module.sys, "platform", platform)

        assert audio_device() == expected
        assert audio_platform_key() == expected

    @pytest.mark.parametrize("function", [audio_device, audio_platform_key])
    def test_a_platform_with_neither_is_refused(self, monkeypatch, function):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd14")

        with pytest.raises(RuntimeError, match="needs CUDA or Apple MPS; platform 'freebsd14'"):
            function()
