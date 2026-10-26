"""The video backend registry: factories by platform and family, and what a family can do on a platform."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import zvisiongenerator.backends as backends_module
from zvisiongenerator.core.video_family import VideoCapabilities
from zvisiongenerator.utils.config import load_config

_ALL_ON = {"i2v": True, "end_frame": True, "extend": True, "upscale": True, "upscale_review": True, "lora": True, "audio": True, "low_memory": True}


def _config(**platform_capabilities: dict[str, bool]) -> dict:
    """Return a config describing an ``ltx`` family and an ``h3`` family with the given platform overlays."""
    ltx = {"label": "LTX", "default_steps": 8, "frame_rule": {"alignment": 8, "offset": 1, "round": "nearest"}, "extend_frame_step": 8, "carry_frames": 25, "capabilities": _ALL_ON}
    h3 = {
        "label": "MiniMax H3",
        "default_steps": 8,
        "frame_rule": {"alignment": 17, "offset": 5, "round": "up"},
        "extend_frame_step": 17,
        "carry_frames": 22,
        "capabilities": _ALL_ON,
        "platform_capabilities": platform_capabilities,
    }
    return {"video_model_presets": {"ltx": ltx, "h3": h3}}


@pytest.fixture(autouse=True)
def _empty_cache():
    backends_module.VIDEO_BACKENDS.clear()
    yield
    backends_module.VIDEO_BACKENDS.clear()


@pytest.fixture
def factories(monkeypatch):
    """Replace the platform map with mock factories for ``ltx`` and ``h3`` on linux and ``ltx`` on darwin."""
    made = {"linux_ltx": MagicMock(), "linux_h3": MagicMock(), "darwin_ltx": MagicMock()}
    registry = {
        "linux": {"ltx": MagicMock(return_value=made["linux_ltx"]), "h3": MagicMock(return_value=made["linux_h3"])},
        "darwin": {"ltx": MagicMock(return_value=made["darwin_ltx"])},
    }
    monkeypatch.setattr(backends_module, "_VIDEO_BACKENDS_MAP", registry)
    monkeypatch.setattr(backends_module.sys, "platform", "linux")
    return registry, made


class TestRegistryShape:
    def test_every_supported_platform_maps_its_families_to_factories(self):
        assert set(backends_module._VIDEO_BACKENDS_MAP) == {"darwin", "win32", "linux"}
        expected = {"darwin": {"ltx", "h3"}, "win32": {"ltx", "h3"}, "linux": {"ltx", "h3"}}
        for platform, by_family in backends_module._VIDEO_BACKENDS_MAP.items():
            assert set(by_family) == expected[platform]
            assert all(callable(factory) for factory in by_family.values())

    def test_video_backends_is_a_dict_by_family(self):
        assert isinstance(backends_module.VIDEO_BACKENDS, dict)


class TestGetVideoBackend:
    def test_a_family_of_the_platform_is_created_and_cached(self, factories):
        registry, made = factories

        first = backends_module.get_video_backend("h3")
        second = backends_module.get_video_backend("h3")

        assert first is second is made["linux_h3"]
        registry["linux"]["h3"].assert_called_once_with()
        assert backends_module.VIDEO_BACKENDS == {"h3": made["linux_h3"]}

    def test_families_are_created_independently(self, factories):
        registry, made = factories

        assert backends_module.get_video_backend("ltx") is made["linux_ltx"]
        registry["linux"]["h3"].assert_not_called()

    def test_a_cached_backend_is_returned_without_a_factory_call(self, factories):
        registry, _ = factories
        cached = MagicMock()
        backends_module.VIDEO_BACKENDS["ltx"] = cached

        assert backends_module.get_video_backend("ltx") is cached
        registry["linux"]["ltx"].assert_not_called()

    def test_a_family_the_platform_lacks_is_refused_with_the_available_families(self, factories, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")

        with pytest.raises(RuntimeError, match=r"^No video backend for model family 'h3'\. Available: \['ltx'\]$"):
            backends_module.get_video_backend("h3")

    def test_an_unsupported_platform_is_refused(self, factories, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd")

        with pytest.raises(RuntimeError, match="Unsupported video platform: freebsd.*macOS, Windows, and Linux"):
            backends_module.get_video_backend("ltx")


class TestPlatformFlagsReadTheMap:
    @pytest.mark.parametrize("platform_key", ["darwin", "win32", "linux"])
    def test_supported_platforms_have_every_flag(self, monkeypatch, platform_key):
        monkeypatch.setattr(backends_module.sys, "platform", platform_key)

        assert (backends_module.supports_video_audio(), backends_module.supports_video_extend(), backends_module.supports_video_upscale_review()) == (True, True, True)

    def test_an_unsupported_platform_has_none(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd")

        assert (backends_module.supports_video_audio(), backends_module.supports_video_extend(), backends_module.supports_video_upscale_review()) == (False, False, False)

    def test_the_flags_follow_the_map(self, monkeypatch):
        monkeypatch.setattr(backends_module, "_VIDEO_BACKENDS_MAP", {"linux": {}})
        monkeypatch.setattr(backends_module.sys, "platform", "linux")

        assert backends_module.supports_video_extend() is True
        monkeypatch.setattr(backends_module, "_VIDEO_BACKENDS_MAP", {})
        assert backends_module.supports_video_extend() is False


class TestVideoFileUpscaleCapability:
    def test_an_upscaler_with_a_video_method_supports_it(self, monkeypatch):
        monkeypatch.setattr(backends_module, "get_seedvr2_upscaler", lambda: MagicMock(spec=["upscale_image", "upscale_video"]))

        assert backends_module.supports_video_file_upscale() is True

    def test_an_upscaler_without_a_video_method_does_not(self, monkeypatch):
        monkeypatch.setattr(backends_module, "get_seedvr2_upscaler", lambda: MagicMock(spec=["upscale_image"]))

        assert backends_module.supports_video_file_upscale() is False

    def test_no_upscaler_does_not(self, monkeypatch):
        monkeypatch.setattr(backends_module, "get_seedvr2_upscaler", lambda: None)

        assert backends_module.supports_video_file_upscale() is False


class TestFamilyAvailability:
    def test_a_family_is_available_where_the_platform_registers_it(self, factories, monkeypatch):
        assert backends_module.video_family_available("h3") is True
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        assert backends_module.video_family_available("h3") is False
        assert backends_module.video_family_available("ltx") is True

    def test_an_unsupported_platform_has_no_family(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "freebsd")

        assert backends_module.video_family_available("ltx") is False


class TestVideoCapabilities:
    def test_the_packaged_ltx_family_keeps_its_capabilities_on_every_supported_platform(self, monkeypatch, tmp_path):
        monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path / ".ziv")
        config = load_config()

        for platform_key in ("darwin", "win32", "linux"):
            monkeypatch.setattr(backends_module.sys, "platform", platform_key)
            assert backends_module.video_capabilities(config, "ltx") == VideoCapabilities(**_ALL_ON)

    def test_a_family_the_config_does_not_describe_has_none(self):
        assert backends_module.video_capabilities({}, "ltx") is None
        assert backends_module.video_capabilities(_config(), "wan") is None

    def test_a_platform_overlay_takes_capabilities_away(self, factories, monkeypatch):
        config = _config(darwin={"upscale": False, "upscale_review": False})
        monkeypatch.setattr(backends_module, "_VIDEO_BACKENDS_MAP", {**factories[0], "darwin": {**factories[0]["darwin"], "h3": MagicMock()}})

        on_linux = backends_module.video_capabilities(config, "h3")
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        on_darwin = backends_module.video_capabilities(config, "h3")

        assert on_linux == VideoCapabilities(**_ALL_ON)
        assert on_darwin == VideoCapabilities(**{**_ALL_ON, "upscale": False, "upscale_review": False})

    def test_a_family_without_a_backend_on_the_platform_can_do_nothing(self, factories, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")

        assert backends_module.video_capabilities(_config(), "h3") == VideoCapabilities(**dict.fromkeys(_ALL_ON, False))

    def test_the_platform_flags_limit_the_family(self, factories, monkeypatch):
        monkeypatch.setattr(backends_module, "supports_video_audio", lambda: False)
        monkeypatch.setattr(backends_module, "supports_video_extend", lambda: False)
        monkeypatch.setattr(backends_module, "supports_video_upscale_review", lambda: False)

        capabilities = backends_module.video_capabilities(_config(), "ltx")

        assert capabilities == VideoCapabilities(**{**_ALL_ON, "audio": False, "extend": False, "upscale_review": False})


class TestMacH3Registration:
    """MiniMax H3 is served on macOS by the MLX backend, with no upscale (the shipped config's overlay)."""

    def test_the_darwin_factory_builds_the_mlx_backend(self):
        from zvisiongenerator.backends.mac.h3_video import H3MlxVideoBackend

        backend = backends_module._VIDEO_BACKENDS_MAP["darwin"]["h3"]()

        assert isinstance(backend, H3MlxVideoBackend)
        assert backend.name == "h3"

    def test_get_video_backend_creates_it_on_macos(self, monkeypatch):
        from zvisiongenerator.backends.mac.h3_video import H3MlxVideoBackend

        monkeypatch.setattr(backends_module.sys, "platform", "darwin")

        assert isinstance(backends_module.get_video_backend("h3"), H3MlxVideoBackend)
        assert backends_module.video_family_available("h3") is True

    def test_the_shipped_capabilities_on_macos_hide_upscale_and_review(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        capabilities = backends_module.video_capabilities(load_config(), "h3")

        assert capabilities == VideoCapabilities(i2v=True, end_frame=True, extend=True, upscale=False, upscale_review=False, lora=True, audio=True, low_memory=False)

    def test_the_ltx_capabilities_on_macos_are_unchanged(self, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        capabilities = backends_module.video_capabilities(load_config(), "ltx")

        assert capabilities is not None
        assert capabilities.upscale is True and capabilities.upscale_review is True
