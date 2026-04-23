"""Tests for declarative Web UI config loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zvisiongenerator.web import config as web_config_module


def _make_app_config() -> dict[str, object]:
    return {
        "generation": {
            "default_ratio": "2:3",
            "default_size": "m",
        },
        "video_generation": {
            "default_ratio": "16:9",
            "default_size": "m",
        },
        "video_sizes": {
            "ltx": {
                "16:9": {
                    "m": {"width": 704, "height": 448, "frames": 49},
                },
            },
        },
        "sizes": {
            "2:3": {
                "m": {"width": 832, "height": 1216},
            },
        },
        "schedulers": {
            "beta": {},
        },
        "model_aliases": {
            "zit": "Tongyi-MAI/Z-Image-Turbo",
            "ltx-8": "dgrauet/ltx-2.3-mlx-q8",
        },
        "ui": {
            "port": 9090,
            "theme": "light",
            "startup_view": "gallery",
            "gallery_page_size": 8,
            "output_dir": "exports",
            "visible_sections": ["image_generation", "gallery_summary", "image_generation"],
            "default_models": {
                "image": "zit",
                "video": "ltx-8",
            },
        },
    }


def test_load_web_config_loads_declarative_ui_settings(monkeypatch, tmp_path):
    """The Web UI config loader should expose typed settings and discovered options."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", _make_app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [SimpleNamespace(name="local-image")])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [SimpleNamespace(name="local-video")])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [SimpleNamespace(name="detail")])
    monkeypatch.setattr(web_config_module, "resolve_model_path", lambda name, **_: _make_app_config()["model_aliases"].get(name, name))
    monkeypatch.setattr(
        web_config_module,
        "detect_video_model",
        lambda value: SimpleNamespace(family="ltx" if "ltx" in str(value) else "unknown"),
    )

    web_config = web_config_module.load_web_config()

    assert web_config.port == 9090
    assert web_config.theme == "light"
    assert web_config.startup_view == "gallery"
    assert web_config.gallery_page_size == 8
    assert web_config.visible_sections == ("image_generation", "gallery_summary")
    assert web_config.default_models.image == "zit"
    assert web_config.default_models.video == "ltx-8"
    assert web_config.image_model_options == ("local-image", "zit")
    assert web_config.video_model_options == ("local-video", "ltx-8")
    assert web_config.lora_options == ("detail",)
    assert web_config.image_ratios == ("2:3",)
    assert web_config.image_size_options == {"2:3": ("m",)}
    assert web_config.video_ratios == ("16:9",)
    assert web_config.video_size_options == {"16:9": ("m",)}
    assert web_config.scheduler_options == ("beta",)
    assert web_config.output_dir == str(tmp_path / "exports")
    assert (tmp_path / "exports").is_dir()


def test_load_web_config_rejects_invalid_ui_mapping(monkeypatch):
    """The Web UI config loader should reject non-mapping UI config values."""
    monkeypatch.setattr(web_config_module, "load_config", lambda: {"ui": "invalid"})

    with pytest.raises(ValueError, match="config 'ui' must be a mapping"):
        web_config_module.load_web_config()
