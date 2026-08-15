"""Tests for declarative Web UI config loading and contract helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import config as web_config_module
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.utils.image_model_detect import ImageModelInfo


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
            "16:9": {
                "m": {"width": 704, "height": 448, "frames": 49},
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
            "startup_view": "gallery",
            "gallery_page_size": 8,
            "output_dir": "exports",
            "default_models": {
                "image": "zit",
                "video": "ltx-8",
            },
        },
    }


def _assert_non_empty_string(value: object) -> None:
    assert isinstance(value, str)
    assert value


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
        "detect_image_model",
        lambda value: ImageModelInfo(family="zimage" if "Z-Image" in str(value) else "unknown", is_distilled=False, size=None),
    )
    monkeypatch.setattr(
        web_config_module,
        "detect_video_model",
        lambda value: SimpleNamespace(family="ltx" if "ltx" in str(value) else "unknown"),
    )

    web_config = web_config_module.load_web_config()

    assert web_config.startup_view == "gallery"
    assert web_config.gallery_page_size == 8
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


def test_load_web_config_uses_image_model_detection_for_aliases(monkeypatch, tmp_path):
    """Image aliases should be accepted by model detection, not name tokens."""
    app_config = _make_app_config()
    app_config["model_aliases"] = {
        "custom-default": str(tmp_path / "models" / "custom-default"),
        "bad-default": str(tmp_path / "models" / "bad-default"),
    }
    app_config["ui"]["default_models"] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", lambda: app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])
    monkeypatch.setattr(web_config_module, "resolve_model_path", lambda name, **_: app_config["model_aliases"].get(name, name))
    monkeypatch.setattr(
        web_config_module,
        "detect_image_model",
        lambda value: ImageModelInfo(family="zimage" if "custom-default" in str(value) else "unknown", is_distilled=False, size=None),
    )
    monkeypatch.setattr(web_config_module, "detect_video_model", lambda _value: SimpleNamespace(family="unknown"))

    web_config = web_config_module.load_web_config()

    assert web_config.default_models.image == "custom-default"
    assert web_config.image_model_options == ("custom-default",)


def test_load_web_config_surfaces_ideo_alias_via_dynamic_inventory(monkeypatch, tmp_path):
    app_config = _make_app_config()
    app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
    }
    app_config["ui"]["default_models"] = {"image": "ideo"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", lambda: app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])
    monkeypatch.setattr(web_config_module, "resolve_model_path", lambda name, **_: app_config["model_aliases"].get(name, name))
    monkeypatch.setattr(
        web_config_module,
        "detect_image_model",
        lambda value: ImageModelInfo(family="ideogram4" if "ideogram-4" in str(value) else "unknown", is_distilled=False, size=None),
    )
    monkeypatch.setattr(web_config_module, "detect_video_model", lambda _value: SimpleNamespace(family="unknown"))

    web_config = web_config_module.load_web_config()

    assert web_config.default_models.image == "ideo"
    assert web_config.image_model_options == ("ideo",)


def test_load_web_config_skips_unavailable_platform_aliases(monkeypatch, tmp_path):
    """Unavailable platform-aware aliases should be ignored during inventory discovery."""
    app_config = _make_app_config()
    app_config["model_aliases"] = {
        "portable-image": "Tongyi-MAI/Z-Image-Turbo",
        "windows-only-blocked": {
            "darwin": "Tongyi-MAI/Z-Image-Turbo",
            "win32": {"message": "Windows build is not available."},
        },
    }
    app_config["ui"]["default_models"] = {"image": "portable-image"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", lambda: app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])
    monkeypatch.setattr(model_inventory_module.sys, "platform", "win32")
    monkeypatch.setattr(
        web_config_module,
        "detect_image_model",
        lambda value: ImageModelInfo(family="zimage" if "Z-Image" in str(value) else "unknown", is_distilled=False, size=None),
    )
    monkeypatch.setattr(web_config_module, "detect_video_model", lambda _value: SimpleNamespace(family="unknown"))

    web_config = web_config_module.load_web_config()

    assert web_config.image_model_options == ("portable-image",)
    assert web_config.default_models.image == "portable-image"


def test_load_web_config_exposes_windows_linux_video_aliases_and_hides_macos_only_ones(monkeypatch, tmp_path):
    """Windows/Linux inventory should include ltx-2.3 and omit macOS-only video aliases."""
    app_config = _make_app_config()
    app_config["model_aliases"] = {
        "ltx-8": {
            "darwin": "dgrauet/ltx-2.3-mlx-q8",
            "win32": {"message": "Alias 'ltx-8' is macOS-only. On Windows, use 'ltx-2.3' for the CUDA diffusers backend."},
            "linux": {"message": "Alias 'ltx-8' is macOS-only. On Linux, use 'ltx-2.3' for the CUDA diffusers backend."},
        },
        "ltx-2.3": {
            "darwin": {"message": "Alias 'ltx-2.3' is available on Windows and Linux only. On macOS, use 'ltx-4' or 'ltx-8'."},
            "win32": "dg845/LTX-2.3-Diffusers",
            "linux": "dg845/LTX-2.3-Diffusers",
        },
    }
    app_config["ui"]["default_models"] = {"video": "ltx-8"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", lambda: app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])
    monkeypatch.setattr(model_inventory_module.sys, "platform", "win32")
    monkeypatch.setattr(
        web_config_module,
        "detect_image_model",
        lambda _value: ImageModelInfo(family="unknown", is_distilled=False, size=None),
    )
    monkeypatch.setattr(
        web_config_module,
        "detect_video_model",
        lambda value: SimpleNamespace(family="ltx" if "Diffusers" in str(value) else "unknown", supports_i2v=True),
    )

    web_config = web_config_module.load_web_config()

    assert web_config.video_model_options == ("ltx-2.3",)
    assert web_config.default_models.video == "ltx-2.3"


def test_load_web_config_rejects_invalid_ui_mapping(monkeypatch):
    """The Web UI config loader should reject non-mapping UI config values."""
    monkeypatch.setattr(web_config_module, "load_config", lambda: {"ui": "invalid"})

    with pytest.raises(ValueError) as exc_info:
        web_config_module.load_web_config()

    assert "ui" in str(exc_info.value)


def test_build_writable_config_schema_exposes_frozen_phase_a_fields(monkeypatch, tmp_path):
    """The writable config schema should include persisted/effective Phase A semantics."""
    monkeypatch.setattr(
        config_contract_module,
        "read_user_config_override",
        lambda: {
            "ui": {
                "default_models": {"image": "zit", "video": "ltx-8"},
                "output_dir": str(tmp_path / "persisted-outputs"),
            },
            "generation": {"default_size": "l"},
        },
    )
    web_config = SimpleNamespace(
        default_models=SimpleNamespace(image="local-image", video="ltx-8"),
        output_dir=str(tmp_path / "effective-outputs"),
        app_config={"generation": {"default_size": "m"}},
    )

    schema = config_contract_module.build_writable_config_schema(web_config)
    fields = {field["key"]: field for field in schema["fields"]}

    assert schema["version"] == 1
    assert schema["semantics"]["omitted"] == "unchanged"
    assert "null" in schema["semantics"]
    _assert_non_empty_string(schema["semantics"]["null"])
    assert sorted(fields) == [
        "generation.default_size",
        "ui.default_models.image",
        "ui.default_models.video",
        "ui.output_dir",
    ]
    assert fields["ui.output_dir"]["clearable"] is True
    assert fields["ui.output_dir"]["type"] == "string"
    assert fields["ui.output_dir"]["empty_string"] == "clear"
    assert fields["ui.output_dir"]["persisted_value"] == str(tmp_path / "persisted-outputs")
    assert fields["ui.output_dir"]["effective_value"] == str(tmp_path / "effective-outputs")
    assert "default_source" in fields["ui.output_dir"]
    _assert_non_empty_string(fields["ui.output_dir"]["default_source"])
    assert fields["ui.output_dir"]["owning_consumer"]
    assert fields["ui.output_dir"]["validation_rules"]
    assert fields["generation.default_size"]["persisted_value"] == "l"
    assert fields["generation.default_size"]["effective_value"] == "m"
    assert "default_source" in fields["generation.default_size"]
    _assert_non_empty_string(fields["generation.default_size"]["default_source"])


def test_build_writable_config_schema_reports_effective_default_size_without_override(monkeypatch):
    """Writable config should report the effective base size even when no override is persisted."""
    monkeypatch.setattr(config_contract_module, "read_user_config_override", lambda: {})
    web_config = SimpleNamespace(
        default_models=SimpleNamespace(image=None, video=None),
        output_dir="/tmp/outputs",
        app_config={"generation": {}},
        image_ratios=("2:3", "1:1"),
        image_size_options={"2:3": ("m", "l"), "1:1": ("s",)},
    )

    schema = config_contract_module.build_writable_config_schema(web_config)
    fields = {field["key"]: field for field in schema["fields"]}

    assert fields["generation.default_size"]["persisted_value"] is None
    assert fields["generation.default_size"]["effective_value"] == "m"


def test_persist_writable_config_patch_clears_and_normalizes_values(monkeypatch, tmp_path):
    """Writable config patches should normalize paths and treat null as clear."""
    captured: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config_contract_module,
        "read_user_config_override",
        lambda: {
            "ui": {
                "default_models": {"image": "zit", "video": "ltx-8"},
                "output_dir": str(tmp_path / "old-output"),
            },
            "generation": {"default_size": "m"},
        },
    )
    monkeypatch.setattr(config_contract_module, "write_user_config_override", lambda payload: captured.setdefault("payload", payload))
    current = SimpleNamespace(
        image_model_options=("zit", "local-image"),
        video_model_options=("ltx-8",),
        app_config={"generation": {"default_ratio": "2:3"}},
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m", "l")},
    )

    config_contract_module.persist_writable_config_patch(
        {
            "ui.default_models.image": None,
            "ui.output_dir": " exports ",
            "generation.default_size": "l",
        },
        current,
    )

    assert captured["payload"] == {
        "ui": {
            "default_models": {"video": "ltx-8"},
            "output_dir": str(tmp_path / "exports"),
        },
        "generation": {"default_size": "l"},
    }


def test_persist_writable_config_patch_clears_output_dir_override(monkeypatch, tmp_path):
    """Clearing the output-dir patch should remove the persisted override entirely."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        config_contract_module,
        "read_user_config_override",
        lambda: {
            "ui": {
                "default_models": {"image": "zit", "video": "ltx-8"},
                "output_dir": str(tmp_path / "old-output"),
            },
        },
    )
    monkeypatch.setattr(config_contract_module, "write_user_config_override", lambda payload: captured.setdefault("payload", payload))

    current = SimpleNamespace(
        image_model_options=("zit",),
        video_model_options=("ltx-8",),
        app_config={"generation": {"default_ratio": "2:3"}},
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m",)},
    )

    config_contract_module.persist_writable_config_patch({"ui.output_dir": None}, current)

    assert captured["payload"] == {
        "ui": {
            "default_models": {"image": "zit", "video": "ltx-8"},
        },
    }
