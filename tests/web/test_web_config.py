"""Tests for declarative Web UI config loading and contract helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import config as web_config_module
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.seedvr2 import MLX_WEIGHTS


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
        "model_alias_families": {},
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


def _patch_minimal_web_config(monkeypatch, tmp_path, app_config=None, *, platform=None, detect_image=None, detect_video=None):
    """Point the web config loader at *app_config* (default: the minimal one) with empty model inventories."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / ".ziv"))  # alias resolution reads the models folder too
    monkeypatch.setattr(web_config_module, "load_config", (lambda: app_config) if app_config is not None else _make_app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])
    if platform is not None:
        monkeypatch.setattr(model_inventory_module.sys, "platform", platform)
    if detect_image is not None:
        monkeypatch.setattr(web_config_module, "detect_image_model", detect_image)
    if detect_video is not None:
        monkeypatch.setattr(web_config_module, "detect_video_model", detect_video)


def test_declared_image_family_returns_configured_family():
    app_config = _make_app_config()
    app_config["model_alias_families"] = {"ideo": "ideogram4"}

    assert model_inventory_module.declared_image_family(app_config, "ideo") == "ideogram4"
    assert model_inventory_module.declared_image_family(app_config, "zit") is None


def test_load_web_config_hides_stored_quants_from_model_options(monkeypatch, tmp_path):
    """Stored quants are picked through their base model and quantize level, never listed or defaulted to."""
    monkeypatch.chdir(tmp_path)

    def _app_config():
        config = _make_app_config()
        config["model_aliases"] = {}
        config["ui"]["default_models"] = {"image": "local-image@q8"}
        return config

    monkeypatch.setattr(web_config_module, "load_config", _app_config)
    monkeypatch.setattr(web_config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
    monkeypatch.setattr(web_config_module, "list_models", lambda _: [SimpleNamespace(name="local-image"), SimpleNamespace(name="local-image@q8")])
    monkeypatch.setattr(web_config_module, "list_video_models", lambda _: [])
    monkeypatch.setattr(web_config_module, "list_loras", lambda _: [])

    web_config = web_config_module.load_web_config()

    assert web_config.image_model_options == ("local-image",)
    assert web_config.default_models.image == "local-image"
    assert [entry.name for entry in web_config.image_inventory] == ["local-image", "local-image@q8"]


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


def test_load_web_config_surfaces_ideo_alias_offline_via_declared_family(monkeypatch, tmp_path):
    app_config = _make_app_config()
    app_config["model_aliases"] = {
        "ideo": {
            "darwin": "ideogram-ai/ideogram-4-fp8",
            "win32": {"message": "Alias 'ideo' is macOS-only."},
            "linux": {"message": "Alias 'ideo' is macOS-only."},
        },
        "custom-default": str(tmp_path / "models" / "custom-default"),
    }
    app_config["model_alias_families"] = {"ideo": "ideogram4"}
    app_config["ui"]["default_models"] = {"image": "ideo"}

    detect_calls: list[str] = []

    def _fake_detect_image_model(value: object) -> ImageModelInfo:
        detect_calls.append(str(value))
        if "ideogram-4" in str(value):
            raise RuntimeError("offline")
        return ImageModelInfo(family="zimage" if "custom-default" in str(value) else "unknown", is_distilled=False, size=None)

    _patch_minimal_web_config(
        monkeypatch,
        tmp_path,
        app_config,
        platform="darwin",
        detect_image=_fake_detect_image_model,
        detect_video=lambda _value: SimpleNamespace(family="unknown"),
    )

    web_config = web_config_module.load_web_config()

    assert web_config.default_models.image == "ideo"
    assert web_config.image_model_options == ("custom-default", "ideo")
    assert detect_calls == [str(tmp_path / "models" / "custom-default")]


_IDEO_ALIASES = {
    "ideo": {
        "darwin": "ideogram-ai/ideogram-4-fp8",
        "win32": {"message": "Alias 'ideo' is macOS-only."},
        "linux": {"message": "Alias 'ideo' is macOS-only."},
    },
    "portable-image": "Tongyi-MAI/Z-Image-Turbo",
}
_BLOCKED_ALIASES = {
    "portable-image": "Tongyi-MAI/Z-Image-Turbo",
    "windows-only-blocked": {
        "darwin": "Tongyi-MAI/Z-Image-Turbo",
        "win32": {"message": "Windows build is not available."},
    },
}
_LTX_ALIASES = {
    "ltx-8": {
        "darwin": "dgrauet/ltx-2.3-mlx-q8",
        "win32": {"message": "Alias 'ltx-8' is macOS-only."},
        "linux": {"message": "Alias 'ltx-8' is macOS-only."},
    },
    "ltx-2.3": {
        "darwin": {"message": "Alias 'ltx-2.3' is available on Windows and Linux only."},
        "win32": "diffusers/LTX-2.3-Distilled-Diffusers",
        "linux": "diffusers/LTX-2.3-Distilled-Diffusers",
    },
}


@pytest.mark.parametrize(
    ("aliases", "families", "default_models", "options_field", "expected"),
    [
        (_IDEO_ALIASES, {"ideo": "ideogram4"}, {"image": "ideo"}, "image_model_options", "portable-image"),
        (_BLOCKED_ALIASES, {}, {"image": "portable-image"}, "image_model_options", "portable-image"),
        (_LTX_ALIASES, {}, {"video": "ltx-8"}, "video_model_options", "ltx-2.3"),
    ],
    ids=["ideo-alias", "blocked-alias", "ltx-video-aliases"],
)
def test_load_web_config_hides_aliases_the_platform_cannot_resolve(monkeypatch, tmp_path, aliases, families, default_models, options_field, expected):
    """A platform-unavailable alias is hidden from the options and the default falls back to an available one."""
    app_config = _make_app_config()
    app_config["model_aliases"] = aliases
    app_config["model_alias_families"] = families
    app_config["ui"]["default_models"] = default_models

    _patch_minimal_web_config(
        monkeypatch,
        tmp_path,
        app_config,
        platform="win32",
        detect_image=lambda value: ImageModelInfo(family="zimage" if "Z-Image" in str(value) else "unknown", is_distilled=False, size=None),
        detect_video=lambda value: SimpleNamespace(family="ltx" if "Diffusers" in str(value) else "unknown", supports_i2v=True),
    )

    web_config = web_config_module.load_web_config()

    assert getattr(web_config, options_field) == (expected,)
    default = web_config.default_models.video if "video" in default_models else web_config.default_models.image
    assert default == expected


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
    assert sorted(fields) == [
        "generation.default_size",
        "prompt_enhancer.user_model",
        "ui.default_models.image",
        "ui.default_models.video",
        "ui.output_dir",
    ]
    assert fields["ui.output_dir"]["clearable"] is True
    assert fields["ui.output_dir"]["type"] == "string"
    assert fields["ui.output_dir"]["empty_string"] == "clear"
    assert fields["ui.output_dir"]["persisted_value"] == str(tmp_path / "persisted-outputs")
    assert fields["ui.output_dir"]["effective_value"] == str(tmp_path / "effective-outputs")
    assert fields["generation.default_size"]["persisted_value"] == "l"
    assert fields["generation.default_size"]["effective_value"] == "m"


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


def _persist_current() -> SimpleNamespace:
    return SimpleNamespace(
        image_model_options=("zit",),
        video_model_options=("ltx-8",),
        app_config={"generation": {"default_ratio": "2:3"}},
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m", "l")},
    )


def test_concurrent_config_saves_do_not_drop_each_others_changes(monkeypatch, tmp_path):
    """Web handlers run in worker threads; overlapping read-modify-write saves must both land."""
    import threading
    import time

    import yaml

    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    real_read = config_contract_module.read_user_config_override

    def _slow_read():
        payload = real_read()
        time.sleep(0.05)  # widen the window between read and write
        return payload

    monkeypatch.setattr(config_contract_module, "read_user_config_override", _slow_read)
    patches = [{"ui.default_models.image": "zit"}, {"ui.default_models.video": "ltx-8"}, {"generation.default_size": "l"}]
    threads = [threading.Thread(target=config_contract_module.persist_writable_config_patch, args=(patch, _persist_current())) for patch in patches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved == {"ui": {"default_models": {"image": "zit", "video": "ltx-8"}}, "generation": {"default_size": "l"}}


def test_user_config_write_is_atomic(monkeypatch, tmp_path):
    """A failed write must leave the previous file intact and no temp files behind (never a truncated file)."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    config_contract_module.write_user_config_override({"ui": {"output_dir": "/keep"}})
    original = (tmp_path / "config.yaml").read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(tmp_path), "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        config_contract_module.write_user_config_override({"ui": {"output_dir": "/new"}})

    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == original
    assert [path.name for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []


def test_user_config_write_preserves_symlink_and_permissions(monkeypatch, tmp_path):
    """Saving from the Web UI must not replace a symlinked config.yaml or widen its permissions."""
    import stat

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = tmp_path / "dotfiles" / "ziv-config.yaml"
    target.parent.mkdir()
    target.write_text("ui: {}\n", encoding="utf-8")
    target.chmod(0o600)
    (data_dir / "config.yaml").symlink_to(target)
    monkeypatch.setenv("ZIV_DATA_DIR", str(data_dir))

    config_contract_module.write_user_config_override({"ui": {"output_dir": "/new"}})

    assert (data_dir / "config.yaml").is_symlink()
    assert "/new" in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.usefixtures("hub_offline")
def test_load_web_config_has_no_seedvr2_where_it_is_unsupported(monkeypatch, tmp_path):
    _patch_minimal_web_config(monkeypatch, tmp_path)

    assert web_config_module.load_web_config().seedvr2 is None


@pytest.mark.parametrize(
    ("budget", "expected_gpu", "expected_system"),
    [
        (SimpleNamespace(gpu_bytes=26_800_000_000, system_bytes=None), 26_800_000_000, None),
        (SimpleNamespace(gpu_bytes=10_700_000_000, system_bytes=33_400_000_000), 10_700_000_000, 33_400_000_000),
        (None, None, None),
    ],
)
@pytest.mark.usefixtures("hub_offline")
def test_load_web_config_builds_seedvr2_setup_from_the_memory_budget_and_engine(monkeypatch, tmp_path, budget, expected_gpu, expected_system):
    _patch_minimal_web_config(monkeypatch, tmp_path)
    setup = object()
    calls: list[dict] = []

    def _build(config, **kwargs):
        calls.append({"config": config, **kwargs})
        return setup

    monkeypatch.setattr(web_config_module, "supports_seedvr2", lambda: True)
    monkeypatch.setattr(web_config_module, "get_seedvr2_upscaler", lambda: SimpleNamespace(weights=MLX_WEIGHTS))
    monkeypatch.setattr(web_config_module, "memory_budget", lambda: budget)
    monkeypatch.setattr(web_config_module, "build_seedvr2_setup", _build)

    web_config = web_config_module.load_web_config()

    assert web_config.seedvr2 is setup
    assert calls[0]["weights"] is MLX_WEIGHTS
    assert (calls[0]["budget_bytes"], calls[0]["system_bytes"]) == (expected_gpu, expected_system)
    assert calls[0]["models_dir"] == tmp_path / ".ziv" / "models"


def _seedvr2_app_config() -> dict[str, object]:
    tiers = [{"model": "seedvr2-7b", "weights_gb": 9.26, "activation_floor_gb": 10.79, "activation_base_gb": 1.88, "activation_gb_per_mp": 0.83}]
    return {**_make_app_config(), "seedvr2": {"model": "auto", "softness": 0.75, "video_softness": 0.0, "reference_megapixels": 16, "tiers": tiers}}


@pytest.mark.usefixtures("hub_offline")
def test_gallery_api_reports_a_seedvr2_copy_stored_after_the_server_started(monkeypatch, tmp_path):
    """The download state is read on every request, so a first-use download shows without a restart."""
    from fastapi.testclient import TestClient
    from PIL import Image

    from zvisiongenerator.utils.stored_quant import MANIFEST_NAME
    from zvisiongenerator.web import server as web_server

    _patch_minimal_web_config(monkeypatch, tmp_path)
    monkeypatch.setattr(web_config_module, "load_config", _seedvr2_app_config)
    monkeypatch.setattr(web_config_module, "supports_seedvr2", lambda: True)
    monkeypatch.setattr(web_config_module, "get_seedvr2_upscaler", lambda: SimpleNamespace(weights=MLX_WEIGHTS))
    monkeypatch.setattr(web_config_module, "memory_budget", lambda: SimpleNamespace(gpu_bytes=26_800_000_000, system_bytes=None))
    outputs = Path(web_config_module.load_web_config().output_dir)
    Image.new("RGB", (64, 64)).save(outputs / "a.png")

    def _seedvr2() -> dict:
        return client.get("/api/gallery").json()["assets"][0]["upscale"]["seedvr2"]

    with TestClient(web_server.app) as client:
        before = _seedvr2()
        copy = tmp_path / ".ziv" / "models" / ".seedvr2" / "seedvr2-7b@q8"
        copy.mkdir(parents=True)
        (copy / MANIFEST_NAME).write_text('{"bits": 8, "backend_format": "mflux-seedvr2", "model": "seedvr2-7b"}', encoding="utf-8")
        after = _seedvr2()

    assert (before["downloaded"], before["download_size_label"]) == (False, "17.0 GB")
    assert (after["downloaded"], after["download_size_label"]) == (True, None)


def test_default_output_dir_follows_ziv_data_dir(monkeypatch, tmp_path):
    """The shipped config leaves ui.output_dir unset so outputs live under the data dir."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ZIV_DATA_DIR", str(data_dir))

    ui_config = load_config()["ui"]

    assert ui_config["output_dir"] is None
    assert config_contract_module.resolve_output_dir(ui_config["output_dir"]) == data_dir / "outputs"


def test_default_output_dir_is_home_ziv_outputs_without_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ZIV_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    ui_config = load_config()["ui"]

    assert config_contract_module.resolve_output_dir(ui_config["output_dir"]) == tmp_path / ".ziv" / "outputs"


def test_explicit_user_output_dir_wins_over_ziv_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ZIV_DATA_DIR", str(data_dir))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    data_dir.mkdir()

    for chosen in (tmp_path / "chosen", "~/.ziv/outputs"):
        (data_dir / "config.yaml").write_text(f"ui:\n  output_dir: {chosen}\n", encoding="utf-8")
        expected = Path(chosen).expanduser()

        assert config_contract_module.resolve_output_dir(load_config()["ui"]["output_dir"]) == expected
