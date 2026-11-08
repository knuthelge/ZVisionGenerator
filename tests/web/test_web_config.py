"""Tests for declarative Web UI config loading and contract helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests._factories import packaged_metadata_copy, use_model_entries
from zvisiongenerator.utils.yaml_round_trip import load_round_trip
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


@pytest.fixture(autouse=True)
def _no_built_in_models(monkeypatch, tmp_path):
    """Serve metadata with no models, so the aliases a test sees are the ones it writes into its config."""
    use_model_entries(monkeypatch, tmp_path, enhancers=True)


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
    assert web_config.scheduler_options == ("beta", "simple")
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


_IDEO = {"kind": "image", "family": "ideogram4", "downloads": [{"ref": "ideogram-ai/ideogram-4-fp8"}]}


def test_load_web_config_surfaces_ideo_alias_offline_via_declared_family(monkeypatch, tmp_path):
    app_config = _make_app_config()
    app_config["model_aliases"] = {"custom-default": str(tmp_path / "models" / "custom-default")}
    app_config["ui"]["default_models"] = {"image": "ideo"}
    use_model_entries(monkeypatch, tmp_path, {"ideo": _IDEO}, enhancers=True)

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


_PORTABLE = {"portable-image": "Tongyi-MAI/Z-Image-Turbo"}
_BLOCKED_ALIASES = {
    **_PORTABLE,
    "windows-only-blocked": {
        "darwin": "Tongyi-MAI/Z-Image-Turbo",
    },
}


def test_load_web_config_hides_aliases_the_platform_cannot_resolve(monkeypatch, tmp_path):
    """A user alias that is unavailable on the platform is hidden from the options and the default falls back to an available one."""
    app_config = _make_app_config()
    app_config["model_aliases"] = _BLOCKED_ALIASES
    app_config["ui"]["default_models"] = {"image": "portable-image"}
    use_model_entries(monkeypatch, tmp_path, enhancers=True)

    _patch_minimal_web_config(
        monkeypatch,
        tmp_path,
        app_config,
        platform="win32",
        detect_image=lambda value: ImageModelInfo(family="zimage" if "Z-Image" in str(value) else "unknown", is_distilled=False, size=None),
        detect_video=lambda value: SimpleNamespace(family="ltx" if "Diffusers" in str(value) else "unknown", supports_i2v=True),
    )

    web_config = web_config_module.load_web_config()

    assert web_config.image_model_options == ("portable-image",)
    assert web_config.default_models.image == "portable-image"


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


def _persist_current() -> SimpleNamespace:
    return SimpleNamespace(
        image_model_options=("zit", "local-image"),
        video_model_options=("ltx-8",),
        app_config={"generation": {"default_ratio": "2:3"}},
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m", "l")},
    )


def _write_user_config(monkeypatch, tmp_path: Path, text: str) -> Path:
    """Point ZIV_DATA_DIR at *tmp_path* and write *text* as its config.yaml."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_persist_writable_config_patch_clears_and_normalizes_values(monkeypatch, tmp_path):
    """Writable config patches should normalize paths and treat null as clear."""
    monkeypatch.chdir(tmp_path)
    path = _write_user_config(
        monkeypatch,
        tmp_path,
        yaml.safe_dump({"ui": {"default_models": {"image": "zit", "video": "ltx-8"}, "output_dir": str(tmp_path / "old-output")}, "generation": {"default_size": "m"}}),
    )

    config_contract_module.persist_writable_config_patch(
        {"ui.default_models.image": None, "ui.output_dir": " exports ", "generation.default_size": "l"},
        _persist_current(),
    )

    assert _load_yaml(path) == {
        "ui": {"default_models": {"video": "ltx-8"}, "output_dir": str(tmp_path / "exports")},
        "generation": {"default_size": "l"},
    }


def test_persist_writable_config_patch_clears_output_dir_override(monkeypatch, tmp_path):
    """Clearing the output-dir patch should remove the persisted override entirely."""
    path = _write_user_config(
        monkeypatch,
        tmp_path,
        yaml.safe_dump({"ui": {"default_models": {"image": "zit", "video": "ltx-8"}, "output_dir": str(tmp_path / "old-output")}}),
    )

    config_contract_module.persist_writable_config_patch({"ui.output_dir": None}, _persist_current())

    assert _load_yaml(path) == {"ui": {"default_models": {"image": "zit", "video": "ltx-8"}}}


# A user file with a header, end-of-line comments, a heading and commented-out keys at the end.
CONFIG_WITH_COMMENTS = """\
# My Ziv config

aliases:  # my own shortcuts
  mine: some/repo  # keep me
  # old: other/repo

discover_content_filter: off

ui:
  output_dir: /old/outputs  # where files go
  gallery_page_size: 12
  default_models:
    image: zit  # my favourite

# Generation
generation:
  default_size: m  # medium

# Tried once:
#   aliases:
#     x: y/z
"""

# The packaged ui block: comment lines between keys and a multi-line comment before the last one.
PACKAGED_UI_BLOCK = """\
ui:
  startup_view: workspace
  gallery_page_size: 12
  output_dir: null  # unset: the data dir's outputs folder
  default_models:
    image: klein9b
    video: ltx-8
  # Set once the first-run dialog is skipped; a user setting.
  first_run_skipped: false
  # Set when Show Hugging Face LoRAs is chosen in Discover LoRAs.
  discover_huggingface: false
  # Discover's content filter on Hugging Face results.
  # Set to off if it hides a LoRA you want to use.
  discover_content_filter: on
"""


def _comments(text: str) -> list[str]:
    """Return the comment text of every line, in file order (a whole-line comment or an end-of-line one)."""
    return [line[line.index("#") :].strip() for line in text.splitlines() if "#" in line]


def _edited(old: dict, path: str, value: object) -> dict:
    """Return a copy of *old* with the dotted *path* set to *value*, or removed (with emptied parents) when it is None."""
    result = copy.deepcopy(old)
    keys = path.split(".")
    parents = [result]
    for key in keys[:-1]:
        if value is None and not isinstance(parents[-1].get(key), dict):
            return result
        parents.append(parents[-1].setdefault(key, {}))
    if value is None:
        parents[-1].pop(keys[-1], None)
        for parent, key in zip(reversed(parents[:-1]), reversed(keys[:-1]), strict=True):
            if not parent[key]:
                del parent[key]
    else:
        parents[-1][keys[-1]] = value
    return result


def _save_patch(monkeypatch, tmp_path: Path, text: str, patch: dict[str, object]) -> tuple[str, str]:
    """Write *text* as config.yaml, apply *patch* through the Web UI save, and return the old and new text."""
    path = _write_user_config(monkeypatch, tmp_path, text)
    config_contract_module.persist_writable_config_patch(patch, _persist_current())
    return text, path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "value", "dropped"),
    [
        pytest.param("generation.default_size", "l", [], id="set an existing key"),
        pytest.param("ui.default_models.image", "local-image", [], id="set an existing nested key"),
        pytest.param("prompt_enhancer.user_model", "owner/name", [], id="set a key under a missing parent"),
        pytest.param("ui.default_models.video", "ltx-8", [], id="set a new key last in its mapping"),
        pytest.param("ui.output_dir", None, ["# where files go"], id="clear the first key of a mapping"),
        pytest.param("ui.default_models.image", None, ["# my favourite"], id="clear the last key before a heading"),
        pytest.param("generation.default_size", None, ["# medium"], id="clear a section's only key"),
    ],
)
def test_config_save_keeps_comments_and_changes_only_the_edited_path(monkeypatch, tmp_path, path, value, dropped):
    """A Config page save keeps every comment except the cleared key's own, and changes only the edited path."""
    old, new = _save_patch(monkeypatch, tmp_path, CONFIG_WITH_COMMENTS, {path: value})

    assert _comments(new) == [comment for comment in _comments(old) if comment not in dropped]
    assert yaml.safe_load(new) == _edited(yaml.safe_load(old), path, value)
    assert new.startswith("# My Ziv config\n")
    assert new.rstrip().endswith("#     x: y/z")  # the commented-out keys stay at the end


def test_config_save_keeps_the_heading_above_its_section(monkeypatch, tmp_path):
    """Adding a key at the end of the ui block does not pull the next section's heading above it."""
    _old, new = _save_patch(monkeypatch, tmp_path, CONFIG_WITH_COMMENTS, {"ui.default_models.video": "ltx-8"})

    assert "# Generation\ngeneration:\n" in new
    assert new.index("video: ltx-8") < new.index("# Generation")


def test_config_save_keeps_the_heading_when_a_nested_section_is_cleared(monkeypatch, tmp_path):
    """Clearing the last key of a section leaves the lines that followed it where they were."""
    _old, new = _save_patch(monkeypatch, tmp_path, CONFIG_WITH_COMMENTS, {"ui.default_models.image": None})

    assert "  gallery_page_size: 12\n\n# Generation\ngeneration:\n" in new


def test_config_save_with_several_new_keys_keeps_trailing_comments_last(monkeypatch, tmp_path):
    """Two keys added in one save both land before the file's trailing comments."""
    _old, new = _save_patch(
        monkeypatch,
        tmp_path,
        CONFIG_WITH_COMMENTS,
        {"prompt_enhancer.user_model": "owner/name", "ui.default_models.video": "ltx-8"},
    )

    assert yaml.safe_load(new)["prompt_enhancer"] == {"user_model": "owner/name"}
    assert new.rstrip().endswith("#     x: y/z")
    assert new.index("user_model") < new.index("# Tried once")


def test_config_save_quotes_strings_that_yaml_one_one_reads_as_another_type(monkeypatch, tmp_path):
    """A value PyYAML would read as a float is written quoted, so it comes back as the string that was saved."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_contract_module, "normalize_user_directory", lambda value: value)
    _old, new = _save_patch(monkeypatch, tmp_path, CONFIG_WITH_COMMENTS, {"ui.output_dir": "1.10"})

    assert yaml.safe_load(new)["ui"]["output_dir"] == "1.10"
    assert '"1.10"' in new


def test_config_save_keeps_the_values_yaml_one_one_reads_in_untouched_keys(monkeypatch, tmp_path):
    """Plain ``off`` and ``on`` stay what PyYAML read them as (booleans) after a save elsewhere."""
    _old, new = _save_patch(monkeypatch, tmp_path, CONFIG_WITH_COMMENTS, {"generation.default_size": "l"})

    assert yaml.safe_load(new)["discover_content_filter"] is False


@pytest.mark.parametrize(
    ("patch", "dropped", "expected"),
    [
        pytest.param({"ui.output_dir": None}, ["# unset: the data dir's outputs folder"], ("ui.output_dir", None), id="clear output_dir"),
        pytest.param({"ui.default_models.video": None}, [], ("ui.default_models.video", None), id="clear the last default model"),
    ],
)
def test_config_save_keeps_the_packaged_ui_block_comments(monkeypatch, tmp_path, patch, dropped, expected):
    """The comment lines between the ui keys survive clearing a key, and the lines after it stay below its section."""
    old, new = _save_patch(monkeypatch, tmp_path, PACKAGED_UI_BLOCK, patch)

    assert _comments(new) == [comment for comment in _comments(old) if comment not in dropped]
    assert yaml.safe_load(new) == _edited(yaml.safe_load(old), *expected)
    assert "  # Set once the first-run dialog is skipped; a user setting.\n  first_run_skipped: false\n" in new


def test_config_save_adds_a_default_model_below_the_ui_comments(monkeypatch, tmp_path):
    """A default video model added to a ui block that lacks it goes with the other default models."""
    text = PACKAGED_UI_BLOCK.replace("    video: ltx-8\n", "")
    old, new = _save_patch(monkeypatch, tmp_path, text, {"ui.default_models.video": "ltx-8"})

    assert _comments(new) == _comments(old)
    assert yaml.safe_load(new) == _edited(yaml.safe_load(old), "ui.default_models.video", "ltx-8")
    assert "    video: ltx-8\n  # Set once the first-run dialog is skipped; a user setting.\n" in new


def test_first_run_skip_and_discover_switch_keep_comments(monkeypatch, tmp_path):
    """The first-run skip and the Discover switch edit their own key and leave the file's comments alone."""
    path = _write_user_config(monkeypatch, tmp_path, PACKAGED_UI_BLOCK)

    config_contract_module.persist_first_run_skipped()
    config_contract_module.persist_discover_huggingface(True)

    new = path.read_text(encoding="utf-8")
    assert _comments(new) == _comments(PACKAGED_UI_BLOCK)
    assert yaml.safe_load(new)["ui"]["first_run_skipped"] is True
    assert config_contract_module.discover_huggingface_enabled() is True
    expected = yaml.safe_load(PACKAGED_UI_BLOCK)
    expected["ui"].update(first_run_skipped=True, discover_huggingface=True)
    assert yaml.safe_load(new) == expected


def test_first_run_skip_and_discover_switch_create_the_file(monkeypatch, tmp_path):
    """A missing or empty config.yaml is created with the one key."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    config_contract_module.persist_first_run_skipped()
    assert _load_yaml(tmp_path / "config.yaml") == {"ui": {"first_run_skipped": True}}

    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    config_contract_module.persist_discover_huggingface(True)
    assert _load_yaml(tmp_path / "config.yaml") == {"ui": {"discover_huggingface": True}}


def test_config_save_keeps_a_file_that_holds_only_comments(monkeypatch, tmp_path):
    """A config.yaml of commented-out settings keeps them when the first key is saved."""
    path = _write_user_config(monkeypatch, tmp_path, "# ui:\n#   output_dir: /somewhere\n")

    config_contract_module.persist_first_run_skipped()

    new = path.read_text(encoding="utf-8")
    assert new.startswith("# ui:\n#   output_dir: /somewhere\n")
    assert yaml.safe_load(new) == {"ui": {"first_run_skipped": True}}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("ui: [unclosed\n", "Failed to read user config override", id="invalid yaml"),
        pytest.param("- just\n- a list\n", "must be a YAML mapping", id="not a mapping"),
        pytest.param("ui:\n  output_dir: /a\nui:\n  output_dir: /b\n", "Failed to read user config override", id="duplicate key"),
    ],
)
def test_config_save_refuses_a_file_it_cannot_round_trip_and_leaves_it_alone(monkeypatch, tmp_path, text, message):
    """An invalid, non-mapping or duplicate-key config.yaml raises ValueError and is not rewritten."""
    path = _write_user_config(monkeypatch, tmp_path, text)

    with pytest.raises(ValueError, match=message):
        config_contract_module.persist_first_run_skipped()
    with pytest.raises(ValueError, match=message):
        config_contract_module.persist_writable_config_patch({"generation.default_size": "l"}, _persist_current())

    assert path.read_text(encoding="utf-8") == text


def test_concurrent_config_saves_do_not_drop_each_others_changes(monkeypatch, tmp_path):
    """Web handlers run in worker threads; overlapping read-modify-write saves must both land."""
    import threading
    import time

    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    real_load = config_contract_module._load_user_config_document

    def _slow_load():
        document = real_load()
        time.sleep(0.05)  # widen the window between read and write
        return document

    monkeypatch.setattr(config_contract_module, "_load_user_config_document", _slow_load)
    patches = [{"ui.default_models.image": "zit"}, {"ui.default_models.video": "ltx-8"}, {"generation.default_size": "l"}]
    threads = [threading.Thread(target=config_contract_module.persist_writable_config_patch, args=(patch, _persist_current())) for patch in patches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    saved = _load_yaml(tmp_path / "config.yaml")
    assert saved == {"ui": {"default_models": {"image": "zit", "video": "ltx-8"}}, "generation": {"default_size": "l"}}


def test_user_config_write_is_atomic(monkeypatch, tmp_path):
    """A failed write must leave the previous file intact and no temp files behind (never a truncated file)."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    config_contract_module.write_user_config_override(load_round_trip("ui:\n  output_dir: /keep\n"))
    original = (tmp_path / "config.yaml").read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(tmp_path), "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        config_contract_module.write_user_config_override(load_round_trip("ui:\n  output_dir: /new\n"))

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

    config_contract_module.write_user_config_override(load_round_trip("ui:\n  output_dir: /new\n"))

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
    return {**_make_app_config(), "seedvr2": {"model": "auto", "softness": 0.75, "video_softness": 0.0, "reference_megapixels": 16}}


@pytest.mark.usefixtures("hub_offline")
def test_gallery_api_reports_a_seedvr2_copy_stored_after_the_server_started(monkeypatch, tmp_path):
    """The download state is read on every request, so a first-use download shows without a restart."""
    from fastapi.testclient import TestClient
    from PIL import Image

    from zvisiongenerator.utils.stored_quant import MANIFEST_NAME
    from zvisiongenerator.web import server as web_server

    _patch_minimal_web_config(monkeypatch, tmp_path)
    use_model_entries(monkeypatch, tmp_path, {"seedvr2-7b": packaged_metadata_copy()["models"]["seedvr2-7b"]}, enhancers=True)
    monkeypatch.setattr("sys.platform", "darwin")  # the tiers, files and size labels are keyed by platform
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
