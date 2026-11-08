"""Tests for the model metadata reader: the cached parse, the entry lookups and the per-platform values."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.downloads.declaration import companion_targets, declared_repo_ids, model_targets
from zvisiongenerator.downloads.plans import audio_plan
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.enhancer_models import enhancer_tier_models, smart_unavailable
from zvisiongenerator.utils.config import load_config, resolve_scheduler_class, video_load_options
from zvisiongenerator.utils.h3_preset import h3_preset
from zvisiongenerator.utils.licences import licences_for, load_licences
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.platform import PLATFORM_LABELS
from zvisiongenerator.utils.seedvr2 import seedvr2_settings
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.model_delete import freed_repo_ids
from zvisiongenerator.utils.model_metadata import (
    METADATA_FILE,
    alias_map,
    built_model_ids,
    builtin_aliases,
    family_block,
    for_platform,
    image_model_family,
    load_model_metadata,
    model_display_name,
    model_download_size_label,
    model_entry,
    model_ids,
)


def test_the_packaged_file_is_read_once_and_shared() -> None:
    assert METADATA_FILE.is_file()
    assert load_model_metadata() is load_model_metadata()


def test_a_copy_edited_and_served_replaces_the_packaged_data_for_one_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = metadata_copy()
    data["models"]["zit"]["licences"] = ["flux-nc"]
    use_model_metadata(monkeypatch, tmp_path, data)

    assert model_entry("zit")["licences"] == ["flux-nc"]
    assert data is not load_model_metadata()


def test_the_packaged_metadata_is_not_changed_by_a_served_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = metadata_copy()
    data["models"].clear()
    use_model_metadata(monkeypatch, tmp_path, data)

    assert model_ids() == ()
    monkeypatch.undo()
    assert "zit" in model_ids()


def test_an_unknown_model_has_no_entry() -> None:
    assert model_entry("no-such-model") is None


def test_model_ids_are_in_file_order_and_can_be_limited_to_a_kind() -> None:
    assert model_ids("upscaler") == ("seedvr2-7b", "seedvr2-3b")
    assert model_ids("matte") == ("matte",)
    assert model_ids()[:2] == ("ltx-8", "ltx-4")
    assert set(model_ids("video")) < set(model_ids())


@pytest.mark.parametrize(
    ("value", "platform_key", "expected"),
    [
        ({"darwin": 1, "win32": 2, "linux": 3}, "win32", 2),
        ({"darwin": 1}, "linux", None),
        ({"weights_gb": 1.6}, "linux", {"weights_gb": 1.6}),
        ("19 GB", "darwin", "19 GB"),
        ([1, 2], "darwin", [1, 2]),
        ({}, "darwin", {}),
    ],
)
def test_for_platform_picks_a_per_platform_value_and_passes_a_plain_one_through(value: object, platform_key: str, expected: object) -> None:
    assert for_platform(value, platform_key) == expected


def test_a_built_in_alias_is_the_source_its_entry_loads_on_each_platform() -> None:
    aliases = builtin_aliases()

    assert aliases["zit"] == "Tongyi-MAI/Z-Image-Turbo"
    assert aliases["ltx-4"]["darwin"].startswith("dgrauet/ltx-2.3-mlx-q4@")
    assert aliases["ltx-4"]["win32"] == aliases["ltx-4"]["linux"] == aliases["ltx-8"]["linux"]
    assert aliases["h3"].startswith("MiniMaxAI/MiniMax-H3@")


def test_a_built_model_is_its_own_alias_source() -> None:
    aliases = builtin_aliases()

    assert [aliases[name] for name in ("ideo", "moss", "moss-sft")] == ["ideo", "moss", "moss-sft"]
    assert built_model_ids() == ("ideo", "moss", "moss-sft")


def test_only_image_and_video_models_are_aliases() -> None:
    assert list(builtin_aliases()) == [*model_ids("video"), *model_ids("image")]


def test_alias_map_lists_the_built_ins_in_file_order_then_the_users() -> None:
    aliases = alias_map({"model_aliases": {"mine": "owner/repo", "other": {"darwin": "owner/mlx"}}})

    assert list(aliases)[-2:] == ["mine", "other"]
    assert list(aliases)[:-2] == list(builtin_aliases())
    assert aliases["other"] == {"darwin": "owner/mlx"}


def test_alias_map_ignores_a_user_alias_named_like_a_built_in() -> None:
    aliases = alias_map({"model_aliases": {"zit": "owner/other", "mine": "owner/repo"}})

    assert aliases["zit"] == builtin_aliases()["zit"]
    assert aliases["mine"] == "owner/repo"


def test_a_user_alias_resolves_and_a_built_in_name_keeps_its_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("model_aliases:\n  mine: owner/repo\n  zit: owner/other\n", encoding="utf-8")
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    aliases = alias_map(load_config())

    assert resolve_model_path("mine", aliases=aliases, platform_key="linux") == "owner/repo"
    assert resolve_model_path("zit", aliases=aliases, platform_key="linux") == builtin_aliases()["zit"]


@pytest.mark.parametrize("config", [{}, {"model_aliases": None}, {"model_aliases": {}}])
def test_alias_map_without_user_aliases_is_the_built_ins(config: dict) -> None:
    assert alias_map(config) == builtin_aliases()


def test_alias_map_gives_the_caller_a_dict_it_may_edit() -> None:
    alias_map({})["extra"] = "owner/repo"

    assert "extra" not in builtin_aliases()


def test_a_model_reads_as_its_entry_name_and_an_unknown_one_as_itself() -> None:
    assert model_display_name("klein9b") == "FLUX.2 Klein 9B"
    assert model_display_name("my-own-alias") == "my-own-alias"


def test_only_some_image_models_name_a_family_the_web_ui_lists_them_by_offline() -> None:
    assert image_model_family("ideo") == "ideogram4"
    assert image_model_family("krea2") == "krea2"
    assert image_model_family("zit") is None
    assert image_model_family("ltx-4") is None
    assert image_model_family("unknown-name") is None


@pytest.mark.parametrize(
    ("name", "platform_key", "label"),
    [
        ("ideo", "darwin", "28 GB"),
        ("ideo", "linux", "24 GB"),
        ("klein4b", "win32", "16 GB"),
        ("seedvr2-7b", "darwin", "17.0 GB"),
        ("seedvr2-3b", "win32", "3.9 GB"),
        ("matte", "linux", "548 MB"),
        ("ltx-4", "darwin", None),
        ("my-own-alias", "darwin", None),
    ],
)
def test_the_download_size_label_is_per_platform_and_absent_where_unset(name: str, platform_key: str, label: str | None) -> None:
    assert model_download_size_label(name, platform_key) == label


def test_family_block_overlays_a_version_the_way_user_config_merges() -> None:
    base = family_block("ltx", None)
    overlaid = family_block("ltx", "ltx25")

    assert overlaid["sampler"] == "euler_ancestral"
    assert overlaid["companions"]["darwin"] == []
    assert [target["role"] for target in overlaid["companions"]["win32"]] == ["latent_upsampler"]
    assert overlaid["frame_rule"] == base["frame_rule"]
    assert overlaid["distilled_lora"]["file"] == "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    assert "sampler" not in base
    assert not {"versions", "model_type"} & overlaid.keys()


def test_family_block_returns_a_copy_nothing_else_sees() -> None:
    block = family_block("ltx", "ltx25")
    block["companions"]["win32"].clear()
    block["frame_rule"]["alignment"] = 99

    fresh = family_block("ltx", "ltx25")
    assert len(fresh["companions"]["win32"]) == 1
    assert fresh["frame_rule"]["alignment"] == 8
    assert load_model_metadata()["video_families"]["ltx"]["frame_rule"]["alignment"] == 8


def test_family_block_of_an_unknown_family_or_version_raises() -> None:
    with pytest.raises(KeyError, match="no-such"):
        family_block("no-such", None)
    with pytest.raises(KeyError, match="no-such"):
        family_block("ltx", "no-such")


_PLATFORMS = ("darwin", "win32", "linux")
_USER_SETS_A_FACT = {
    "model_aliases.zit": "model_aliases:\n  zit: owner/other\n  ltx-4: owner/other\n",
    "model_labels.klein9b": "model_labels:\n  klein9b:\n    name: Mine\n    download_size_label: 1 GB\n",
    "video_model_presets.ltx.diffusers.default_repo": "video_model_presets:\n  ltx:\n    diffusers:\n      default_repo: owner/other\n",
    "video_model_presets.h3.repo_ref": "video_model_presets:\n  h3:\n    repo_ref: owner/other\n    diffusers:\n      attention: sdpa\n",
    "prompt_enhancer.model": "prompt_enhancer:\n  model: owner/other\n  revision: main\n",
    "prompt_enhancer.smart.min_memory_gb": "prompt_enhancer:\n  smart:\n    min_memory_gb:\n      darwin: 1\n      win32: 1\n      linux: 1\n",
    "seedvr2.tiers": "seedvr2:\n  tiers:\n    - {model: seedvr2-7b, weights_gb: 1, working_gb: 1}\n",
    "audio_model_presets.yue2.files": "audio_model_presets:\n  yue2:\n    files:\n      - {repo_id: owner/other, patterns: ['*']}\n",
    "platforms.darwin": "platforms:\n  darwin: Mine\n",
    "schedulers.beta.mflux_class": "schedulers:\n  beta:\n    mflux_class: owner.Other\n    diffusers_class: owner.Other\n",
    "models section": "models:\n  zit:\n    name: Mine\n    licences: [flux-nc]\n",
    "licences section": "licences:\n  zimage-apache:\n    name: Mine\n    supplier: Me\n    summary: Free\n    url: https://example.com\n",
}


def _derived_facts(config: dict[str, Any]) -> dict[str, Any]:
    """Return the values the app derives from the model facts, for *config* on every platform."""
    aliases = alias_map(config)
    facts: dict[str, Any] = {
        "aliases": {name: aliases[name] for name in builtin_aliases()},
        "name": model_display_name("klein9b"),
        "image_family": [image_model_family("ideo"), image_model_family("zit")],
        "licences": [record.id for record in licences_for(load_licences(), "zit")],
        "yue2": parse_audio_presets(config),
        "yue2_plan": audio_plan(next(preset for preset in parse_audio_presets(config) if preset.id == "yue2")),
        "scheduler": [resolve_scheduler_class(name, backend) for name in ("beta", "simple") for backend in ("mflux", "diffusers")],
        "detected": detect_video_model("diffusers/ltx-2.3-distilled-diffusers"),
    }
    for platform_key in _PLATFORMS:
        facts[platform_key] = {
            "label": PLATFORM_LABELS[platform_key],
            "ltx-4": (model_targets("ltx-4", platform_key), companion_targets("ltx", None, platform_key), video_load_options("ltx", config, "ltx-4", platform_key=platform_key, upscale=True)),
            "yue2": (model_targets("yue2", platform_key), declared_repo_ids("yue2", platform_key)),
            "enhancer": (enhancer_tier_models(platform_key=platform_key), smart_unavailable(platform_key=platform_key, budget_bytes=2**30)),
            "seedvr2": seedvr2_settings(config, platform_key).tiers,
            "freed": freed_repo_ids(
                SimpleNamespace(name="ltx-4", family="ltx", resolved_path=builtin_aliases()["ltx-4"][platform_key]), (), models_dir=Path("/nonexistent"), platform_key=platform_key
            ),
        }
    facts["h3"] = [h3_preset(config, platform_key) for platform_key in ("win32", "linux")]
    return facts


@pytest.mark.parametrize("settings", _USER_SETS_A_FACT.values(), ids=_USER_SETS_A_FACT.keys())
def test_a_user_config_cannot_change_a_model_fact(settings: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty, user = tmp_path / "empty", tmp_path / "user"
    empty.mkdir()
    user.mkdir()
    (user / "config.yaml").write_text(settings, encoding="utf-8")
    monkeypatch.setenv("ZIV_DATA_DIR", str(empty))
    baseline = _derived_facts(load_config())

    monkeypatch.setenv("ZIV_DATA_DIR", str(user))
    config = load_config()

    assert _derived_facts(config) == baseline
