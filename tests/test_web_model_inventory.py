"""Video inventory built from the packaged config, per platform (the LTX-2.5 alias surfaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zvisiongenerator.utils.config import load_config, model_display_name
from zvisiongenerator.web.first_run import recommended_first_run_model
from zvisiongenerator.web.model_inventory import discover_video_inventory

_LTX25_REPO = "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"


@pytest.fixture
def packaged_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Return the packaged config with the user's data dir (and so any user config) swapped for an empty one."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
    return load_config()


def _video_inventory(config: dict[str, Any], tmp_path: Path, platform: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr("sys.platform", platform)
    entries = discover_video_inventory(config, tmp_path, list_installed=lambda _data_dir: [])
    return {entry.name: entry for entry in entries}


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_ltx25_alias_is_listed_beside_the_ltx23_aliases(packaged_config, tmp_path, monkeypatch, platform):
    inventory = _video_inventory(packaged_config, tmp_path, platform, monkeypatch)

    entry = inventory["ltx25-4"]
    assert (entry.source, entry.family, entry.supports_i2v) == ("alias", "ltx", True)
    assert entry.resolved_path == _LTX25_REPO
    assert {"ltx-4", "ltx-8"} <= set(inventory)


def test_ltx23_aliases_resolve_unchanged_off_macos(packaged_config, tmp_path, monkeypatch):
    inventory = _video_inventory(packaged_config, tmp_path, "linux", monkeypatch)

    default_repo = packaged_config["video_model_presets"]["ltx"]["diffusers"]["default_repo"]
    for name in ("ltx-4", "ltx-8"):
        assert inventory[name].family == "ltx"
        assert inventory[name].resolved_path == default_repo
    assert "2.5" not in default_repo


def test_ltx25_alias_is_hidden_on_macos_and_the_ltx23_aliases_remain(packaged_config, tmp_path, monkeypatch):
    inventory = _video_inventory(packaged_config, tmp_path, "darwin", monkeypatch)

    assert "ltx25-4" not in inventory
    assert {"ltx-4", "ltx-8"} <= set(inventory)
    assert inventory["ltx-8"].resolved_path == packaged_config["model_aliases"]["ltx-8"]["darwin"]


def test_ltx25_alias_has_its_label(packaged_config):
    assert model_display_name(packaged_config, "ltx25-4") == "LTX-2.5 4-bit"
    assert model_display_name(packaged_config, "ltx-4") == "LTX-2.3 4-bit"
    assert model_display_name(packaged_config, "ltx-8") == "LTX-2.3 8-bit"


def test_default_video_model_and_first_run_recommendation_are_unchanged(packaged_config):
    assert packaged_config["ui"]["default_models"]["video"] == "ltx-8"
    assert packaged_config["ui"]["default_models"]["image"] == "klein9b"
    assert recommended_first_run_model(packaged_config, None, ["klein4b", "klein9b"]) == "klein4b"
    assert packaged_config["first_run"]["large_model"] == "klein9b"
