"""Video inventory built from the packaged config, per platform (the LTX-2.5 alias surfaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._factories import load_packaged_config

from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.web import model_inventory as model_inventory_module
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, discover_video_inventory, selectable_image_inventory, stored_quant_of


@pytest.fixture
def packaged_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Return the packaged config with the user's data dir (and so any user config) swapped for an empty one."""
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
    return load_packaged_config()


def _video_inventory(config: dict[str, Any], tmp_path: Path, platform: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr("sys.platform", platform)
    entries = discover_video_inventory(config, tmp_path, list_installed=lambda _data_dir: [])
    return {entry.name: entry for entry in entries}


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_ltx25_alias_is_listed_beside_the_ltx23_aliases(packaged_config, tmp_path, monkeypatch, platform):
    inventory = _video_inventory(packaged_config, tmp_path, platform, monkeypatch)

    entry = inventory["ltx25-4"]
    assert (entry.source, entry.family, entry.supports_i2v) == ("alias", "ltx", True)
    assert entry.resolved_path == packaged_config["model_aliases"]["ltx25-4"]["darwin" if platform == "darwin" else "linux"]
    assert {"ltx-4", "ltx-8"} <= set(inventory)


def test_ltx23_aliases_resolve_unchanged_off_macos(packaged_config, tmp_path, monkeypatch):
    inventory = _video_inventory(packaged_config, tmp_path, "linux", monkeypatch)

    default_repo = packaged_config["video_model_presets"]["ltx"]["diffusers"]["default_repo"]
    for name in ("ltx-4", "ltx-8"):
        assert inventory[name].family == "ltx"
        assert inventory[name].resolved_path == default_repo
    assert "2.5" not in default_repo


def test_h3_alias_is_listed_on_macos_as_an_h3_model(packaged_config, tmp_path, monkeypatch):
    inventory = _video_inventory(packaged_config, tmp_path, "darwin", monkeypatch)

    entry = inventory["h3"]
    assert (entry.source, entry.family, entry.supports_i2v) == ("alias", "h3", True)
    assert entry.resolved_path == packaged_config["video_model_presets"]["h3"]["repo_ref"]


def _installed(name: str) -> ImageInventoryEntry:
    return ImageInventoryEntry(name=name, family="unknown", size=None, source="installed", resolved_path=f"/models/{name}")


def test_the_macos_h3_text_encoder_copy_is_a_stored_quant_not_a_model():
    entries = (_installed("custom"), _installed("h3@mlx-te4"), _installed("custom@q4"))

    assert stored_quant_of(entries[1]) == ("h3", 4)
    assert [entry.name for entry in selectable_image_inventory(entries)] == ["custom"]


def test_the_macos_h3_text_encoder_copy_is_not_listed_as_an_installed_video_model(tmp_path, packaged_config, monkeypatch):
    from zvisiongenerator.utils.video_model_detect import detect_video_model

    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
    models_dir = tmp_path / "ziv" / "models"
    for name in ("h3@mlx-te4", "h3-local"):
        (models_dir / name).mkdir(parents=True)
        (models_dir / name / "model_index.json").write_text('{"_class_name": "MiniMaxH3ModularPipeline"}', encoding="utf-8")
    assert detect_video_model(str(models_dir / "h3@mlx-te4")).family == "h3"  # nothing but the name keeps it out

    inventory = discover_video_inventory(packaged_config, tmp_path / "ziv")

    installed = {entry.name for entry in inventory if entry.source == "installed"}
    assert "h3-local" in installed
    assert "h3@mlx-te4" not in installed


def test_image_inventory_does_not_probe_video_aliases(tmp_path):
    """Video aliases are recognised offline; probing them as image repos cost a Hub request per page load."""
    probed: list[str] = []

    def _detect(path: str) -> ImageModelInfo:
        probed.append(path)
        return ImageModelInfo(family="zimage", is_distilled=False, size=None)

    inventory = model_inventory_module.discover_image_inventory(
        {"model_aliases": {"ltx-8": "dgrauet/ltx-2.3-mlx-q8", "zit": "Tongyi-MAI/Z-Image-Turbo"}},
        tmp_path,
        list_installed=lambda _data_dir: [],
        resolve_alias_path=lambda name, **_kwargs: {"ltx-8": "dgrauet/ltx-2.3-mlx-q8", "zit": "Tongyi-MAI/Z-Image-Turbo"}[name],
        detect_model=_detect,
    )

    assert probed == ["Tongyi-MAI/Z-Image-Turbo"]
    assert [entry.name for entry in inventory] == ["zit"]


def test_image_inventory_still_probes_local_paths_containing_ltx(tmp_path):
    """Only remote video repos are skipped; a local image model under e.g. /Volumes/Voltx must still be detected."""
    local_dir = str(tmp_path / "Voltx" / "my-image-model")
    inventory = model_inventory_module.discover_image_inventory(
        {"model_aliases": {"mine": local_dir}},
        tmp_path,
        list_installed=lambda _data_dir: [],
        resolve_alias_path=lambda _name, **_kwargs: local_dir,
        detect_model=lambda _path: ImageModelInfo(family="zimage", is_distilled=False, size=None),
    )
    assert [entry.name for entry in inventory] == ["mine"]
