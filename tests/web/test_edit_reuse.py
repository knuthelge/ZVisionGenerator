"""The gallery's Reuse of a saved edit: it returns to Edit with the picture it started from, and an upscale of it does not."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tests._factories import make_edit_payload
from tests.web._helpers import ffprobe_cannot_read_files
from zvisiongenerator.utils.provenance import embed_png_config, read_png_config, recorded_settings
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import gallery_asset_to_json, list_gallery_assets


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch):
    ffprobe_cannot_read_files(monkeypatch)
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)


def _web_config() -> SimpleNamespace:
    return SimpleNamespace(image_model_options=("zit",), video_model_options=("ltx-8",), default_models=WebUiDefaultModels(image="zit", video="ltx-8"), seedvr2=None)


def _write(path: Path, payload: dict) -> None:
    info = PngInfo()
    embed_png_config(info, payload)
    Image.new("RGB", (64, 48), "blue").save(path, pnginfo=info)


def _reuse(tmp_path: Path, payload: dict, *, source_exists: bool = True) -> tuple[dict, dict[str, str]]:
    source = Path(payload["source"]["path"])
    if source_exists:
        Image.new("RGB", (8, 8)).save(source)
    _write(tmp_path / "result.png", payload)
    asset = next(asset for asset in list_gallery_assets(str(tmp_path)) if asset.name == "result.png")
    result = gallery_asset_to_json(asset, _web_config())
    query = urlsplit(result["reuse_workspace_url"].removeprefix("#")).query
    return result, {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items()}


class TestWhatAnEditRecords:
    def test_the_source_block_has_no_workflow_so_the_gallery_s_override_leaves_the_edit_alone(self, tmp_path):
        """The exemption from the source-workflow override is by data: only an upscale's ``source`` names a workflow."""
        payload = make_edit_payload(tmp_path / "door.png")

        assert "workflow" not in payload["source"]
        assert payload["workflow"] == "edit"

    def test_the_edit_block_is_read_back_and_a_picture_that_is_not_an_edit_has_none(self, tmp_path):
        payload = make_edit_payload(tmp_path / "door.png", extend=("top", 256))
        _write(tmp_path / "result.png", payload)

        recorded = recorded_settings(read_png_config(str(tmp_path / "result.png")))

        assert recorded.edit == {"mode": "extend", "extend": {"side": "top", "amount": 256}}
        assert recorded_settings({"workflow": "txt2img"}).edit is None


class TestReuse:
    def test_a_paint_edit_returns_to_edit_with_its_picture_and_asks_for_paint(self, tmp_path):
        payload = make_edit_payload(tmp_path / "door.png", painted=True)

        result, params = _reuse(tmp_path, payload)

        assert params["workflow"] == "edit"
        assert params["edit_source_path"] == str(tmp_path / "door.png")
        assert params["edit_repaint"] == "true"
        assert {"edit_extend_side", "edit_extend_amount"}.isdisjoint(params)
        assert result["workflow"] == "edit"
        assert result["reuse_state"]["resolved_workflow"] == "edit"
        assert result["reuse_state"]["workflow_available"] is True
        assert result["reuse_state"]["fallback_reasons"] == []

    @pytest.mark.parametrize(
        ("mask", "extend", "repaint", "extended"),
        [(True, None, True, False), (False, ("left", 128), False, True), (False, None, False, False)],
        ids=["paint", "extend", "whole"],
    )
    def test_paint_is_asked_for_again_only_when_the_edit_painted(self, tmp_path, mask, extend, repaint, extended):
        _result, params = _reuse(tmp_path, make_edit_payload(tmp_path / "door.png", painted=mask, extend=extend))

        assert ("edit_repaint" in params) is repaint
        assert ("edit_extend_side" in params) is extended
        if extend:
            assert (params["edit_extend_side"], params["edit_extend_amount"]) == (extend[0], str(extend[1]))

    def test_the_text_model_and_run_settings_carry_over_and_nothing_an_edit_does_not_use(self, tmp_path):
        _result, params = _reuse(tmp_path, make_edit_payload(tmp_path / "door.png", painted=True))

        assert (params["prompt"], params["model"], params["seed"], params["steps"], params["lora"]) == ("a red door", "zit", "7", "9", "")
        assert {"ratio", "size", "width", "height", "guidance", "scheduler", "negative_prompt", "transparent", "reference_path", "sharpen_enabled", "upscale"}.isdisjoint(params)

    def test_a_source_that_moved_stays_edit_names_a_blank_source_and_says_so(self, tmp_path):
        result, params = _reuse(tmp_path, make_edit_payload(tmp_path / "gone.png", painted=True), source_exists=False)

        assert params["workflow"] == "edit"
        assert params["edit_source_path"] == ""
        assert "missing_edit_source" not in params
        assert params["edit_repaint"] == "true"
        assert result["reuse_state"]["workflow_available"] is False
        assert result["reuse_state"]["fallback_reasons"] == ["missing_edit_source"]

    def test_an_edit_of_a_picture_in_the_output_folder_reuses_that_picture(self, tmp_path):
        (tmp_path / "shots").mkdir()
        payload = make_edit_payload(tmp_path / "shots" / "door.png")

        _result, params = _reuse(tmp_path, payload)

        assert params["edit_source_path"] == str(tmp_path / "shots" / "door.png")

    def test_an_extend_reuses_its_side_and_amount_and_the_viewer_gets_the_source_picture(self, tmp_path):
        result, params = _reuse(tmp_path, make_edit_payload(tmp_path / "door.png", extend=("top", 256)))

        assert (params["edit_extend_side"], params["edit_extend_amount"]) == ("top", "256")
        assert (result["details"]["source"]["path"], result["details"]["source"]["width"], result["details"]["source"]["height"]) == (str(tmp_path / "door.png"), 512, 480)

    def test_an_upscale_of_an_edit_regenerates_as_text_to_image(self, tmp_path):
        """An upscale records the workflow of the picture it enlarged (never ``edit``), so its Reuse regenerates that picture."""
        _write(
            tmp_path / "result.png",
            {
                "schema": "zvisiongenerator.config.v1",
                "workflow": "upscale",
                "prompt": "a red door",
                "model": "zit",
                "seed": 7,
                "steps": 9,
                "source": {"path": str(tmp_path / "door.png"), "width": 640, "height": 480, "workflow": "txt2img"},
            },
        )
        asset = next(asset for asset in list_gallery_assets(str(tmp_path)) if asset.name == "result.png")

        result = gallery_asset_to_json(asset, _web_config())

        assert result["reuse_state"]["resolved_workflow"] == "txt2img"
        assert "edit" not in json.dumps(parse_qs(urlsplit(result["reuse_workspace_url"].removeprefix("#")).query))

    def test_an_image_that_is_not_an_edit_has_no_edit_block_and_reuses_as_before(self, tmp_path):
        _write(tmp_path / "result.png", {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "prompt": "a castle", "model": "zit", "seed": 1, "steps": 8, "ratio": "2:3", "size": "m"})
        asset = next(asset for asset in list_gallery_assets(str(tmp_path)) if asset.name == "result.png")

        result = gallery_asset_to_json(asset, _web_config())
        params = {key: values[0] for key, values in parse_qs(urlsplit(result["reuse_workspace_url"].removeprefix("#")).query, keep_blank_values=True).items()}

        assert params["workflow"] == "txt2img"
        assert {"edit_source_path", "edit_repaint", "missing_edit_source"}.isdisjoint(params)
        assert params["ratio"] == "2:3"
