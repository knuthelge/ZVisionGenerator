"""The gallery's Reuse of a saved edit: it returns to Edit with the picture it started from, and an upscale of it does not."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tests.web._helpers import ffprobe_cannot_read_files
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.utils.provenance import build_image_config_payload, embed_png_config, read_png_config, recorded_settings
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.gallery import gallery_asset_to_json, list_gallery_assets


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch):
    ffprobe_cannot_read_files(monkeypatch)
    monkeypatch.setattr("zvisiongenerator.web.gallery._asset_upscale_json", lambda *_args: None)


def _web_config() -> SimpleNamespace:
    return SimpleNamespace(image_model_options=("zit",), video_model_options=("ltx-8",), default_models=WebUiDefaultModels(image="zit", video="ltx-8"), seedvr2=None)


def _edit_payload(source: Path, *, mask: bool = False, extend: tuple[str, int] | None = None, size: tuple[int, int] = (640, 480)) -> dict:
    """Return the config an edit of *source* embeds, built by the real provenance writer."""
    request = ImageGenerationRequest(
        backend=MagicMock(),
        model=object(),
        prompt="a red door",
        model_name="zit",
        edit_source=str(source),
        edit_paint=np.ones((4, 4), dtype=bool) if mask else None,
        edit_extend=extend,
        seed=7,
        steps=9,
        guidance=3.5,
        scheduler="beta",
        sharpen=False,
        width=832,
        height=1216,
    )
    artifacts = ImageWorkingArtifacts(image=Image.new("RGB", size))
    artifacts.metadata["edit_source"] = {"path": str(source), "width": 512, "height": 480}
    return build_image_config_payload(request, artifacts)


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
        payload = _edit_payload(tmp_path / "door.png")

        assert "workflow" not in payload["source"]
        assert payload["workflow"] == "edit"

    def test_the_edit_block_is_read_back_and_a_malformed_one_is_dropped(self, tmp_path):
        payload = _edit_payload(tmp_path / "door.png", mask=True, extend=("top", 256))
        _write(tmp_path / "result.png", payload)

        recorded = recorded_settings(read_png_config(str(tmp_path / "result.png")))

        assert recorded.edit == {"mode": "both", "extend": {"side": "top", "amount": 256}}
        assert recorded_settings({"edit": {"mode": "sideways"}}).edit is None
        assert recorded_settings({"edit": "paint"}).edit is None
        assert recorded_settings({"workflow": "txt2img"}).edit is None
        assert recorded_settings({"edit": {"mode": "extend", "extend": {"side": "top", "amount": 7}}}).edit == {"mode": "extend"}


class TestReuse:
    def test_a_paint_edit_returns_to_edit_with_its_picture_and_asks_for_paint(self, tmp_path):
        payload = _edit_payload(tmp_path / "door.png", mask=True)

        result, params = _reuse(tmp_path, payload)

        assert params["workflow"] == "edit"
        assert params["edit_source_path"] == str(tmp_path / "door.png")
        assert params["edit_repaint"] == "true"
        assert {"edit_extend_side", "edit_extend_amount"}.isdisjoint(params)
        assert result["workflow"] == "edit"
        assert result["reuse_state"]["resolved_workflow"] == "edit"
        assert result["reuse_state"]["workflow_available"] is True
        assert result["reuse_state"]["fallback_reasons"] == []

    def test_the_override_does_not_rewrite_an_edit_to_text_to_image(self, tmp_path):
        result, params = _reuse(tmp_path, _edit_payload(tmp_path / "door.png", mask=True))

        assert (result["reuse_state"]["requested_workflow"], params["workflow"]) == ("edit", "edit")
        assert result["details"]["recorded_workflow"] == "edit"

    @pytest.mark.parametrize(
        ("mask", "extend", "repaint", "extended"),
        [(True, None, True, False), (False, ("left", 128), False, True), (True, ("bottom", 512), True, True), (False, None, False, False)],
        ids=["paint", "extend", "both", "whole"],
    )
    def test_paint_is_asked_for_again_only_when_the_edit_painted(self, tmp_path, mask, extend, repaint, extended):
        _result, params = _reuse(tmp_path, _edit_payload(tmp_path / "door.png", mask=mask, extend=extend))

        assert ("edit_repaint" in params) is repaint
        assert ("edit_extend_side" in params) is extended
        if extend:
            assert (params["edit_extend_side"], params["edit_extend_amount"]) == (extend[0], str(extend[1]))

    def test_the_text_model_and_run_settings_carry_over_and_nothing_an_edit_does_not_use(self, tmp_path):
        _result, params = _reuse(tmp_path, _edit_payload(tmp_path / "door.png", mask=True))

        assert (params["prompt"], params["model"], params["seed"], params["steps"], params["lora"]) == ("a red door", "zit", "7", "9", "")
        assert {"ratio", "size", "width", "height", "guidance", "scheduler", "negative_prompt", "transparent", "reference_path", "sharpen_enabled", "upscale"}.isdisjoint(params)

    def test_a_source_that_moved_stays_edit_names_a_blank_source_and_says_so(self, tmp_path):
        result, params = _reuse(tmp_path, _edit_payload(tmp_path / "gone.png", mask=True), source_exists=False)

        assert params["workflow"] == "edit"
        assert params["edit_source_path"] == ""
        assert "missing_edit_source" not in params
        assert params["edit_repaint"] == "true"
        assert result["reuse_state"]["workflow_available"] is False
        assert result["reuse_state"]["fallback_reasons"] == ["missing_edit_source"]

    def test_an_edit_of_a_picture_in_the_output_folder_reuses_that_picture(self, tmp_path):
        (tmp_path / "shots").mkdir()
        payload = _edit_payload(tmp_path / "shots" / "door.png")

        _result, params = _reuse(tmp_path, payload)

        assert params["edit_source_path"] == str(tmp_path / "shots" / "door.png")

    def test_the_viewer_gets_the_edit_block_and_the_source_picture(self, tmp_path):
        result, _params = _reuse(tmp_path, _edit_payload(tmp_path / "door.png", mask=True, extend=("top", 256)))

        assert result["details"]["edit"] == {"mode": "both", "extend": {"side": "top", "amount": 256}}
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

        assert result["details"]["edit"] is None
        assert params["workflow"] == "txt2img"
        assert {"edit_source_path", "edit_repaint", "missing_edit_source"}.isdisjoint(params)
        assert params["ratio"] == "2:3"
