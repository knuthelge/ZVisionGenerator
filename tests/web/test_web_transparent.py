"""The transparent background in the Web UI: the form field, the workflow contract and the preset key."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from starlette.datastructures import FormData

from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import embed_png_config
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner
from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS


def _submit(monkeypatch, tmp_path, entries: list[tuple[str, str]]):
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="zimage", is_distilled=True, size="6b"),
        defaults=make_resolved_image_defaults(),
        submitted=submitted,
    )
    response = web_server._submit_image_job(FormData([("model", "zit"), ("prompt", "a fox"), *entries]), web_config)
    return submitted[0], response


@pytest.mark.parametrize(("entries", "expected"), [([("transparent", "true")], True), ([("transparent", "false")], False), ([], False)])
def test_transparent_form_field_reaches_the_request(monkeypatch, tmp_path, entries, expected):
    job, _response = _submit(monkeypatch, tmp_path, entries)

    assert job["request"].transparent is expected
    assert job["args"].transparent is expected


@pytest.mark.parametrize(("entries", "recorded"), [([("transparent", "true")], "true"), ([("transparent", "false")], "false"), ([], None)])
def test_the_submitted_transparent_field_is_recorded_for_load_settings(monkeypatch, tmp_path, entries, recorded):
    _job, response = _submit(monkeypatch, tmp_path, entries)

    assert response["context"]["settings"].get("transparent") == recorded


@pytest.mark.parametrize("workflow", ["txt2img", "img2img"])
def test_image_workflows_offer_the_transparent_control(workflow):
    assert "postprocess_transparent" in WORKFLOW_DEFINITIONS[workflow]["visible_controls"]


@pytest.mark.parametrize("workflow", ["txt2vid", "img2vid", "extend"])
def test_video_workflows_clear_the_transparent_field_and_do_not_offer_the_control(workflow):
    assert "transparent" in WORKFLOW_DEFINITIONS[workflow]["clear_fields"]
    assert "postprocess_transparent" not in WORKFLOW_DEFINITIONS[workflow]["visible_controls"]


def test_a_preset_stores_the_transparent_key(client):
    body = {"name": "Cut-out", "mode": "image", "model": "my-image-model", "settings": {"transparent": True}}
    response = client.post("/api/presets", json=body)

    assert response.status_code == 200
    assert response.json()["settings"]["transparent"] is True


@pytest.mark.parametrize("value", ["true", 1])
def test_a_preset_transparent_key_must_be_a_boolean(client, value):
    body = {"name": "Cut-out", "mode": "image", "model": "my-image-model", "settings": {"transparent": value}}

    assert client.post("/api/presets", json=body).status_code == 422


@pytest.mark.parametrize(("generation", "expected"), [({"transparent": True}, True), ({}, False)])
def test_a_job_output_entry_carries_transparent(monkeypatch, tmp_path, generation, expected):
    info = PngInfo()
    config = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2img", "model": "zit", "seed": 1, "prompt": "a fox", "generation": generation}
    embed_png_config(info, config)
    output = tmp_path / "fox.png"
    Image.new("RGBA", (16, 16)).save(output, pnginfo=info)
    monkeypatch.setattr(web_runner, "load_web_config", lambda: make_web_config())

    payload = web_runner._output_asset_payload({"output_dir": str(tmp_path)}, str(output))

    assert payload is not None
    assert payload["transparent"] is expected


@pytest.fixture
def client():
    with TestClient(web_server.app) as test_client:
        yield test_client
