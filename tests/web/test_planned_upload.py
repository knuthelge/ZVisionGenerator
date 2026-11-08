"""The stand-ins a download plan uses for a form's uploads."""

from __future__ import annotations

import io

from starlette.datastructures import FormData, UploadFile

from zvisiongenerator.web.planned_upload import PlannedUpload, planned_form, planned_upload_path


def _upload(name: str | None, body: bytes = b"data") -> UploadFile:
    return UploadFile(file=io.BytesIO(body), filename=name)


def test_each_named_upload_becomes_a_stand_in_and_text_is_kept():
    form = FormData([("prompt", "a lake"), ("image_file", _upload("a.png")), ("reference", _upload("r1.png")), ("reference", "/typed/path.png"), ("reference", _upload("r2.png"))])

    planned = planned_form(form)

    assert planned.get("prompt") == "a lake"
    assert planned.get("image_file") == PlannedUpload("a.png")
    assert planned.getlist("reference") == [PlannedUpload("r1.png"), "/typed/path.png", PlannedUpload("r2.png")]


def test_a_file_part_without_a_name_is_left_alone():
    empty = _upload("", b"")

    planned = planned_form(FormData([("image_file", empty)]))

    assert planned.get("image_file") is empty
    assert not isinstance(planned.get("image_file"), PlannedUpload)


def test_a_plain_mapping_is_accepted():
    planned = planned_form({"prompt": "x", "voice_file": _upload("v.wav")})

    assert planned.get("voice_file") == PlannedUpload("v.wav")


def test_the_stand_in_path_is_under_the_output_folder_and_keeps_only_the_file_name(tmp_path):
    path = planned_upload_path(PlannedUpload("../../etc/voice.wav"), str(tmp_path))

    assert path == str(tmp_path / ".web_uploads" / "planned" / "voice.wav")
    assert not (tmp_path / ".web_uploads").exists()


def test_a_plan_with_a_project_puts_the_stand_in_under_the_root(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies
    from zvisiongenerator.utils.image_model_detect import ImageModelInfo
    from zvisiongenerator.web import server as web_server

    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="zimage", is_distilled=True, size=None), defaults=make_resolved_image_defaults(supports_img2img=True))
    planned = []
    monkeypatch.setattr(web_server, "job_download_parts", lambda request, config, **kwargs: planned.append(kwargs["args"]) or ())
    monkeypatch.setattr(web_server, "_summarize_job_parts", lambda parts: {})

    with TestClient(web_server.app) as client:
        response = client.post("/api/generate/plan", data={"model": "zimage", "prompt": "x", "project": "A"}, files={"image_file": ("ref.png", b"data", "image/png")})

    assert response.status_code == 200, response.text
    (args,) = planned
    assert args.image_path == str(tmp_path / ".web_uploads" / "planned" / "ref.png")
    assert args.output == str(tmp_path / "A")
    assert not (tmp_path / ".web_uploads").exists()
