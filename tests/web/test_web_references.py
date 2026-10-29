"""Reference images in the Web UI: the repeated ``reference`` form field, its refusals and the model capability."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from starlette.datastructures import FormData, UploadFile

from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies, write_png
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.workspace_contract import CANONICAL_WORKFLOW_VALUES, WORKFLOW_DEFINITIONS


def _upload(tmp_path: Path, name: str) -> UploadFile:
    source = tmp_path / f"src-{name}"
    write_png(source)
    return UploadFile(file=io.BytesIO(source.read_bytes()), filename=name)


def _submit(monkeypatch, tmp_path, entries: list[tuple[str, object]], *, cap: int = 4):
    """Submit an image job on a model that takes *cap* references and return the queued job and the response."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    submitted: list[dict[str, object]] = []
    patch_image_submit_dependencies(
        monkeypatch,
        model_info=ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b"),
        defaults=make_resolved_image_defaults(max_reference_images=cap),
        submitted=submitted,
    )
    response = web_server._submit_image_job(FormData([("model", "zit"), ("prompt", "the woman from image 1"), *entries]), web_config)
    return submitted[0], response


def test_references_keep_the_order_uploads_and_paths_were_submitted_in(monkeypatch, tmp_path):
    first, third = tmp_path / "first.png", tmp_path / "third.png"
    write_png(first)
    write_png(third)

    job, response = _submit(monkeypatch, tmp_path, [("reference", str(first)), ("reference", _upload(tmp_path, "second.png")), ("reference", str(third))])

    paths = job["args"].reference_paths
    assert [paths[0], paths[2]] == [str(first.resolve()), str(third.resolve())]
    assert Path(paths[1]).parent == tmp_path / "out" / ".web_uploads"
    assert job["request"].reference_paths == tuple(paths)
    assert response["context"]["settings"]["reference_path"] == paths
    assert "reference" not in response["context"]["settings"]


def test_references_leave_the_job_type_to_the_start_image(monkeypatch, tmp_path):
    ref = tmp_path / "ref.png"
    write_png(ref)

    _job, response = _submit(monkeypatch, tmp_path, [("reference", str(ref))])
    assert response["context"]["job_type"] == "Text to Image"

    _job, response = _submit(monkeypatch, tmp_path, [("reference", str(ref)), ("image_file", _upload(tmp_path, "start.png"))])
    assert response["context"]["job_type"] == "Image to Image"


def test_blank_reference_values_are_skipped(monkeypatch, tmp_path):
    job, response = _submit(monkeypatch, tmp_path, [("reference", "  ")])

    assert job["args"].reference_paths == []
    assert "reference_path" not in response["context"]["settings"]


def test_more_references_than_the_model_takes_are_refused(monkeypatch, tmp_path):
    cap, count = 4, 5
    refs = []
    for index in range(count):
        path = tmp_path / f"r{index}.png"
        write_png(path)
        refs.append(("reference", str(path)))

    with pytest.raises(ValueError):
        _submit(monkeypatch, tmp_path, refs, cap=cap)
    # The same list is accepted by a model that takes that many.
    job, _response = _submit(monkeypatch, tmp_path, refs, cap=count)
    assert len(job["args"].reference_paths) == count


def test_a_reference_path_that_is_missing_or_not_an_image_is_refused(monkeypatch, tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("x")

    for bad in (str(tmp_path / "nope.png"), str(text), "https://example.test/a.png"):
        with pytest.raises(ValueError):
            _submit(monkeypatch, tmp_path, [("reference", bad)])


def test_an_uploaded_reference_that_is_not_a_readable_image_is_refused(monkeypatch, tmp_path):
    broken = UploadFile(file=io.BytesIO(b"not an image"), filename="broken.png")

    with pytest.raises(ValueError):
        _submit(monkeypatch, tmp_path, [("reference", broken)])

    assert _uploaded_files(tmp_path) == []


def _uploaded_files(tmp_path: Path) -> list[Path]:
    folder = tmp_path / "out" / ".web_uploads"
    return sorted(folder.iterdir()) if folder.exists() else []


def test_the_same_image_uploaded_twice_is_saved_once(monkeypatch, tmp_path):
    """An upload is named by its content, so a repeat reuses the file whatever the upload was called or which job sent it."""
    first_job, _ = _submit(monkeypatch, tmp_path, [("reference", _upload(tmp_path, "a.png")), ("image_file", _upload(tmp_path, "start.png"))])
    saved = _uploaded_files(tmp_path)
    mtimes = [path.stat().st_mtime_ns for path in saved]
    second_job, _ = _submit(monkeypatch, tmp_path, [("reference", _upload(tmp_path, "b.png")), ("reference", _upload(tmp_path, "c.png"))])

    assert len(saved) == 1
    assert _uploaded_files(tmp_path) == saved
    assert [path.stat().st_mtime_ns for path in saved] == mtimes
    assert second_job["args"].reference_paths == [str(saved[0]), str(saved[0])]
    assert first_job["args"].reference_paths == [str(saved[0])]
    assert first_job["args"].image_path == str(saved[0])


def test_images_with_different_content_are_saved_apart(monkeypatch, tmp_path):
    other = tmp_path / "src-other.png"
    Image.new("RGB", (8, 8), "red").save(other)

    _submit(monkeypatch, tmp_path, [("reference", _upload(tmp_path, "a.png")), ("reference", UploadFile(file=io.BytesIO(other.read_bytes()), filename="a.png"))])

    assert len(_uploaded_files(tmp_path)) == 2


def test_the_image_model_defaults_carry_the_reference_cap(monkeypatch):
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b"))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_a: make_resolved_image_defaults(max_reference_images=4))

    assert workspace_api_module._build_image_bootstrap_defaults("zit", web_config)["max_reference_images"] == 4


def test_a_model_that_cannot_be_detected_offers_no_references(monkeypatch):
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("gone")))

    assert workspace_api_module._build_image_bootstrap_defaults("zit", make_web_config())["max_reference_images"] == 0


def test_only_the_image_workflows_offer_references_and_no_workflow_clears_them():
    offering = [name for name in CANONICAL_WORKFLOW_VALUES if "references" in WORKFLOW_DEFINITIONS[name]["visible_controls"]]
    clearing = [name for name in CANONICAL_WORKFLOW_VALUES if "reference_path" in WORKFLOW_DEFINITIONS[name]["clear_fields"]]

    assert offering == ["txt2img", "img2img"]
    assert clearing == []


def test_check_reports_the_reference_paths_that_are_gone(tmp_path):
    kept, gone = tmp_path / "kept.png", tmp_path / "gone.png"
    write_png(kept)

    with TestClient(web_server.app) as client:
        response = client.post("/api/references/check", json={"paths": [str(kept), str(gone), str(gone)]})

    assert response.status_code == 200
    assert response.json() == {"missing": [str(gone)]}


@pytest.mark.parametrize("body", [{"paths": "a.png"}, {"paths": [1]}, []])
def test_check_refuses_a_body_without_a_list_of_paths(body):
    with TestClient(web_server.app) as client:
        assert client.post("/api/references/check", json=body).status_code == 422
