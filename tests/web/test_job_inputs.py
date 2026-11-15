"""The inputs a Web UI job shows: the records its context keeps and the names its snapshots carry."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import threading
from typing import Any

from PIL import Image
import pytest
from starlette.datastructures import FormData, UploadFile

from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies, stub_video_submission, wait_for_status, write_png
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.web import job_inputs
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_contract import public_job_snapshot
from zvisiongenerator.web.upload_folders import UPLOADS_FOLDER

_KLEIN = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")


@pytest.fixture(autouse=True)
def _no_real_memory_release(monkeypatch):
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)


def _image(path: Path, size: tuple[int, int] = (64, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "teal").save(path)
    return path


def _upload(name: str) -> UploadFile:
    out = io.BytesIO()
    Image.new("RGB", (8, 8), "teal").save(out, format="PNG")
    out.seek(0)
    return UploadFile(file=out, filename=name)


# ── The records ──────────────────────────────────────────────────────────────


def test_a_record_is_named_by_its_file_unless_a_name_is_given(tmp_path):
    assert job_inputs.job_input("source", tmp_path / "door.png", "image")["name"] == "door.png"
    assert job_inputs.job_input("start", tmp_path / "3f9a.png", "image", name="holiday.png")["name"] == "holiday.png"


def test_a_given_name_is_cut_to_its_last_part_and_a_blank_one_falls_back_to_the_file(tmp_path):
    assert job_inputs.job_input("start", tmp_path / "a.png", "image", name="../../etc/holiday.png")["name"] == "holiday.png"
    assert job_inputs.job_input("start", tmp_path / "a.png", "image", name="")["name"] == "a.png"


def test_a_record_holds_a_role_a_name_and_a_kind_and_never_a_path(tmp_path):
    record = job_inputs.job_input("source", tmp_path / "private" / "door.png", "image")

    assert record == {"role": "source", "name": "door.png", "kind": "image"}


HASH = hashlib.sha256(b"a picture").hexdigest()


def test_an_image_stored_under_its_hash_is_shown_as_an_uploaded_image(tmp_path):
    stored = tmp_path / UPLOADS_FOLDER / f"{HASH}.png"

    assert job_inputs.job_input("source", stored, "image")["name"] == "Uploaded image"
    # A name the sender gave wins over the stored one.
    assert job_inputs.job_input("source", stored, "image", name="holiday.png")["name"] == "holiday.png"


@pytest.mark.parametrize(
    "path",
    [
        f"{UPLOADS_FOLDER}/{HASH}/door.png",
        f"{UPLOADS_FOLDER}/not-a-hash.png",
        f"elsewhere/{HASH}.png",
        f"{UPLOADS_FOLDER}/{HASH[:-1]}.png",
        f"{UPLOADS_FOLDER}/{HASH.upper()}.png",
    ],
)
def test_only_a_hash_named_file_directly_in_the_uploads_folder_is_hidden(tmp_path, path):
    assert job_inputs.job_input("source", tmp_path / path, "image")["name"] == Path(path).name


# ── Snapshots ────────────────────────────────────────────────────────────────


def _snapshot(context: dict[str, Any]) -> dict[str, Any]:
    return public_job_snapshot(
        job_id="job-1",
        status="queued",
        workflow="img2img",
        supported_controls=(),
        context=context,
        created_at=0.0,
        completed_at=None,
        event_count=0,
        last_event=None,
        paused=False,
        result_path=None,
    )


def test_a_snapshot_carries_the_inputs_of_its_context_by_name(tmp_path):
    snapshot = _snapshot({"inputs": [job_inputs.job_input("start", tmp_path / "start.png", "image")]})

    assert snapshot["inputs"] == [{"role": "start", "name": "start.png", "kind": "image"}]


def test_a_snapshot_of_a_job_without_inputs_has_an_empty_list():
    assert _snapshot({})["inputs"] == []


def _runner(**kwargs) -> web_runner_module.WebRunner:
    return web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, **kwargs)


def _hold_the_worker(runner: web_runner_module.WebRunner):
    """Occupy the worker with a job that is not a generation job, so the generation jobs submitted next wait."""
    started, release = threading.Event(), threading.Event()
    runner._submit_job(job_type="hold", exclusive=False, target_factory=lambda _cb: (started.set(), release.wait(timeout=5.0)))
    assert started.wait(timeout=2.0)
    return release


def test_the_active_job_and_the_queued_ones_each_list_their_own_inputs(tmp_path):
    runner = _runner()
    release = _hold_the_worker(runner)
    try:
        active = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None, context={"inputs": [job_inputs.job_input("source", tmp_path / "a.png", "image")]})
        queued = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None, context={"inputs": [job_inputs.job_input("source", tmp_path / "b.png", "image")]})

        jobs = runner.list_jobs()

        assert jobs["active_job"]["job_id"] == active
        assert [entry["name"] for entry in jobs["active_job"]["inputs"]] == ["a.png"]
        assert [(job["job_id"], [entry["name"] for entry in job["inputs"]]) for job in jobs["queued_jobs"]] == [(queued, ["b.png"])]
    finally:
        release.set()
        runner.shutdown()


def test_a_pruned_job_still_names_its_inputs(tmp_path):
    runner = _runner(terminal_retention_seconds=0.0)
    try:
        job_id = runner._submit_job(job_type="test", target_factory=lambda _cb: None, context={"inputs": [job_inputs.job_input("source", tmp_path / "a.png", "image")]})
        assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
        with runner._jobs_lock:
            runner._prune_terminal_jobs_locked()
            assert job_id not in runner._jobs

        assert runner.get_job_snapshot(job_id)["inputs"] == [{"role": "source", "name": "a.png", "kind": "image"}]
    finally:
        runner.shutdown()


# ── What each kind of job records ────────────────────────────────────────────


def _records(context: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(entry["role"], entry["name"], entry["kind"]) for entry in context["inputs"]]


def _submit_image(monkeypatch, tmp_path, entries: list[tuple[str, object]]) -> dict[str, Any]:
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    submitted: list[dict[str, Any]] = []
    patch_image_submit_dependencies(monkeypatch, model_info=_KLEIN, defaults=make_resolved_image_defaults(max_reference_images=4, supports_mask_edit=True), submitted=submitted)
    web_server._submit_image_job(FormData([("model", "zit"), ("prompt", "a red door"), *entries]), web_config)
    return submitted[0]["context"]


def test_a_text_to_image_job_has_no_inputs(monkeypatch, tmp_path):
    assert _submit_image(monkeypatch, tmp_path, [])["inputs"] == []


def test_an_image_job_lists_its_source_image_then_its_references_in_order_under_the_names_they_were_sent_as(monkeypatch, tmp_path):
    typed = tmp_path / "typed.png"
    write_png(typed)

    context = _submit_image(
        monkeypatch, tmp_path, [("image_file", _upload("start.png")), ("reference", _upload("first.png")), ("reference", str(typed)), ("reference", "  "), ("reference", _upload("last.png"))]
    )

    assert _records(context) == [
        ("source", "start.png", "image"),
        ("reference", "first.png", "image"),
        ("reference", "typed.png", "image"),
        ("reference", "last.png", "image"),
    ]


def test_an_image_job_started_from_a_typed_path_names_the_file(monkeypatch, tmp_path):
    start = tmp_path / "pond.png"
    write_png(start)

    context = _submit_image(monkeypatch, tmp_path, [("image_path", str(start))])

    assert _records(context) == [("source", "pond.png", "image")]


def test_an_edit_job_lists_its_source_picture(monkeypatch, tmp_path):
    root = tmp_path / "out"
    source = _image(root / "door.png")

    context = _submit_image(monkeypatch, tmp_path, [("workflow", "edit"), ("edit_source_path", str(source))])

    assert _records(context) == [("source", "door.png", "image")]


def test_an_edit_job_from_an_upload_keeps_the_name_it_was_sent_as(monkeypatch, tmp_path):
    context = _submit_image(monkeypatch, tmp_path, [("workflow", "edit"), ("edit_source_file", _upload("my door.png"))])

    assert _records(context) == [("source", "my door.png", "image")]


def test_an_edit_job_resubmitted_with_a_stored_upload_path_calls_it_an_uploaded_image(monkeypatch, tmp_path):
    stored = _image(tmp_path / "out" / UPLOADS_FOLDER / f"{HASH}.png")

    context = _submit_image(monkeypatch, tmp_path, [("image_path", str(stored))])

    assert _records(context) == [("source", "Uploaded image", "image")]


def test_an_image_to_video_job_lists_its_start_image_and_its_end_image(monkeypatch, tmp_path):
    submitted = stub_video_submission(monkeypatch, tmp_path)
    end = _image(tmp_path / "end.png")
    web_config = web_server.load_web_config()

    web_server._submit_video_job(FormData([("model", "ltx-8"), ("prompt", "a lake"), ("image_file", _upload("first.png")), ("end_image_path", str(end))]), web_config)

    assert _records(submitted[0]["context"]) == [("start", "first.png", "image"), ("end", "end.png", "image")]


def test_a_text_to_video_job_has_no_inputs(monkeypatch, tmp_path):
    submitted = stub_video_submission(monkeypatch, tmp_path)

    web_server._submit_video_job(FormData([("model", "ltx-8"), ("prompt", "a lake")]), web_server.load_web_config())

    assert submitted[0]["context"]["inputs"] == []
