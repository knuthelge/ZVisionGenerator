"""The Web UI's Edit job: its form, its refusals and the download plan."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import FormData, UploadFile

from tests._factories import _make_plan
from tests._images import SIDEWAYS_UPRIGHT_SIZE, mask_png, painted_box, write_cutout, write_sideways_photo
from tests.web._helpers import make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies
from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.edit_request import check_edit_inputs
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import read_png_config
from zvisiongenerator.web import edit_job
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.workflows.generation_prompt import EXTEND_INSTRUCTION

_SOURCE_SIZE = (64, 48)
_KLEIN = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")


@pytest.fixture(autouse=True)
def _no_real_memory_release(monkeypatch):
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)


def _png_bytes(size: tuple[int, int], paint: tuple[int, int, int, int] | None = (4, 4, 20, 20), *, mode: str = "L") -> bytes:
    """Return a mask PNG of *size* that is black with a white box at *paint* (left, top, right, bottom); ``None`` paints nothing."""
    return mask_png(painted_box(size[1], size[0], *([paint] if paint else [])), mode=mode)


def _upload(data: bytes, name: str = "mask.png") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


class _Env(SimpleNamespace):
    """What an Edit submit test needs: the output root, a source picture in it and the jobs handed to the runner."""

    def submit(self, *entries: tuple[str, object], model: str = "zit") -> dict:
        """Submit an Edit form on the source picture and return the job the runner was handed."""
        fields = [("workflow", "edit"), ("model", model), ("prompt", "a red door"), ("edit_source_path", str(self.source)), *entries]
        response = web_server._submit_image_job(FormData(fields), self.web_config)
        self.response = response
        return self.submitted[-1]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Route a submit to a recording runner on a model that edits, with a source picture inside the output root."""
    root = tmp_path / "out"
    root.mkdir()
    source = root / "door.png"
    Image.new("RGB", _SOURCE_SIZE, "teal").save(source)
    web_config = make_web_config()
    web_config.output_dir = str(root)
    submitted: list[dict] = []
    patch_image_submit_dependencies(monkeypatch, model_info=_KLEIN, defaults=make_resolved_image_defaults(max_reference_images=4, supports_mask_edit=True), submitted=submitted)
    return _Env(root=root, source=source, web_config=web_config, submitted=submitted, response=None, tmp_path=tmp_path)


def _use_model(monkeypatch, info: ImageModelInfo, **defaults) -> None:
    from zvisiongenerator.web import server

    monkeypatch.setattr(server, "detect_image_model", lambda _model: info)
    monkeypatch.setattr(server, "resolve_defaults", lambda *_args: make_resolved_image_defaults(**defaults))


# -- R8.1: the job -----------------------------------------------------------------------------------------------


class TestTheJob:
    def test_a_path_source_and_a_mask_queue_a_job_with_both(self, env):
        job = env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        args = job["args"]
        assert args.edit_source == str(env.source.resolve())
        assert args.edit_paint.dtype == bool and args.edit_paint.shape == (_SOURCE_SIZE[1], _SOURCE_SIZE[0])
        assert args.edit_paint[4:20, 4:20].all() and args.edit_paint.sum() == 16 * 16
        assert args.edit_extend is None
        assert env.response["context"]["workflow"] == "edit"
        assert env.response["context"]["job_type"] == "Edit"

    def test_the_painted_area_is_not_among_the_settings_the_job_shows(self, env):
        env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))), ("edit_painted", "true"))

        assert "edit_mask_file" not in env.response["context"]["settings"]
        assert env.response["context"]["settings"]["edit_painted"] == "true"
        assert env.response["context"]["settings"]["edit_source_path"] == str(env.source)

    def test_an_edit_runs_no_post_processing_and_takes_no_other_inputs(self, env):
        job = env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        args, request = job["args"], job["request"]
        assert args.sharpen is False and args.contrast is False and args.saturation is False
        assert args.upscale is None and args.image_path is None
        assert (request.sharpen, request.contrast, request.saturation, request.transparent, request.upscale_factor) == (False, False, False, False, None)
        assert request.reference_paths == () and request.image_path is None

    def test_enhancement_is_off_so_the_encoded_prompt_is_the_sent_prompt(self, env):
        job = env.submit(("enhance_auto", "true"), ("enhance_settings", "{}"), ("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        args = job["args"]
        assert args.no_enhance is True
        assert job["enhance_by_set"] is None
        assert job["prompts_data"] == {"web": [("a red door", None)]}

    def test_a_job_without_paint_or_extend_runs_whole_picture_on_a_model_with_references(self, env):
        job = env.submit()

        assert job["args"].edit_paint is None and job["args"].edit_extend is None

    def test_extend_is_carried_as_a_side_and_an_amount(self, env):
        job = env.submit(("edit_extend_side", "top"), ("edit_extend_amount", "256"))

        assert job["args"].edit_extend == ("top", 256)

    def test_the_output_lands_beside_a_source_in_the_output_folder(self, env):
        (env.root / "keepers").mkdir()
        nested = env.root / "keepers" / "door.png"
        Image.new("RGB", _SOURCE_SIZE).save(nested)

        job = env.submit(("edit_source_path", str(nested)))

        assert job["args"].output == str(nested.parent.resolve()) == job["request"].output_dir
        assert env.response["context"]["save_project"] == "keepers"

    def test_a_source_outside_the_output_folder_saves_into_the_picked_project(self, env):
        outside = env.tmp_path / "elsewhere.png"
        Image.new("RGB", _SOURCE_SIZE).save(outside)

        job = env.submit(("edit_source_path", str(outside)), ("project", "shots"))

        assert job["args"].output == str(env.root / "shots")

    def test_an_uploaded_source_is_staged_and_recorded_like_a_start_image(self, env):
        source_bytes = io.BytesIO()
        Image.new("RGB", _SOURCE_SIZE, "red").save(source_bytes, format="PNG")
        fields = [("workflow", "edit"), ("model", "zit"), ("prompt", "x"), ("edit_source_file", _upload(source_bytes.getvalue(), "pic.png"))]

        web_server._submit_image_job(FormData(fields), env.web_config)

        job = env.submitted[-1]
        staged = Path(job["args"].edit_source)
        assert staged.parent.parent == env.root / ".web_uploads"
        assert staged.name == "pic.png"
        assert job["args"].output == str(env.root)
        assert env.submitted[-1]["context"]["settings"]["edit_source_path"] == job["args"].edit_source

    @pytest.mark.parametrize(
        ("entries", "shown"),
        [
            ((("edit_mask_file", "mask"),), _SOURCE_SIZE),  # paint: the picture's own size
            ((("edit_extend_side", "top"), ("edit_extend_amount", "128")), (_SOURCE_SIZE[0], _SOURCE_SIZE[1] + 128)),  # extend: the amount added on its side
            ((), _SOURCE_SIZE),  # the whole picture: the picture's own size
        ],
        ids=["paint", "extend", "whole picture"],
    )
    def test_the_job_shows_the_size_the_picture_will_have(self, env, entries, shown):
        env.submit(*((key, _upload(_png_bytes(_SOURCE_SIZE))) if value == "mask" else (key, value) for key, value in entries))

        context = env.response["context"]
        assert (context["width"], context["height"]) == shown
        assert context["meta"].startswith(f"{shown[0]}x{shown[1]} ")

    def test_a_mask_at_another_size_of_the_same_shape_is_scaled_to_the_source(self, env):
        paint = env.submit(("edit_mask_file", _upload(_png_bytes((128, 96), (8, 8, 40, 40)))))["args"].edit_paint

        assert paint.shape == (_SOURCE_SIZE[1], _SOURCE_SIZE[0])
        assert set(np.unique(paint)) == {True, False}
        assert paint[4:20, 4:20].all() and paint.sum() == 16 * 16

    def test_a_browser_mask_with_a_transparent_background_keeps_only_the_strokes(self, env):
        image = Image.new("RGBA", _SOURCE_SIZE, (0, 0, 0, 0))
        image.paste((255, 255, 255, 255), (4, 4, 20, 20))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        paint = env.submit(("edit_mask_file", _upload(buffer.getvalue())))["args"].edit_paint

        assert paint[4:20, 4:20].all() and paint.sum() == 16 * 16

    def test_a_turned_source_is_read_upright_and_its_mask_follows_the_upright_shape(self, env):
        turned = write_sideways_photo(env.root / "turned.jpg")
        width, height = SIDEWAYS_UPRIGHT_SIZE

        paint = env.submit(("edit_source_path", turned), ("edit_mask_file", _upload(_png_bytes((width * 2, height * 2)))))["args"].edit_paint

        assert paint.shape == (height, width)
        assert (env.response["context"]["width"], env.response["context"]["height"]) == (width, height)

    def test_a_mask_is_never_staged_as_an_upload(self, env):
        env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        assert not (env.root / ".web_uploads").exists()


# -- R8.4: the refusals ------------------------------------------------------------------------------------------


def _shared_refusal(**inputs) -> str:
    with pytest.raises(ValueError) as refusal:
        check_edit_inputs(**inputs)
    return str(refusal.value)


class TestRefusals:
    def test_a_cut_out_cannot_be_extended_in_the_words_the_cli_uses(self, env):
        env.submit(("edit_source_path", write_cutout(env.root / "cut.png", _SOURCE_SIZE)), ("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))  # paint alone is fine

        with pytest.raises(ValueError) as refusal:
            env.submit(("edit_source_path", str(env.root / "cut.png")), ("edit_extend_side", "left"), ("edit_extend_amount", "128"))

        assert str(refusal.value) == _shared_refusal(has_mask=False, extend=("left", 128), cutout=True, supports_mask_edit=True)

    def test_whole_picture_on_a_cut_out_asks_for_paint(self, env):
        with pytest.raises(ValueError) as refusal:
            env.submit(("edit_source_path", write_cutout(env.root / "cut.png", _SOURCE_SIZE)))

        assert str(refusal.value) == _shared_refusal(has_mask=False, extend=None, cutout=True, supports_mask_edit=True)

    def test_a_model_that_cannot_edit_is_refused(self, env, monkeypatch):
        _use_model(monkeypatch, ImageModelInfo(family="flux1", is_distilled=False, size=None), supports_mask_edit=False, max_reference_images=0)

        with pytest.raises(ValueError) as refusal:
            env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        assert str(refusal.value) == _shared_refusal(has_mask=True, extend=None, cutout=False, supports_mask_edit=False)

    def test_an_empty_mask_is_refused(self, env):
        with pytest.raises(ValueError, match="no painted area"):
            env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE, None))))

        assert env.submitted == []

    def test_a_mask_of_another_shape_is_refused(self, env):
        with pytest.raises(ValueError, match="does not match the shape"):
            env.submit(("edit_mask_file", _upload(_png_bytes((64, 64)))))

    def test_paint_and_extend_together_are_refused_before_a_job_is_queued(self, env):
        with pytest.raises(ValueError, match="paints or extends, not both"):
            env.submit(("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))), ("edit_extend_side", "left"), ("edit_extend_amount", "128"))

        assert env.submitted == []

    @pytest.mark.parametrize(
        ("side", "amount", "message"),
        [("middle", "256", "side must be"), ("top", "100", "from 128 to 512"), ("top", "1024", "from 128 to 512"), ("top", None, "whole number"), (None, "256", "side must be")],
    )
    def test_an_extend_that_is_not_allowed_is_refused(self, env, side, amount, message):
        entries = [(key, value) for key, value in (("edit_extend_side", side), ("edit_extend_amount", amount)) if value is not None]

        with pytest.raises(ValueError, match=message):
            env.submit(*entries)

    def test_a_form_without_a_source_is_refused(self, env):
        form = FormData([("workflow", "edit"), ("model", "zit"), ("prompt", "x")])

        with pytest.raises(ValueError, match="Pick the picture to edit"):
            web_server._submit_image_job(form, env.web_config)

    def test_a_source_that_is_not_an_image_is_refused(self, env):
        broken = env.root / "broken.png"
        broken.write_bytes(b"not an image")

        with pytest.raises(ValueError, match="not a readable image"):
            env.submit(("edit_source_path", str(broken)))

    def test_a_form_without_a_prompt_is_refused(self, env):
        form = FormData([("workflow", "edit"), ("model", "zit"), ("edit_source_path", str(env.source)), ("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE)))])

        with pytest.raises(ValueError, match="prompt"):
            web_server._submit_image_job(form, env.web_config)

        assert env.submitted == []

    def test_the_route_answers_a_refusal_with_422_and_its_message(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "load_web_config", lambda: env.web_config)
        with TestClient(web_server.app) as client:
            response = client.post("/api/generate", data={"workflow": "edit", "model": "zit", "prompt": "x", "edit_source_path": write_cutout(env.root / "cut.png", _SOURCE_SIZE)})

        assert response.status_code == 422
        assert response.json()["detail"] == "Paint the area to change."


# -- R8.3: the download plan -------------------------------------------------------------------------------------


class TestThePlan:
    @pytest.fixture
    def plan_client(self, env, monkeypatch):
        monkeypatch.setattr(web_server, "load_web_config", lambda: env.web_config)
        monkeypatch.setattr(web_server, "job_download_parts", lambda *_a, **_kw: ())
        with TestClient(web_server.app) as client:
            yield client

    def test_empty_file_parts_plan_without_reading_or_writing_any_file(self, env, plan_client, monkeypatch):
        def forbidden(*_a, **_kw):
            raise AssertionError("a plan must not touch the files")

        monkeypatch.setattr(web_server, "read_edit_source", forbidden)
        monkeypatch.setattr(edit_job, "decode_paint", forbidden)
        before = sorted(path.relative_to(env.tmp_path).as_posix() for path in env.tmp_path.rglob("*"))

        response = plan_client.post(
            "/api/generate/plan",
            data={"mode": "image", "workflow": "edit", "model": "zit", "prompt": "a red door", "edit_painted": "true"},
            files={"edit_source_file": ("door.png", b"", "image/png"), "edit_mask_file": ("mask.png", b"", "image/png")},
        )

        assert response.status_code == 200
        assert sorted(path.relative_to(env.tmp_path).as_posix() for path in env.tmp_path.rglob("*")) == before
        assert env.submitted == []

    def test_a_plan_still_applies_the_checks_on_the_fields(self, env, plan_client):
        response = plan_client.post(
            "/api/generate/plan",
            data={"mode": "image", "workflow": "edit", "model": "zit", "prompt": "x", "edit_extend_side": "top", "edit_extend_amount": "100"},
            files={"edit_source_file": ("door.png", b"", "image/png")},
        )

        assert response.status_code == 422
        assert "from 128 to 512" in response.json()["detail"]

    def test_a_plan_with_a_typed_source_reads_it_and_decodes_no_mask(self, env, plan_client, monkeypatch):
        monkeypatch.setattr(edit_job, "decode_paint", lambda *_a, **_kw: pytest.fail("a plan decodes no mask"))

        response = plan_client.post(
            "/api/generate/plan",
            data={"mode": "image", "workflow": "edit", "model": "zit", "prompt": "x", "edit_source_path": str(env.source)},
            files={"edit_mask_file": ("mask.png", b"", "image/png")},
        )

        assert response.status_code == 200

    def test_a_plan_of_a_cut_out_without_paint_is_refused_like_a_submit(self, env, plan_client):
        response = plan_client.post(
            "/api/generate/plan",
            data={"mode": "image", "workflow": "edit", "model": "zit", "prompt": "x", "edit_source_path": write_cutout(env.root / "cut.png", _SOURCE_SIZE)},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == _shared_refusal(has_mask=False, extend=None, cutout=True, supports_mask_edit=True)


# -- R8.6: the contract ------------------------------------------------------------------------------------------


class TestTheContract:
    def test_the_contract_lists_edit_with_its_definition_on_every_platform(self):
        from zvisiongenerator.web.workspace_contract import build_workflow_contract

        for kwargs in ({}, {"video_extend": False, "video_audio": False}):
            contract = build_workflow_contract(**kwargs)

            assert "edit" in contract["values"]
            assert contract["definitions"]["edit"] == {
                "mode": "image",
                "model_kind": "image",
                "visible_controls": ["workflow", "model", "quantize", "loras", "prompt_inline", "runs", "steps", "seed"],
                "requires_reference_image": False,
                "requires_source_video": False,
                "clear_fields": [],
            }

    def test_edit_is_an_image_workflow(self):
        from zvisiongenerator.web.workspace_contract import canonicalize_workflow, workflow_mode

        assert canonicalize_workflow("edit") == "edit"
        assert workflow_mode("edit") == "image"

    def test_a_form_may_post_edit_as_its_workflow(self):
        assert web_server._workflow_from_form("edit", fallback="txt2img") == "edit"


# -- the parsed job runs ------------------------------------------------------------------------------------------


class TestTheParsedJobRuns:
    @staticmethod
    def _run_form(env, backend: MagicMock, prompt: str, *entries: tuple[str, object]) -> None:
        """Parse an Edit form on the source picture and run the job it builds through ``run_batch`` on *backend*."""
        backend.name = "mock"
        fields = [("workflow", "edit"), ("model", "zit"), ("prompt", prompt), ("edit_source_path", str(env.source)), *entries]
        parsed = web_server._parse_image_job(FormData(fields), env.web_config)
        config = {"generation": {"seed_min": 1, "seed_max": 100}, "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4}}
        run_batch(
            backend,
            MagicMock(spec=[]),
            parsed.prompts_data,
            config,
            parsed.args,
            model_info=_KLEIN,
            plan=_make_plan(parsed.prompts_data, parsed.args, config=config),
            progress_callback=lambda _event: None,
        )

    def test_the_args_and_request_the_form_builds_run_through_run_batch(self, env):
        """The job's namespace carries everything ``run_batch`` reads: an edit lands beside its source with its own provenance."""
        painted_at_run: list[bool] = []
        backend = MagicMock(spec=ImageBackend)

        def edit_region(**kwargs):
            painted_at_run.append(bool(kwargs["paint"].any()))
            return Image.new("RGB", (555, 777), "white")

        backend.edit_region.side_effect = edit_region

        self._run_form(env, backend, "a red door", ("edit_mask_file", _upload(_png_bytes(_SOURCE_SIZE))))

        saved = env.root / "door_edit.png"
        assert saved.is_file() and painted_at_run == [True]
        recorded = read_png_config(str(saved))
        assert recorded["workflow"] == "edit"
        assert recorded["edit"] == {"mode": "paint"}
        assert "sharpen" not in recorded["generation"]
        assert '"edit_paint"' not in json.dumps(recorded)

    def test_an_extend_form_reaches_the_extend_the_model_draws_and_saves_the_grown_picture(self, env):
        backend = MagicMock(spec=ImageBackend)
        backend.edit_region.side_effect = lambda **kwargs: kwargs["image"].copy()

        self._run_form(env, backend, "more door", ("edit_extend_side", "left"), ("edit_extend_amount", "128"))

        call = backend.edit_region.call_args.kwargs
        assert call["fix_join"] is True and call["prompt"] == EXTEND_INSTRUCTION + "more door"
        assert call["reference"].size == call["image"].size and call["paint"].shape == call["image"].size[::-1]
        with Image.open(next(env.root.glob("door_edit*.png"))) as saved:
            assert saved.size == (_SOURCE_SIZE[0] + 128, _SOURCE_SIZE[1])
