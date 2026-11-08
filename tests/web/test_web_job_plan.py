"""Tests for the plan routes: what a generate form or an upscale would download, asked before the job is queued."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tests._factories import load_packaged_config
from tests.web._helpers import ffprobe_cannot_read_files, make_resolved_image_defaults, make_web_config, patch_image_submit_dependencies, stub_video_submission
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.downloads.space import SpaceNeed
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.provenance import embed_png_config
from zvisiongenerator.web import model_download_plan as plan_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import upscale_api
from zvisiongenerator.web import job_downloads as job_downloads_module

_KLEIN = "unsloth/FLUX.2-klein-9B"
_ENHANCER = "owner/enhancer"
_GIB = 1024**3


def _use_fake_planners(monkeypatch, env) -> None:
    """Plan one target per model and (when ``env.enhance``) the enhancer, without the Hub."""
    monkeypatch.setattr(job_downloads_module, "image_plan", lambda model_ref, *, family, platform_key: (DownloadTarget(_KLEIN if model_ref == "klein9b" else f"owner/{model_ref}"),))
    monkeypatch.setattr(job_downloads_module, "video_plan", lambda model_ref, *, family, config, model_name, platform_key, upscale: (DownloadTarget(f"owner/{model_ref}"),))
    monkeypatch.setattr(job_downloads_module, "matte_plan", lambda: (DownloadTarget("owner/matte"),))
    monkeypatch.setattr(job_downloads_module, "enhancer_plan", lambda repo, revision: (DownloadTarget(repo, revision),))
    monkeypatch.setattr(job_downloads_module.prompt_enhancer_session, "plan_job_enhancer", lambda *_a, **_kw: (_ENHANCER, None) if env.enhance else None)


@pytest.fixture
def plan_env(monkeypatch, tmp_path):
    """Route the plan through fake planners, a fake Hub and a fake disk, and give the test a handle on each."""
    env = SimpleNamespace(
        present=set(), unreachable=False, enhance=True, hub_calls=[], flights=DownloadFlights(), submitted=[], web_config=make_web_config(), root=tmp_path, queued_jobs=(), queued_downloads=()
    )
    env.web_config.output_dir = str(tmp_path)
    packaged = load_packaged_config()
    env.web_config.app_config.update(audio_model_presets=packaged["audio_model_presets"])
    monkeypatch.setattr(web_server, "load_web_config", lambda: env.web_config)
    patch_image_submit_dependencies(monkeypatch, model_info=ImageModelInfo(family="flux2", is_distilled=False, size=None), defaults=make_resolved_image_defaults(), submitted=env.submitted)
    monkeypatch.setattr(web_server.download_flights, "active", env.flights.active)
    monkeypatch.setattr(web_server.web_runner, "committed_targets", lambda: env.queued_jobs)
    monkeypatch.setattr(web_server.download_lane, "committed_targets", lambda: env.queued_downloads)

    _use_fake_planners(monkeypatch, env)

    def check(target):
        env.hub_calls.append(target.repo_id)
        return None if env.unreachable else TargetCheck(missing_files=3, missing_bytes=2 * _GIB)

    real_summary = plan_module.parts_summary
    monkeypatch.setattr(
        web_server,
        "parts_summary",
        lambda parts, **kwargs: real_summary(
            parts,
            **kwargs,
            check=check,
            present=lambda target: target.repo_id in env.present,
            space=lambda items, *, cache_dir: SpaceNeed(path=cache_dir, needed_bytes=sum(c.missing_bytes for _, c in items), free_bytes=100 * _GIB),
            cache_dir=lambda: tmp_path,
        ),
    )
    with TestClient(web_server.app) as client:
        env.client = client
        yield env


def _form(**fields):
    return {"mode": "image", "model": "klein9b", "prompt": "a barn", **fields}


class TestGeneratePlan:
    def test_a_missing_model_and_its_enhancer_are_listed_enhancer_first(self, plan_env):
        response = plan_env.client.post("/api/generate/plan", data=_form())

        assert response.status_code == 200
        plan = response.json()
        assert [(part["label"], part["model"], part["bytes"]) for part in plan["parts"]] == [
            (_ENHANCER, _ENHANCER, 2 * _GIB),
            ("FLUX.2 Klein 9B", "klein9b", 2 * _GIB),
        ]
        assert plan["total_bytes"] == 4 * _GIB
        assert plan["space"]["enough"] is True
        assert plan["unchecked"] is False
        assert plan_env.submitted == []  # a plan never queues

    def test_a_restricted_model_s_part_carries_its_licence_record(self, plan_env):
        plan_env.enhance = False

        licences = plan_env.client.post("/api/generate/plan", data=_form()).json()["licences"]

        assert [(entry["model"], entry["name"], entry["supplier"], entry["summary"]) for entry in licences] == [
            ("klein9b", "FLUX Non-Commercial License v2.1", "Black Forest Labs", "Non-commercial use of the model; images may be used commercially")
        ]
        assert licences[0]["url_label"] == "Model page"
        assert licences[0]["url"] == "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
        assert licences[0]["attribution"] is None

    def test_a_model_of_the_user_s_own_has_no_licences(self, plan_env):
        plan_env.enhance = False

        assert plan_env.client.post("/api/generate/plan", data=_form(model="owner/my-finetune")).json()["licences"] == []

    def test_only_the_parts_that_download_carry_licences(self, plan_env):
        plan_env.enhance = False
        plan_env.present = {_KLEIN}

        plan = plan_env.client.post("/api/generate/plan", data=_form(transparent="on")).json()

        assert [part["model"] for part in plan["parts"]] == ["matte"]
        assert [(entry["model"], entry["url_label"]) for entry in plan["licences"]] == [("matte", "Dataset terms")]

    def test_a_video_model_with_two_records_carries_both(self, plan_env, monkeypatch):
        stub_video_submission(monkeypatch, plan_env.root)
        monkeypatch.setattr(web_server, "load_web_config", lambda: plan_env.web_config)  # the stub's own config has no licences
        plan_env.enhance = False

        licences = plan_env.client.post("/api/generate/plan", data={"mode": "video", "model": "ltx-4", "prompt": "a lake"}).json()["licences"]

        assert [(entry["model"], entry["supplier"]) for entry in licences] == [("ltx-4", "Lightricks"), ("ltx-4", "Google")]

    def test_a_transparent_job_adds_the_matte_models_last(self, plan_env):
        plan_env.enhance = False

        parts = plan_env.client.post("/api/generate/plan", data=_form(transparent="on")).json()["parts"]

        assert [part["model"] for part in parts] == ["klein9b", "matte"]

    def test_a_present_model_has_no_parts_and_no_hub_call(self, plan_env):
        plan_env.enhance = False
        plan_env.present = {"owner/zit"}

        plan = plan_env.client.post("/api/generate/plan", data=_form(model="zit")).json()

        assert plan == {"parts": [], "total_bytes": 0, "space": None, "unchecked": False, "licences": []}
        assert plan_env.hub_calls == []

    def test_a_model_that_is_already_downloading_is_left_out(self, plan_env):
        plan_env.enhance = False
        plan_env.flights.claim(DownloadTarget(_KLEIN))

        plan = plan_env.client.post("/api/generate/plan", data=_form()).json()

        assert plan["parts"] == []
        assert plan_env.hub_calls == []

    def test_only_the_part_that_is_downloading_is_left_out(self, plan_env):
        plan_env.flights.claim(DownloadTarget(_ENHANCER))

        parts = plan_env.client.post("/api/generate/plan", data=_form()).json()["parts"]

        assert [part["model"] for part in parts] == ["klein9b"]

    def test_a_model_that_a_queued_job_will_download_is_left_out_so_the_second_generate_asks_nothing(self, plan_env):
        plan_env.enhance = False
        plan_env.queued_jobs = (DownloadTarget(_KLEIN),)

        plan = plan_env.client.post("/api/generate/plan", data=_form()).json()

        assert plan["parts"] == []
        assert plan_env.hub_calls == []

    def test_a_model_that_the_models_page_will_download_is_left_out(self, plan_env):
        plan_env.enhance = False
        plan_env.queued_downloads = (DownloadTarget(_KLEIN),)

        assert plan_env.client.post("/api/generate/plan", data=_form()).json()["parts"] == []

    def test_a_part_with_every_missing_target_on_its_way_is_left_out(self, plan_env, monkeypatch):
        plan_env.enhance = False
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *_a, **_kw: (DownloadTarget("owner/one"), DownloadTarget("owner/two")))
        plan_env.flights.claim(DownloadTarget("owner/one"))
        plan_env.queued_jobs = (DownloadTarget("owner/two"),)

        assert plan_env.client.post("/api/generate/plan", data=_form()).json()["parts"] == []

    def test_a_model_that_shares_a_file_with_one_on_its_way_is_still_asked_about(self, plan_env, monkeypatch):
        plan_env.enhance = False
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *_a, **_kw: (DownloadTarget("owner/shared"), DownloadTarget("owner/own")))
        plan_env.queued_jobs = (DownloadTarget("owner/shared"),)

        parts = plan_env.client.post("/api/generate/plan", data=_form()).json()["parts"]

        assert [part["model"] for part in parts] == ["klein9b"]

    def test_a_job_that_is_queued_when_it_is_submitted_keeps_its_planned_targets_for_the_next_plan(self, plan_env, monkeypatch):
        plan_env.enhance = False
        kept = []
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", lambda **kwargs: kept.append(kwargs["planned_targets"]) or "job-1")

        plan_env.client.post("/api/generate", data=_form())

        assert kept == [(DownloadTarget(_KLEIN),)]

    def test_the_job_is_queued_with_the_model_part_it_was_planned_with_so_the_worker_does_not_plan_again(self, plan_env, monkeypatch):
        plan_env.enhance = False
        kept = []
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", lambda **kwargs: kept.append(kwargs) or "job-1")
        planned = []
        real = job_downloads_module.image_plan
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *a, **kw: planned.append(a) or real(*a, **kw))

        plan_env.client.post("/api/generate", data=_form())

        (kwargs,) = kept
        assert [(part.role, part.targets) for part in [kwargs["model_part"]]] == [("model", (DownloadTarget(_KLEIN),))]
        assert len(planned) == 1  # one plan for the submit, whatever the targets and the part are used for

    def test_a_model_that_cannot_be_planned_offline_is_planned_once_per_submit(self, plan_env, monkeypatch):
        plan_env.enhance = False
        kept = []
        monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", lambda **kwargs: kept.append(kwargs) or "job-1")
        planned = []
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *a, **kw: planned.append(a) or None)  # offline: the file list cannot be read

        plan_env.client.post("/api/generate", data=_form())

        (kwargs,) = kept
        assert kwargs["model_part"].targets is None and kwargs["planned_targets"] == ()
        assert len(planned) == 1

    def test_a_hub_that_cannot_be_reached_marks_the_plan_unchecked(self, plan_env):
        plan_env.unreachable = True
        plan_env.enhance = False

        response = plan_env.client.post("/api/generate/plan", data=_form())

        assert response.status_code == 200
        plan = response.json()
        assert plan["unchecked"] is True
        assert plan["total_bytes"] is None
        assert [(part["model"], part["bytes"]) for part in plan["parts"]] == [("klein9b", None)]

    def test_a_part_that_cannot_be_planned_is_unchecked(self, plan_env, monkeypatch):
        plan_env.enhance = False
        monkeypatch.setattr(job_downloads_module, "image_plan", lambda *_a, **_kw: None)

        plan = plan_env.client.post("/api/generate/plan", data=_form(model="zit")).json()

        assert plan["unchecked"] is True
        assert plan["parts"] == [{"label": "Z-Image Turbo", "model": "zit", "bytes": None}]

    def test_a_refusal_by_the_hub_is_a_424_with_its_message(self, plan_env, monkeypatch):
        def refuse(_parts, **_kwargs):
            raise RuntimeError("Hugging Face refused owner/gated")

        monkeypatch.setattr(web_server, "parts_summary", refuse)

        response = plan_env.client.post("/api/generate/plan", data=_form())

        assert response.status_code == 424
        assert "owner/gated" in response.json()["detail"]

    def test_an_unconfigured_enhancer_is_a_422_with_its_message(self, plan_env, monkeypatch):
        def unconfigured(*_args, **_kwargs):
            raise ValueError("No prompt enhancer model is configured for platform 'linux'. Set prompt_enhancer.user_model.")

        monkeypatch.setattr(web_server, "job_download_parts", unconfigured)

        response = plan_env.client.post("/api/generate/plan", data=_form())

        assert response.status_code == 422
        assert "prompt_enhancer.user_model" in response.json()["detail"]

    @pytest.mark.parametrize("path", ["/api/generate/plan", "/api/generate"])
    def test_an_unknown_mode_is_a_422_and_never_an_image(self, plan_env, path):
        response = plan_env.client.post(path, data=_form(mode="hologram"))

        assert response.status_code == 422
        assert "hologram" in response.json()["detail"]
        assert plan_env.submitted == []

    def test_a_form_without_a_mode_is_an_image(self, plan_env):
        plan_env.enhance = False

        response = plan_env.client.post("/api/generate/plan", data={"model": "klein9b", "prompt": "a barn"})

        assert [part["model"] for part in response.json()["parts"]] == ["klein9b"]

    @pytest.mark.parametrize("fields", [{"steps": "-4"}, {"width": "999"}, {"prompt": ""}], ids=["steps", "width", "prompt"])
    def test_an_invalid_form_is_refused_as_the_submit_refuses_it(self, plan_env, fields):
        form = _form(**fields)

        planned = plan_env.client.post("/api/generate/plan", data=form)
        submitted = plan_env.client.post("/api/generate", data=form)

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    def test_a_submit_and_its_plan_parse_the_same_request(self, plan_env, monkeypatch):
        seen = []
        real_parts = web_server.job_download_parts
        monkeypatch.setattr(web_server, "job_download_parts", lambda request, *rest, **kwargs: seen.append((request, kwargs.get("args"))) or real_parts(request, *rest, **kwargs))
        form = _form(seed="7", runs="2", steps="6", ratio="2:3", size="l", transparent="on")

        plan_env.client.post("/api/generate/plan", data=form)
        plan_env.client.post("/api/generate", data=form)

        planned_request, planned_args = seen[0]
        queued = plan_env.submitted[0]
        assert planned_request == queued["request"]
        assert vars(planned_args) == vars(queued["args"])

    def test_a_video_form_plans_the_video_model_and_its_enhancer(self, plan_env, monkeypatch):
        stub_video_submission(monkeypatch, plan_env.root)

        response = plan_env.client.post("/api/generate/plan", data={"mode": "video", "model": "ltx-8", "prompt": "a lake"})

        assert response.status_code == 200
        assert [part["model"] for part in response.json()["parts"]] == [_ENHANCER, "ltx-8"]

    def test_a_video_submit_and_its_plan_parse_the_same_request(self, plan_env, monkeypatch):
        submitted = stub_video_submission(monkeypatch, plan_env.root)
        seen = []
        real_parts = web_server.job_download_parts
        monkeypatch.setattr(web_server, "job_download_parts", lambda request, *rest, **kwargs: seen.append((request, kwargs.get("args"))) or real_parts(request, *rest, **kwargs))
        form = {"mode": "video", "model": "ltx-8", "prompt": "a lake", "seed": "5", "steps": "6"}

        assert plan_env.client.post("/api/generate/plan", data=form).status_code == 200
        assert plan_env.client.post("/api/generate", data=form).status_code == 200

        planned_request, planned_args = seen[0]
        assert planned_request == submitted[0]["request"]
        assert vars(planned_args) == vars(submitted[0]["args"])

    def test_a_plan_saves_no_upload_and_the_submit_saves_it_once(self, plan_env):
        files = {"image_file": ("start.png", _png_bytes(), "image/png")}
        form = _form(workflow="img2img")

        planned = plan_env.client.post("/api/generate/plan", data=form, files=files)

        assert planned.status_code == 200
        assert not (plan_env.root / ".web_uploads").exists()
        queued = plan_env.client.post("/api/generate", data=form, files=files)
        assert queued.status_code == 200
        assert len(list((plan_env.root / ".web_uploads").iterdir())) == 1


def _png_bytes() -> bytes:
    png = io.BytesIO()
    Image.new("RGB", (64, 64), "teal").save(png, format="PNG")
    return png.getvalue()


def _without_bodies(files):
    """Return *files* as the Workspace sends them to the plan: each part keeps its field and file name and has no body."""
    return [(field, (name, b"", kind)) for field, (name, _body, kind) in files]


def _plan(plan_env, data, files):
    return plan_env.client.post("/api/generate/plan", data=data, files=files)


class TestPlanWithoutFileBodies:
    """The Workspace plans a form with its file bodies left out; the plan is the one the full form gets, and no upload is read or saved."""

    @pytest.fixture
    def video_env(self, plan_env, monkeypatch):
        ffprobe_cannot_read_files(monkeypatch)  # no track in these tests is a readable file
        stub_video_submission(monkeypatch, plan_env.root)
        monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: plan_env.submitted.append(kwargs) or "job-123")
        return plan_env

    def _same_parts(self, plan_env, data, files):
        full = _plan(plan_env, data, files)
        bare = _plan(plan_env, data, _without_bodies(files))

        assert full.status_code == bare.status_code == 200, (full.text, bare.text)
        assert bare.json() == full.json()
        assert bare.json()["parts"]  # something to download, so the comparison is not of two empty plans
        assert not (plan_env.root / ".web_uploads").exists()

    def test_an_image_form_with_a_start_image_plans_the_same_parts(self, plan_env):
        self._same_parts(plan_env, _form(workflow="img2img"), [("image_file", ("start.png", _png_bytes(), "image/png"))])

    def test_an_image_form_with_references_plans_the_same_parts(self, plan_env, monkeypatch):
        monkeypatch.setattr(web_server, "resolve_defaults", lambda *_args: make_resolved_image_defaults(max_reference_images=3))
        files = [("reference", (f"ref{index}.png", _png_bytes(), "image/png")) for index in range(2)]

        self._same_parts(plan_env, _form(transparent="on"), files)

    def test_a_video_form_with_both_frames_and_a_track_plans_the_same_parts(self, video_env):
        files = [
            ("image_file", ("first.png", _png_bytes(), "image/png")),
            ("end_image_file", ("last.png", _png_bytes(), "image/png")),
            ("audio_track_file", ("song.m4a", b"not really audio", "audio/mp4")),
        ]
        data = {"mode": "video", "model": "ltx-8", "prompt": "a lake", "workflow": "img2vid"}

        self._same_parts(video_env, data, files)

    def test_the_plan_refuses_the_image_type_the_submit_refuses(self, plan_env):
        files = [("image_file", ("start.gif", b"", "image/gif"))]
        form = _form(workflow="img2img")

        planned = _plan(plan_env, form, files)
        submitted = plan_env.client.post("/api/generate", data=form, files=files)

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    def test_an_image_that_does_not_decode_is_left_to_the_submit(self, plan_env):
        files = [("image_file", ("start.png", b"", "image/png"))]
        form = _form(workflow="img2img")

        planned = _plan(plan_env, form, files)
        submitted = plan_env.client.post("/api/generate", data=form, files=[("image_file", ("start.png", b"not an image", "image/png"))])

        assert planned.status_code == 200
        assert submitted.status_code == 422

    def test_a_track_type_the_submit_refuses_is_refused_by_the_plan(self, video_env):
        data = {"mode": "video", "model": "ltx-8", "prompt": "a lake"}
        files = [("audio_track_file", ("song.aac", b"", "audio/aac"))]

        planned = _plan(video_env, data, files)

        assert planned.status_code == 422

    def test_an_unreadable_track_is_refused_by_the_submit_and_not_by_the_plan(self, video_env):
        files = [("audio_track_file", ("song.m4a", b"", "audio/mp4"))]
        data = {"mode": "video", "model": "ltx-8", "prompt": "a lake"}

        planned = _plan(video_env, data, files)
        submitted = video_env.client.post("/api/generate", data=data, files=[("audio_track_file", ("song.m4a", b"not audio", "audio/mp4"))])

        assert planned.status_code == 200
        assert submitted.status_code == 422

    @pytest.mark.parametrize("fields", [{"steps": "-4"}, {"width": "999"}, {"prompt": ""}], ids=["steps", "width", "prompt"])
    def test_a_field_error_still_refuses_the_plan(self, plan_env, fields):
        files = [("image_file", ("start.png", b"", "image/png"))]
        data = _form(workflow="img2img", **fields)

        planned = _plan(plan_env, data, files)
        submitted = plan_env.client.post("/api/generate", data=data, files=[("image_file", ("start.png", _png_bytes(), "image/png"))])

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    def test_a_start_image_workflow_without_any_image_is_still_refused(self, plan_env):
        response = plan_env.client.post("/api/generate/plan", data=_form(workflow="img2img"))

        assert response.status_code == 422
        assert "start image" in response.json()["detail"]

    def test_a_typed_track_path_that_cannot_be_read_is_refused_by_the_plan(self, video_env):
        data = {"mode": "video", "model": "ltx-8", "prompt": "a lake", "audio_track_path": str(video_env.root / "missing.m4a")}

        planned = _plan(video_env, data, None)
        submitted = video_env.client.post("/api/generate", data=data)

        assert planned.status_code == submitted.status_code == 422
        assert planned.json() == submitted.json()

    def test_a_video_field_error_still_refuses_the_plan(self, video_env):
        files = [("end_image_file", ("last.png", b"", "image/png"))]
        data = {"mode": "video", "model": "ltx-8", "prompt": "a lake"}

        response = _plan(video_env, data, files)

        assert response.status_code == 422
        assert "first frame" in response.json()["detail"]


def _write_png(path, config=None):
    info = PngInfo()
    if config is not None:
        embed_png_config(info, config)
    Image.new("RGB", (832, 1216)).save(path, pnginfo=info)


@pytest.fixture
def upscale_env(plan_env, monkeypatch):
    plan_env.web_config.app_config.update(
        {
            "upscale": {"default_denoise_2x": 0.4, "default_denoise_4x": 0.4, "max_megapixels": 16},
            "sharpening": {"normal": 1.0, "upscaled": 1.2, "pre_upscale": 0.8},
        }
    )
    monkeypatch.setattr(web_server, "_media_output_root", lambda: plan_env.root.resolve())
    monkeypatch.setattr(upscale_api, "resolve_model_path", lambda name, **_: name)
    monkeypatch.setattr(upscale_api, "detect_image_model", lambda _ref: ImageModelInfo(family="flux2", is_distilled=False, size=None))
    monkeypatch.setattr(upscale_api, "resolve_defaults", lambda *_a: {**make_resolved_image_defaults(), "upscale_steps": 3})
    _write_png(plan_env.root / "a.png", config={"prompt": "a barn", "model": "zit", "seed": 3})
    return plan_env


class TestUpscalePlan:
    def test_a_creative_upscale_lists_the_model_it_refines_with(self, upscale_env):
        response = upscale_env.client.post("/api/upscale/plan", json={"asset_id": "a.png", "factor": 2})

        assert response.status_code == 200
        assert [part["model"] for part in response.json()["parts"]] == ["zit"]

    def test_nothing_is_listed_when_the_model_is_present(self, upscale_env):
        upscale_env.present = {"owner/zit"}

        plan = upscale_env.client.post("/api/upscale/plan", json={"asset_id": "a.png", "factor": 2}).json()

        assert plan["parts"] == []
        assert upscale_env.hub_calls == []

    def test_a_model_that_is_already_downloading_is_left_out(self, upscale_env):
        upscale_env.flights.claim(DownloadTarget("owner/zit"))

        assert upscale_env.client.post("/api/upscale/plan", json={"asset_id": "a.png", "factor": 2}).json()["parts"] == []

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            ({"asset_id": "missing.png", "factor": 2}, 404),
            ({"asset_id": "../outside.png", "factor": 2}, 422),
            ({"asset_id": "a.png", "factor": 3}, 422),
            ({"asset_id": "a.png", "factor": 2, "method": "seedvr2"}, 422),
        ],
    )
    def test_a_request_the_submit_refuses_is_refused_the_same_way(self, upscale_env, body, status):
        planned = upscale_env.client.post("/api/upscale/plan", json=body)
        submitted = upscale_env.client.post("/api/upscale", json=body)

        assert planned.status_code == submitted.status_code == status
        assert planned.json() == submitted.json()

    def test_a_submit_and_its_plan_parse_the_same_upscale(self, upscale_env, monkeypatch):
        seen, queued = [], []
        real_parts = web_server.job_download_parts
        monkeypatch.setattr(web_server, "job_download_parts", lambda request, *rest, **kwargs: seen.append((request, kwargs["model_ref"])) or real_parts(request, *rest, **kwargs))
        monkeypatch.setattr(web_server.web_runner, "submit_upscale_job", lambda **kwargs: queued.append(kwargs) or "job-9")
        body = {"asset_id": "a.png", "factor": 2}

        assert upscale_env.client.post("/api/upscale/plan", json=body).status_code == 200
        assert upscale_env.client.post("/api/upscale", json=body).status_code == 200

        assert seen[0] == (queued[0]["request"], queued[0]["model_ref"])

    def test_a_body_that_is_not_json_is_a_422(self, upscale_env):
        assert upscale_env.client.post("/api/upscale/plan", content=b"nope").status_code == 422
