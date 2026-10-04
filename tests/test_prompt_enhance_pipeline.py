"""Tests for auto prompt enhancement in workflows, runners, provenance, prompt files, and CLIs."""

from __future__ import annotations

import warnings
from argparse import Namespace
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from conftest import _make_args, _make_plan, _make_video_args
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.job_plan import EnhanceStatus, JobPlan
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.enhance_cli import add_enhance_arguments, parse_enhance_args
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, enhancement_requested, resolve_item_enhance
from zvisiongenerator.utils.prompts import inspect_prompts_text
from zvisiongenerator.utils.provenance import build_image_config_payload, build_video_config_payload
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import build_video_workflow, build_workflow
from zvisiongenerator.workflows.image_stages import enhance_prompt_stage, resolve_prompt_stage
from zvisiongenerator.workflows.video_stages import enhance_prompt_stage as video_enhance_prompt_stage
from zvisiongenerator.workflows.video_stages import resolve_prompt_stage as video_resolve_prompt_stage


def _image_request(**overrides) -> ImageGenerationRequest:
    values = dict(backend=None, model=None, prompt="a red fox", seed=11, resolved_prompt="a red fox", enhanced_prompt="A red fox in deep snow, golden light.")
    values.update(overrides)
    return ImageGenerationRequest(**values)


# ── Workflow composition ────────────────────────────────────────────────────


class TestWorkflowComposition:
    def test_image_stage_only_when_enhancing(self):
        args = _make_args()
        assert enhance_prompt_stage not in build_workflow(args).stages
        stages = build_workflow(args, enhance=True).stages
        assert stages[0] is resolve_prompt_stage and stages[1] is enhance_prompt_stage

    def test_video_stage_only_when_enhancing(self):
        args = _make_video_args()
        assert video_enhance_prompt_stage not in build_video_workflow(args).stages
        assert build_video_workflow(args, enhance=True).stages[1] is video_enhance_prompt_stage


# ── Stage behaviour ─────────────────────────────────────────────────────────


class TestEnhanceStage:
    """The stage applies the preflight rewrite; it never calls an enhancer."""

    def test_applies_planned_rewrite_and_reports(self):
        seen: list[str] = []
        request = _image_request(on_prompt_enhanced=seen.append)
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        assert enhance_prompt_stage(request, artifacts) is StageOutcome.success
        assert artifacts.resolved_prompt == "A red fox in deep snow, golden light."
        assert artifacts.metadata["enhanced_prompt"] == artifacts.resolved_prompt
        assert seen == [artifacts.resolved_prompt]

    def test_not_enhanced_is_noop(self):
        seen: list[str] = []
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        assert enhance_prompt_stage(_image_request(enhanced_prompt=None, on_prompt_enhanced=seen.append), artifacts) is StageOutcome.success
        assert artifacts.resolved_prompt == "a red fox" and "enhanced_prompt" not in artifacts.metadata and seen == []

    def test_resolve_stage_uses_planned_choice(self):
        artifacts = ImageWorkingArtifacts()
        resolve_prompt_stage(_image_request(prompt="a {red|grey} fox", resolved_prompt="a grey fox"), artifacts)
        assert artifacts.resolved_prompt == "a grey fox"

    def test_resolve_stage_without_plan_expands(self):
        artifacts = ImageWorkingArtifacts()
        resolve_prompt_stage(_image_request(prompt="a {red|red} fox", resolved_prompt=None), artifacts)
        assert artifacts.resolved_prompt == "a red fox"

    def test_video_stages(self):
        seen: list[str] = []
        request = VideoGenerationRequest(backend=None, model=None, prompt="a {fox|fox} runs", resolved_prompt="a fox runs", enhanced_prompt="A fox sprints.", on_prompt_enhanced=seen.append)
        artifacts = VideoWorkingArtifacts()
        video_resolve_prompt_stage(request, artifacts)
        assert artifacts.resolved_prompt == "a fox runs"
        video_enhance_prompt_stage(request, artifacts)
        assert artifacts.resolved_prompt == "A fox sprints." and artifacts.metadata["enhanced_prompt"] == "A fox sprints." and seen == ["A fox sprints."]


# ── Provenance (REQ-10) ─────────────────────────────────────────────────────


class TestProvenance:
    def test_image_payload_stores_enhanced_prompt(self):
        artifacts = ImageWorkingArtifacts(metadata={"enhanced_prompt": "An enhanced fox."})
        assert build_image_config_payload(_image_request(prompt="a {red|grey} fox"), artifacts)["prompt"] == "An enhanced fox."

    def test_image_payload_without_enhancement_keeps_template(self):
        assert build_image_config_payload(_image_request(prompt="a {red|grey} fox"), ImageWorkingArtifacts())["prompt"] == "a {red|grey} fox"

    def test_video_payload_stores_enhanced_prompt(self):
        request = VideoGenerationRequest(backend=None, model=None, prompt="a fox runs")
        artifacts = VideoWorkingArtifacts(metadata={"enhanced_prompt": "A fox sprints."})
        assert build_video_config_payload(request, artifacts)["prompt"] == "A fox sprints."


# ── Precedence ──────────────────────────────────────────────────────────────


class TestPrecedence:
    ENTRY = EnhanceSettings(style="anime")
    OVERRIDE = EnhanceSettings(style="photo")

    @pytest.mark.parametrize(
        ("disabled", "override", "entry", "expected"),
        [
            (True, OVERRIDE, ENTRY, None),
            (False, OVERRIDE, ENTRY, OVERRIDE),
            (False, None, ENTRY, ENTRY),
            (False, None, None, None),
        ],
    )
    def test_item(self, disabled, override, entry, expected):
        assert resolve_item_enhance(disabled=disabled, override=override, entry=entry) == expected

    def test_requested(self):
        assert enhancement_requested(disabled=False, override=None, enhance_by_set={"a": [None, self.ENTRY]})
        assert not enhancement_requested(disabled=False, override=None, enhance_by_set={"a": [None]})
        assert not enhancement_requested(disabled=True, override=self.OVERRIDE, enhance_by_set=None)


# ── Runners ─────────────────────────────────────────────────────────────────

_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    "generation": {"seed_min": 1, "seed_max": 100},
    "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
    "model_presets": {"flux1": {"enhance_max_words": 180}},
}


def _enhance_all(plan: JobPlan, text: str = "ENHANCED") -> JobPlan:
    """Mark every iteration that requested enhancement as enhanced with *text* (as preflight would)."""
    return JobPlan(iterations=tuple(replace(it, enhanced_prompt=text, enhance_status=EnhanceStatus.ENHANCED) if it.enhance is not None else it for it in plan.iterations))


class TestImageRunner:
    def _run(self, *, args, plan, prompts):
        requests: list[ImageGenerationRequest] = []
        events: list[dict] = []

        def _stage(request, artifacts):
            requests.append(request)
            if request.on_prompt_enhanced is not None and request.enhanced_prompt is not None:
                request.on_prompt_enhanced(request.enhanced_prompt)
            return StageOutcome.success

        with patch("zvisiongenerator.image_runner.build_workflow") as build:
            build.return_value = GenerationWorkflow(name="t", stages=[MagicMock(side_effect=_stage)])
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                prompts,
                _CONFIG,
                args,
                model_info=ImageModelInfo(family="flux1", is_distilled=False, size=None),
                plan=plan,
                progress_callback=events.append,
            )
        return requests, events, build

    def test_planned_rewrites_reach_requests(self):
        prompts = {"a": [("p1", None), ("p2", None)]}
        args = _make_args(enhance=None, no_enhance=False)
        plan = _enhance_all(_make_plan(prompts, args, config=_CONFIG, enhance_by_set={"a": [EnhanceSettings(style="anime"), None]}))
        requests, events, build = self._run(args=args, plan=plan, prompts=prompts)
        assert [r.enhanced_prompt for r in requests] == ["ENHANCED", None]
        assert build.call_args.kwargs == {"enhance": True}
        enhanced = [e for e in events if e["type"] == "prompt_enhanced"]
        assert enhanced == [
            {"type": "prompt_enhanced", "mode": "image", "run_index": 0, "ran_iterations": 1, "total_iterations": 2, "set_name": "a", "prompt_index": 0, "seed": 42, "enhanced_prompt": "ENHANCED"}
        ]

    def test_no_rewrites_builds_without_stage(self):
        prompts = {"a": [("p1", None)]}
        requests, _events, build = self._run(args=_make_args(), plan=_make_plan(prompts, _make_args(), config=_CONFIG), prompts=prompts)
        assert build.call_args.kwargs == {"enhance": False}
        assert requests[0].enhanced_prompt is None


class TestVideoRunner:
    def test_planned_rewrite_and_event(self):
        requests: list[VideoGenerationRequest] = []
        events: list[dict] = []

        def _stage(request, artifacts):
            requests.append(request)
            request.on_prompt_enhanced(request.enhanced_prompt)
            return StageOutcome.success

        prompts = {"v": [("a fox runs", None)]}
        args = _make_video_args(enhance=EnhanceSettings(style="cinematic"), no_enhance=False)
        run_video_batch(
            backend=MagicMock(),
            model=MagicMock(),
            model_info=VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
            workflow=GenerationWorkflow(name="t", stages=[MagicMock(side_effect=_stage)]),
            prompts_data=prompts,
            config={},
            args=args,
            plan=_enhance_all(_make_plan(prompts, args), "A fox sprints."),
            progress_callback=events.append,
        )
        assert requests[0].enhanced_prompt == "A fox sprints." and requests[0].resolved_prompt == "a fox runs"
        assert [e["enhance_status"] for e in events if e["type"] == "prompt_started"] == ["enhanced"]
        assert any(e["type"] == "prompt_enhanced" and e["mode"] == "video" for e in events)

    def test_requires_matching_plan(self):
        args = _make_video_args()
        with pytest.raises(ValueError, match="iterations"):
            run_video_batch(MagicMock(), MagicMock(), MagicMock(), MagicMock(), {"v": [("x", None)]}, {}, args, plan=JobPlan(iterations=()))
        with pytest.raises(ValueError, match="cancelled"):
            run_video_batch(MagicMock(), MagicMock(), MagicMock(), MagicMock(), {}, {}, args, plan=JobPlan(iterations=(), cancelled=True))


# ── Prompt files ────────────────────────────────────────────────────────────


class TestPromptFileEnhance:
    def test_entries_parsed_and_aligned(self):
        inspection = inspect_prompts_text(
            "a:\n  - prompt: x\n    enhance: {style: anime, motion: [pacing]}\n  - prompt: y\n    active: false\n    enhance: true\n  - prompt: z\n",
            source_name="f.yaml",
        )
        assert inspection.enhance_by_set == {"a": [EnhanceSettings(style="anime", motion=("pacing",)), None]}
        assert inspection.options[0].enhance.style == "anime"

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("{style: watercolor}", "entry 0 of prompt set 'a' in f.yaml"),
            ("'yes'", "expected true, false, or a mapping"),
            ("{style: keep, details: [], length: same, motion: []}", "Nothing to enhance"),
        ],
    )
    def test_invalid_warns_and_keeps_file_usable(self, value, message):
        with pytest.warns(UserWarning, match=message):
            inspection = inspect_prompts_text(f"a:\n  - prompt: x\n    enhance: {value}\n  - prompt: y\n", source_name="f.yaml")
        assert inspection.prompts_data == {"a": [("x", None), ("y", None)]}
        assert inspection.enhance_by_set == {"a": [None, None]}

    def test_web_preview_exposes_spec(self, tmp_path):
        from zvisiongenerator.web.prompt_files import inspect_prompt_file

        path = tmp_path / "p.yaml"
        path.write_text("a:\n  - prompt: x\n    enhance: {length: longer}\n  - prompt: y\n", encoding="utf-8")
        options = inspect_prompt_file(str(path), accepted_extensions=(".yaml",)).options
        assert options[0]["enhance"] == "style=keep,mood=keep,details=lighting+composition,length=longer,motion=action"
        assert options[1]["enhance"] is None


# ── CLI flags ───────────────────────────────────────────────────────────────


def _cli(argv: list[str], *, mode: str = "image") -> Namespace:
    import argparse

    parser = argparse.ArgumentParser(prog="t")
    add_enhance_arguments(parser, mode=mode)
    args = parser.parse_args(argv)
    parse_enhance_args(parser, args, mode=mode)
    return args


class TestCliFlags:
    def test_absent(self):
        args = _cli([])
        assert args.enhance is None and args.no_enhance is False and args.enhance_model is None

    def test_bare_flag_is_defaults(self):
        assert _cli(["--enhance"]).enhance == EnhanceSettings()

    def test_spec(self):
        assert _cli(["--enhance", "style=anime,length=extra"]).enhance == EnhanceSettings(style="anime", length="extra")

    def test_model(self):
        assert _cli(["--enhance-model", "me/llm@v1"]).enhance_model == "me/llm@v1"

    @pytest.mark.parametrize(
        ("argv", "mode"),
        [
            (["--enhance", "style=nope"], "image"),
            (["--enhance", "motion=action"], "image"),
            (["--enhance", "--no-enhance"], "image"),
            (["--enhance", "details=,length=same"], "image"),
        ],
    )
    def test_errors_exit(self, argv, mode, capsys):
        with pytest.raises(SystemExit):
            _cli(argv, mode=mode)

    def test_video_motion(self):
        assert _cli(["--enhance", "motion=pacing"], mode="video").enhance.motion == ("pacing",)


class _CliEnhancer:
    def __init__(self, order: list[str]):
        self.order = order

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        yield "A red fox in deep snow, golden light, shallow depth of field."

    def close(self) -> None:
        self.order.append("close")


class TestImageCliLifecycle:
    """The CLI runs preflight (enhancer load → rewrites → unload → free memory) before loading the image model."""

    def _patch(self, monkeypatch, order: list[str], *, run_batch=None, load_error=None):
        from zvisiongenerator import image_cli
        from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
        from zvisiongenerator.utils.prompts import PromptFileInspection

        backend = MagicMock(name="mflux")
        backend.name = "mflux"
        backend.load_model.side_effect = lambda *a, **k: (order.append("load_model"), (MagicMock(), MagicMock(family="zimage")))[1]

        def _factory(repo, revision):
            if load_error is not None:
                raise load_error
            return _CliEnhancer(order)

        session = PromptEnhancerSession(_factory, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
        skip = MagicMock(name="SkipSignal")
        skip.check.return_value = False
        skip.consume.return_value = None
        skip.start.side_effect = lambda: order.append("listener_start")
        skip.stop.side_effect = lambda: order.append("listener_stop")
        monkeypatch.setattr("sys.argv", ["ziv-image", "-m", "zimage", "--prompt", "a fox", "--enhance", "style=photo"])
        monkeypatch.setattr(
            image_cli,
            "load_config",
            lambda: {
                "sizes": {"2:3": {"m": {"width": 512, "height": 768}}},
                "generation": {"default_ratio": "2:3", "default_size": "m"},
                "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
            },
        )
        monkeypatch.setattr(image_cli, "resolve_model_path", lambda p, **kw: p)
        monkeypatch.setattr(image_cli, "detect_image_model", lambda _: MagicMock(family="zimage", size=None))
        monkeypatch.setattr(image_cli, "get_backend", lambda: backend)
        monkeypatch.setattr(image_cli, "resolve_defaults", lambda *a, **k: {"steps": 4, "guidance": 1.0, "scheduler": None})
        monkeypatch.setattr(image_cli, "validate_scheduler", lambda *a: None)
        monkeypatch.setattr(image_cli, "inspect_prompts_file", lambda _: PromptFileInspection(prompts_data={}, options=[]))
        monkeypatch.setattr(image_cli, "SkipSignal", lambda: skip)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, rev: True)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", lambda: order.append("release_memory"))
        captured: dict = {}
        monkeypatch.setattr(image_cli, "run_batch", run_batch or (lambda *a, **k: (order.append("run_batch"), captured.update(k))))
        return image_cli, captured, skip

    def test_preflight_before_model_load(self, monkeypatch):
        order: list[str] = []
        image_cli, captured, skip = self._patch(monkeypatch, order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_cli.main()
        assert order == ["listener_start", "close", "release_memory", "load_model", "run_batch", "listener_stop"]
        plan = captured["plan"]
        assert plan.iterations[0].enhance_status is EnhanceStatus.ENHANCED and plan.has_rewrites
        assert captured["skip_signal"] is skip

    def test_enhancer_load_error_is_a_usage_error(self, monkeypatch):
        order: list[str] = []
        image_cli, _captured, _skip = self._patch(monkeypatch, order, load_error=RuntimeError("Could not load prompt enhancer model a/b"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(SystemExit):
                image_cli.main()
        assert "load_model" not in order and "release_memory" not in order
        assert order[-1] == "listener_stop"

    def test_generation_errors_propagate_and_stop_the_listener(self, monkeypatch):
        order: list[str] = []
        image_cli, _captured, _skip = self._patch(monkeypatch, order, run_batch=MagicMock(side_effect=RuntimeError("MPS out of memory")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="MPS out of memory"):
                image_cli.main()
        assert order[-1] == "listener_stop"

    def test_quit_during_preflight_skips_model_load(self, monkeypatch):
        order: list[str] = []
        image_cli, _captured, skip = self._patch(monkeypatch, order)
        skip.consume.return_value = "quit"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_cli.main()
        assert order == ["listener_start", "close", "release_memory", "listener_stop"]
