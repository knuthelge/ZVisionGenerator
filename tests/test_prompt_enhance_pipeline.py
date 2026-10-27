"""Tests for auto prompt enhancement in workflows, runners, provenance, prompt files, and CLIs."""

from __future__ import annotations

import warnings
from argparse import Namespace
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_args, _make_plan, _make_video_args

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
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import build_video_workflow, build_workflow
from zvisiongenerator.workflows.enhance_stage import apply_planned_enhancement
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


# ── Shared reporting helper ─────────────────────────────────────────────────


class TestApplyPlannedEnhancement:
    """The helper both enhance stages report through."""

    def test_image_defaults_store_print_and_report_the_rewrite_unchanged(self, capsys):
        seen: list[str] = []
        request = _image_request(enhanced_prompt="A red fox\nin deep snow.", on_prompt_enhanced=seen.append)
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        apply_planned_enhancement(request, artifacts)
        assert artifacts.resolved_prompt == "A red fox\nin deep snow."
        assert artifacts.metadata == {"enhanced_prompt": "A red fox\nin deep snow."}
        assert seen == ["A red fox\nin deep snow."]
        assert capsys.readouterr().out == "Enhanced prompt:\nA red fox in deep snow.\n\n"

    def test_not_enhanced_prints_nothing_and_changes_nothing(self, capsys):
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        apply_planned_enhancement(_image_request(enhanced_prompt=None), artifacts)
        assert artifacts.resolved_prompt == "a red fox" and artifacts.metadata == {}
        assert capsys.readouterr().out == ""

    def test_composed_text_is_shown_and_reported_while_the_rewrite_is_stored(self, capsys):
        seen: list[str] = []
        request = _image_request(enhanced_prompt="A fox sprints.", on_prompt_enhanced=seen.append)
        artifacts = ImageWorkingArtifacts()
        apply_planned_enhancement(request, artifacts, resolved_prompt="A field. A fox sprints.", extra_metadata={"enhanced_scene": "A field."})
        assert artifacts.resolved_prompt == "A field. A fox sprints."
        assert artifacts.metadata == {"enhanced_prompt": "A fox sprints.", "enhanced_scene": "A field."}
        assert seen == ["A field. A fox sprints."]
        assert capsys.readouterr().out == "Enhanced prompt:\nA field. A fox sprints.\n\n"


# ── Stage behaviour ─────────────────────────────────────────────────────────


class TestEnhanceStage:
    """The stage applies the preflight rewrite; it never calls an enhancer."""

    def test_video_stage_without_a_scene_adds_no_scene_metadata(self):
        request = VideoGenerationRequest(backend=None, model=None, prompt="a fox runs", enhanced_prompt="A fox sprints.")
        artifacts = VideoWorkingArtifacts()

        assert video_enhance_prompt_stage(request, artifacts) is StageOutcome.success

        assert artifacts.resolved_prompt == "A fox sprints."
        assert artifacts.metadata == {"enhanced_prompt": "A fox sprints."}

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

    def test_video_stages_with_a_scene_compose_only_the_resolved_prompt(self):
        seen: list[str] = []
        request = VideoGenerationRequest(
            backend=None,
            model=None,
            prompt="a {fox|fox} runs",
            resolved_prompt="a fox runs",
            scene="A {snowy|snowy} field.",
            resolved_scene="A snowy field.",
            enhanced_prompt="A fox sprints.",
            enhanced_scene="A snowy field at dusk.",
            on_prompt_enhanced=seen.append,
        )
        artifacts = VideoWorkingArtifacts()
        video_resolve_prompt_stage(request, artifacts)
        assert artifacts.resolved_prompt == "A snowy field. a fox runs"
        video_enhance_prompt_stage(request, artifacts)
        assert artifacts.resolved_prompt == "A snowy field at dusk. A fox sprints."
        assert artifacts.metadata["enhanced_prompt"] == "A fox sprints."
        assert artifacts.metadata["enhanced_scene"] == "A snowy field at dusk."
        assert seen == ["A snowy field at dusk. A fox sprints."]


# ── Provenance (REQ-10) ─────────────────────────────────────────────────────


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
            "a:\n  - prompt: x\n    enhance: {style: anime, motion: [secondary]}\n  - prompt: y\n    active: false\n    enhance: true\n  - prompt: z\n",
            source_name="f.yaml",
        )
        assert inspection.enhance_by_set == {"a": [EnhanceSettings(style="anime", motion=("secondary",)), None]}
        assert inspection.options[0].enhance.style == "anime"

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("{style: vaporwave}", "entry 0 of prompt set 'a' in f.yaml"),
            ("'yes'", "expected true, false, or a mapping"),
        ],
    )
    def test_invalid_warns_and_keeps_file_usable(self, value, message):
        with pytest.warns(UserWarning, match=message):
            inspection = inspect_prompts_text(f"a:\n  - prompt: x\n    enhance: {value}\n  - prompt: y\n", source_name="f.yaml")
        assert inspection.prompts_data == {"a": [("x", None), ("y", None)]}
        assert inspection.enhance_by_set == {"a": [None, None]}

    def test_parts_action_is_parsed_and_a_keep_all_entry_with_it_is_kept(self):
        """``parts: action`` always changes the prompt (the action is rewritten), so a keep-all entry is not refused as a no-op."""
        keep = "style: keep, mood: keep, details: [], length: same, motion: []"
        inspection = inspect_prompts_text(
            f"a:\n  - scene: A castle.\n    prompt: x\n    enhance: {{{keep}, parts: action}}\n  - prompt: y\n    enhance: {{parts: both}}\n",
            source_name="f.yaml",
        )
        assert inspection.enhance_by_set["a"][0] == EnhanceSettings(style="keep", mood="keep", details=(), length="same", motion=(), parts="action")
        assert inspection.enhance_by_set["a"][1] == EnhanceSettings(parts="both")

    def test_an_invalid_parts_value_warns_and_keeps_file_usable(self):
        with pytest.warns(UserWarning, match="entry 0 of prompt set 'a' in f.yaml"):
            inspection = inspect_prompts_text("a:\n  - prompt: x\n    enhance: {parts: scene}\n", source_name="f.yaml")
        assert inspection.enhance_by_set == {"a": [None]}

    def test_web_preview_writes_parts_only_when_set(self, tmp_path):
        from zvisiongenerator.web.prompt_files import inspect_prompt_file

        path = tmp_path / "p.yaml"
        path.write_text("a:\n  - prompt: x\n    enhance: {parts: action}\n", encoding="utf-8")
        assert inspect_prompt_file(str(path), accepted_extensions=(".yaml",)).options[0]["enhance"] == "parts=action"

    def test_web_preview_exposes_spec(self, tmp_path):
        from zvisiongenerator.web.prompt_files import inspect_prompt_file

        path = tmp_path / "p.yaml"
        path.write_text("a:\n  - prompt: x\n    enhance: {length: longer}\n  - prompt: y\n", encoding="utf-8")
        options = inspect_prompt_file(str(path), accepted_extensions=(".yaml",)).options
        assert options[0]["enhance"] == "length=longer"
        assert options[1]["enhance"] is None

    def test_web_preview_gives_the_image_run_spec(self, tmp_path):
        from zvisiongenerator.web.prompt_files import inspect_prompt_file

        path = tmp_path / "p.yaml"
        path.write_text("a:\n  - prompt: x\n    enhance: {style: anime, camera: orbit}\n  - prompt: y\n    enhance: {camera: orbit}\n  - prompt: z\n    enhance: true\n", encoding="utf-8")
        options = inspect_prompt_file(str(path), accepted_extensions=(".yaml",)).options
        assert [option["enhance"] for option in options] == ["style=anime,camera=orbit", "camera=orbit", "style=keep"]
        assert [option["enhance_image"] for option in options] == ["style=anime", None, "style=keep"]
        assert [option["enhance_caption"] for option in options] == ["style=anime", "style=keep", "style=keep"]


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
        assert _cli(["--enhance-model", "smart"]).enhance_model == "smart"

    def test_json_format_implies_enhance_and_is_never_a_no_op(self):
        args = _cli(["--enhance-format", "json"])
        assert (args.enhance_format, args.enhance) == ("json", EnhanceSettings())
        assert _cli(["--enhance", "details=,length=same", "--enhance-format", "json"]).enhance is not None

    def test_json_format_alone_leaves_a_prompt_file_to_its_entries(self):
        from zvisiongenerator.enhance_cli import defer_format_to_entries

        args = _cli(["--enhance-format", "json"])
        defer_format_to_entries(args, {"set": [EnhanceSettings(style="anime"), None]})
        assert args.enhance is None

        with_spec = _cli(["--enhance", "style=photo", "--enhance-format", "json"])
        defer_format_to_entries(with_spec, {"set": [EnhanceSettings(style="anime")]})
        assert with_spec.enhance == EnhanceSettings(style="photo")

    def test_json_format_alone_warns_when_no_entry_enhances(self):
        from zvisiongenerator.enhance_cli import defer_format_to_entries

        with pytest.warns(UserWarning, match="none have one"):
            defer_format_to_entries(_cli(["--enhance-format", "json"]), {"set": [None]})

    def test_json_format_is_image_only(self, capsys):
        with pytest.raises(SystemExit):
            _cli(["--enhance-format", "json"], mode="video")

    def test_json_format_with_no_enhance_exits(self, capsys):
        with pytest.raises(SystemExit):
            _cli(["--enhance-format", "json", "--no-enhance"])

    @pytest.mark.parametrize(
        ("argv", "mode"),
        [
            (["--enhance", "style=nope"], "image"),
            (["--enhance", "motion=action"], "image"),
            (["--enhance", "--no-enhance"], "image"),
        ],
    )
    def test_errors_exit(self, argv, mode, capsys):
        with pytest.raises(SystemExit):
            _cli(argv, mode=mode)

    def test_video_motion(self):
        assert _cli(["--enhance", "motion=secondary"], mode="video").enhance.motion == ("secondary",)

    def test_video_sound(self):
        assert _cli(["--enhance", "sound=ambient+dialogue"], mode="video").enhance.sound == ("ambient", "dialogue")

    def test_video_parts(self):
        assert _cli(["--enhance", "parts=action"], mode="video").enhance.parts == "action"
        assert _cli(["--enhance", "style=anime,parts=both"], mode="video").enhance == EnhanceSettings(style="anime", parts="both")

    def test_parts_alone_is_a_change_with_nothing_else_set(self):
        assert _cli(["--enhance", "style=keep,details=,length=same,motion=,parts=action"], mode="video").enhance.parts == "action"

    def test_parts_is_unset_by_default_so_the_job_decides(self):
        assert _cli(["--enhance"], mode="video").enhance.parts is None

    def test_parts_is_video_only_and_validated(self, capsys):
        with pytest.raises(SystemExit):
            _cli(["--enhance", "parts=action"], mode="image")
        with pytest.raises(SystemExit):
            _cli(["--enhance", "parts=scene"], mode="video")

    def test_note_implies_enhance(self):
        assert _cli(["--enhance-note", "at night,  in the rain"]).enhance == EnhanceSettings(note="at night, in the rain")

    def test_note_with_spec(self):
        assert _cli(["--enhance", "style=anime", "--enhance-note", "at dusk"]).enhance == EnhanceSettings(style="anime", note="at dusk")

    def test_note_alone_counts_as_a_change(self):
        assert _cli(["--enhance", "details=,length=same", "--enhance-note", "at dusk"]).enhance.note == "at dusk"

    def test_note_with_no_enhance_exits(self):
        with pytest.raises(SystemExit):
            _cli(["--enhance-note", "x", "--no-enhance"])


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
