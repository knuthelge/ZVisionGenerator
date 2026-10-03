"""Tests for auto prompt enhancement in workflows, runners, provenance, prompt files, and CLIs."""

from __future__ import annotations

import warnings
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from conftest import _make_args, _make_video_args
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
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


class _FakeEnhancer:
    repo = "fake/repo"
    revision = None

    def __init__(self, text: str = "A red fox in deep snow, golden light, shallow depth of field."):
        self.text = text
        self.seeds: list[int] = []
        self.systems: list[str] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.seeds.append(seed)
        self.systems.append(messages[0]["content"])
        yield self.text

    def close(self) -> None:
        pass


def _image_request(**overrides) -> ImageGenerationRequest:
    values = dict(backend=None, model=None, prompt="a red fox", seed=11, enhance=EnhanceSettings(), prompt_enhancer=_FakeEnhancer())
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
    def test_rewrites_resolved_prompt_and_reports(self):
        seen: list[str] = []
        request = _image_request(on_prompt_enhanced=seen.append)
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        assert enhance_prompt_stage(request, artifacts) is StageOutcome.success
        assert artifacts.resolved_prompt.startswith("A red fox in deep snow")
        assert artifacts.metadata["enhanced_prompt"] == artifacts.resolved_prompt
        assert seen == [artifacts.resolved_prompt]
        assert request.prompt_enhancer.seeds == [11]

    def test_uses_resolved_choice_not_template(self):
        request = _image_request(prompt="a {red|grey} fox")
        artifacts = ImageWorkingArtifacts()
        resolve_prompt_stage(request, artifacts)
        enhance_prompt_stage(request, artifacts)
        assert "Placeholders" not in request.prompt_enhancer.systems[0]

    def test_off_is_noop(self):
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        enhance_prompt_stage(_image_request(enhance=None), artifacts)
        assert artifacts.resolved_prompt == "a red fox" and "enhanced_prompt" not in artifacts.metadata

    def test_json_caption_skipped(self):
        artifacts = ImageWorkingArtifacts(resolved_prompt='{"a": 1}')
        request = _image_request(prompt='{"a": 1}', json_prompt=True)
        enhance_prompt_stage(request, artifacts)
        assert artifacts.resolved_prompt == '{"a": 1}' and request.prompt_enhancer.seeds == []

    def test_failure_warns_and_keeps_prompt(self):
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        with pytest.warns(UserWarning, match="Prompt enhancement failed"):
            outcome = enhance_prompt_stage(_image_request(prompt_enhancer=_FakeEnhancer(text="")), artifacts)
        assert outcome is StageOutcome.success
        assert artifacts.resolved_prompt == "a red fox"

    def test_unexpected_adapter_error_falls_back(self):
        class _Broken(_FakeEnhancer):
            def generate(self, messages, **kwargs):
                raise TypeError("chat template rejected the system role")

        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        with pytest.warns(UserWarning, match="system role"):
            assert enhance_prompt_stage(_image_request(prompt_enhancer=_Broken()), artifacts) is StageOutcome.success
        assert artifacts.resolved_prompt == "a red fox"

    def test_missing_enhancer_warns(self):
        artifacts = ImageWorkingArtifacts(resolved_prompt="a red fox")
        with pytest.warns(UserWarning, match="no enhancer is loaded"):
            enhance_prompt_stage(_image_request(prompt_enhancer=None), artifacts)

    def test_ceiling_and_options_are_used(self):
        enhancer = _FakeEnhancer()
        request = _image_request(prompt_enhancer=enhancer, enhance=EnhanceSettings(length="extra"), enhance_ceiling=180)
        enhance_prompt_stage(request, ImageWorkingArtifacts(resolved_prompt="fox " * 100))
        assert "153-180 words" in enhancer.systems[0]  # upper bound capped at the FLUX.1 ceiling

    def test_video_stage_uses_video_mode(self):
        enhancer = _FakeEnhancer()
        request = VideoGenerationRequest(backend=None, model=None, prompt="a fox runs", seed=3, enhance=EnhanceSettings(), prompt_enhancer=enhancer)
        artifacts = VideoWorkingArtifacts(resolved_prompt="a fox runs")
        video_enhance_prompt_stage(request, artifacts)
        assert "text-to-video" in enhancer.systems[0]
        assert artifacts.metadata["enhanced_prompt"]


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


class TestImageRunner:
    def _run(self, *, args, enhance_by_set=None, prompts=None):
        requests: list[ImageGenerationRequest] = []
        events: list[dict] = []

        def _stage(request, artifacts):
            requests.append(request)
            if request.on_prompt_enhanced is not None and request.enhance is not None:
                request.on_prompt_enhanced("ENHANCED")
            return StageOutcome.success

        enhancer = _FakeEnhancer()
        with patch("zvisiongenerator.image_runner.build_workflow") as build:
            build.return_value = GenerationWorkflow(name="t", stages=[MagicMock(side_effect=_stage)])
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                prompts or {"a": [("p1", None), ("p2", None)]},
                _CONFIG,
                args,
                model_info=ImageModelInfo(family="flux1", is_distilled=False, size=None),
                progress_callback=events.append,
                enable_interactive_controls=False,
                prompt_enhancer=enhancer,
                enhance_by_set=enhance_by_set,
            )
        return requests, events, build, enhancer

    def test_yaml_entries_apply_per_prompt(self):
        entry = EnhanceSettings(style="anime")
        requests, events, build, enhancer = self._run(args=_make_args(enhance=None, no_enhance=False), enhance_by_set={"a": [entry, None]})
        assert [r.enhance for r in requests] == [entry, None]
        assert requests[0].prompt_enhancer is enhancer and requests[1].prompt_enhancer is None
        assert requests[0].enhance_ceiling == 180
        assert requests[0].enhance_options["length"]["percent"]["extra"] == 300
        assert build.call_args.kwargs == {"enhance": True}
        enhanced = [e for e in events if e["type"] == "prompt_enhanced"]
        assert enhanced == [
            {"type": "prompt_enhanced", "mode": "image", "run_index": 0, "ran_iterations": 1, "total_iterations": 2, "set_name": "a", "prompt_index": 0, "seed": 42, "enhanced_prompt": "ENHANCED"}
        ]

    def test_override_applies_to_all(self):
        override = EnhanceSettings(style="photo")
        requests, *_ = self._run(args=_make_args(enhance=override, no_enhance=False), enhance_by_set={"a": [EnhanceSettings(style="anime"), None]})
        assert [r.enhance for r in requests] == [override, override]

    def test_no_enhance_disables_all(self):
        requests, _events, build, _ = self._run(args=_make_args(enhance=None, no_enhance=True), enhance_by_set={"a": [EnhanceSettings(), None]})
        assert [r.enhance for r in requests] == [None, None]
        assert build.call_args.kwargs == {"enhance": False}

    def test_args_without_enhance_fields(self):
        requests, *_ = self._run(args=_make_args())
        assert all(r.enhance is None for r in requests)


class TestVideoRunner:
    def test_override_and_event(self):
        requests: list[VideoGenerationRequest] = []
        events: list[dict] = []

        def _stage(request, artifacts):
            requests.append(request)
            request.on_prompt_enhanced("ENHANCED")
            return StageOutcome.success

        override = EnhanceSettings(style="cinematic")
        args = _make_video_args(enhance=override, no_enhance=False)
        run_video_batch(
            backend=MagicMock(),
            model=MagicMock(),
            model_info=VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
            workflow=GenerationWorkflow(name="t", stages=[MagicMock(side_effect=_stage)]),
            prompts_data={"v": [("a fox runs", None)]},
            config={},
            args=args,
            progress_callback=events.append,
            prompt_enhancer=_FakeEnhancer(),
        )
        assert requests[0].enhance == override and requests[0].enhance_ceiling == 300
        assert any(e["type"] == "prompt_enhanced" and e["mode"] == "video" for e in events)


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
        assert options[0]["enhance"] == "style=keep,details=lighting+composition,length=longer,motion=action"
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


class TestImageCliLifecycle:
    """The CLI preflights before loading the image model and holds the enhancer for the batch."""

    def test_enhance_order_and_release(self, monkeypatch):
        from zvisiongenerator import image_cli
        from zvisiongenerator.utils.prompts import PromptFileInspection

        order: list[str] = []
        backend = MagicMock(name="mflux")
        backend.name = "mflux"
        backend.load_model.side_effect = lambda *a, **k: (order.append("load_model"), (MagicMock(), MagicMock(family="zimage")))[1]
        session = MagicMock()
        session.acquire.return_value.__enter__.return_value = "ENHANCER"
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
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, rev: order.append(f"preflight:{repo}") or True)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
        captured = {}
        monkeypatch.setattr(image_cli, "run_batch", lambda *a, **k: (order.append("run_batch"), captured.update(k)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_cli.main()
        assert order == ["preflight:a/b", "load_model", "run_batch"]
        assert captured["prompt_enhancer"] == "ENHANCER"
        session.release.assert_called_once()

    def test_generation_errors_are_not_reported_as_enhancer_failures(self, monkeypatch):
        """A RuntimeError from the batch propagates; only enhancer loading becomes a usage error."""
        self._patch(monkeypatch, run_batch=MagicMock(side_effect=RuntimeError("MPS out of memory")))
        from zvisiongenerator import image_cli

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="MPS out of memory"):
                image_cli.main()

    def test_enhancer_load_error_is_a_usage_error(self, monkeypatch):
        session = self._patch(monkeypatch, run_batch=MagicMock())
        session.acquire.return_value.__enter__.side_effect = RuntimeError("Could not load prompt enhancer model a/b")
        from zvisiongenerator import image_cli

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(SystemExit):
                image_cli.main()
        session.release.assert_called_once()

    def _patch(self, monkeypatch, *, run_batch):
        from zvisiongenerator import image_cli
        from zvisiongenerator.utils.prompts import PromptFileInspection

        backend = MagicMock(name="mflux")
        backend.name = "mflux"
        backend.load_model.return_value = (MagicMock(), MagicMock(family="zimage"))
        session = MagicMock()
        session.acquire.return_value.__enter__.return_value = "ENHANCER"
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
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, rev: True)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr(image_cli, "run_batch", run_batch)
        return session
