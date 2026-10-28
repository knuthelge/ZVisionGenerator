"""Tests for the preflight phase: iteration planning, the enhancer lifecycle, controls and events."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import make_extend_source

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.job_plan import EnhancerUsed, EnhanceStatus
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.utils.config import load_config, video_family
from zvisiongenerator.preflight import _iteration_context, _job_context, plan_iterations, run_preflight
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, entry_scene, resolve_enhance_ceiling
from zvisiongenerator.utils.prompt_enhance_video import CONTINUE_NATURALLY


def _plan(prompts_data, **overrides):
    values = dict(runs=1, seed=42, seed_min=4, seed_max=2**32 - 1, json_prompt=False, disabled=False, override=None, enhance_by_set=None)
    values.update(overrides)
    return plan_iterations(prompts_data, **values)


class TestPlanIterations:
    def test_order_is_runs_sets_prompts(self):
        plan = _plan({"a": [("a1", None), ("a2", None)], "b": [("b1", "neg")]}, runs=2)
        assert [(it.run_index, it.set_name, it.prompt_index, it.prompt) for it in plan] == [
            (0, "a", 0, "a1"),
            (0, "a", 1, "a2"),
            (0, "b", 0, "b1"),
            (1, "a", 0, "a1"),
            (1, "a", 1, "a2"),
            (1, "b", 0, "b1"),
        ]

    def test_fixed_seed(self):
        assert {it.seed for it in _plan({"a": [("x", None), ("y", None)]}, seed=9)} == {9}

    def test_random_seed_uses_bounds(self):
        with patch("zvisiongenerator.preflight.random.randint", side_effect=[11, 12]) as randint:
            plan = _plan({"a": [("x", None), ("y", None)]}, seed=None, seed_min=5, seed_max=50)
        assert [it.seed for it in plan] == [11, 12]
        randint.assert_called_with(5, 50)

    def test_random_choices_expanded(self):
        (iteration,) = _plan({"a": [("{red|red} fox", None)]})
        assert iteration.prompt == "{red|red} fox" and iteration.resolved_prompt == "red fox"

    def test_json_caption_verbatim_and_never_enhanced(self):
        caption = '{"a": "{x|y}"}'
        (iteration,) = _plan({"prompt": [(caption, None)]}, json_prompt=True, override=EnhanceSettings())
        assert iteration.resolved_prompt == caption and iteration.enhance is None

    def test_enhance_precedence(self):
        entry, override = EnhanceSettings(style="anime"), EnhanceSettings(style="photo")
        by_set = {"a": [entry, None]}
        data = {"a": [("x", None), ("y", None)]}
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set)] == [entry, None]
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set, override=override)] == [override, override]
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set, override=override, disabled=True)] == [None, None]

    def test_starts_unenhanced(self):
        (iteration,) = _plan({"a": [("x", None)]}, override=EnhanceSettings())
        assert iteration.enhance_status is EnhanceStatus.OFF and iteration.enhanced_prompt is None

    def test_scene_defaults_to_none(self):
        (iteration,) = _plan({"a": [("x", None)]})
        assert iteration.scene is None and iteration.resolved_scene is None and iteration.enhanced_scene is None

    def test_job_scene_applies_to_every_entry(self):
        plan = _plan({"a": [("x", None), ("y", None)]}, scene="  A castle at dusk.  ")
        assert [(it.scene, it.resolved_scene) for it in plan] == [("A castle at dusk.", "A castle at dusk.")] * 2

    def test_entry_scene_beats_job_scene(self):
        plan = _plan({"a": [("x", None), ("y", None), ("z", None)]}, scene="Job scene.", scene_by_set={"a": [None, "Entry scene.", "  "]})
        assert [it.scene for it in plan] == ["Job scene.", "Entry scene.", "Job scene."]

    def test_scene_lookup_is_aligned_per_set_and_index(self):
        plan = _plan({"a": [("x", None)], "b": [("y", None)]}, runs=2, scene_by_set={"b": ["Scene B."]})
        assert [(it.run_index, it.set_name, it.scene) for it in plan] == [(0, "a", None), (0, "b", "Scene B."), (1, "a", None), (1, "b", "Scene B.")]

    @pytest.mark.parametrize(
        ("own", "job", "expected"),
        [(None, None, None), ("  ", " ", None), (" Own ", "Job", "Own"), ("", " Job ", "Job"), ("  ", "Job", "Job")],
    )
    def test_entry_scene_trims_and_skips_blanks(self, own, job, expected):
        assert entry_scene({"a": [own]}, "a", 0, job) == expected

    def test_entry_scene_out_of_range_or_unknown_set_uses_the_job_scene(self):
        assert entry_scene({"a": ["x"]}, "a", 5, "Job") == "Job"
        assert entry_scene({"a": ["x"]}, "b", 0, None) is None
        assert entry_scene(None, "a", 0, "Job") == "Job"

    def test_scene_choices_are_resolved_apart_from_the_prompt(self):
        (iteration,) = _plan({"a": [("a {red|red} fox", None)]}, scene="A {night|night} street.")
        assert iteration.scene == "A {night|night} street." and iteration.resolved_scene == "A night street."
        assert iteration.prompt == "a {red|red} fox" and iteration.resolved_prompt == "a red fox"

    def test_scene_only_entry_plans_with_an_empty_prompt(self):
        (iteration,) = _plan({"prompt": [("", None)]}, scene="Night.")
        assert iteration.resolved_prompt == "" and iteration.resolved_scene == "Night."

    def test_json_caption_has_no_scene(self):
        (iteration,) = _plan({"prompt": [('{"a": 1}', None)]}, json_prompt=True, scene="Night.", scene_by_set={"prompt": ["Own."]})
        assert iteration.scene is None and iteration.resolved_scene is None

    def test_empty_input(self):
        assert _plan({}) == ()
        assert _plan({"a": []}) == ()
        assert _plan({"a": [("x", None)]}, runs=0) == ()


# ── run_preflight helpers ───────────────────────────────────────────────────

_CONFIG = {
    "generation": {"seed_min": 1, "seed_max": 100},
    "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
    "model_presets": {"flux1": {"enhance_max_words": 180}},
}


class _FakeEnhancer:
    """Yield a rewrite per call; *during* holds a step (or list of steps) run mid-generate per call."""

    def __init__(self, order: list[str], *, texts=None, during=None, on_cuda=True):
        self.order = order
        self.texts = list(texts or [])
        self.during = list(during or [])
        self.on_cuda = on_cuda
        self.calls: list[int] = []
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.calls.append(seed)
        self.prompts.append(messages[-1]["content"])
        self.systems.append(messages[0]["content"])
        hook = self.during.pop(0) if self.during else None
        # Each step is followed by a cancellation poll, like the per-token poll of a real LLM;
        # steps left after a stop run as keys pressed after the LLM stopped.
        stopped = False
        for step in hook if isinstance(hook, list) else [hook]:
            if step is not None:
                step()
            stopped = stopped or (cancelled is not None and cancelled())
        if stopped:
            return
        text = self.texts.pop(0) if self.texts else f"A vivid rewrite number {len(self.calls)} with soft light."
        if isinstance(text, Exception):
            raise text
        yield text

    def close(self) -> None:
        self.order.append("close")


def _session(enhancer, *, downloaded=True):
    return PromptEnhancerSession(lambda repo, revision: enhancer, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: downloaded)


def _args(**overrides):
    values = dict(runs=1, seed=None, no_enhance=False, enhance=EnhanceSettings(), enhance_model=None)
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def _available(monkeypatch):
    monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)


def _run(prompts=None, *, enhancer=None, order=None, args=None, control=None, family="zimage", mode="image", session=None, **kwargs):
    order = order if order is not None else []
    enhancer = enhancer or _FakeEnhancer(order)
    events: list[dict] = []
    release = MagicMock(side_effect=lambda: order.append("release_memory"))

    def _record(event):
        events.append(event)
        if event["type"] == "batch_cancelled":
            order.append("batch_cancelled")

    plan = run_preflight(
        prompts if prompts is not None else {"a": [("a red fox", None), ("a grey wolf", None)]},
        _CONFIG,
        args or _args(),
        mode=mode,
        model_family=family,
        control=control,
        progress_callback=_record,
        session=session or _session(enhancer),
        release_memory=release,
        **kwargs,
    )
    return plan, events, release, enhancer


_PAIR_TEXT = "SCENE: A snowy field at dusk, soft light.\nACTION: A red fox runs across it."


def _capture_rewrite(monkeypatch, name):
    """Replace the preflight's enhance function *name* with one that records its keyword arguments and returns a fixed rewrite."""
    captured: list[dict] = []

    def _fake(*_args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(prompt="A vivid rewrite with soft light.", scene="A snowy field at dusk.", format_fallback=False)

    monkeypatch.setattr(f"zvisiongenerator.preflight.{name}", _fake)
    return captured


def _types(events):
    return [event["type"] for event in events]


# ── Lifecycle and rewrite rules ─────────────────────────────────────────────


class TestRunPreflight:
    def test_json_format_writes_captions(self):
        reply = "SUMMARY: A fox.\nAESTHETICS: calm\nLIGHTING: soft\nCAMERA: 50mm\nCOLORS: red\nBACKGROUND: Snow.\nOBJECT: A red fox."
        enhancer = _FakeEnhancer([], texts=[reply, reply])
        plan, _events, _release, _enhancer = _run(enhancer=enhancer, args=_args(enhance_format="json"), family="ideogram4")
        assert all(json.loads(it.enhanced_prompt)["high_level_description"] == "A fox." for it in plan.iterations)

    def test_text_format_stays_prose(self):
        plan, _events, _release, _enhancer = _run(args=_args(enhance_format=None))
        assert all(not it.enhanced_prompt.startswith("{") for it in plan.iterations)

    def test_plan_names_the_enhancer_that_ran(self):
        plan, _events, _release, _enhancer = _run()
        assert plan.enhancer == EnhancerUsed("a/b", None, tier="fast", format="text")
        quiet, _events, _release, _enhancer = _run(args=_args(no_enhance=True))
        assert quiet.enhancer is None

    def test_rewrites_then_closes_then_releases(self):
        order: list[str] = []
        plan, _events, release, enhancer = _run(order=order)
        assert [it.enhance_status for it in plan.iterations] == [EnhanceStatus.ENHANCED, EnhanceStatus.ENHANCED]
        assert all(it.enhanced_prompt for it in plan.iterations)
        assert order == ["close", "release_memory"]
        release.assert_called_once()
        assert not plan.cancelled and plan.has_rewrites

    def test_one_call_per_enhanced_iteration_with_its_seed(self):
        prompts = {"a": [("a {red|red} fox", None), ("plain", None)]}
        plan, _events, _release, enhancer = _run(prompts, args=_args(enhance=None, runs=2), enhance_by_set={"a": [EnhanceSettings(), None]})
        enhanced = [it for it in plan.iterations if it.enhance is not None]
        assert enhancer.calls == [it.seed for it in enhanced]
        assert len(enhancer.calls) == 2
        assert all("a red fox" in prompt for prompt in enhancer.prompts)
        assert [it.enhance_status for it in plan.iterations] == [EnhanceStatus.ENHANCED, EnhanceStatus.OFF] * 2

    def test_no_rewrites_loads_nothing_and_unloads_resident(self):
        session = MagicMock()
        plan, events, release, _ = _run(args=_args(enhance=None), session=session)
        session.acquire.assert_not_called()
        session.release.assert_called_once()
        release.assert_not_called()
        assert _types(events) == ["preflight_started", "preflight_finished"]
        assert all(it.enhance_status is EnhanceStatus.OFF for it in plan.iterations)

    def test_no_rewrites_without_session_releases_process_enhancer(self, monkeypatch):
        released = MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.release_resident_enhancer", released)
        run_preflight({"a": [("x", None)]}, _CONFIG, _args(no_enhance=True), mode="image", model_family=None, release_memory=MagicMock())
        released.assert_called_once()

    def test_pre_resident_enhancer_is_reused_silently(self):
        order: list[str] = []
        enhancer = _FakeEnhancer(order)
        session = _session(enhancer)
        with session.acquire("a/b", None, idle_seconds=None):
            pass
        plan, events, release, _ = _run(order=order, enhancer=enhancer, session=session)
        assert "enhancer_loading" not in _types(events)
        assert plan.has_rewrites
        release.assert_called_once()

    def test_load_phase_events(self):
        _plan, events, *_ = _run(session=_session(_FakeEnhancer([]), downloaded=False))
        assert [e for e in events if e["type"] == "enhancer_loading"] == [{"type": "enhancer_loading", "mode": "image", "phase": "downloading"}]

    def test_the_download_hook_runs_with_the_planned_enhancer_before_it_loads(self):
        order: list[str] = []
        calls: list[tuple[str, str | None]] = []
        enhancer = _FakeEnhancer(order)
        session = PromptEnhancerSession(lambda repo, revision: (order.append("load"), enhancer)[1], scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)

        def hook(repo, revision):
            calls.append((repo, revision))
            order.append("ensure")
            return True

        plan, events, _release, _enhancer = _run(session=session, ensure_downloads=hook)
        assert calls == [("a/b", None)]
        assert order[:2] == ["ensure", "load"]
        assert plan.has_rewrites and not plan.cancelled
        assert [e["phase"] for e in events if e["type"] == "enhancer_loading"] == ["loading"]

    def test_a_hook_that_ends_the_job_cancels_without_loading_or_emitting_a_cancellation(self):
        session = MagicMock()
        plan, events, release, _enhancer = _run(session=session, ensure_downloads=lambda repo, revision: False)
        assert plan.cancelled and plan.enhancer is None
        assert len(plan.iterations) == 2
        session.acquire.assert_not_called()
        release.assert_not_called()
        assert _types(events) == ["preflight_started"]

    def test_the_hook_is_not_called_when_nothing_is_rewritten(self):
        hook = MagicMock(return_value=True)
        _run(args=_args(enhance=None), session=MagicMock(), ensure_downloads=hook)
        hook.assert_not_called()

    def test_without_a_hook_a_first_use_download_is_the_downloading_load_phase(self):
        _plan, events, *_ = _run(session=_session(_FakeEnhancer([]), downloaded=False), ensure_downloads=None)
        assert [e["phase"] for e in events if e["type"] == "enhancer_loading"] == ["downloading"]

    def test_cpu_phase_event(self):
        enhancer = _FakeEnhancer([], on_cuda=False)
        _plan, events, *_ = _run(enhancer=enhancer)
        phases = [e["phase"] for e in events if e["type"] == "enhancer_loading"]
        assert phases == ["loading", "cpu"]

    def test_plan_job_enhancer_error_loads_and_releases_nothing(self):
        session = MagicMock()
        release = MagicMock()
        with pytest.raises(ValueError, match="enhancer"):
            run_preflight({"a": [("x", None)]}, {}, _args(), mode="image", model_family=None, session=session, release_memory=release)
        session.acquire.assert_not_called()
        session.release.assert_not_called()
        release.assert_not_called()

    def test_load_failure_propagates_without_release_memory(self):
        session = MagicMock()
        session.acquire.return_value.__enter__.side_effect = RuntimeError("Could not load prompt enhancer model a/b")
        release = MagicMock()
        with pytest.raises(RuntimeError, match="Could not load"):
            run_preflight({"a": [("x", None)]}, _CONFIG, _args(), mode="image", model_family=None, session=session, release_memory=release)
        release.assert_not_called()
        session.release.assert_called_once()

    def test_failure_falls_back_and_continues(self):
        order: list[str] = []
        enhancer = _FakeEnhancer(order, texts=[TypeError("chat template rejected the system role")])
        with pytest.warns(UserWarning, match="Prompt enhancement failed \\(chat template rejected the system role\\); using the original prompt"):
            plan, events, *_ = _run(order=order, enhancer=enhancer)
        assert [it.enhance_status for it in plan.iterations] == [EnhanceStatus.FAILED, EnhanceStatus.ENHANCED]
        assert plan.iterations[0].enhanced_prompt is None
        failed = [e for e in events if e["type"] == "prompt_enhance_failed"]
        assert failed == [{"type": "prompt_enhance_failed", "mode": "image", "index": 1, "total": 2, "message": "chat template rejected the system role"}]

    def test_ceiling_comes_from_model_family(self, monkeypatch):
        captured = _capture_rewrite(monkeypatch, "enhance_prompt")
        _run({"a": [("fox " * 100, None)]}, args=_args(enhance=EnhanceSettings(length="extra")), family="flux1")
        _run({"a": [("fox " * 100, None)]}, args=_args(enhance=EnhanceSettings(length="extra")), family="zimage")
        assert [call["ceiling"] for call in captured] == [180, resolve_enhance_ceiling(_CONFIG, family="zimage", mode="image")]
        assert captured[0]["ceiling"] != captured[1]["ceiling"]

    def test_video_jobs_are_rewritten_as_scene_and_action_pairs(self, monkeypatch):
        images = _capture_rewrite(monkeypatch, "enhance_prompt")
        videos = _capture_rewrite(monkeypatch, "enhance_video_prompt")
        _run({"v": [("a fox runs", None)]}, mode="video")
        assert (len(images), len(videos)) == (0, 1)

    def test_video_job_shape_reaches_the_rewrite_context(self, monkeypatch):
        captured = _capture_rewrite(monkeypatch, "enhance_video_prompt")
        args = _args(enhance=EnhanceSettings(sound=("ambient",)), width=704, height=448, num_frames=97, fps=24, no_audio=False, image_path="first.png")
        _run({"v": [("a fox runs", None)]}, mode="video", args=args)
        context = captured[0]["context"]
        assert (context.audio, context.orientation, context.reference_image) == (True, "landscape", True)
        assert context.seconds == pytest.approx(97 / 24)

    def test_video_without_audio_has_no_audio_in_the_context(self, monkeypatch):
        captured = _capture_rewrite(monkeypatch, "enhance_video_prompt")
        _run({"v": [("a fox runs", None)]}, mode="video", args=_args(enhance=EnhanceSettings(sound=("ambient",)), no_audio=True))
        assert captured[0]["context"].audio is False

    def test_image_ratio_sets_the_frame_in_the_context(self, monkeypatch):
        captured = _capture_rewrite(monkeypatch, "enhance_prompt")
        _run({"a": [("a fox", None)]}, args=_args(ratio="9:16", width=None, height=None))
        assert captured[0]["context"].orientation == "portrait"

    def test_json_captions_are_not_enhanced(self):
        session = MagicMock()
        plan, *_ = _run({"prompt": [('{"a": 1}', None)]}, args=_args(json_prompt_enabled=True), session=session)
        session.acquire.assert_not_called()
        assert plan.iterations[0].enhance is None

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            run_preflight({}, {}, _args(), mode="audio", model_family=None)

    def test_default_release_error_becomes_warning(self, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", MagicMock(side_effect=RuntimeError("metal gone")))
        with pytest.warns(UserWarning, match="metal gone"):
            plan = run_preflight({"a": [("x", None)]}, _CONFIG, _args(), mode="image", model_family=None, session=_session(_FakeEnhancer([])))
        assert plan.has_rewrites


# ── Video scene ─────────────────────────────────────────────────────────────


class TestVideoScene:
    def test_pair_lands_in_the_plan_with_action_only_in_enhanced_prompt(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        plan, *_ = _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video", scene="A snowy field.")
        (iteration,) = plan.iterations
        assert iteration.enhance_status is EnhanceStatus.ENHANCED
        assert iteration.enhanced_prompt == "A red fox runs across it."
        assert iteration.enhanced_scene == "A snowy field at dusk, soft light."
        assert "snowy" not in iteration.enhanced_prompt

    def test_enhancer_receives_the_scene_and_prompt_with_choices_resolved(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        _run({"v": [("a {fox|fox} runs", None)]}, enhancer=enhancer, mode="video", scene="A {snowy|snowy} field.")
        assert enhancer.prompts[0] == "Scene: A snowy field.\nPrompt: a fox runs"

    def test_entry_scene_beats_the_job_scene_in_the_rewrite(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT, _PAIR_TEXT])
        _run({"v": [("a fox runs", None), ("a wolf runs", None)]}, enhancer=enhancer, mode="video", scene="Job scene.", scene_by_set={"v": ["Entry scene.", None]})
        assert [p.splitlines()[0] for p in enhancer.prompts] == ["Scene: Entry scene.", "Scene: Job scene."]

    def test_blank_scene_splits_the_prompt(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        plan, *_ = _run({"v": [("a fox runs in a snowy field", None)]}, enhancer=enhancer, mode="video")
        assert enhancer.prompts[0].splitlines()[0] == "Scene: (none)"
        assert plan.iterations[0].enhanced_scene == "A snowy field at dusk, soft light."

    def test_scene_only_run_is_enhanced(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        plan, *_ = _run({"prompt": [("", None)]}, enhancer=enhancer, mode="video", scene="A snowy field.")
        assert enhancer.prompts[0] == "Scene: A snowy field.\nPrompt: (none)"
        assert plan.iterations[0].enhanced_prompt == "A red fox runs across it."

    def test_unformatted_reply_keeps_the_scene_and_warns(self):
        """The fallback does not rewrite the scene, so no enhanced scene is planned (it once echoed the user's scene)."""
        reply = "A fox sprints across the snow and the camera follows."
        enhancer = _FakeEnhancer([], texts=[reply, reply])
        with pytest.warns(UserWarning, match="did not return a scene and an action"):
            plan, *_ = _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video", scene="A snowy field.")
        assert plan.iterations[0].enhanced_scene is None
        assert plan.iterations[0].enhanced_prompt == reply

    def test_action_only_keeps_the_scene_and_plans_no_enhanced_scene(self):
        action = "A red fox sprints across the snow, the camera tracking low beside it."
        enhancer = _FakeEnhancer([], texts=[action])
        args = _args(enhance=EnhanceSettings(parts="action"))
        plan, *_ = _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video", args=args, scene="A snowy field.")
        (iteration,) = plan.iterations
        assert iteration.enhance_status is EnhanceStatus.ENHANCED
        assert iteration.enhanced_scene is None and iteration.enhanced_prompt == action
        assert iteration.scene == "A snowy field." and enhancer.prompts[0] == "a fox runs"
        assert "A snowy field." in enhancer.systems[0]

    def test_action_only_with_a_blank_scene_splits_and_warns_once_per_iteration(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        args = _args(enhance=EnhanceSettings(parts="action"))
        with pytest.warns(UserWarning, match="parts=action needs a scene"):
            plan, *_ = _run({"v": [("a fox runs in a snowy field", None)]}, enhancer=enhancer, mode="video", args=args)
        assert plan.iterations[0].enhanced_scene == "A snowy field at dusk, soft light."
        assert plan.iterations[0].enhanced_prompt == "A red fox runs across it."

    def test_both_rewrites_the_scene_whatever_the_workflow(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        args = _args(enhance=EnhanceSettings(parts="both"))
        plan, *_ = _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video", args=args, scene="A snowy field.")
        assert plan.iterations[0].enhanced_scene == "A snowy field at dusk, soft light."

    def test_failed_rewrite_leaves_the_scene_unenhanced(self):
        enhancer = _FakeEnhancer([], texts=[TypeError("template rejected")])
        with pytest.warns(UserWarning, match="Prompt enhancement failed"):
            plan, *_ = _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video", scene="A snowy field.")
        (iteration,) = plan.iterations
        assert iteration.enhance_status is EnhanceStatus.FAILED and iteration.enhanced_scene is None
        assert iteration.scene == "A snowy field." and iteration.resolved_scene == "A snowy field."

    def test_no_enhancement_still_plans_the_scene(self):
        plan, *_ = _run({"v": [("a fox runs", None)]}, args=_args(enhance=None), mode="video", scene="A snowy field.", session=MagicMock())
        (iteration,) = plan.iterations
        assert iteration.scene == "A snowy field." and iteration.enhance_status is EnhanceStatus.OFF

    def test_image_jobs_ignore_the_scene(self):
        enhancer = _FakeEnhancer([])
        plan, *_ = _run({"a": [("a red fox", None)]}, enhancer=enhancer, scene="A snowy field.", scene_by_set={"a": ["Own."]})
        (iteration,) = plan.iterations
        assert iteration.scene is None and iteration.enhanced_scene is None
        assert "snowy" not in enhancer.prompts[0] and "Own." not in enhancer.prompts[0]


class TestVideoExtend:
    """Verify the rewrite context and prompts of a job that extends a clip."""

    _OLD = "A woman in a red raincoat stands on a rainy street. She says to the camera that it is late."
    _SCENE = "A woman in a red raincoat on a rainy street at night."
    _ACTION_TEXT = "She turns and walks slowly down the wet pavement, her umbrella tilting against the rain."

    def _source(self, **overrides) -> ExtendSource:
        return make_extend_source(**{"scene": self._SCENE, **overrides})

    def _extend_args(self, source=None, **overrides):
        source = source or self._source()
        values = dict(extend_source=source, extend_frames=48, width=source.width, height=source.height, num_frames=25 + 48, no_audio=False)
        values.update(overrides)
        return _args(**values)

    # -- the job context ---------------------------------------------------

    def test_context_describes_the_window_at_the_source_frame_rate(self):
        context = _job_context(self._extend_args(), mode="video")
        assert context.extends is True
        assert context.seconds == pytest.approx(73 / 24)
        assert context.new_seconds == pytest.approx(2.0)
        assert context.orientation == "landscape"
        assert context.reference_image is False
        assert context.audio is True

    def test_context_spans_the_window_of_the_family(self):
        family = video_family(load_config(), "h3")

        context = _job_context(self._extend_args(extend_frames=17, num_frames=39), mode="video", family=family)

        assert context.seconds == pytest.approx(39 / 24)
        assert context.new_seconds == pytest.approx(17 / 24)

    def test_context_knows_about_a_last_frame(self):
        with_both = _job_context(_args(width=704, height=448, num_frames=97, fps=24, no_audio=True, image_path="a.png", end_image_path="b.png"), mode="video")
        without = _job_context(_args(width=704, height=448, num_frames=97, fps=24, no_audio=True, image_path="a.png"), mode="video")

        assert (with_both.reference_image, with_both.end_image) == (True, True)
        assert (without.reference_image, without.end_image) == (True, False)

    def test_context_counts_the_references_apart_from_the_start_image(self):
        references = _job_context(_args(width=512, height=512, reference_paths=["a.png", "b.png"]), mode="image")
        start_only = _job_context(_args(width=512, height=512, image_path="a.png"), mode="image")

        assert (references.reference_image, references.references) == (False, 2)
        assert (start_only.reference_image, start_only.references) == (True, 0)

    def test_other_jobs_do_not_extend(self):
        context = _job_context(_args(width=704, height=448, num_frames=97, fps=24, no_audio=False), mode="video")
        assert context.extends is False and context.previous is None and context.new_seconds is None
        assert context.seconds == pytest.approx(97 / 24)

    def test_image_jobs_ignore_the_extend_arguments(self):
        assert _job_context(self._extend_args(), mode="image").extends is False

    def test_prompt_is_original_is_decided_per_prompt(self):
        context = _job_context(self._extend_args(self._source(scene=None, prompt=self._OLD)), mode="video")
        unchanged = _plan({"a": [(f"  {self._OLD}\n", None)]})[0]
        edited = _plan({"a": [("She waves.", None)]})[0]
        assert _iteration_context(context, unchanged).prompt_is_original is True
        assert _iteration_context(context, edited).prompt_is_original is False
        scene_context = _job_context(self._extend_args(), mode="video")
        assert _iteration_context(scene_context, unchanged) is scene_context

    # -- the rewrite -------------------------------------------------------

    def test_scene_is_kept_and_only_the_action_is_written(self):
        """The kept scene is not an enhanced scene: the plan carries none, so the user's scene is the one used."""
        enhancer = _FakeEnhancer([], texts=[self._ACTION_TEXT])
        plan, *_ = _run({"v": [("she leaves", None)]}, enhancer=enhancer, mode="video", args=self._extend_args(), scene=self._SCENE)
        (iteration,) = plan.iterations
        assert iteration.enhance_status is EnhanceStatus.ENHANCED
        assert iteration.enhanced_scene is None and iteration.enhanced_prompt == self._ACTION_TEXT
        assert enhancer.prompts[0] == "she leaves"
        assert self._SCENE in enhancer.systems[0] and "She opens a black umbrella." in enhancer.systems[0]

    def test_an_empty_continuation_is_allowed_and_asks_to_continue(self):
        enhancer = _FakeEnhancer([], texts=[self._ACTION_TEXT])
        plan, *_ = _run({"v": [("", None)]}, enhancer=enhancer, mode="video", args=self._extend_args(), scene=self._SCENE)
        assert enhancer.prompts[0] == CONTINUE_NATURALLY
        assert plan.iterations[0].enhance_status is EnhanceStatus.ENHANCED

    def test_an_empty_scene_and_prompt_are_allowed_for_a_source_without_a_scene(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        args = self._extend_args(self._source(scene=None, prompt=None))
        plan, *_ = _run({"v": [("", None)]}, enhancer=enhancer, mode="video", args=args)
        assert plan.iterations[0].enhance_status is EnhanceStatus.ENHANCED
        assert plan.iterations[0].enhanced_scene == "A snowy field at dusk, soft light."

    def test_an_empty_prompt_and_scene_still_fail_outside_extend(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        with pytest.warns(UserWarning, match="Enter a scene or a prompt"):
            plan, *_ = _run({"v": [("", None)]}, enhancer=enhancer, mode="video", args=_args(width=704, height=448, num_frames=97, fps=24))
        assert plan.iterations[0].enhance_status is EnhanceStatus.FAILED
        assert enhancer.calls == []

    def test_legacy_source_with_the_unchanged_prompt_extracts_the_scene_and_continues_naturally(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        args = self._extend_args(self._source(scene=None, prompt=self._OLD))
        plan, *_ = _run({"v": [(self._OLD, None)]}, enhancer=enhancer, mode="video", args=args)
        assert self._OLD in enhancer.systems[0]
        assert CONTINUE_NATURALLY in enhancer.prompts[0] and "She says to the camera" not in enhancer.prompts[0]
        assert plan.iterations[0].enhanced_scene == "A snowy field at dusk, soft light."
        assert plan.iterations[0].enhanced_prompt == "A red fox runs across it."

    def test_legacy_source_with_an_edited_prompt_sends_the_users_text(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        args = self._extend_args(self._source(scene=None, prompt=self._OLD))
        _run({"v": [("She picks up the umbrella.", None)]}, enhancer=enhancer, mode="video", args=args)
        assert "She picks up the umbrella." in enhancer.prompts[0] and CONTINUE_NATURALLY not in enhancer.prompts[0]

    def test_each_prompt_of_a_job_is_compared_to_the_old_prompt_on_its_own(self):
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT, _PAIR_TEXT])
        args = self._extend_args(self._source(scene=None, prompt=self._OLD))
        _run({"v": [(self._OLD, None), ("She picks up the umbrella.", None)]}, enhancer=enhancer, mode="video", args=args)
        assert CONTINUE_NATURALLY in enhancer.prompts[0] and CONTINUE_NATURALLY not in enhancer.prompts[1]

    def test_an_action_that_repeats_the_previous_clip_is_retried(self):
        enhancer = _FakeEnhancer([], texts=["She opens a black umbrella.", self._ACTION_TEXT])
        plan, *_ = _run({"v": [("she leaves", None)]}, enhancer=enhancer, mode="video", args=self._extend_args(), scene=self._SCENE)
        assert len(enhancer.calls) == 2 and plan.iterations[0].enhanced_prompt == self._ACTION_TEXT


# ── Controls ────────────────────────────────────────────────────────────────


def _resume_on_wait(control: SkipSignal, *, then: str | None = None):
    """Make wait_for_key return at once, optionally queueing *then* while paused."""

    def _wait():
        if then is not None:
            control.queue_action(then)

    control.wait_for_key = _wait


class TestPreflightControls:
    def test_next_mid_rewrite_skips_and_continues(self):
        control = SkipSignal()
        enhancer = _FakeEnhancer([], during=[lambda: control.queue_action("skip")])
        plan, events, *_ = _run(enhancer=enhancer, control=control)
        assert [it.enhance_status for it in plan.iterations] == [EnhanceStatus.SKIPPED, EnhanceStatus.ENHANCED]
        assert control.consume() is None
        assert "prompt_enhance_failed" not in _types(events)

    def test_next_at_boundary_makes_no_call(self):
        control = SkipSignal()
        control.queue_action("skip")
        plan, events, _release, enhancer = _run(enhancer=_FakeEnhancer([]), control=control)
        assert len(enhancer.calls) == 1
        assert [it.enhance_status for it in plan.iterations] == [EnhanceStatus.SKIPPED, EnhanceStatus.ENHANCED]
        assert _types(events).count("prompts_enhancing") == 2

    def test_next_during_last_rewrite_leaves_signal_empty(self):
        control = SkipSignal()
        enhancer = _FakeEnhancer([], during=[None, lambda: control.queue_action("skip")])
        plan, *_ = _run(enhancer=enhancer, control=control)
        assert plan.iterations[-1].enhance_status is EnhanceStatus.SKIPPED
        assert control.consume() is None

    def test_next_then_pause_mid_rewrite_is_skipped_and_pauses(self):
        control = SkipSignal()
        _resume_on_wait(control)

        enhancer = _FakeEnhancer([], during=[[lambda: control.queue_action("skip"), lambda: control.queue_action("pause")]])
        plan, events, *_ = _run(enhancer=enhancer, control=control)
        assert plan.iterations[0].enhance_status is EnhanceStatus.SKIPPED
        assert "prompt_enhance_failed" not in _types(events)
        assert _types(events).count("job_paused") == 1 and _types(events).count("job_resumed") == 1

    def test_pause_at_boundary_waits_then_continues(self):
        control = SkipSignal()
        _resume_on_wait(control)
        control.queue_action("pause")
        plan, events, *_ = _run(control=control)
        types = _types(events)
        assert types.index("job_paused") < types.index("job_resumed") < types.index("prompts_enhancing")
        assert plan.has_rewrites
        paused = next(e for e in events if e["type"] == "job_paused")
        assert paused == {"type": "job_paused", "mode": "image", "completed_iterations": 0, "total_iterations": 2}

    @pytest.mark.parametrize("where", ["boundary", "mid", "final"])
    def test_quit_cancels_after_release(self, where):
        order: list[str] = []
        control = SkipSignal()
        quit_now = lambda: control.queue_action("quit")  # noqa: E731
        if where == "boundary":
            control.queue_action("quit")
            enhancer = _FakeEnhancer(order)
        elif where == "mid":
            enhancer = _FakeEnhancer(order, during=[quit_now])
        else:
            # Pause is not an interrupt, so the rewrite finishes; queue quit during the last rewrite via pause→quit.
            enhancer = _FakeEnhancer(order, during=[None, lambda: control.queue_action("pause")])
            _resume_on_wait(control, then="quit")
        plan, events, release, _ = _run(order=order, enhancer=enhancer, control=control)
        assert plan.cancelled
        assert order[-3:] == ["close", "release_memory", "batch_cancelled"]
        release.assert_called_once()
        assert _types(events)[-1] == "batch_cancelled"
        assert "preflight_finished" not in _types(events)

    def test_final_boundary_pause_happens_after_release(self):
        order: list[str] = []
        control = SkipSignal()

        def _wait():
            order.append("paused")

        control.wait_for_key = _wait
        enhancer = _FakeEnhancer(order, during=[None, lambda: control.queue_action("pause")])
        plan, events, *_ = _run(order=order, enhancer=enhancer, control=control)
        assert order == ["close", "release_memory", "paused"]
        assert not plan.cancelled and _types(events)[-1] == "preflight_finished"

    def test_quit_while_paused(self):
        control = SkipSignal()
        _resume_on_wait(control, then="quit")
        control.queue_action("pause")
        plan, events, _release, enhancer = _run(control=control)
        assert plan.cancelled and enhancer.calls == []
        assert _types(events)[-1] == "batch_cancelled"

    def test_repeat_is_dropped(self):
        control = SkipSignal()
        control.queue_action("repeat")
        plan, *_ = _run(control=control)
        assert all(it.enhance_status is EnhanceStatus.ENHANCED for it in plan.iterations)
        assert control.consume() is None

    def test_failure_with_queued_pause_fails_then_pauses(self):
        control = SkipSignal()
        _resume_on_wait(control)
        enhancer = _FakeEnhancer([], texts=[RuntimeError("boom")], during=[lambda: control.queue_action("pause")])
        with pytest.warns(UserWarning, match="boom"):
            plan, events, *_ = _run(enhancer=enhancer, control=control)
        assert plan.iterations[0].enhance_status is EnhanceStatus.FAILED
        types = _types(events)
        assert types.index("prompt_enhance_failed") < types.index("job_paused")
        assert [i for i, t in enumerate(types) if t == "prompts_enhancing"][1] > types.index("job_resumed")

    def test_without_control(self):
        plan, *_ = _run(control=None)
        assert plan.has_rewrites and not plan.cancelled

    def test_no_rewrites_leaves_signal_untouched(self):
        control = SkipSignal()
        control.queue_action("skip")
        _run(args=_args(enhance=None), control=control, session=MagicMock())
        assert control.consume() == "skip"


# ── Events ──────────────────────────────────────────────────────────────────


class TestPreflightEvents:
    def test_sequence_and_payloads(self):
        _plan, events, *_ = _run(session=_session(_FakeEnhancer([]), downloaded=False))
        assert events == [
            {"type": "preflight_started", "mode": "image", "total_iterations": 2, "total_rewrites": 2},
            {"type": "enhancer_loading", "mode": "image", "phase": "downloading"},
            {"type": "prompts_enhancing", "mode": "image", "index": 1, "total": 2},
            {"type": "prompts_enhancing", "mode": "image", "index": 2, "total": 2},
            {"type": "preflight_finished", "mode": "image", "total_iterations": 2, "enhanced": 2, "failed": 0, "skipped": 0},
        ]

    def test_counts_only_rewrites(self):
        prompts = {"a": [("x", None), ("y", None), ("z", None)]}
        enhancer = _FakeEnhancer([], texts=[_PAIR_TEXT])
        _plan, events, *_ = _run(prompts, enhancer=enhancer, args=_args(enhance=None), enhance_by_set={"a": [None, EnhanceSettings(), None]}, mode="video")
        assert events[0] == {"type": "preflight_started", "mode": "video", "total_iterations": 3, "total_rewrites": 1}
        assert [e for e in events if e["type"] == "prompts_enhancing"] == [{"type": "prompts_enhancing", "mode": "video", "index": 1, "total": 1}]

    def test_empty_job(self):
        _plan, events, *_ = _run({}, session=MagicMock())
        assert events == [
            {"type": "preflight_started", "mode": "image", "total_iterations": 0, "total_rewrites": 0},
            {"type": "preflight_finished", "mode": "image", "total_iterations": 0, "enhanced": 0, "failed": 0, "skipped": 0},
        ]

    def test_no_timing_keys(self):
        control = SkipSignal()
        _resume_on_wait(control)
        control.queue_action("pause")
        enhancer = _FakeEnhancer([], texts=[RuntimeError("boom")])
        with pytest.warns(UserWarning):
            _plan, events, *_ = _run(enhancer=enhancer, control=control)
        assert {"job_paused", "prompt_enhance_failed", "preflight_finished"} <= set(_types(events))
        for event in events:
            assert not {"eta_secs", "avg_secs", "elapsed_secs"} & event.keys()
