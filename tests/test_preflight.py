"""Tests for the preflight phase: iteration planning, the enhancer lifecycle, controls and events."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
from zvisiongenerator.core.job_plan import EnhanceStatus
from zvisiongenerator.preflight import plan_iterations, run_preflight
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings


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


def _types(events):
    return [event["type"] for event in events]


# ── Lifecycle and rewrite rules ─────────────────────────────────────────────


class TestRunPreflight:
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

    def test_ceiling_comes_from_model_family(self):
        enhancer = _FakeEnhancer([])
        _run({"a": [("fox " * 100, None)]}, enhancer=enhancer, args=_args(enhance=EnhanceSettings(length="extra")), family="flux1")
        assert "153-180 words" in enhancer.systems[0]

    def test_video_mode_prompt(self):
        enhancer = _FakeEnhancer([])
        _run({"v": [("a fox runs", None)]}, enhancer=enhancer, mode="video")
        assert "text-to-video" in enhancer.systems[0]

    def test_json_captions_are_not_enhanced(self):
        session = MagicMock()
        plan, *_ = _run({"prompt": [('{"a": 1}', None)]}, args=_args(json_prompt_enabled=True), session=session)
        session.acquire.assert_not_called()
        assert plan.iterations[0].enhance is None

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            run_preflight({}, {}, _args(), mode="audio", model_family=None)

    def test_default_release_patch_target(self, monkeypatch):
        released = MagicMock()
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", released)
        run_preflight({"a": [("x", None)]}, _CONFIG, _args(), mode="image", model_family=None, session=_session(_FakeEnhancer([])))
        released.assert_called_once()

    def test_default_release_error_becomes_warning(self, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", MagicMock(side_effect=RuntimeError("metal gone")))
        with pytest.warns(UserWarning, match="metal gone"):
            plan = run_preflight({"a": [("x", None)]}, _CONFIG, _args(), mode="image", model_family=None, session=_session(_FakeEnhancer([])))
        assert plan.has_rewrites


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
        _plan, events, *_ = _run(prompts, args=_args(enhance=None), enhance_by_set={"a": [None, EnhanceSettings(), None]}, mode="video")
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
