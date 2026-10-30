"""The prompts run_batch() hands to backend.prepare_prompts(): what _job_prompts() enumerates and when the call happens."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tests._factories import _make_args, _make_plan

from zvisiongenerator.core import JobPrompt
from zvisiongenerator.core.job_plan import EnhanceStatus, JobPlan
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.image_runner import _job_prompts, run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.workflows import build_workflow

_MODEL_INFO = ImageModelInfo(family="zimage", is_distilled=False, size=None)


def _config(*, supports_negative: bool = True, max_references: int = 0) -> dict:
    return {
        "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
        "generation": {"seed_min": 1, "seed_max": 100},
        "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
        "model_presets": {"zimage": {"supports_negative_prompt": supports_negative, "max_reference_images": max_references}},
    }


def _plan(prompts, *, runs=1, **plan_args):
    return _make_plan(prompts, _make_args(runs=runs, **plan_args), config=_config())


def _enhance(plan: JobPlan, index: int, *, resolved: str | None = None, enhanced: str | None) -> JobPlan:
    iterations = list(plan.iterations)
    status = EnhanceStatus.ENHANCED if enhanced is not None else EnhanceStatus.OFF
    iterations[index] = replace(
        iterations[index],
        resolved_prompt=iterations[index].resolved_prompt if resolved is None else resolved,
        enhance=EnhanceSettings() if enhanced is not None else None,
        enhanced_prompt=enhanced,
        enhance_status=status,
    )
    return JobPlan(iterations=tuple(iterations))


class TestJobPrompts:
    def test_resolved_text_with_the_prompt_files_negative(self):
        prompts = {"s": [("a {red|blue} cat", "blurry")]}
        plan = _enhance(_plan(prompts), 0, resolved="a red cat", enhanced=None)
        assert _job_prompts(plan, prompts, supports_negative=True) == (JobPrompt("a red cat", "blurry"),)

    def test_enhanced_text_replaces_the_resolved_one(self):
        prompts = {"s": [("a cat", None)]}
        plan = _enhance(_plan(prompts), 0, resolved="a cat", enhanced="A tabby cat in warm light.")
        assert _job_prompts(plan, prompts, supports_negative=True) == (JobPrompt("A tabby cat in warm light.", None),)

    def test_an_empty_resolved_text_falls_back_to_the_template(self):
        prompts = {"s": [("a cat", None)]}
        plan = _enhance(_plan(prompts), 0, resolved="", enhanced=None)
        assert [p.prompt for p in _job_prompts(plan, prompts, supports_negative=True)] == ["a cat"]

    def test_an_empty_enhanced_text_falls_back_to_the_template(self):
        prompts = {"s": [("a cat", None)]}
        plan = _enhance(_plan(prompts), 0, resolved="a cat", enhanced="")
        assert [p.prompt for p in _job_prompts(plan, prompts, supports_negative=True)] == ["a cat"]

    def test_negatives_are_dropped_when_the_model_takes_none(self):
        prompts = {"s": [("a cat", "blurry")]}
        assert _job_prompts(_plan(prompts), prompts, supports_negative=False) == (JobPrompt("a cat", None),)

    def test_runs_and_repeated_prompts_are_listed_once_in_first_use_order(self):
        prompts = {"s": [("b", None), ("a", None), ("b", None)], "t": [("a", None)]}
        result = _job_prompts(_plan(prompts, runs=2), prompts, supports_negative=True)
        assert [p.prompt for p in result] == ["b", "a"]

    def test_same_text_with_different_negatives_stays_distinct(self):
        prompts = {"s": [("a", "x"), ("a", "y")]}
        result = _job_prompts(_plan(prompts), prompts, supports_negative=True)
        assert [p.negative_prompt for p in result] == ["x", "y"]

    def test_no_prompts_gives_no_entries(self):
        assert _job_prompts(JobPlan(iterations=()), {}, supports_negative=True) == ()


def _run(backend, model, prompts, args, *, plan=None, skip_signal=None, config=None, workflow=None):
    cfg = config or _config()
    plan = plan or _make_plan(prompts, args, config=cfg)
    patcher = patch("zvisiongenerator.image_runner.build_workflow", return_value=workflow) if workflow is not None else patch("zvisiongenerator.image_runner.build_workflow", wraps=build_workflow)
    with patcher:
        run_batch(backend, model, prompts, cfg, args, model_info=_MODEL_INFO, plan=plan, skip_signal=skip_signal)


def _noop_workflow() -> GenerationWorkflow:
    return GenerationWorkflow(name="t", stages=[MagicMock(return_value=StageOutcome.success)])


class TestPrepareSeam:
    def test_prompts_are_prepared_once_before_the_first_generation(self):
        backend = MagicMock()
        calls: list[str] = []
        backend.prepare_prompts.side_effect = lambda *a, **kw: calls.append("prepare")
        workflow = GenerationWorkflow(name="t", stages=[lambda r, a: calls.append("generate") or StageOutcome.success])
        model = MagicMock(spec=[])
        prompts = {"s": [("a cat", "blurry"), ("a dog", None)]}
        _run(backend, model, prompts, _make_args(runs=2, width=64, height=64), workflow=workflow)
        assert calls == ["prepare", *["generate"] * 4]
        backend.prepare_prompts.assert_called_once()
        assert backend.prepare_prompts.call_args.args[0] is model
        assert set(backend.prepare_prompts.call_args.args[1]) == {JobPrompt("a cat", "blurry"), JobPrompt("a dog", None)}
        assert (backend.prepare_prompts.call_args.kwargs["width"], backend.prepare_prompts.call_args.kwargs["height"]) == (64, 64)

    def test_nothing_is_prepared_without_a_model(self):
        backend = MagicMock()
        _run(backend, None, {"s": [("a cat", None)]}, _make_args(), workflow=_noop_workflow())
        backend.prepare_prompts.assert_not_called()

    def test_nothing_is_prepared_for_zero_iterations(self):
        backend = MagicMock()
        _run(backend, MagicMock(spec=[]), {}, _make_args(), plan=JobPlan(iterations=()), workflow=_noop_workflow())
        backend.prepare_prompts.assert_not_called()

    def test_cancelled_reads_a_queued_quit_without_consuming_it(self):
        skip = SkipSignal()
        backend = MagicMock()
        _run(backend, MagicMock(spec=[]), {"s": [("a cat", None)]}, _make_args(), skip_signal=skip, workflow=_noop_workflow())
        cancelled = backend.prepare_prompts.call_args.kwargs["cancelled"]
        assert cancelled() is False
        skip.queue_action("quit")
        assert cancelled() is True
        assert skip.pending() == "quit"

    def test_there_is_no_cancel_check_without_a_skip_signal(self):
        backend = MagicMock()
        _run(backend, MagicMock(spec=[]), {"s": [("a cat", None)]}, _make_args(), workflow=_noop_workflow())
        assert backend.prepare_prompts.call_args.kwargs["cancelled"] is None

    def test_a_quit_queued_during_preparing_cancels_the_batch(self):
        skip = SkipSignal()
        backend = MagicMock()
        backend.prepare_prompts.side_effect = lambda *a, **kw: skip.queue_action("quit")
        events: list[dict] = []
        workflow = _noop_workflow()
        prompts = {"s": [("a cat", None)]}
        args = _make_args()
        with patch("zvisiongenerator.image_runner.build_workflow", return_value=workflow):
            run_batch(
                backend, MagicMock(spec=[]), prompts, _config(), args, model_info=_MODEL_INFO, plan=_make_plan(prompts, args, config=_config()), skip_signal=skip, progress_callback=events.append
            )
        assert [e["type"] for e in events if e["type"].startswith("batch_")] == ["batch_started", "batch_cancelled"]
        workflow.stages[0].assert_not_called()


def _recording_backend() -> MagicMock:
    """Return a mock backend that returns images of the requested size and records its prompts."""
    backend = MagicMock()
    backend.name = "mock"
    backend.text_to_image.side_effect = lambda **kw: Image.new("RGB", (kw["width"], kw["height"]))
    backend.image_to_image.side_effect = lambda **kw: Image.new("RGB", kw["image"].size)
    return backend


def _with_one_retry(args) -> GenerationWorkflow:
    """Return the real workflow plus a last stage that asks for a retry on its first call."""
    workflow = build_workflow(args, enhance=True)
    outcomes = iter([StageOutcome.retry])
    workflow.stages.append(lambda request, artifacts: next(outcomes, StageOutcome.success))
    return workflow


def _prompts_sent(backend) -> list[tuple[str, str | None]]:
    calls = [*backend.text_to_image.call_args_list, *backend.image_to_image.call_args_list]
    return [(c.kwargs["prompt"], c.kwargs["negative_prompt"]) for c in calls]


@pytest.mark.parametrize(
    ("scenario", "supports_negative", "args_overrides", "retry"),
    [
        ("choices_runs_and_negatives", True, {"runs": 2, "width": 64, "height": 64}, False),
        ("negatives_the_model_ignores", False, {"runs": 2, "width": 64, "height": 64}, False),
        ("img2img", True, {"width": 64, "height": 64}, False),
        ("retry", True, {"width": 64, "height": 64}, True),
        ("upscale_of_an_unaligned_base_size", True, {"width": 1000, "height": 1000, "upscale": 2, "upscale_denoise": 0.3, "upscale_steps": 2}, False),
        ("upscale_of_a_preset_size", True, {"upscale": 2, "upscale_denoise": 0.3, "upscale_steps": 2, "upscale_guidance": 2.5}, False),
        ("review_before_the_upscale", True, {"width": 512, "height": 512, "upscale": 2, "upscale_denoise": 0.3, "upscale_steps": 2, "review": True}, False),
        ("references", True, {"width": 64, "height": 64}, False),
        ("references_with_an_upscale", True, {"width": 512, "height": 512, "upscale": 2, "upscale_denoise": 0.3, "upscale_steps": 2}, False),
        ("json_prompt", True, {"width": 64, "height": 64, "json_prompt_enabled": True}, False),
    ],
)
def test_every_prompt_and_generation_size_a_generation_sends_was_prepared(tmp_path, scenario, supports_negative, args_overrides, retry):
    config = _config(supports_negative=supports_negative, max_references=4 if scenario.startswith("references") else 0)
    prompts = {"s": [("a {red|blue|green|black} cat", "blurry"), ("a dog", None)], "t": [("a {x|y|z} bird", "")]}
    args = _make_args(output=str(tmp_path / "out"), sharpen=False, seed=None, **args_overrides)
    if scenario == "img2img":
        start = tmp_path / "start.png"
        Image.new("RGB", (80, 120)).save(start)
        args.image_path = str(start)
    if scenario.startswith("references"):
        args.reference_paths = []
        for index in range(2):
            reference = tmp_path / f"ref{index}.png"
            Image.new("RGB", (40 + index * 10, 60)).save(reference)
            args.reference_paths.append(str(reference))
    skip_signal = None
    if scenario == "review_before_the_upscale":
        skip_signal = SkipSignal()
        skip_signal.wait_for_review = MagicMock(return_value="approve")
    plan = _make_plan(prompts, args, config=config)
    plan = _enhance(plan, 1, enhanced="A loyal dog on a hill.")
    backend = _recording_backend()
    workflow = _with_one_retry(args) if retry else build_workflow(args, enhance=True)

    _run(backend, MagicMock(spec=[]), prompts, args, plan=plan, config=config, workflow=workflow, skip_signal=skip_signal)

    prepared = backend.prepare_prompts.call_args.args[1]
    assert backend.prepare_prompts.call_count == 1
    sent = _prompts_sent(backend)
    assert len(sent) >= len(plan.iterations) + (1 if retry else 0)
    assert set(sent) <= {(p.prompt, p.negative_prompt) for p in prepared}
    assert backend.text_to_image.call_args_list or backend.image_to_image.call_args_list
    size = (backend.prepare_prompts.call_args.kwargs["width"], backend.prepare_prompts.call_args.kwargs["height"])
    for call in backend.text_to_image.call_args_list:
        assert (call.kwargs["width"], call.kwargs["height"]) == size
    if scenario == "review_before_the_upscale":
        skip_signal.wait_for_review.assert_called()
        assert backend.image_to_image.call_args_list
    if scenario == "json_prompt":
        assert all("{" in prompt for prompt, _ in sent if prompt.startswith("a {"))
    if scenario.startswith("references"):
        assert all(call.kwargs["reference_images"] for call in [*backend.text_to_image.call_args_list, *backend.image_to_image.call_args_list])
