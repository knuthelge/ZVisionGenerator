"""Tests for the review before upscaling of videos: LTX backend seam, stages, batch runner and CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_mock_video_backend, _make_plan, _make_video_args
from tests.video._helpers import make_ltx_info, make_video_request, patch_video_cli_for_preflight

from zvisiongenerator.backends import supports_video_upscale_review
from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend
from zvisiongenerator.backends.mac.video import LtxVideoBackend, _LtxModel
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.video_runner import run_video_batch
from zvisiongenerator.workflows import video_stages
from zvisiongenerator.workflows.video_stages import image_to_video_stage, text_to_video_stage

_FPS = 24.0
_STAGE1_RESULT = SimpleNamespace(video_tokens="video_tokens", audio_tokens="audio_tokens", latent_dims=(3, 7, 11))


# ---------------------------------------------------------------------------
# LTX MLX backend: the per-call wrapper over the pipeline's _stage1
# ---------------------------------------------------------------------------


class _FakeDistilledPipeline:
    """Stand-in for ``DistilledPipeline`` with the methods the review wrapper touches, recording their order."""

    def __init__(self, *, low_memory: bool, fail_in_stage2: bool = False):
        self.low_memory = low_memory
        self.fail_in_stage2 = fail_in_stage2
        self.dit: object | None = "dit"
        self._loaded = True
        self.upsampler: object | None = "upsampler"
        self.prompt_encoder = MagicMock()
        self.image_conditioner = MagicMock()
        self.vae_decoder: object | None = None
        self.audio_decoder: object | None = None
        self.vocoder: object | None = None
        self.stepwise = None
        self.video_patchifier = MagicMock()
        self.audio_patchifier = MagicMock()
        self.events: list[str] = []
        self.generate_calls = 0
        self.stage1_return = (_STAGE1_RESULT, 49, 480, 704)
        self.seen_by_stage2: tuple[object, bool, tuple] | None = None
        self.decoded_to: Path | None = None
        self.decoded_seed: int | None = None

    def _stage1(self, *args, **kwargs):
        self.events.append("stage1")
        return self.stage1_return

    def _load_decoders(self):
        self.events.append("load_decoders")
        self.vae_decoder, self.audio_decoder, self.vocoder = "vae", "audio", "vocoder"

    def _decode_and_save_video(self, video_latent, audio_latent, output_path, *, frame_rate, seed=0):
        self.events.append("decode")
        self.decoded_to = Path(output_path)
        self.decoded_seed = seed
        Path(output_path).write_bytes(b"mp4")
        if self.low_memory:
            self.dit = None
            self._loaded = False
        return output_path

    def generate_and_save(self, **kwargs):
        self.generate_calls += 1
        result = self._stage1("prompt")
        self.events.append("stage2")
        self.seen_by_stage2 = (self.dit, self._loaded, result)
        if self.fail_in_stage2:
            raise RuntimeError("stage 2 failed")
        return kwargs["output_path"]


@pytest.fixture
def ltx_memory(monkeypatch):
    """Replace the LTX modules the backend imports lazily; no pipeline code is ever loaded."""
    memory = MagicMock()
    for name in ("ltx_core_mlx", "ltx_core_mlx.utils", "ltx_pipelines_mlx"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.utils.memory", memory)
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.scheduler", SimpleNamespace(STAGE_2_SIGMAS=[0.9, 0.7, 0.4, 0.0]))
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils", MagicMock())
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils.args", SimpleNamespace(ImageConditioningInput=lambda **fields: SimpleNamespace(**fields)))
    return memory


def _generate(pipe, review, *, stage1_steps: int | None = 6, upscale: bool = True, image_path: str | None = None, output_path: str = "/out/o.mp4"):
    backend = LtxVideoBackend()
    kwargs = dict(
        model=_LtxModel(pipeline=pipe, frame_rate=_FPS, upscale=upscale, low_ram_streaming=False),
        prompt="p",
        width=704,
        height=480,
        num_frames=49,
        seed=7,
        steps=8,
        output_path=output_path,
        stage1_steps=stage1_steps,
    )
    if review is not None:
        kwargs["review"] = review
    return backend.text_to_video(**kwargs) if image_path is None else backend.image_to_video(image_path=image_path, **kwargs)


class TestLtxReviewWrapper:
    @pytest.mark.parametrize("image_path", [None, "/in/start.png"])
    def test_review_runs_between_the_stages_and_generate_and_save_runs_once(self, ltx_memory, image_path):
        pipe = _FakeDistilledPipeline(low_memory=True)
        reviewed: list[str] = []

        _generate(pipe, lambda path: (pipe.events.append("review"), reviewed.append(str(path))), image_path=image_path)

        assert pipe.generate_calls == 1
        assert pipe.events == ["stage1", "load_decoders", "decode", "review", "stage2"]
        assert len(reviewed) == 1

    def test_the_small_render_exists_during_the_review_outside_the_output_folder_and_is_gone_after(self, ltx_memory, tmp_path):
        pipe = _FakeDistilledPipeline(low_memory=False)
        seen: dict[str, object] = {}

        def _review(path: Path) -> None:
            seen["path"] = path
            seen["exists"] = path.is_file()
            seen["content"] = path.read_bytes()

        _generate(pipe, _review, output_path=str(tmp_path / "out" / "o.mp4"))

        path = seen["path"]
        assert seen["exists"] and seen["content"] == b"mp4"
        assert tmp_path not in path.parents
        assert not path.exists() and not path.parent.exists()

    def test_the_half_size_latents_are_decoded_with_the_clip_settings(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=False)

        _generate(pipe, lambda path: None)

        pipe.video_patchifier.unpatchify.assert_called_once_with("video_tokens", (3, 7, 11))
        pipe.audio_patchifier.unpatchify.assert_called_once_with("audio_tokens")
        assert pipe.decoded_seed == 7

    def test_the_stage_one_result_reaches_stage_two_unchanged(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)

        _generate(pipe, lambda path: None)

        assert pipe.seen_by_stage2 is not None
        assert pipe.seen_by_stage2[2] is pipe.stage1_return

    def test_the_dit_and_loaded_flag_survive_the_preview_decode(self, ltx_memory):
        """Under low_memory the decode empties the DiT slot; stage 2 and the review must still see it."""
        pipe = _FakeDistilledPipeline(low_memory=True)
        during_review: list[tuple] = []

        _generate(pipe, lambda path: during_review.append((pipe.dit, pipe._loaded)))

        assert during_review == [("dit", True)]
        assert pipe.seen_by_stage2[:2] == ("dit", True)

    def test_a_failing_preview_decode_releases_the_model_under_low_memory_and_removes_the_shadow(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)
        pipe._decode_and_save_video = MagicMock(side_effect=lambda *a, **k: setattr(pipe, "dit", None) or (_ for _ in ()).throw(RuntimeError("decode failed")))

        with pytest.raises(RuntimeError, match="decode failed"):
            _generate(pipe, lambda path: None)

        assert (pipe.dit, pipe._loaded, pipe.upsampler) == (None, False, None)
        assert "_stage1" not in vars(pipe)

    def test_the_decoders_are_freed_before_the_review_under_low_memory(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)
        during_review: list[tuple] = []

        _generate(pipe, lambda path: during_review.append((pipe.vae_decoder, pipe.audio_decoder, pipe.vocoder)))

        assert during_review == [(None, None, None)]
        ltx_memory.aggressive_cleanup.assert_called()

    def test_the_decoders_stay_loaded_without_low_memory(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=False)

        _generate(pipe, lambda path: None)

        assert (pipe.vae_decoder, pipe.audio_decoder, pipe.vocoder) == ("vae", "audio", "vocoder")

    def test_stage1_is_restored_after_success(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)

        _generate(pipe, lambda path: None)

        assert "_stage1" not in vars(pipe)

    def test_stage1_is_restored_when_the_review_raises(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)
        seen: list[Path] = []

        def _review(path: Path) -> None:
            seen.append(path)
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _generate(pipe, _review)

        assert "_stage1" not in vars(pipe)
        assert not seen[0].exists()
        assert pipe.generate_calls == 1

    def test_stage1_is_restored_when_stage_two_raises(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True, fail_in_stage2=True)

        with pytest.raises(RuntimeError, match="stage 2 failed"):
            _generate(pipe, lambda path: None)

        assert "_stage1" not in vars(pipe)

    def test_without_review_the_pipeline_is_left_alone(self, ltx_memory):
        pipe = _FakeDistilledPipeline(low_memory=True)

        _generate(pipe, None)

        assert pipe.events == ["stage1", "stage2"]

    def test_review_without_stage1_steps_is_refused_before_generating(self, ltx_memory):
        for upscale in (True, False):
            pipe = _FakeDistilledPipeline(low_memory=True)

            with pytest.raises(ValueError, match="review"):
                _generate(pipe, lambda path: None, stage1_steps=None, upscale=upscale)

            assert pipe.generate_calls == 0 and pipe.events == []


# ---------------------------------------------------------------------------
# Diffusers backend and platform support
# ---------------------------------------------------------------------------


class TestDiffusersReviewNeedsAnUpscale:
    @pytest.mark.parametrize("method", ["text_to_video", "image_to_video"])
    def test_review_without_stage1_steps_raises_before_any_work(self, method):
        model = MagicMock(upsample_pipeline=None)
        kwargs = dict(model=model, prompt="p", width=704, height=448, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4", review=lambda path: None)
        if method == "image_to_video":
            kwargs["image_path"] = "/tmp/in.png"

        with pytest.raises(ValueError, match="review"):
            getattr(DiffusersVideoBackend(), method)(**kwargs)

        model.text_to_video.assert_not_called()
        model.image_to_video.assert_not_called()


class TestSupportsVideoUpscaleReview:
    @pytest.mark.parametrize(("platform", "expected"), [("darwin", True), ("win32", True), ("linux", True), ("freebsd", False)])
    def test_every_registered_video_backend_supports_review(self, monkeypatch, platform, expected):
        monkeypatch.setattr(sys, "platform", platform)

        assert supports_video_upscale_review() is expected


# ---------------------------------------------------------------------------
# Stages: review is forwarded only with upscale, and its wait is not generation time
# ---------------------------------------------------------------------------


class _Clock:
    """A fake ``time.monotonic``."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _request(**overrides) -> VideoGenerationRequest:
    return make_video_request(**{"prompt": "a sunset", "image_path": "/in/start.png", "output_dir": "/out", **overrides})


def _artifacts() -> VideoWorkingArtifacts:
    return VideoWorkingArtifacts(filename="clip.mp4", resolved_prompt="a sunset")


class TestStageReview:
    @pytest.mark.parametrize(("stage", "method"), [(text_to_video_stage, "text_to_video"), (image_to_video_stage, "image_to_video")])
    def test_review_is_forwarded_with_upscale(self, stage, method):
        review = MagicMock()
        request = _request(upscale=2, upscale_steps=6, review=review)
        getattr(request.backend, method).return_value = Path("/out/clip.mp4")

        stage(request, _artifacts())

        kwargs = getattr(request.backend, method).call_args.kwargs
        assert kwargs["stage1_steps"] == 6
        kwargs["review"](Path("/tmp/small.mp4"))
        review.assert_called_once_with(Path("/tmp/small.mp4"))

    @pytest.mark.parametrize(("stage", "method"), [(text_to_video_stage, "text_to_video"), (image_to_video_stage, "image_to_video")])
    def test_review_is_not_forwarded_without_upscale(self, stage, method):
        request = _request(review=MagicMock())
        getattr(request.backend, method).return_value = Path("/out/clip.mp4")

        stage(request, _artifacts())

        kwargs = getattr(request.backend, method).call_args.kwargs
        assert "review" not in kwargs and "stage1_steps" not in kwargs

    @pytest.mark.parametrize(("stage", "method"), [(text_to_video_stage, "text_to_video"), (image_to_video_stage, "image_to_video")])
    def test_no_review_key_when_none_is_set(self, stage, method):
        request = _request(upscale=2, upscale_steps=6)
        getattr(request.backend, method).return_value = Path("/out/clip.mp4")

        stage(request, _artifacts())

        assert "review" not in getattr(request.backend, method).call_args.kwargs

    @pytest.mark.parametrize(("stage", "method"), [(text_to_video_stage, "text_to_video"), (image_to_video_stage, "image_to_video")])
    def test_the_wait_is_not_generation_time(self, monkeypatch, stage, method):
        clock = _Clock()
        monkeypatch.setattr(video_stages.time, "monotonic", clock)
        request = _request(upscale=2, upscale_steps=6, review=lambda path: clock.advance(100.0))

        def _backend_call(**kwargs):
            clock.advance(3.0)
            kwargs["review"](Path("/tmp/small.mp4"))
            clock.advance(4.0)
            return Path("/out/clip.mp4")

        setattr(request.backend, method, _backend_call)
        artifacts = _artifacts()

        outcome = stage(request, artifacts)

        assert outcome is StageOutcome.success
        assert artifacts.generation_time == pytest.approx(7.0)
        assert artifacts.review_wait == pytest.approx(100.0)

    def test_the_wait_is_counted_when_the_review_raises(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(video_stages.time, "monotonic", clock)

        def _review(path: Path) -> None:
            clock.advance(30.0)
            raise KeyboardInterrupt

        request = _request(upscale=2, upscale_steps=6, review=_review, image_path=None)
        request.backend.text_to_video.side_effect = lambda **kwargs: kwargs["review"](Path("/tmp/small.mp4"))
        artifacts = _artifacts()

        with pytest.raises(KeyboardInterrupt):
            text_to_video_stage(request, artifacts)

        assert artifacts.review_wait == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# run_video_batch
# ---------------------------------------------------------------------------


def _review_workflow(clock: _Clock | None = None, *, render_seconds: float = 5.0, wait_seconds: float = 100.0, after_seconds: float = 20.0):
    """Return a workflow that runs the real text-to-video stage against a backend which reviews its small render."""
    backend = MagicMock()

    def _text_to_video(**kwargs):
        if clock is not None:
            clock.advance(render_seconds)
        if "review" in kwargs:
            kwargs["review"](Path("/tmp/ziv-review-x/small-render.mp4"))
        if clock is not None:
            clock.advance(after_seconds)
        return Path(kwargs["output_path"])

    backend.text_to_video.side_effect = _text_to_video
    workflow = MagicMock()

    def _run(request, artifacts):
        artifacts.filename = "clip.mp4"
        return text_to_video_stage(request, artifacts)

    workflow.run.side_effect = _run
    return backend, workflow


def _run_batch(skip, *, prompts=1, review=True, callback=None, args=None, workflow=None, backend=None):
    backend = backend or _make_mock_video_backend()
    if workflow is None:
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
    args = args or _make_video_args(upscale=2, upscale_steps=6, review=review, output="/out")
    prompt_data = {"set": [(f"prompt {i}", None) for i in range(prompts)]}
    config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}
    run_video_batch(
        backend,
        MagicMock(),
        make_ltx_info(),
        workflow,
        prompt_data,
        config,
        args,
        plan=_make_plan(prompt_data, args, config=config),
        progress_callback=callback,
        skip_signal=skip,
    )


def _decided(*decisions: str) -> SkipSignal:
    """Return a signal whose review answers *decisions* in order."""
    skip = SkipSignal()
    answers = iter(decisions)
    skip.wait_for_review = lambda: next(answers)  # type: ignore[method-assign]
    return skip


class TestRunVideoBatchReview:
    def test_review_callable_is_built_only_with_upscale_and_review(self):
        for upscale, review, expect in ((2, True, True), (2, False, False), (None, True, False)):
            workflow = MagicMock()
            workflow.run.return_value = StageOutcome.success
            args = _make_video_args(upscale=upscale, upscale_steps=6 if upscale else None, review=review)

            _run_batch(SkipSignal(), args=args, workflow=workflow)

            assert (workflow.run.call_args.args[0].review is not None) is expect

    def test_review_without_a_signal_is_refused(self):
        with pytest.raises(ValueError, match="skip_signal"):
            _run_batch(None)

    def test_approve_pauses_resumes_and_finishes_with_success(self):
        events: list[dict] = []
        backend, workflow = _review_workflow()

        _run_batch(_decided("approve"), callback=events.append, backend=backend, workflow=workflow)

        types = [e["type"] for e in events]
        assert types.index("job_paused") < types.index("job_resumed") < types.index("generation_finished")
        paused = next(e for e in events if e["type"] == "job_paused")
        assert paused["review"] == {"iteration": 1}
        assert paused["mode"] == "video"
        assert "preview" not in paused
        assert paused["review_path"] == str(Path("/tmp/ziv-review-x/small-render.mp4"))
        finished = next(e for e in events if e["type"] == "generation_finished")
        assert finished["status"] == "success"
        assert types[-1] == "batch_completed"

    def test_discard_is_skipped_not_failed_and_the_batch_completes(self):
        events: list[dict] = []
        backend, workflow = _review_workflow()

        _run_batch(_decided("discard", "approve"), prompts=2, callback=events.append, backend=backend, workflow=workflow)

        statuses = [e["status"] for e in events if e["type"] == "generation_finished"]
        assert statuses == ["skipped", "success"]
        skipped = next(e for e in events if e["type"] == "generation_finished" and e["status"] == "skipped")
        assert "output_path" not in skipped or skipped["output_path"] is None
        assert events[-1]["type"] == "batch_completed"
        assert events[-1]["completed_iterations"] == 2

    def test_a_batch_where_everything_is_discarded_still_completes(self):
        events: list[dict] = []
        backend, workflow = _review_workflow()

        _run_batch(_decided("discard", "discard"), prompts=2, callback=events.append, backend=backend, workflow=workflow)

        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["skipped", "skipped"]
        assert events[-1]["type"] == "batch_completed"
        assert "batch_failed" not in [e["type"] for e in events]

    def test_quit_during_a_review_cancels_the_batch(self):
        events: list[dict] = []
        backend, workflow = _review_workflow()
        skip = _decided("quit")
        skip.queue_action("quit")

        _run_batch(skip, prompts=2, callback=events.append, backend=backend, workflow=workflow)

        types = [e["type"] for e in events]
        assert types[-1] == "batch_cancelled"
        assert "generation_finished" not in types

    def test_quit_chosen_while_waiting_cancels_the_batch_and_no_second_video_starts(self):
        events: list[dict] = []
        backend, workflow = _review_workflow()
        skip = SkipSignal()
        skip.wait_for_review = lambda: (skip.queue_action("quit"), "quit")[1]  # type: ignore[method-assign]

        _run_batch(skip, prompts=2, callback=events.append, backend=backend, workflow=workflow)

        types = [e["type"] for e in events]
        assert types.count("generation_started") == 1
        assert types[-1] == "batch_cancelled"
        assert "generation_finished" not in types

    def test_timing_excludes_the_wait_for_approved_and_discarded_videos(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(video_stages.time, "monotonic", clock)
        events: list[dict] = []
        backend, workflow = _review_workflow(clock)
        skip = SkipSignal()
        answers = iter(["discard", "approve"])

        def _wait() -> str:
            clock.advance(100.0)
            return next(answers)

        skip.wait_for_review = _wait  # type: ignore[method-assign]

        _run_batch(skip, prompts=2, callback=events.append, backend=backend, workflow=workflow)

        finished = [e for e in events if e["type"] == "generation_finished"]
        assert [e["status"] for e in finished] == ["skipped", "success"]
        assert finished[0]["generation_time"] == pytest.approx(5.0)
        assert finished[1]["generation_time"] == pytest.approx(25.0)
        second_started = [e for e in events if e["type"] == "generation_started"][1]
        assert second_started["avg_secs"] == pytest.approx(5.0)

    def test_the_path_is_printed_only_without_a_callback(self, capsys):
        path = str(Path("/tmp/ziv-review-x/small-render.mp4"))
        backend, workflow = _review_workflow()
        _run_batch(_decided("approve"), backend=backend, workflow=workflow)
        assert path in capsys.readouterr().out

        backend, workflow = _review_workflow()
        _run_batch(_decided("approve"), callback=lambda event: None, backend=backend, workflow=workflow)
        assert path not in capsys.readouterr().out

    def test_the_quit_key_still_cancels_between_videos_without_review(self):
        events: list[dict] = []
        skip = SkipSignal()
        skip.queue_action("quit")

        _run_batch(skip, review=False, callback=events.append, args=_make_video_args(review=False))

        assert events[-1]["type"] == "batch_cancelled"


# ---------------------------------------------------------------------------
# ziv-video --review
# ---------------------------------------------------------------------------


class _FakeSkipSignal:
    """Records start and stop of the key listener."""

    instances: list[_FakeSkipSignal] = []

    def __init__(self, *, quit_only: bool = False):
        self.quit_only = quit_only
        self.calls: list[str] = []
        _FakeSkipSignal.instances.append(self)

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")


def _tty(monkeypatch, value: bool = True) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: value))


class TestVideoCliReview:
    def test_review_flag_is_parsed(self):
        from zvisiongenerator.video_cli import _build_video_parser

        parser = _build_video_parser()

        assert parser.parse_args(["-m", "model", "--upscale", "2", "--review"]).review is True
        assert parser.parse_args(["-m", "model"]).review is False

    @pytest.mark.parametrize(
        ("argv", "tty", "supported"),
        [(["--review"], True, True), (["--review", "--upscale", "2"], False, True), (["--review", "--upscale", "2"], True, False)],
    )
    def test_review_errors_before_any_heavy_work(self, monkeypatch, argv, tty, supported):
        import zvisiongenerator.video_cli as video_cli

        _tty(monkeypatch, tty)
        monkeypatch.setattr(video_cli, "supports_video_upscale_review", lambda: supported)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", MagicMock(side_effect=AssertionError("ffmpeg checked")))

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", *argv]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

    def _patched_main(self, monkeypatch, tmp_path, order, *, batch_error: Exception | None = None):
        video_cli, captured, _workflows = patch_video_cli_for_preflight(monkeypatch, order)
        _FakeSkipSignal.instances.clear()
        monkeypatch.setattr(video_cli, "SkipSignal", _FakeSkipSignal)
        monkeypatch.setattr(video_cli, "supports_video_upscale_review", lambda: True)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: order.append("ensure_ffmpeg"))

        def _run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
            order.append("run_video_batch")
            captured["skip_signal"] = skip_signal
            captured["listener_calls_at_batch"] = list(skip_signal.calls) if skip_signal is not None else None
            if batch_error is not None:
                raise batch_error

        monkeypatch.setattr(video_cli, "run_video_batch", _run_video_batch)
        return video_cli, captured

    def test_the_listener_starts_after_preflight_and_stops_after_the_batch(self, monkeypatch, tmp_path):
        order: list[str] = []
        _tty(monkeypatch)
        video_cli, captured = self._patched_main(monkeypatch, tmp_path, order)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", "--upscale", "2", "--review", "-o", str(tmp_path)]):
            video_cli.main()

        (listener,) = _FakeSkipSignal.instances
        assert listener.quit_only is True
        assert captured["skip_signal"] is listener
        assert captured["listener_calls_at_batch"] == ["start"]
        assert listener.calls == ["start", "stop"]
        assert order.index("ensure_ffmpeg") < order.index("load_model") < order.index("run_video_batch")

    def test_the_terminal_is_restored_when_the_batch_raises(self, monkeypatch, tmp_path):
        order: list[str] = []
        _tty(monkeypatch)
        video_cli, _captured = self._patched_main(monkeypatch, tmp_path, order, batch_error=KeyboardInterrupt())

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", "--upscale", "2", "--review", "-o", str(tmp_path)]):
            with pytest.raises(KeyboardInterrupt):
                video_cli.main()

        assert _FakeSkipSignal.instances[0].calls == ["start", "stop"]

    def test_no_listener_without_review(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, captured = self._patched_main(monkeypatch, tmp_path, order)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", "--upscale", "2", "-o", str(tmp_path)]):
            video_cli.main()

        assert _FakeSkipSignal.instances == []
        assert captured["skip_signal"] is None
