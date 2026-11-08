"""The memory guard of a transparent job: when the image model is released and loaded again around the cut-out."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_args, _make_plan

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.matte_room import FOREGROUND_BYTES_PER_PIXEL, MIN_FREE_BYTES, MatteRoom, needs_swap, required_free_bytes
from zvisiongenerator.utils.image_model_detect import ImageModelInfo

_GB = 1024**3
_SMALL = 1024 * 1024
_4X_UPSCALE = 4096 * 4096
_MODEL_INFO = ImageModelInfo(family="zimage", is_distilled=False, size=None)


def _backend() -> MagicMock:
    """Return a mock image backend that has ``release_model`` and a name."""
    backend = MagicMock(spec=ImageBackend)
    backend.name = "mflux"
    return backend


def _room(free, *, reload=True) -> tuple[MatteRoom, MagicMock]:
    """Return a room over a mock backend whose free memory is *free*, plus a mock that records the calls in order."""
    calls = MagicMock()
    backend = _backend()
    calls.attach_mock(backend.release_model, "release_model")
    release_memory = MagicMock()
    calls.attach_mock(release_memory, "release_memory")
    reloader = MagicMock(return_value="reloaded") if reload else None
    if reloader is not None:
        calls.attach_mock(reloader, "reload")
    room = MatteRoom(backend, "model", reloader, lambda: free, release_memory)
    return room, calls


class TestNeedsSwap:
    def test_threshold_is_two_gigabytes(self):
        assert MIN_FREE_BYTES == 2 * _GB

    @pytest.mark.parametrize(("free", "expected"), [(1 * _GB, True), (2 * _GB - 1, True), (2 * _GB, False), (3 * _GB, False), (None, False)])
    def test_only_known_short_memory_asks_for_a_swap(self, free, expected):
        assert needs_swap(free, MIN_FREE_BYTES) is expected


class TestRequiredFreeBytes:
    def test_grows_with_the_image_size(self):
        assert required_free_bytes(0) == MIN_FREE_BYTES
        assert required_free_bytes(_4X_UPSCALE) == MIN_FREE_BYTES + _4X_UPSCALE * FOREGROUND_BYTES_PER_PIXEL
        assert required_free_bytes(_4X_UPSCALE) > 5 * _GB


class TestMakeRoom:
    def test_three_gigabytes_free_releases_the_model_before_a_4x_upscaled_cut_out(self):
        room, calls = _room(3 * _GB)

        room.make_room(_4X_UPSCALE)

        assert calls.release_model.call_count == 1
        assert room.swapping and room.released

    def test_one_gigabyte_free_releases_the_model_and_then_the_memory(self):
        room, calls = _room(1 * _GB)

        room.make_room(_SMALL)

        assert [c[0] for c in calls.mock_calls] == ["release_model", "release_memory"]
        assert calls.release_model.call_args.args == ("model",)
        assert room.swapping and room.released

    def test_three_gigabytes_free_releases_nothing(self):
        room, calls = _room(3 * _GB)

        room.make_room(_SMALL)

        assert calls.mock_calls == []
        assert not room.swapping and not room.released

    def test_unknown_free_memory_never_releases(self):
        room, calls = _room(None)

        room.make_room(_SMALL)

        assert calls.mock_calls == []

    def test_once_swapping_every_later_call_releases_even_with_plenty_free(self):
        free = [1 * _GB]
        room, calls = _room(0)
        room.free_bytes = lambda: free[0]
        room.make_room(_SMALL)
        room.model_for_render(MagicMock(), MagicMock())
        free[0] = 8 * _GB

        room.make_room(_SMALL)

        assert calls.release_model.call_count == 2
        assert room.released

    def test_without_a_reloader_nothing_is_released(self):
        room, calls = _room(1 * _GB, reload=False)

        room.make_room(_SMALL)

        assert calls.mock_calls == []
        assert not room.swapping and not room.released


class TestModelForRender:
    def test_an_unreleased_model_is_returned_as_is(self):
        room, calls = _room(3 * _GB)
        chain, prepare = MagicMock(), MagicMock()

        assert room.model_for_render(chain, prepare) == "model"

        chain.close.assert_not_called()
        prepare.assert_not_called()
        calls.reload.assert_not_called()

    def test_a_released_model_is_reloaded_after_the_chain_is_closed_and_prompts_are_prepared_again(self):
        room, calls = _room(1 * _GB)
        room.make_room(_SMALL)
        calls.reset_mock()
        chain, prepare = MagicMock(), MagicMock()
        calls.attach_mock(chain.close, "close")
        calls.attach_mock(prepare, "prepare")

        model = room.model_for_render(chain, prepare)

        assert model == "reloaded" == room.model
        assert [c[0] for c in calls.mock_calls] == ["close", "release_memory", "reload", "prepare"]
        prepare.assert_called_once_with("reloaded")
        assert not room.released

    def test_the_reload_happens_once_per_release(self):
        room, calls = _room(1 * _GB)
        room.make_room(_SMALL)
        room.model_for_render(MagicMock(), MagicMock())

        room.model_for_render(MagicMock(), MagicMock())

        calls.reload.assert_called_once_with()


def _stage_that_makes_room(request, artifacts) -> StageOutcome:
    """Stand in for the matte stage: ask for room as the real one does while the chain is not loaded."""
    if request.make_room is not None:
        request.make_room(_SMALL)
    return StageOutcome.success


def _run(prompts, args, *, backend, model, chain, reload_model=None, free=1 * _GB, stage=_stage_that_makes_room):
    config = {
        "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
        "generation": {"seed_min": 1, "seed_max": 100},
        "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
    }
    plan = _make_plan(prompts, args, config=config)
    workflow = GenerationWorkflow(name="t", stages=[stage])
    with (
        patch("zvisiongenerator.image_runner.build_workflow", return_value=workflow),
        patch("zvisiongenerator.image_runner.accelerator_free_bytes", return_value=free),
        patch("zvisiongenerator.image_runner.release_accelerator_memory"),
    ):
        run_batch(backend, model, prompts, config, args, model_info=_MODEL_INFO, plan=plan, reload_model=reload_model, matte_factory=lambda: chain)


class TestRunBatchTransparent:
    def test_a_short_job_reloads_and_prepares_again_before_the_next_render_with_the_chain_closed_first(self):
        backend = _backend()
        chain = MagicMock()
        events = MagicMock()
        events.attach_mock(backend.prepare_prompts, "prepare")
        events.attach_mock(backend.release_model, "release_model")
        events.attach_mock(chain.close, "close")
        reloaded = MagicMock(name="reloaded")
        reload_model = MagicMock(return_value=reloaded)
        events.attach_mock(reload_model, "reload")

        _run({"s": [("a cat", None), ("a dog", None)]}, _make_args(transparent=True), backend=backend, model="first", chain=chain, reload_model=reload_model)

        names = [c[0] for c in events.mock_calls]
        assert names == ["prepare", "release_model", "close", "reload", "prepare", "release_model", "close"]
        assert [c.args[0] for c in backend.prepare_prompts.call_args_list] == ["first", reloaded]
        assert [c.args[0] for c in backend.release_model.call_args_list] == ["first", reloaded]

    def test_plenty_of_free_memory_never_releases_or_reloads(self):
        backend = _backend()
        reload_model = MagicMock()

        _run({"s": [("a cat", None), ("a dog", None)]}, _make_args(transparent=True), backend=backend, model="m", chain=MagicMock(), reload_model=reload_model, free=8 * _GB)

        backend.release_model.assert_not_called()
        reload_model.assert_not_called()
        backend.prepare_prompts.assert_called_once()

    def test_without_a_reloader_the_model_is_never_released(self):
        backend = _backend()

        _run({"s": [("a cat", None)]}, _make_args(transparent=True), backend=backend, model="m", chain=MagicMock(), reload_model=None)

        backend.release_model.assert_not_called()

    def test_the_request_carries_the_chain_and_the_room(self):
        seen = {}

        def stage(request, artifacts):
            seen["matte"], seen["make_room"], seen["transparent"] = request.matte, request.make_room, request.transparent
            return StageOutcome.success

        chain = MagicMock()
        _run({"s": [("a cat", None)]}, _make_args(transparent=True), backend=_backend(), model="m", chain=chain, stage=stage)

        assert seen["matte"] is chain
        assert callable(seen["make_room"])
        assert seen["transparent"] is True

    def test_the_chain_is_closed_when_a_stage_raises(self):
        chain = MagicMock()

        def stage(request, artifacts):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _run({"s": [("a cat", None)]}, _make_args(transparent=True), backend=_backend(), model="m", chain=chain, stage=stage)

        chain.close.assert_called_once_with()

    def test_an_opaque_job_creates_no_chain(self):
        backend = _backend()
        factory = MagicMock()
        seen = {}

        def stage(request, artifacts):
            seen["matte"], seen["make_room"] = request.matte, request.make_room
            return StageOutcome.success

        config = {
            "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
        }
        args = _make_args()
        prompts = {"s": [("a cat", None)]}
        with patch("zvisiongenerator.image_runner.build_workflow", return_value=GenerationWorkflow(name="t", stages=[stage])):
            run_batch(backend, "m", prompts, config, args, model_info=_MODEL_INFO, plan=_make_plan(prompts, args, config=config), matte_factory=factory)

        factory.assert_not_called()
        assert seen == {"matte": None, "make_room": None}
