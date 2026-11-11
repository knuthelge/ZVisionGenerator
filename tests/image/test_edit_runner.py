"""Test that run_batch runs an edit job: one prompt preparation, no release, saved provenance, and Next and Stop."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._factories import _make_args, _make_plan
from tests._images import noise_picture, painted_box
from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import JobPrompt
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import read_png_config
from zvisiongenerator.workflows.generation_prompt import EXTEND_INSTRUCTION

_MODEL_INFO = ImageModelInfo(family="zimage", is_distilled=False, size=None)
# An edit takes its size from the picture, so the config has no size presets.
_CONFIG = {"generation": {"seed_min": 1, "seed_max": 100}, "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4}}


def _source(tmp_path: Path, name: str = "door.png") -> Path:
    path = tmp_path / name
    noise_picture(480, 320, seed=7).save(path)
    return path


def _mask():
    return painted_box(320, 480, (150, 100, 260, 200))


def _backend() -> MagicMock:
    backend = MagicMock(spec=ImageBackend)
    backend.name = "mock"
    backend.edit_region.side_effect = lambda **kwargs: noise_picture(777, 555, seed=1)
    return backend


def _run(tmp_path: Path, backend: MagicMock, *, runs: int = 2, skip_signal: SkipSignal | None = None, source_name: str = "door.png", output: Path | None = None, **overrides) -> list[dict]:
    values = {"edit_paint": _mask(), "edit_extend": None, **overrides}
    args = _make_args(edit_source=str(_source(tmp_path, source_name)), edit_size=(480, 320), output=str(output or tmp_path / "out"), runs=runs, **values)
    prompts = {"set1": [("a red door", None)]}
    events: list[dict] = []
    run_batch(
        backend,
        MagicMock(spec=[]),
        prompts,
        _CONFIG,
        args,
        model_info=_MODEL_INFO,
        plan=_make_plan(prompts, args, config=_CONFIG),
        progress_callback=events.append,
        skip_signal=skip_signal,
    )
    return events


def _outputs(tmp_path: Path) -> list[str]:
    folder = tmp_path / "out"
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def test_an_edit_batch_prepares_the_prompts_once_and_never_releases_the_model(tmp_path):
    backend = _backend()

    events = _run(tmp_path, backend, runs=3)

    backend.prepare_prompts.assert_called_once()
    prepared = backend.prepare_prompts.call_args.args[1]
    assert prepared == (JobPrompt("a red door", None),)
    assert [call.kwargs["prompt"] for call in backend.edit_region.call_args_list] == ["a red door"] * 3
    backend.release_model.assert_not_called()
    backend.load_model.assert_not_called()
    assert [event["status"] for event in events if event["type"] == "generation_finished"] == ["success"] * 3


@pytest.mark.parametrize(
    ("overrides", "sent"),
    [
        ({}, "a red door"),  # Paint
        ({"edit_paint": None, "edit_extend": ("left", 128)}, EXTEND_INSTRUCTION + "a red door"),  # Extend
        ({"edit_paint": None}, "a red door"),  # the whole picture: no instruction, only an Extend has one
    ],
    ids=["paint", "extend", "whole picture"],
)
def test_the_prompt_an_edit_sends_is_among_the_prompts_the_batch_prepared(tmp_path, overrides, sent):
    """A backend that pre-encodes its prompts (macOS) frees the encoder after them, so a prompt it was not given fails the run."""
    backend = _backend()
    backend.text_to_image.side_effect = lambda **kwargs: Image.new("RGB", (kwargs["width"], kwargs["height"]))

    _run(tmp_path, backend, runs=1, **overrides)

    drawn = backend.edit_region if backend.edit_region.called else backend.text_to_image
    prepared = [job_prompt.prompt for job_prompt in backend.prepare_prompts.call_args.args[1]]
    assert drawn.call_args.kwargs["prompt"] == sent
    assert sent in prepared


def test_the_outputs_are_named_after_the_source_with_a_counter(tmp_path):
    _run(tmp_path, _backend(), runs=2)

    assert _outputs(tmp_path) == ["door_edit.png", "door_edit_2.png"]


@pytest.mark.parametrize(
    ("source_name", "taken", "saved"),
    [
        ("door.png", [], "door_edit.png"),
        ("door_edit.png", [], "door_edit_2.png"),
        ("door_edit_2.png", ["door_edit.png"], "door_edit_3.png"),
    ],
)
def test_editing_an_edit_again_takes_the_counter_and_never_a_second_edit_suffix(tmp_path, source_name, taken, saved):
    for name in taken:
        (tmp_path / name).write_bytes(b"taken")

    # The edit sits beside its source, as it does by default.
    _run(tmp_path, _backend(), runs=1, source_name=source_name, output=tmp_path)

    assert {path.name for path in tmp_path.glob("*.png")} == {source_name, *taken, saved}


def test_the_edit_needs_no_size_preset_and_sends_the_job_settings(tmp_path):
    backend = _backend()

    _run(tmp_path, backend, runs=1, steps=6, guidance=2.5, seed=77)

    call = backend.edit_region.call_args.kwargs
    assert (call["steps"], call["guidance"], call["seed"]) == (6, 2.5, 77)


def test_the_saved_file_records_the_edit_and_no_post_processing(tmp_path):
    _run(tmp_path, _backend(), runs=1, sharpen=True, contrast=1.3, saturation=1.2, transparent=True)

    saved = tmp_path / "out" / "door_edit.png"
    config = read_png_config(saved)
    with Image.open(saved) as image:
        size, mode = image.size, image.mode
    assert config["workflow"] == "edit"
    assert config["source"] == {"path": str(tmp_path / "door.png"), "width": 480, "height": 320}
    assert config["edit"] == {"mode": "paint"}
    assert (config["width"], config["height"]) == size == (480, 320)
    assert "sharpen" not in config["generation"] and "contrast" not in config["generation"] and "transparent" not in config["generation"]
    assert mode == "RGB"  # an opaque source is never cut out


def test_next_during_the_edit_skips_it_without_saving_and_the_batch_goes_on(tmp_path):
    backend = _backend()
    first = backend.edit_region.side_effect
    backend.edit_region.side_effect = [None, first(), first()]

    with pytest.warns(UserWarning, match="skipped"):
        events = _run(tmp_path, backend, runs=3)

    assert [event["status"] for event in events if event["type"] == "generation_finished"] == ["skipped", "success", "success"]
    assert _outputs(tmp_path) == ["door_edit.png", "door_edit_2.png"]


def test_stop_during_the_edit_quits_the_batch(tmp_path):
    backend = _backend()
    skip = SkipSignal()

    def _stop(**kwargs):
        skip.queue_action("quit")
        return None

    backend.edit_region.side_effect = _stop

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = _run(tmp_path, backend, runs=3, skip_signal=skip)

    assert backend.edit_region.call_count == 1
    assert _outputs(tmp_path) == []
    assert events[-1]["type"] == "batch_cancelled"
