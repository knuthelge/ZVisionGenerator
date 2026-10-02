"""Tests for shared progress event helpers."""

from __future__ import annotations

from PIL import Image
import pytest

from zvisiongenerator.core.progress_events import preview_milestone_steps
from zvisiongenerator.workflows.image_stages import _emit_step_progress


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [
        (0, set()),
        (1, set()),
        (2, {1}),
        (4, {1, 2, 3}),
        (8, {2, 4, 6}),
        (9, {3, 5, 7}),
        (20, {5, 10, 15}),
    ],
)
def test_preview_milestone_steps_cover_quarters_but_not_the_final_step(total_steps, expected):
    assert preview_milestone_steps(total_steps) == expected


def test_stage_step_progress_forwards_preview_images():
    events: list[dict] = []
    preview = Image.new("RGB", (8, 8))
    callback = _emit_step_progress(events.append, phase="image_generate", total_steps=4)

    callback({"current_step": 1, "total_steps": 4})
    callback({"current_step": 2, "total_steps": 4, "preview": preview})

    assert "preview" not in events[0]
    assert events[1]["preview"] is preview
    assert events[1]["phase"] == "image_generate"
