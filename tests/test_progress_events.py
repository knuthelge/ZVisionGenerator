"""Tests for shared progress event helpers."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import pytest

from zvisiongenerator.core.progress_events import (
    FAILURE_MESSAGE_LIMIT,
    emit_generation_finished,
    exception_message,
    failure_with_detail,
    one_line_message,
    preview_steps,
    run_workflow_with_progress,
)
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.workflows.image_stages import emit_step_progress


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [
        (0, set()),
        (1, set()),
        (2, {1}),
        (4, {1, 2, 3}),
        (8, {1, 2, 3, 4, 5, 6, 7}),
    ],
)
def test_preview_steps_cover_every_step_but_the_final_one(total_steps, expected):
    assert preview_steps(total_steps) == expected


def test_stage_step_progress_forwards_preview_images():
    events: list[dict] = []
    preview = Image.new("RGB", (8, 8))
    callback = emit_step_progress(events.append, phase="image_generate", total_steps=4)

    callback({"current_step": 1, "total_steps": 4})
    callback({"current_step": 2, "total_steps": 4, "preview": preview})

    assert "preview" not in events[0]
    assert events[1]["preview"] is preview
    assert events[1]["phase"] == "image_generate"


@dataclass
class _StageRequest:
    step_callback: object = None


def test_workflow_started_lists_every_stage_name_in_order():
    def text_to_image_stage(request, artifacts):
        return StageOutcome.success

    def upscale_stage(request, artifacts):
        return StageOutcome.success

    events: list[dict] = []
    workflow = GenerationWorkflow(name="dynamic", stages=[text_to_image_stage, upscale_stage])

    run_workflow_with_progress(workflow, _StageRequest(), object(), progress_callback=events.append, event_context={})

    started = next(event for event in events if event["type"] == "workflow_started")
    assert started["stage_names"] == ["text_to_image", "upscale"]
    assert started["total_stages"] == 2


BANNER = (
    "ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers\n  built with gcc 14 (GCC)\n  configuration: "
    + "--enable-foo " * 80
    + "\n  libavutil      59. 39.100 / 59. 39.100\n  libpostproc    57.  1.100 / 57.  1.100\n"
)


class TestFailureWithDetail:
    def test_keeps_the_prefix_and_the_last_line_of_a_long_output(self):
        detail = BANNER + "[mov @ 0x1] stuff\n" * 100 + "out.mp4: Invalid data found when processing input\n"
        message = failure_with_detail("Joining the clips failed", detail)
        assert message.startswith("Joining the clips failed: ")
        assert message.endswith("Invalid data found when processing input")
        assert "\n" not in message
        assert len(message) <= FAILURE_MESSAGE_LIMIT

    def test_drops_the_banner_when_other_lines_exist(self):
        message = failure_with_detail("Failed", BANNER + "Conversion failed!\n")
        assert message == "Failed: Conversion failed!"

    def test_a_banner_only_output_is_still_reported(self):
        assert "ffmpeg version" in failure_with_detail("Failed", "ffmpeg version 7.1")

    def test_an_empty_detail_gives_just_the_prefix(self):
        assert failure_with_detail("Failed", " \n ") == "Failed"

    @pytest.mark.parametrize("limit", [1, 2, 5, 11, 12, 21, 60, 500])
    @pytest.mark.parametrize("prefix_length", [0, 1, 6, 20, 600])
    @pytest.mark.parametrize("detail_length", [0, 1, 13, 14, 1000])
    def test_never_exceeds_the_limit_and_stays_on_one_line(self, limit, prefix_length, detail_length):
        message = failure_with_detail("p" * prefix_length, "d" * detail_length, limit=limit)
        assert len(message) <= limit
        assert "\n" not in message

    def test_a_prefix_longer_than_the_limit_is_cut_and_the_detail_is_left_out(self):
        message = failure_with_detail("p" * 600, "detail")
        assert len(message) == FAILURE_MESSAGE_LIMIT
        assert message.endswith("…")
        assert "detail" not in message

    def test_windows_line_endings_become_single_spaces(self):
        message = failure_with_detail("Failed", "first\r\nsecond\r\n\r\nthird\r\n")
        assert message == "Failed: first second third"

    def test_a_single_long_line_keeps_its_end(self):
        message = failure_with_detail("Failed", "a" * 2000 + "z" * 10)
        assert message.startswith("Failed: …")
        assert message.endswith("z" * 10)
        assert len(message) == FAILURE_MESSAGE_LIMIT

    @pytest.mark.parametrize(
        "line",
        [
            "Error opening input files: No such file or directory",
            "Invalid configuration: unknown option",
            "The libavcodec: encoder not found",
            "libavcodec: bad thing happened",
            "libpostproc: bad thing happened",
        ],
    )
    def test_real_error_lines_are_not_mistaken_for_the_banner(self, line):
        assert failure_with_detail("Failed", f"ffmpeg version 7.1\n  configuration: --enable-foo\n{line}\n") == f"Failed: {line}"

    def test_a_cut_detail_starts_with_an_ellipsis_and_respects_the_limit(self):
        message = failure_with_detail("Failed", "x" * 1000 + " the end", limit=60)
        assert message.startswith("Failed: …")
        assert message.endswith("the end")
        assert len(message) <= 60


class TestOneLineMessage:
    def test_collapses_newlines_tabs_and_runs_of_spaces(self):
        assert one_line_message("  first\nsecond\t\tthird   fourth\r\n") == "first second third fourth"

    def test_empty_and_blank_text_give_an_empty_string(self):
        assert one_line_message("") == ""
        assert one_line_message(" \n\t ") == ""

    def test_text_at_the_limit_is_kept(self):
        text = "a" * FAILURE_MESSAGE_LIMIT
        assert one_line_message(text) == text

    def test_longer_text_is_cut_to_the_limit_with_an_ellipsis(self):
        result = one_line_message("a" * (FAILURE_MESSAGE_LIMIT + 50))
        assert len(result) == FAILURE_MESSAGE_LIMIT
        assert result == "a" * (FAILURE_MESSAGE_LIMIT - 1) + "…"

    def test_custom_limit(self):
        assert one_line_message("abcdefghij", limit=5) == "abcd…"


class TestExceptionMessage:
    def test_long_tool_error_keeps_the_head_and_the_final_line(self):
        text = "Could not join the clips: " + "[mov @ 0x1] noise\n" * 100 + "out.mp4: Invalid data found when processing input\n"
        message = exception_message(text)
        assert message.startswith("Could not join the clips: ")
        assert message.endswith("Invalid data found when processing input")
        assert "\n" not in message
        assert len(message) <= FAILURE_MESSAGE_LIMIT

    def test_short_message_is_only_collapsed(self):
        assert exception_message("Bad  input:\n  nope\n") == "Bad input: nope"

    def test_long_message_without_a_head_keeps_its_beginning(self):
        message = exception_message("a" * (FAILURE_MESSAGE_LIMIT + 50))
        assert message == one_line_message("a" * (FAILURE_MESSAGE_LIMIT + 50))
        assert len(message) == FAILURE_MESSAGE_LIMIT

    def test_long_message_with_a_distant_colon_keeps_its_beginning(self):
        text = "a" * 300 + ": " + "z" * 600
        message = exception_message(text)
        assert message.startswith("a" * 100)
        assert len(message) <= FAILURE_MESSAGE_LIMIT

    def test_empty_text_gives_an_empty_string(self):
        assert exception_message("  ") == ""

    def test_url_in_the_head_is_not_split_at_its_scheme(self):
        head = "Failed to download https://huggingface.co/org/model/resolve/main/weights.safetensors"
        text = head + ": " + "noise line\n" * 100 + "401 Client Error: Unauthorized\n"
        message = exception_message(text)
        assert message.startswith(head + ": ")
        assert message.endswith("Unauthorized")
        assert "\n" not in message
        assert len(message) <= FAILURE_MESSAGE_LIMIT

    def test_windows_path_in_the_head_is_not_split_at_the_drive_letter(self):
        head = "C:\\models\\x.safetensors"
        text = head + ": " + "noise line\n" * 100 + "invalid header length\n"
        message = exception_message(text)
        assert message.startswith(head + ": ")
        assert message.endswith("invalid header length")
        assert "\n" not in message
        assert len(message) <= FAILURE_MESSAGE_LIMIT


class TestEmitGenerationFinishedMessage:
    CONTEXT = {"mode": "image", "run_index": 0}

    def _emit(self, **kwargs):
        events: list[dict] = []
        emit_generation_finished(events.append, event_context=self.CONTEXT, **kwargs)
        return events[0]

    def test_message_is_included_when_given(self):
        assert self._emit(status="failed", message="boom")["message"] == "boom"

    @pytest.mark.parametrize("message", [None, ""])
    def test_no_message_key_without_text(self, message):
        assert "message" not in self._emit(status="failed", message=message)

    def test_default_payload_has_no_message_key(self):
        assert "message" not in self._emit(status="success", filename="a.png")
