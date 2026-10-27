"""Tests for zvisiongenerator.image_review — the review callable of one image generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.image_review import make_image_review
from zvisiongenerator.review import REVIEW_PROMPT, ReviewOutcome
from zvisiongenerator.utils.interactive import SkipSignal


def _build(skip, outcome, *, progress_callback=None):
    return make_image_review(skip, outcome, progress_callback=progress_callback, ran_iterations=3, attempt=2, completed_iterations=lambda: 2, total_iterations=8)


class TestTerminalReview:
    """Without a progress callback the job runs in a terminal and the render goes to a temporary file."""

    def test_preview_file_exists_during_the_prompt_and_is_gone_after(self, tmp_path, capsys):
        seen: dict[str, object] = {}
        skip = MagicMock(spec=SkipSignal)

        def _decide():
            printed = capsys.readouterr().out
            path = Path(next(line for line in printed.splitlines() if line.startswith("Small render: ")).removeprefix("Small render: "))
            seen["path"] = path
            seen["exists"] = path.is_file()
            seen["size"] = Image.open(path).size
            seen["prompt_printed"] = REVIEW_PROMPT in printed
            return "approve"

        skip.wait_for_review.side_effect = _decide
        outcome = ReviewOutcome()

        approved = _build(skip, outcome)(Image.new("RGB", (48, 32)))

        assert approved is True
        assert seen["exists"] is True
        assert seen["size"] == (48, 32)
        assert seen["prompt_printed"] is True
        assert not seen["path"].exists()
        assert outcome.decision == "approve"

    def test_preview_file_is_outside_the_output_folder(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        skip = MagicMock(spec=SkipSignal)
        skip.wait_for_review.return_value = "discard"

        _build(skip, ReviewOutcome())(Image.new("RGB", (8, 8)))

        printed = capsys.readouterr().out
        path = Path(next(line for line in printed.splitlines() if line.startswith("Small render: ")).removeprefix("Small render: "))
        assert tmp_path not in path.parents

    def test_preview_file_is_deleted_when_the_wait_raises(self, capsys):
        skip = MagicMock(spec=SkipSignal)
        skip.wait_for_review.side_effect = KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _build(skip, ReviewOutcome())(Image.new("RGB", (8, 8)))

        printed = capsys.readouterr().out
        path = Path(next(line for line in printed.splitlines() if line.startswith("Small render: ")).removeprefix("Small render: "))
        assert not path.exists()


class TestWebReview:
    """With a progress callback the render travels as the event's preview and no path is written or sent."""

    def test_emits_paused_with_review_and_preview_then_resumed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        events: list[dict] = []
        skip = SkipSignal()
        skip.set_review_decision("approve")
        image = Image.new("RGB", (8, 8))

        approved = _build(skip, ReviewOutcome(), progress_callback=events.append)(image)

        assert approved is True
        assert [event["type"] for event in events] == ["job_paused", "job_resumed"]
        assert events[0]["review"] == {"iteration": 3, "attempt": 2}
        assert events[0]["preview"] is image
        assert events[0]["completed_iterations"] == 2
        assert events[0]["total_iterations"] == 8
        assert list(tmp_path.iterdir()) == []


class TestDecisions:
    def test_discard_returns_false_and_marks_the_outcome_declined(self):
        skip = SkipSignal()
        skip.set_review_decision("discard")
        outcome = ReviewOutcome()

        assert _build(skip, outcome, progress_callback=lambda _e: None)(Image.new("RGB", (8, 8))) is False
        assert outcome.decision == "discard"
        assert outcome.declined is True

    def test_quit_returns_false_and_leaves_the_quit_queued(self):
        skip = SkipSignal()
        skip.queue_action("quit")
        outcome = ReviewOutcome()

        assert _build(skip, outcome, progress_callback=lambda _e: None)(Image.new("RGB", (8, 8))) is False
        assert outcome.decision == "quit"
        assert outcome.declined is True
        assert skip.pending() == "quit"

    def test_approve_is_not_declined(self):
        skip = SkipSignal()
        skip.set_review_decision("approve")
        outcome = ReviewOutcome()

        _build(skip, outcome, progress_callback=lambda _e: None)(Image.new("RGB", (8, 8)))

        assert outcome.declined is False

    def test_wait_is_measured(self):
        import threading

        skip = SkipSignal()
        outcome = ReviewOutcome()
        threading.Timer(0.15, skip.set_review_decision, args=("approve",)).start()

        _build(skip, outcome, progress_callback=lambda _e: None)(Image.new("RGB", (8, 8)))

        assert outcome.wait_seconds >= 0.1
