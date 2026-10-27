"""Tests for zvisiongenerator.review — the shared mechanics of a review before the upscale."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.review import await_review_decision, review_enabled


@pytest.mark.parametrize(
    ("upscale", "review", "expected"),
    [(2, True, True), (2, False, False), (None, True, False), (0, True, False), (None, False, False)],
)
def test_review_is_enabled_only_with_an_upscale_and_the_flag(upscale, review, expected):
    assert review_enabled(SimpleNamespace(upscale=upscale, review=review)) is expected


def test_review_is_disabled_for_args_without_the_attributes():
    assert review_enabled(SimpleNamespace()) is False


def test_await_review_decision_opens_the_review_before_announcing_it():
    calls: list[str] = []
    skip = MagicMock()
    skip.begin_review.side_effect = lambda: calls.append("begin")
    skip.wait_for_review.side_effect = lambda: (calls.append("wait"), "approve")[1]

    def _callback(event):
        calls.append(event["type"])

    decision = await_review_decision(skip, progress_callback=_callback, mode="image", completed_iterations=1, total_iterations=2, review={"iteration": 2})

    assert decision == "approve"
    assert calls == ["begin", "job_paused", "wait", "job_resumed"]
