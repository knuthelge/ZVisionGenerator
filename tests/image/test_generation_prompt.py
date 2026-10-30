"""Tests for the prompt and negative prompt an image generation sends to the backend."""

from __future__ import annotations

import pytest

from zvisiongenerator.workflows.generation_prompt import generation_negative, generation_prompt


@pytest.mark.parametrize(
    ("template", "resolved", "enhanced", "expected"),
    [
        ("{a|b} cat", "a cat", None, "a cat"),
        ("{a|b} cat", "a cat", "a sleepy cat", "a sleepy cat"),
        ("{a|b} cat", None, None, "{a|b} cat"),
        ("{a|b} cat", "", None, "{a|b} cat"),
        ("{a|b} cat", "a cat", "", "{a|b} cat"),
        ("", None, None, ""),
    ],
)
def test_the_rewrite_beats_the_resolved_prompt_and_both_beat_the_template(template, resolved, enhanced, expected):
    assert generation_prompt(template, resolved, enhanced) == expected


@pytest.mark.parametrize(
    ("negative", "supports", "expected"),
    [
        ("blurry", True, "blurry"),
        ("blurry", False, None),
        (None, True, None),
        (None, False, None),
    ],
)
def test_the_negative_is_sent_only_when_the_model_takes_negative_prompts(negative, supports, expected):
    assert generation_negative(negative, supports_negative=supports) == expected
