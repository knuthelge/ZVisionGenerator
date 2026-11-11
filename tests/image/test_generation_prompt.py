"""Tests for the prompt and negative prompt an image generation sends to the backend."""

from __future__ import annotations

import pytest

from zvisiongenerator.workflows.generation_prompt import BACKGROUND_SUFFIX, conditioning_prompt, generation_negative, generation_prompt


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


def test_the_background_phrase_is_added_only_when_transparent():
    assert conditioning_prompt("a cat", transparent=True) == "a cat, isolated on a plain light grey studio background"
    assert conditioning_prompt("a cat", transparent=False) == "a cat"


def test_an_extend_puts_its_instruction_in_front_and_nothing_else_changes():
    assert conditioning_prompt("more sky", transparent=False, extend=True) == "Expand the image to fill the empty canvas.\n\nmore sky"
    assert conditioning_prompt("more sky", transparent=False) == "more sky"


def test_a_json_caption_is_never_given_the_background_phrase():
    caption = '{"summary": "a cat"}'

    assert conditioning_prompt(caption, transparent=True) == caption


def test_the_background_phrase_is_not_deduplicated():
    assert conditioning_prompt("a cat" + BACKGROUND_SUFFIX, transparent=True) == "a cat" + BACKGROUND_SUFFIX * 2
