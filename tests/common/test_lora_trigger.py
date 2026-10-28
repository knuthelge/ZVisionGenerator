"""Tests for reading a LoRA's trigger text from a model card and a file header."""

from __future__ import annotations

import json

import pytest

from zvisiongenerator.utils.lora_trigger import MAX_TRIGGER_LENGTH, caption_trigger, card_triggers, clean_trigger


class TestCardTriggers:
    @pytest.mark.parametrize(
        ("prompt", "words"),
        [
            ("t3chnic4lly", "t3chnic4lly"),
            ("t3chnic4lly, 1960s film", "t3chnic4lly, 1960s film"),
            ("  Pixel Art ;  retro", "Pixel Art, retro"),
            ("first line\nsecond line", "first line, second line"),
            (["", "ohwx person", "other, ohwx PERSON"], "ohwx person, other"),
            ("Clay animation, Clay.", "Clay animation, Clay"),
        ],
    )
    def test_splits_the_prompt_into_comma_separated_words(self, prompt, words):
        assert card_triggers(prompt) == words

    @pytest.mark.parametrize("prompt", [None, "", "   ", ",", 5, [], [""]])
    def test_names_nothing_for_an_empty_or_non_text_prompt(self, prompt):
        assert card_triggers(prompt) is None

    def test_a_long_prompt_keeps_whole_words_up_to_the_length_limit(self):
        words = [f"word{index}" for index in range(100)]
        line = card_triggers(", ".join(words))
        assert line is not None and len(line) <= MAX_TRIGGER_LENGTH
        assert line.split(", ") == words[: len(line.split(", "))]

    def test_a_single_word_over_the_limit_is_cut(self):
        assert len(card_triggers("word" * 100) or "") <= MAX_TRIGGER_LENGTH


class TestCaptionTrigger:
    def test_one_distinct_caption_is_the_trigger(self):
        metadata = {"ss_tag_frequency": json.dumps({"10_a": {"t3chnic4lly": 40}, "20_b": {"t3chnic4lly": 12}})}

        assert caption_trigger(metadata) == "t3chnic4lly"

    def test_a_vocabulary_of_tags_names_no_trigger(self):
        metadata = {"ss_tag_frequency": json.dumps({"10_a": {"1girl": 5, "smile": 3}})}

        assert caption_trigger(metadata) is None

    @pytest.mark.parametrize("metadata", [{}, {"ss_tag_frequency": "not json"}, {"ss_tag_frequency": "[1]"}, {"ss_tag_frequency": json.dumps({"a": {}})}])
    def test_missing_or_unreadable_frequencies_name_no_trigger(self, metadata):
        assert caption_trigger(metadata) is None


def test_clean_trigger_keeps_one_trimmed_line():
    assert clean_trigger("  a \n b ") == "a b"
    assert clean_trigger("   ") is None
    assert clean_trigger(None) is None
