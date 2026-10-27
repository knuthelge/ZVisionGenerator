"""Tests for the Ideogram 4 settings both image backends share."""

from __future__ import annotations

import json

import pytest

from zvisiongenerator.core.ideogram4 import caption_text, sampler_settings, wrap_ideogram4_prompt


class TestCaptions:
    def test_plain_text_becomes_a_minimal_structured_caption(self):
        assert wrap_ideogram4_prompt("a red car") == {
            "high_level_description": "a red car",
            "compositional_deconstruction": {"background": "a red car", "elements": []},
        }

    @pytest.mark.parametrize("prompt", ['{"high_level_description": "x"}', '  {\n  "high_level_description": "x"\n}\n'])
    def test_a_json_caption_keeps_its_content_on_one_line(self, prompt):
        assert wrap_ideogram4_prompt(prompt) == '{"high_level_description":"x"}'
        assert caption_text(prompt) == '{"high_level_description":"x"}'

    def test_text_that_only_looks_like_json_is_kept_as_written(self):
        assert caption_text("{a red car") == "{a red car"

    def test_the_text_for_a_string_backend_is_compact_json_that_keeps_unicode(self):
        text = caption_text("blå bil")

        assert json.loads(text)["high_level_description"] == "blå bil"
        assert " " not in text.replace("blå bil", "")  # no padding between tokens
        assert "å" in text


class TestSamplerSettings:
    def test_the_run_uses_the_given_steps_and_instants_noise_schedule(self):
        sampler = sampler_settings(8, 1.0)

        assert (sampler.steps, sampler.mu, sampler.std) == (8, 0.0, 1.75)

    def test_a_guidance_other_than_one_is_ignored_with_a_warning(self):
        with pytest.warns(UserWarning, match="guidance 7 is ignored"):
            sampler = sampler_settings(8, 7.0)

        assert sampler.steps == 8

    def test_the_built_in_guidance_raises_no_warning(self, recwarn):
        sampler_settings(8, 1.0)

        assert not recwarn.list
