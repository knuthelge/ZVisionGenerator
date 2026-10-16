"""Tests for the Ideogram 4 settings both image backends share."""

from __future__ import annotations

import json

import pytest

from zvisiongenerator.core.ideogram4 import IDEOGRAM4_INITIAL_SIGMA, caption_text, sampler_settings, wrap_ideogram4_prompt


class TestCaptions:
    def test_plain_text_becomes_a_minimal_structured_caption(self):
        assert wrap_ideogram4_prompt("a red car") == {
            "high_level_description": "a red car",
            "compositional_deconstruction": {"background": "a red car", "elements": []},
        }

    @pytest.mark.parametrize("prompt", ['{"high_level_description": "x"}', '  {"a": 1}'])
    def test_a_json_caption_is_kept_as_written(self, prompt):
        assert wrap_ideogram4_prompt(prompt) == prompt
        assert caption_text(prompt) == prompt

    def test_the_text_for_a_string_backend_is_compact_json_that_keeps_unicode(self):
        text = caption_text("blå bil")

        assert json.loads(text)["high_level_description"] == "blå bil"
        assert " " not in text.replace("blå bil", "")  # no padding between tokens
        assert "å" in text


class TestSamplerSettings:
    def test_the_default_is_the_20_step_preset_with_two_polish_steps(self):
        sampler = sampler_settings(20, 7.0, steps_explicit=False, guidance_explicit=False)

        assert sampler.steps == 20
        assert sampler.guidance_schedule == (7.0,) * 18 + (3.0,) * 2
        assert sampler.guidance_scale is None
        assert (sampler.mu, sampler.std) == (0.0, 1.75)

    def test_the_configured_steps_are_ignored_unless_asked_for(self):
        assert sampler_settings(50, 7.0, steps_explicit=False, guidance_explicit=False).steps == 20

    def test_explicit_steps_keep_the_polish_steps_at_the_end(self):
        sampler = sampler_settings(30, 7.0, steps_explicit=True, guidance_explicit=False)

        assert len(sampler.guidance_schedule) == 30
        assert sampler.guidance_schedule[:28] == (7.0,) * 28 and sampler.guidance_schedule[28:] == (3.0, 3.0)

    @pytest.mark.parametrize("steps", [1, 2, 3])
    def test_a_very_short_run_keeps_at_least_one_main_step(self, steps):
        schedule = sampler_settings(steps, 7.0, steps_explicit=True, guidance_explicit=False).guidance_schedule

        assert len(schedule) == steps and schedule[0] == 7.0

    def test_explicit_guidance_is_one_constant_for_every_step(self):
        sampler = sampler_settings(20, 5.5, steps_explicit=False, guidance_explicit=True)

        assert (sampler.guidance_scale, sampler.guidance_schedule, sampler.steps) == (5.5, None, 20)

    def test_either_a_schedule_or_a_constant_is_set_never_both(self):
        for guidance_explicit in (False, True):
            sampler = sampler_settings(20, 7.0, steps_explicit=False, guidance_explicit=guidance_explicit)

            assert (sampler.guidance_schedule is None) != (sampler.guidance_scale is None)


def test_the_first_sigma_default_is_just_above_pure_noise():
    assert IDEOGRAM4_INITIAL_SIGMA == 1.004
