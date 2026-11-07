"""Tests for the audio option matrix the Web UI reads as ``prompt_enhancer.audio_matrix``."""

from __future__ import annotations

import json

from tests._paths import FIXTURES
from zvisiongenerator.utils.prompt_enhance_audio import audio_matrix_contract, audio_settings_from_mapping

SOUND_AXES = ("space", "distance", "over_time", "movement")


class TestAudioMatrixFixture:
    def test_the_web_ui_fixture_is_the_served_matrix(self):
        # The Vitest suites read this file as the payload's `audio_matrix`. After a change to the axes, rewrite it with
        # `json.dump(audio_matrix_contract(), fp, indent=2, ensure_ascii=False)`.
        fixture = json.loads((FIXTURES / "audio_enhance_matrix.json").read_text(encoding="utf-8"))

        assert fixture == audio_matrix_contract()

    def test_a_sound_popover_offers_four_single_choice_axes_that_keep_by_default(self):
        axes = {axis["key"]: axis for axis in audio_matrix_contract()["axes"] if axis["kind"] == "sound"}

        assert tuple(axes) == SOUND_AXES
        assert all(not axis["multi"] and axis["default"] == ["keep"] and axis["options"][0]["slug"] == "keep" for axis in axes.values())

    def test_every_option_the_popover_offers_is_accepted_for_a_sound(self):
        for axis in audio_matrix_contract()["axes"]:
            if axis["kind"] != "sound":
                continue
            for option in axis["options"]:
                assert audio_settings_from_mapping({axis["key"]: option["slug"]}, task="sound")
