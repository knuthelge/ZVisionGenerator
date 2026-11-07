"""The Stable Audio 3 caption: the music TrackType prefix, the music words, the Length suffix and the text it must leave alone."""

from __future__ import annotations

import pytest

from zvisiongenerator.backends.audio.sa3_caption import sa3_caption


class TestSound:
    def test_a_sound_gets_only_the_length_suffix(self):
        assert sa3_caption("steady rain on a tin roof", kind="sound", seconds=12) == "steady rain on a tin roof. Length: 12 seconds"

    def test_a_length_that_is_not_whole_has_one_decimal(self):
        assert sa3_caption("a door slam", kind="sound", seconds=2.5).endswith("Length: 2.5 seconds")
        assert sa3_caption("a door slam", kind="sound", seconds=7.25).endswith("Length: 7.2 seconds")

    def test_a_whole_length_given_as_a_float_has_no_decimal(self):
        assert sa3_caption("a door slam", kind="sound", seconds=5.0).endswith("Length: 5 seconds")

    def test_music_settings_are_ignored_for_a_sound(self):
        assert sa3_caption("rain", kind="sound", seconds=5, tempo=120, key="C") == "rain. Length: 5 seconds"


class TestMusic:
    def test_music_names_no_vocals_and_has_no_tempo_or_key_on_auto(self):
        assert sa3_caption("warm lo-fi beat", kind="music", seconds=30) == "TrackType: Music, warm lo-fi beat, no vocals. Length: 30 seconds"

    def test_tempo_alone(self):
        assert sa3_caption("warm lo-fi beat", kind="music", seconds=30, tempo=84) == "TrackType: Music, warm lo-fi beat, 84 BPM, no vocals. Length: 30 seconds"

    @pytest.mark.parametrize(("key", "words"), [("F#m", "F# minor"), ("C", "C major"), ("Bbm", "Bb minor")])
    def test_key_alone_is_written_in_words(self, key, words):
        assert sa3_caption("warm lo-fi beat", kind="music", seconds=30, key=key) == f"TrackType: Music, warm lo-fi beat, {words}, no vocals. Length: 30 seconds"

    def test_tempo_and_key_together(self):
        assert sa3_caption("warm lo-fi beat", kind="music", seconds=45.5, tempo=84, key="Am") == "TrackType: Music, warm lo-fi beat, 84 BPM, A minor, no vocals. Length: 45.5 seconds"


class TestUserText:
    def test_one_trailing_period_is_dropped_and_no_more(self):
        assert sa3_caption("rain on a roof.", kind="sound", seconds=5) == "rain on a roof. Length: 5 seconds"
        assert sa3_caption("rain...", kind="sound", seconds=5) == "rain... Length: 5 seconds"

    def test_a_music_prompt_that_starts_with_tracktype_gets_no_prefix(self):
        assert sa3_caption("TrackType: Jazz, a ballad", kind="music", seconds=3) == "TrackType: Jazz, a ballad, no vocals. Length: 3 seconds"
        assert sa3_caption("tracktype: Music, a ballad", kind="music", seconds=3) == "tracktype: Music, a ballad, no vocals. Length: 3 seconds"

    def test_a_prompt_with_a_length_gets_no_suffix(self):
        assert sa3_caption("a footstep. Length: 3 seconds", kind="sound", seconds=9) == "a footstep. Length: 3 seconds"
        assert sa3_caption("a footstep, LENGTH: 3 seconds", kind="sound", seconds=9) == "a footstep, LENGTH: 3 seconds"

    def test_a_prompt_with_both_is_returned_as_it_is(self):
        text = "a footstep. Length: 3 seconds"

        assert sa3_caption(text, kind="sound", seconds=9) == text

    def test_a_long_prompt_is_never_cut(self):
        prompt = " ".join(f"word{index}" for index in range(400))

        assert prompt in sa3_caption(prompt, kind="music", seconds=30, tempo=100, key="C")

    def test_an_empty_prompt_leaves_no_stray_comma(self):
        assert sa3_caption("", kind="music", seconds=5) == "TrackType: Music, no vocals. Length: 5 seconds"

    def test_a_kind_stable_audio_does_not_make_is_refused(self):
        with pytest.raises(ValueError, match="sound or music"):
            sa3_caption("hello", kind="speech", seconds=5)
