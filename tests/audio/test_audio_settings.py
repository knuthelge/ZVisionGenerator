"""Tests for the vocabulary of the audio settings: tempo, key, speech delivery and the type of a sound."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from tests._paths import FIXTURES
from zvisiongenerator.core.audio_types import SpeechDelivery
from zvisiongenerator.utils.audio_settings import (
    AUTO_SOUND_SECONDS,
    EMOTIONS,
    MARKS,
    MUSIC_KEYS,
    PACE_READING_FACTOR,
    PACES,
    SOUND_TYPES,
    VIBES,
    audio_settings_contract,
    format_reading_time,
    key_words,
    parse_delivery,
    parse_key,
    parse_sound_type,
    parse_tempo,
    parse_vibe,
    reading_characters,
    reading_seconds,
)


class TestTempo:
    @pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), ("  ", None), ("92", 92), (" 92 ", 92), (40, 40), ("220", 220), (120.0, 120)])
    def test_a_tempo_in_range_is_a_whole_number_and_a_blank_is_auto(self, value, expected):
        assert parse_tempo(value) == expected

    @pytest.mark.parametrize("value", ["39", "221", 0, -5, "fast", "92.5", 92.5, True, "1e2"])
    def test_anything_else_is_refused_in_the_users_words(self, value):
        with pytest.raises(ValueError, match=r"Tempo must be 40 to 220 BPM\."):
            parse_tempo(value)


class TestKey:
    def test_there_are_twelve_majors_and_twelve_minors_without_repeats(self):
        assert len(MUSIC_KEYS) == len(set(MUSIC_KEYS)) == 24
        assert sum(key.endswith("m") for key in MUSIC_KEYS) == 12

    @pytest.mark.parametrize("key", MUSIC_KEYS)
    def test_every_key_id_parses_to_itself(self, key):
        assert parse_key(key) == key

    @pytest.mark.parametrize("value", [None, "", " "])
    def test_a_blank_key_is_auto(self, value):
        assert parse_key(value) is None

    @pytest.mark.parametrize("value", ["H", "c", "F♯m", "C minor", 3])
    def test_an_unknown_key_is_refused_with_the_list(self, value):
        with pytest.raises(ValueError, match=r"Key must be one of: C, Db"):
            parse_key(value)

    @pytest.mark.parametrize(("key", "words"), [("F#m", "F# minor"), ("C", "C major"), ("Bbm", "Bb minor"), ("Bm", "B minor"), ("Db", "Db major")])
    def test_a_key_in_words(self, key, words):
        assert key_words(key) == words


class TestDelivery:
    def test_nothing_given_is_the_neutral_delivery(self):
        assert parse_delivery({}) == SpeechDelivery() == SpeechDelivery("neutral", "natural", "normal", "normal", "normal")
        assert parse_delivery({"emotion": "", "pace": None}) == SpeechDelivery()

    def test_each_value_is_read_by_its_setting(self):
        values = {"emotion": "sad", "manner": "whisper", "pace": "very-slow", "pitch": "low", "expressiveness": "high"}

        assert parse_delivery(values) == SpeechDelivery("sad", "whisper", "very-slow", "low", "high")

    @pytest.mark.parametrize(("field", "value"), [("emotion", "grumpy"), ("manner", "sing"), ("pace", "fast-ish"), ("pitch", "very-high"), ("expressiveness", "Normal")])
    def test_an_unknown_value_is_refused_with_its_setting_named(self, field, value):
        with pytest.raises(ValueError, match=f"^{field.capitalize()} must be one of: "):
            parse_delivery({field: value})

    def test_every_vibe_is_made_of_known_values(self):
        for delivery in VIBES.values():
            assert parse_delivery(asdict(delivery)) == delivery

    def test_the_emotions_are_neutral_and_twenty_grouped_ones(self):
        assert EMOTIONS[0] == "neutral" and len(EMOTIONS) == 21 and len(set(EMOTIONS)) == 21


class TestMarks:
    def test_a_mark_sets_one_field_and_as_set_sets_none(self):
        assert MARKS["sad"] == ("emotion", "sad")
        assert MARKS["whisper"] == ("manner", "whisper") and MARKS["shout"] == ("manner", "shout")
        assert MARKS["as set"] is None
        assert "neutral" not in MARKS

    def test_every_mark_targets_a_value_the_delivery_accepts(self):
        for target in filter(None, MARKS.values()):
            assert parse_delivery({target[0]: target[1]})


class TestSoundType:
    @pytest.mark.parametrize(("value", "expected"), [(None, "auto"), ("", "auto"), ("effect", "effect"), (" one-shot ", "one-shot")])
    def test_a_sound_type_is_read_and_a_blank_is_auto(self, value, expected):
        assert parse_sound_type(value) == expected

    def test_an_unknown_type_is_refused_with_the_list(self):
        with pytest.raises(ValueError, match=r"Sound type must be one of: auto, ambience, effect, one-shot\."):
            parse_sound_type("music")

    def test_every_type_has_an_auto_length(self):
        assert set(AUTO_SOUND_SECONDS) == set(SOUND_TYPES)
        assert AUTO_SOUND_SECONDS == {"auto": 12.0, "ambience": 12.0, "effect": 5.0, "one-shot": 2.0}


class TestVibe:
    @pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), (" announcer ", "announcer"), ("whispered-secret", "whispered-secret")])
    def test_a_vibe_is_read_and_a_blank_is_none(self, value, expected):
        assert parse_vibe(value) == expected

    def test_an_unknown_vibe_is_refused_with_the_list(self):
        with pytest.raises(ValueError, match=r"Vibe must be one of: calm-narrator, storyteller, excited-host, announcer, sad-confession, whispered-secret\."):
            parse_vibe("villain")

    def test_the_vibes_the_command_takes_are_the_ones_the_contract_serves_to_the_web_ui(self):
        assert [vibe["id"] for vibe in audio_settings_contract()["vibes"]] == list(VIBES)


class TestContract:
    def test_the_contract_is_json_ready_and_lists_every_vocabulary(self):
        contract = audio_settings_contract()

        assert json.loads(json.dumps(contract)) == contract
        assert contract["tempo"] == {"min": 40, "max": 220}
        assert [key["id"] for key in contract["keys"]] == list(MUSIC_KEYS)
        assert contract["keys"][6] == {"id": "F#", "words": "F# major", "mode": "major"}
        assert [pace["id"] for pace in contract["paces"]] == list(PACES)
        assert contract["auto_seconds"] == AUTO_SOUND_SECONDS and "default_music_seconds" not in contract
        assert contract["delivery_defaults"] == {"emotion": "neutral", "manner": "natural", "pace": "normal", "pitch": "normal", "expressiveness": "normal"}

    def test_emotions_carry_their_group_and_neutral_has_none(self):
        emotions = {emotion["id"]: emotion["group"] for emotion in audio_settings_contract()["emotions"]}

        assert emotions["neutral"] is None and emotions["elated"] == "Bright" and emotions["disgusted"] == "Heavy"

    def test_marks_include_as_set_with_no_target(self):
        marks = {mark["id"]: mark for mark in audio_settings_contract()["marks"]}

        assert marks["as set"] == {"id": "as set", "field": None, "value": None}
        assert marks["shout"] == {"id": "shout", "field": "manner", "value": "shout"}

    def test_reading_rates_cover_every_pace(self):
        reading = audio_settings_contract()["reading"]

        assert reading["chars_per_second"] == 14.0 and set(reading["pace_factor"]) == set(PACES) == set(PACE_READING_FACTOR)

    def test_reading_time_and_characters_are_each_others_inverse_at_every_pace(self):
        assert reading_seconds(280) == 20.0 and reading_characters(20) == 280
        assert reading_seconds(0) == 0.0
        for pace in PACES:
            assert abs(reading_seconds(reading_characters(30, pace), pace) - 30) < 0.1
        assert reading_characters(20, "slow") == round(20 * 14 * 0.81) and reading_seconds(280, "no-such-pace") == 20.0

    @pytest.mark.parametrize(("seconds", "text"), [(0, "0:00"), (5, "0:05"), (59.6, "1:00"), (84, "1:24"), (600, "10:00")])
    def test_a_reading_time_is_written_as_minutes_and_seconds(self, seconds, text):
        assert format_reading_time(seconds) == text

    def test_the_web_ui_fixture_is_the_served_vocabulary(self):
        # The Vitest suites read this file as the payload's `audio_settings`. After a change to the vocabulary, rewrite it with
        # `json.dump(audio_settings_contract(), fp, indent=2, ensure_ascii=False)`.
        fixture = json.loads((FIXTURES / "audio_settings_contract.json").read_text(encoding="utf-8"))

        assert fixture == audio_settings_contract()
