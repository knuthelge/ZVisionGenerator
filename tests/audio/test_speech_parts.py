"""Tests for reading long text in parts: the text splitter, the Delivery marks that cut it into segments, and the joiner of the parts' audio."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests._paths import FIXTURES
from zvisiongenerator.core.audio_types import SpeechDelivery
from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import SpeechPart, has_marks, known_marks_only, marks_used, speech_parts, strip_marks, unknown_mark_message, unknown_marks, without_trailing_marks

RATE = 24000
SETTINGS = {"join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}


def _split(text, max_chars=500, *, max_text_chars=10_000):
    return tuple(part.text for part in speech_parts(text, None, part_chars=max_chars, max_text_chars=max_text_chars))


def _sentence(length: int, letter: str) -> str:
    """Return one sentence of exactly *length* characters (words of one letter, a full stop)."""
    body = " ".join([letter * 9] * length)[: length - 1].rstrip()
    return body + "." * (length - len(body))


class TestSplit:
    def test_whole_sentences_are_packed_up_to_the_limit(self):
        sentences = [_sentence(200, letter) for letter in "abcde"]

        parts = _split(" ".join(sentences))

        assert parts == (f"{sentences[0]} {sentences[1]}", f"{sentences[2]} {sentences[3]}", sentences[4])
        assert all(len(part) <= 500 for part in parts)

    def test_short_text_is_one_part(self):
        assert _split("Hello there. How are you?") == ("Hello there. How are you?",)

    def test_a_long_sentence_is_cut_into_clauses_packed_with_the_sentence_before(self):
        short = _sentence(100, "s")
        clauses = [_sentence(300, "a")[:-1] + ",", _sentence(300, "b")[:-1] + ",", _sentence(100, "c")]
        long_sentence = " ".join(clauses)
        assert len(long_sentence) > 500

        parts = _split(f"{short} {long_sentence}")

        assert parts == (f"{short} {clauses[0]}", f"{clauses[1]} {clauses[2]}")

    @pytest.mark.parametrize("mark", [";", ":", "—", ","])
    def test_every_clause_mark_ends_a_clause(self, mark):
        first, second = "a" * 300 + mark, "b" * 300 + "."

        assert _split(f"{first} {second}") == (first, second)

    def test_a_clause_without_marks_is_cut_at_the_last_space_before_the_limit(self):
        text = " ".join(["word"] * 300)

        parts = _split(text)

        assert len(parts) == 3
        assert all(len(part) <= 500 for part in parts)
        assert " ".join(parts) == text

    def test_a_clause_without_spaces_is_cut_at_the_limit(self):
        assert [len(part) for part in _split("a" * 1100)] == [500, 500, 100]

    def test_line_breaks_end_a_sentence(self):
        assert _split("One\nTwo", max_chars=5) == ("One", "Two")

    def test_full_width_stops_need_no_space(self):
        assert _split("你好。再见。", max_chars=3) == ("你好。", "再见。")

    def test_a_decimal_point_is_not_a_sentence_end(self):
        assert _split("Pi is 3.14. Done.", max_chars=11) == ("Pi is 3.14.", "Done.")

    def test_blank_text_has_no_parts(self):
        assert _split(" \n ") == ()


class TestLineBreaks:
    """A line break the writer put between two units of a part reaches the model: Long pause is a line break."""

    def test_a_line_break_between_sentences_in_one_part_stays_a_line_break(self):
        assert _split("One.\nTwo.") == ("One.\nTwo.",)

    def test_a_pause_followed_by_a_line_break_keeps_both(self):
        assert _split("Wait…\nWhat?") == ("Wait…\nWhat?",)

    def test_a_blank_line_between_sentences_is_kept_as_a_line_break(self):
        assert _split("One.\n\nTwo.") == ("One.\nTwo.",)

    def test_units_the_writer_ended_with_a_space_are_still_joined_by_a_space(self):
        assert _split("One. Two.\nThree. Four.") == ("One. Two.\nThree. Four.",)

    def test_only_the_breaks_the_writer_made_are_line_breaks(self):
        assert _split("One. Two.\nThree.", max_chars=9) == ("One. Two.", "Three.")

    def test_a_break_at_the_end_of_a_long_sentence_follows_its_last_clause(self):
        first, second = "a" * 300 + ",", "b" * 150 + "."

        assert _split(f"{first} {second}\nTail.", max_chars=400) == (first, f"{second}\nTail.")

    def test_a_break_at_the_edge_of_a_part_is_dropped_with_the_part_boundary(self):
        assert _split("One.\nTwo.", max_chars=5) == ("One.", "Two.")


DEFAULT = SpeechDelivery()


def _parts(text, delivery=DEFAULT, *, part_chars=500, max_text_chars=10_000):
    return speech_parts(text, delivery, part_chars=part_chars, max_text_chars=max_text_chars)


class TestMarks:
    def test_text_without_marks_is_the_split_text_with_the_settings_delivery(self):
        delivery = SpeechDelivery(emotion="sad")

        assert _parts("Hello. How are you?", delivery) == (SpeechPart("Hello. How are you?", delivery),)

    def test_a_mark_starts_a_segment_with_that_marks_value_on_top_of_the_settings(self):
        settings = SpeechDelivery(pace="slow", pitch="low")

        parts = _parts("Hello. [sad] I am sad. [whisper] Quiet now.", settings)

        assert parts == (
            SpeechPart("Hello.", settings),
            SpeechPart("I am sad.", SpeechDelivery(emotion="sad", pace="slow", pitch="low")),
            SpeechPart("Quiet now.", SpeechDelivery(manner="whisper", pace="slow", pitch="low")),
        )

    def test_marks_do_not_stack(self):
        parts = _parts("[sad] One. [shout] Two.")

        assert parts[1].delivery == SpeechDelivery(manner="shout")  # not sad as well

    def test_a_mark_holds_until_the_next_mark_or_the_end(self):
        parts = _parts("[angry] One. Two. Three. [afraid] Four.", part_chars=6)

        assert [(part.text, part.delivery.emotion) for part in parts] == [("One.", "angry"), ("Two.", "angry"), ("Three.", "angry"), ("Four.", "afraid")]

    def test_as_set_returns_to_the_settings_delivery(self):
        settings = SpeechDelivery(emotion="proud", pace="fast")

        parts = _parts("[sad] Down. [as set] Back.", settings)

        assert parts[0].delivery == SpeechDelivery(emotion="sad", pace="fast")
        assert parts[1].delivery == settings

    def test_marks_are_case_insensitive_and_forgive_extra_spaces(self):
        parts = _parts("[SAD] One. [ As  Set ] Two.", SpeechDelivery(emotion="proud"))

        assert [part.delivery.emotion for part in parts] == ["sad", "proud"]

    def test_every_mark_in_the_vocabulary_is_recognised(self):
        from zvisiongenerator.utils.audio_settings import MARKS

        for name, target in MARKS.items():
            (part,) = _parts(f"[{name}] Text.", SpeechDelivery(emotion="proud"))
            expected = SpeechDelivery(emotion="proud") if target is None else SpeechDelivery(**{"emotion": "proud", target[0]: target[1]})
            assert part.delivery == expected, name

    def test_a_part_never_crosses_a_mark_even_when_both_sides_would_fit_in_one(self):
        parts = _parts("Short. [sad] Also short.")

        assert [part.text for part in parts] == ["Short.", "Also short."]

    def test_each_segment_is_split_by_the_part_limit_on_its_own(self):
        long_segment = " ".join(["Word one two."] * 4)

        parts = _parts(f"[sad] {long_segment} [as set] End.", part_chars=30)

        assert [part.text for part in parts] == ["Word one two. Word one two.", "Word one two. Word one two.", "End."]
        assert [part.delivery.emotion for part in parts] == ["sad", "sad", "neutral"]

    def test_the_marks_themselves_do_not_count_against_the_part_limit(self):
        (part,) = _parts("[determined] Exactly twenty chars", part_chars=len("Exactly twenty chars"))

        assert part.text == "Exactly twenty chars"

    def test_empty_segments_make_no_parts(self):
        assert _parts("[sad] [angry]   [whisper]") == ()
        assert [part.text for part in _parts("[sad][angry] Hi.")] == ["Hi."]

    def test_a_line_break_inside_a_segment_survives_in_its_part(self):
        parts = _parts("[sad] First…\nSecond. [as set] Third.")

        assert [part.text for part in parts] == ["First…\nSecond.", "Third."]

    def test_a_line_break_just_before_a_mark_ends_the_part(self):
        assert [part.text for part in _parts("One.\n[sad] Two.")] == ["One.", "Two."]

    def test_other_brackets_are_plain_text(self):
        text = "See [1] and [a1b2] and [this is a much too long bracketed run] and (sad)."

        (part,) = _parts(text)

        assert part.text == text

    def test_the_text_budget_counts_the_text_without_marks(self):
        assert _parts("[sad]" + "a" * 10, max_text_chars=10)
        with pytest.raises(ValueError, match="Shorten the text"):
            _parts("[sad]" + "a" * 11, max_text_chars=10)

    def test_text_over_the_budget_is_refused_with_the_splitters_message(self):
        with pytest.raises(ValueError, match="over 10,000 characters"):
            _parts("a" * 10_001)


class TestRecordingReadsAsRecorded:
    def test_without_a_delivery_the_marks_are_taken_out_and_every_part_has_none(self):
        parts = speech_parts("Hello. [sad] I am sad. [shout] Loud.", None, part_chars=10, max_text_chars=10_000)

        assert [(part.text, part.delivery) for part in parts] == [("Hello.", None), ("I am sad.", None), ("Loud.", None)]

    def test_the_text_a_mark_sat_in_is_read_whole(self):
        (part,) = speech_parts("I feel [sad] today.", None, part_chars=500, max_text_chars=10_000)

        assert part.text.split() == ["I", "feel", "today."]

    def test_the_text_budget_still_counts_the_text_without_marks(self):
        with pytest.raises(ValueError, match="Shorten"):
            speech_parts("[sad]" + "a" * 11, None, part_chars=500, max_text_chars=10)

    def test_other_brackets_stay_in_the_text(self):
        (part,) = speech_parts("Note [1] here.", None, part_chars=500, max_text_chars=10_000)

        assert part.text == "Note [1] here."


class TestFindMarks:
    def test_has_marks_sees_known_marks_only(self):
        assert has_marks("A [sad] day.") and has_marks("[AS SET]")
        assert not has_marks("A [sadd] day.") and not has_marks("No marks [1].")

    def test_the_refusal_of_a_mark_names_what_to_write_for_a_pause_or_a_laugh_and_otherwise_the_way_out(self):
        assert "type …" in unknown_mark_message("[pause]") and "write haha" in unknown_mark_message("[Laughing]")
        assert unknown_mark_message("[sadd]") == "[sadd] isn't a Delivery mark. Use one of the Delivery marks, or remove the brackets."

    def test_unknown_marks_lists_bracketed_words_that_are_not_marks_as_written(self):
        assert unknown_marks("A [sadd] day [Whisper] [really quiet] [x] [1] ([whispr])") == ("[sadd]", "[really quiet]", "[whispr]")

    def test_unknown_marks_ignores_brackets_that_are_not_two_to_twenty_letters_and_spaces(self):
        assert unknown_marks("[a] [ ] [a1] [abcdefghijklmnopqrstu] [sad] [x y]") == ("[x y]",)

    def test_text_with_only_known_marks_has_no_unknown_ones(self):
        assert unknown_marks("[sad] One [as set] two [Shout]") == ()

    def test_strip_marks_takes_out_the_known_marks_and_leaves_other_brackets(self):
        assert strip_marks("[sad] One [Whisper] two [as set] three [sadd] [1].") == " One  two  three [sadd] [1]."
        assert strip_marks("No marks.") == "No marks."

    def test_known_marks_only_writes_marks_in_their_form_and_drops_the_others(self):
        assert known_marks_only("[Sad] One. [sighs] Two [AS  SET] three [1].") == "[sad] One. Two [as set] three [1]."
        assert known_marks_only("[oops] Start\n [oops] next") == "Start\nnext"

    def test_known_marks_only_gives_text_that_has_no_unknown_marks(self):
        text = "[Sad] A [really quiet] B [whispr] C [as set] D"

        assert unknown_marks(known_marks_only(text)) == ()
        assert known_marks_only("Plain text.") == "Plain text."

    def test_marks_used_lists_each_known_mark_once_in_order(self):
        assert marks_used("[Sad] A [whisper] B [sad] C [oops] [as  set]") == ("sad", "whisper", "as set")
        assert marks_used("No marks [1].") == ()

    def test_known_marks_only_can_keep_just_the_marks_it_is_given(self):
        assert known_marks_only("[sad] A [whisper] B [oops] C", only=("sad",)) == "[sad] A B C"
        assert known_marks_only("[sad] A [whisper] B", only=()) == "A B"
        assert known_marks_only("[sad] A", only=None) == "[sad] A"

    def test_trailing_marks_are_dropped_and_marks_before_words_are_not(self):
        assert without_trailing_marks("Hello. [sad] [whisper]") == "Hello."
        assert without_trailing_marks("[sad] Hello.") == "[sad] Hello."
        assert without_trailing_marks("Hello [1] [Shout]  ") == "Hello [1]"
        assert without_trailing_marks("Hello.  ") == "Hello.  " and without_trailing_marks("[sad]") == ""

    def test_every_mark_the_web_ui_offers_is_read_as_that_mark(self):
        from zvisiongenerator.utils.audio_settings import audio_settings_contract

        base = SpeechDelivery(emotion="sad", manner="shout")
        for mark in audio_settings_contract()["marks"]:
            text = f"One. [{mark['id']}] Two."  # as the Insert menu writes it: at the start of a sentence, a space after it

            first, second = speech_parts(text, base, part_chars=500, max_text_chars=10_000)

            assert unknown_marks(text) == () and has_marks(text)
            assert (first.text, second.text) == ("One.", "Two.")
            expected = base if mark["field"] is None else SpeechDelivery(**{**base.__dict__, mark["field"]: mark["value"]})
            assert (first.delivery, second.delivery) == (base, expected)


def _tone(seconds: float, amplitude: float = 0.4) -> np.ndarray:
    """Return a cosine tone, which starts at full amplitude so a fade-in shows."""
    n = round(seconds * RATE)
    return (amplitude * np.cos(2 * np.pi * 220 * np.arange(n) / RATE)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE), dtype=np.float32)


class TestJoin:
    def test_a_long_tail_is_trimmed_to_the_edge_limit(self):
        part = np.concatenate([_tone(1.0), _silence(1.9)])

        joined = join_speech_parts([part], RATE, SETTINGS)

        assert len(joined) == RATE + round(0.150 * RATE)
        assert not joined[RATE:].any()

    def test_a_long_lead_is_trimmed_to_the_edge_limit(self):
        part = np.concatenate([_silence(1.0), _tone(1.0)])

        assert len(join_speech_parts([part], RATE, SETTINGS)) == RATE + round(0.150 * RATE)

    def test_a_short_edge_silence_is_kept(self):
        part = np.concatenate([_silence(0.05), _tone(1.0), _silence(0.1)])

        assert len(join_speech_parts([part], RATE, SETTINGS)) == len(part)

    def test_the_pause_between_parts_is_exactly_300_ms(self):
        joined = join_speech_parts([_tone(0.5), _tone(0.5)], RATE, SETTINGS)

        assert len(joined) == 2 * 12000 + 7200
        assert not joined[12000:19200].any()
        assert joined[19200 + 240 : 19200 + 480].any()

    def test_each_edge_is_faded_over_10_ms_and_nothing_else_changes(self):
        tone = _tone(0.5)

        joined = join_speech_parts([tone, tone], RATE, SETTINGS)

        first = joined[:12000]
        assert first[0] == 0.0
        assert np.all(np.diff(np.abs(first[:240]) / np.abs(tone[:240]).clip(1e-9)) >= -1e-6)
        np.testing.assert_allclose(first[240:-240], tone[240:-240], atol=1e-6)
        assert first[-1] == 0.0
        assert abs(first[-241]) > 0.0
        np.testing.assert_allclose(joined[19200:], first)

    def test_the_join_is_the_trimmed_parts_plus_the_pauses(self):
        parts = [
            np.concatenate([_silence(0.3), _tone(1.0), _silence(1.9)]),
            np.concatenate([_silence(0.05), _tone(0.5), _silence(0.2)]),
            _tone(0.7),
        ]
        trimmed = (3600 + 24000 + 3600) + (1200 + 12000 + 3600) + 16800

        assert len(join_speech_parts(parts, RATE, SETTINGS)) == trimmed + 2 * 7200

    def test_a_part_far_below_the_mean_is_raised_by_the_cap_only(self):
        loud, quiet = _tone(0.5, 0.4), _tone(0.5, 0.2)  # the quiet one is 6 dB down; the mean is 2 dB below the loud ones

        joined = join_speech_parts([loud, loud, quiet], RATE, SETTINGS)

        quiet_part = joined[38400:50400]
        assert np.abs(quiet_part[240:-240]).max() == pytest.approx(0.2 * 10 ** (3 / 20), rel=0.01)
        loud_part = joined[:12000]
        assert np.abs(loud_part[240:-240]).max() == pytest.approx(0.4 * 10 ** (-2 / 20), rel=0.01)

    def test_equal_parts_are_left_at_their_level(self):
        tone = _tone(0.5)

        joined = join_speech_parts([tone, tone], RATE, SETTINGS)

        assert np.abs(joined[240:11760]).max() == pytest.approx(0.4, rel=0.001)

    def test_a_boost_never_clips(self):
        peaky = _tone(0.5, 0.1)
        peaky[6000] = 0.9

        joined = join_speech_parts([_tone(0.5), _tone(0.5), peaky], RATE, SETTINGS)

        assert np.abs(joined).max() <= 1.0 + 1e-6

    def test_a_silent_part_is_kept_as_it_is_and_does_not_move_the_level(self):
        joined = join_speech_parts([_tone(0.5), _silence(0.5)], RATE, SETTINGS)

        assert len(joined) == 12000 + 7200 + 12000
        assert np.abs(joined[240:11760]).max() == pytest.approx(0.4, rel=0.001)

    def test_empty_parts_are_left_out(self):
        joined = join_speech_parts([np.zeros(0, dtype=np.float32), _tone(0.5)], RATE, SETTINGS)

        assert len(joined) == 12000

    def test_no_samples_is_refused(self):
        with pytest.raises(ValueError, match="no speech"):
            join_speech_parts([np.zeros(0, dtype=np.float32)], RATE, SETTINGS)


_COUNTS = json.loads((FIXTURES / "speech_character_counts.json").read_text(encoding="utf-8"))
_UNKNOWN_MARKS = json.loads((FIXTURES / "speech_unknown_marks.json").read_text(encoding="utf-8"))


class TestUnknownMarksFixture:
    """The bracketed runs the server refuses; the Web UI's ``unknownMarks`` reads the same fixture so Generate is off for exactly those texts."""

    @pytest.mark.parametrize("case", _UNKNOWN_MARKS, ids=[case["text"][:24] or "empty" for case in _UNKNOWN_MARKS])
    def test_the_runs_that_look_like_a_mark_and_are_not_one(self, case):
        assert unknown_marks(case["text"]) == tuple(case["unknown"])


class TestCharacterCount:
    """The limit counts the characters that are read; the Web UI's count (``stripMarks``) reads the same fixture."""

    @pytest.mark.parametrize("case", _COUNTS, ids=[case["text"][:24] or "empty" for case in _COUNTS])
    @pytest.mark.parametrize("delivery", [SpeechDelivery(), None], ids=["built-in", "recording"])
    def test_a_text_is_over_the_limit_by_one_character_more_than_the_fixture_says(self, case, delivery):
        count = case["characters"]

        speech_parts(case["text"], delivery, part_chars=500, max_text_chars=count)
        if count:
            with pytest.raises(ValueError, match="Shorten the text"):
                speech_parts(case["text"], delivery, part_chars=500, max_text_chars=count - 1)
