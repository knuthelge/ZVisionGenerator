"""Tests for reading long text in parts: the text splitter and the joiner of the parts' audio."""

from __future__ import annotations

import numpy as np
import pytest

from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import split_speech_text

RATE = 24000
SETTINGS = {"join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150}
TAG = "<|sfx:laughter|>"


def _split(text, max_chars=500, *, max_text_chars=10_000):
    return split_speech_text(text, max_chars, max_text_chars=max_text_chars)


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

    def test_a_tag_is_never_cut_by_the_limit(self):
        text = "x" * 495 + TAG + "y" * 100

        assert _split(text) == ("x" * 495, TAG + "y" * 100)

    def test_a_tag_is_not_split_at_its_punctuation(self):
        text = f"Hello {TAG} world <|speed:0.8|> again."

        assert _split(text) == (text,)
        parts = _split(text, max_chars=20)
        assert len(parts) > 1
        assert all(part.count("<|") == part.count("|>") for part in parts)
        assert " ".join(parts) == text

    def test_line_breaks_end_a_sentence(self):
        assert _split("One\nTwo", max_chars=5) == ("One", "Two")

    def test_full_width_stops_need_no_space(self):
        assert _split("你好。再见。", max_chars=3) == ("你好。", "再见。")

    def test_a_decimal_point_is_not_a_sentence_end(self):
        assert _split("Pi is 3.14. Done.", max_chars=11) == ("Pi is 3.14.", "Done.")

    def test_blank_text_has_no_parts(self):
        assert _split(" \n ") == ()


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
