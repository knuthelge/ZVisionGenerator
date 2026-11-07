"""Tests for the pure ABC score helpers of the YuE2 backend, against scores the model really wrote.

The fixtures in ``tests/fixtures/yue2_scores`` are model output: a planned song, a planned instrumental and its silenced
form, and two scores the model wrote after being given a header.
"""

from __future__ import annotations

import pytest

from tests._paths import FIXTURES
from zvisiongenerator.backends.audio.yue2_score import planned_meter, planned_tempo, reads_as_score, score_header, silence_vocal

SCORES = FIXTURES / "yue2_scores"


def _score(name: str) -> str:
    return (SCORES / f"{name}.abc").read_text(encoding="utf-8")


def _voice_lines(abc: str, voice: str) -> list[str]:
    """Return the music lines of *voice* after the key line."""
    lines = abc.split("\n")
    current, found = "", []
    for line in lines[next(i for i, text in enumerate(lines) if text.startswith("K:")) + 1 :]:
        if line.startswith("V:"):
            current = line[2:].strip().split()[0]
        elif line and not line.startswith("%") and current == voice:
            found.append(line)
    return found


class TestScoreHeader:
    @pytest.mark.parametrize(("name", "tempo", "key"), [("song_header_am", 90, "Am"), ("song_header_g", 90, "G")])
    def test_the_header_is_what_the_model_continued_in_the_real_scores(self, name, tempo, key):
        assert _score(name).startswith(score_header(tempo, key))

    def test_the_tempo_only_header_ends_after_the_voice_lines(self):
        header = score_header(112)

        assert header.endswith('snm="Inst."\n')
        assert "K:" not in header
        assert header.splitlines()[:5] == ["X:1", "T:", "M:4/4", "L:1/32", "Q:1/4=112"]

    def test_the_key_goes_after_the_voice_lines(self):
        assert score_header(90, "F#m") == score_header(90) + "K:F#m\n"

    def test_the_meter_and_note_length_can_be_chosen(self):
        assert score_header(76, "D", "3/4", "1/16").splitlines()[2:5] == ["M:3/4", "L:1/16", "Q:1/4=76"]


class TestPlannedMeter:
    @pytest.mark.parametrize("name", ["song_planned", "instrumental_planned", "song_header_am"])
    def test_the_meter_and_length_are_read_from_the_plans_own_lines(self, name):
        assert planned_meter(_score(name)) == ("4/4", "1/32")

    def test_the_first_meter_line_counts(self):
        assert planned_meter("X:1\nM:3/4\nL:1/16\nQ:1/4=90\nK:C\nM:4/4\nL:1/8\n") == ("3/4", "1/16")

    @pytest.mark.parametrize("abc", ["", "X:1\nQ:1/4=90\n", "X:1\nM:fast\nL:short\n"])
    def test_a_score_without_a_readable_meter_gives_4_4_and_1_32(self, abc):
        assert planned_meter(abc) == ("4/4", "1/32")


class TestPlannedTempo:
    @pytest.mark.parametrize(("name", "tempo"), [("song_planned", 142), ("instrumental_planned", 70), ("song_header_am", 90)])
    def test_the_tempo_is_read_from_the_quarter_note_line(self, name, tempo):
        assert planned_tempo(_score(name)) == tempo

    @pytest.mark.parametrize("abc", ["", "K:C\nV: Vocal\nA2|", "X:1\nQ:1/4=fast\nK:C\n", "X:1\nQ:1/8=90\nK:C\n"])
    def test_a_score_without_a_readable_tempo_gives_none(self, abc):
        assert planned_tempo(abc) is None


class TestSilenceVocal:
    def test_the_instrumentals_vocal_voice_becomes_rests_exactly_as_the_checked_edit(self):
        assert silence_vocal(_score("instrumental_planned")) == _score("instrumental_silenced")

    def test_every_other_line_stays_byte_identical(self):
        original = _score("song_planned")
        silenced = silence_vocal(original)

        assert len(silenced.split("\n")) == len(original.split("\n"))
        for before, after in zip(original.split("\n"), silenced.split("\n"), strict=True):
            if before not in _voice_lines(original, "Vocal"):
                assert after == before

    def test_each_vocal_line_keeps_its_bar_count(self):
        original = _score("song_planned")
        silenced = silence_vocal(original)

        before = _voice_lines(original, "Vocal")
        after = _voice_lines(silenced, "Vocal")
        assert len(after) == len(before)
        for source, rests in zip(before, after, strict=True):
            bars = len([segment for segment in source.split("|") if segment.strip()])
            assert rests == (f"Z{bars}|" if bars > 1 else "Z|")

    def test_the_instrumental_voice_keeps_its_notes(self):
        original = _score("song_header_g")

        assert _voice_lines(silence_vocal(original), "Ins") == _voice_lines(original, "Ins")

    def test_a_silenced_score_is_unchanged_by_silencing_again(self):
        once = silence_vocal(_score("song_header_am"))

        assert silence_vocal(once) == once

    def test_multi_bar_rests_count_as_their_bars(self):
        assert silence_vocal("K:C\nV: Vocal\nZ3|z4|\nV: Ins\nA2|") == "K:C\nV: Vocal\nZ4|\nV: Ins\nA2|"

    def test_a_closing_bar_line_is_not_a_bar(self):
        assert silence_vocal("K:C\nV: Vocal\nA2|B2|]\nV: Ins\nA2|") == "K:C\nV: Vocal\nZ2|\nV: Ins\nA2|"

    def test_a_voice_declared_without_a_space_is_read(self):
        assert silence_vocal("K:C\nV:Vocal\nA2|B2|\nV:Ins\nA2|") == "K:C\nV:Vocal\nZ2|\nV:Ins\nA2|"

    @pytest.mark.parametrize(
        "abc",
        ["", "V: Vocal\nA2|", "X:1\nK:C\n", "K:C\nV: Ins\nA2|B2|", "K:C\nV: Vocal\n% only a comment\nV: Ins\nA2|"],
        ids=["empty", "no key line", "no music", "no vocal voice", "empty vocal voice"],
    )
    def test_a_score_that_cannot_be_read_raises(self, abc):
        with pytest.raises(ValueError, match="score has no"):
            silence_vocal(abc)


class TestReadsAsScore:
    @pytest.mark.parametrize("name", ["song_planned", "instrumental_planned", "song_header_am", "song_header_g"])
    def test_the_real_scores_read_as_scores(self, name):
        assert reads_as_score(_score(name))

    @pytest.mark.parametrize("abc", ["", "X:1\nQ:1/4=90\n", "K:C\n", "K:C\nV: Ins\n% comment\n"], ids=["empty", "no key", "no music", "only fields"])
    def test_text_without_a_key_and_music_does_not(self, abc):
        assert not reads_as_score(abc)
