"""Tests for the live progress line of ``ziv audio``: one rewritten line on a terminal, a line per phase otherwise."""

from __future__ import annotations

from tests._streams import FakeStream
from zvisiongenerator.utils.audio_console import ConsoleProgress, progress_text
from zvisiongenerator.utils.live_line import LiveLine


def _console(*, terminal: bool) -> tuple[ConsoleProgress, FakeStream]:
    out = FakeStream(terminal=terminal)
    return ConsoleProgress(LiveLine(out)), out


class TestText:
    def test_a_phase_with_a_total_shows_the_count_and_the_time_left_in_the_batch(self):
        text = progress_text("Synthesis", 12, 64, 1, 1, 120.0)

        assert "Synthesis 12/64" in text and text.endswith("about 2 min left in the batch")

    def test_a_phase_without_a_total_shows_the_count_with_a_thousands_separator_and_no_time_before_a_take_finished(self):
        assert progress_text("Song tokens", 1840, None, 1, 1, None).strip() == "Song tokens 1,840"

    def test_a_part_of_several_names_itself(self):
        text = progress_text("Speech tokens", 5, 40, 2, 3, None)

        assert "part 2 of 3" in text and "5/40" in text

    def test_the_decode_phase_reads_finishing_without_a_count(self):
        assert progress_text("Decode", 0, 1, 1, 1, None).strip() == "Finishing..."
        assert progress_text("Decode", 1, 1, 2, 3, None).strip() == "Finishing (part 2 of 3)..."


class TestTerminal:
    def test_one_line_is_rewritten_in_place_while_a_phase_runs(self):
        console, out = _console(terminal=True)

        console("Synthesis", 1, 3, 1, 1, None)
        console("Synthesis", 2, 3, 1, 1, None)

        assert out.getvalue().count("\r") == 2 and "\n" not in out.getvalue()
        assert out.getvalue().rstrip().endswith("Synthesis 2/3")

    def test_a_new_phase_starts_a_new_line(self):
        console, out = _console(terminal=True)

        console("Plan", 4, None, 1, 1, None)
        console("Song tokens", 1, None, 1, 1, None)

        assert out.getvalue().count("\n") == 1

    def test_a_total_that_moves_while_the_count_runs_is_shown_as_the_first_one(self):
        console, out = _console(terminal=True)

        console("Speech tokens", 1, 220, 1, 2, None)
        console("Speech tokens", 200, 230, 1, 2, None)
        console("Speech tokens", 220, 230, 1, 2, None)

        assert [text.strip() for text in out.getvalue().split("\r") if text.strip()] == [
            "Speech tokens (part 1 of 2) 1/220",
            "Speech tokens (part 1 of 2) 200/220",
            "Speech tokens (part 1 of 2) 220/220",
        ]

    def test_past_the_first_total_only_the_count_is_shown(self):
        console, out = _console(terminal=True)

        console("Speech tokens", 220, 220, 1, 1, None)
        console("Speech tokens", 230, 240, 1, 1, None)
        console("Speech tokens", 238, 238, 1, 1, None)  # the part ended: it reports the count as its total

        assert out.getvalue().rstrip().endswith("Speech tokens 238")

    def test_a_new_part_takes_its_own_total(self):
        console, out = _console(terminal=True)

        console("Speech tokens", 5, 100, 1, 2, None)
        console("Speech tokens", 1, 40, 2, 2, None)

        assert out.getvalue().rstrip().endswith("(part 2 of 2) 1/40")

    def test_a_phase_that_never_reports_a_total_shows_only_the_count(self):
        console, out = _console(terminal=True)

        console("Song tokens", 10, None, 1, 1, None)
        console("Song tokens", 20, None, 1, 1, None)

        assert out.getvalue().rstrip().endswith("Song tokens 20")

    def test_closing_ends_an_open_line_once(self):
        console, out = _console(terminal=True)
        console("Plan", 1, None, 1, 1, None)

        console.close()
        console.close()

        assert out.getvalue().count("\n") == 1


class TestPipe:
    def test_one_line_is_printed_when_the_phase_or_part_changes_and_none_between(self):
        console, out = _console(terminal=False)

        for step in range(1, 5):
            console("Synthesis", step, 4, 1, 1, None)
        console("Decode", 0, 1, 1, 1, None)

        assert out.getvalue() == "  Synthesis...\n  Finishing...\n"

    def test_a_new_part_of_the_same_phase_prints_again(self):
        console, out = _console(terminal=False)

        console("Speech tokens", 1, 10, 1, 2, None)
        console("Speech tokens", 1, 10, 2, 2, None)

        assert out.getvalue().count("\n") == 2
