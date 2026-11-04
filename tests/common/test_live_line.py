"""Tests for the console's live line: the single owner of the unfinished terminal row."""

from __future__ import annotations

import logging
import sys

from tests._streams import FakeStream
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.live_line import LiveLine, LiveLineStream, route_stderr_through_live_line, shared_line


class TestOnATerminal:
    def test_a_rewrite_replaces_the_line_and_blanks_what_is_left_of_a_longer_one(self):
        out = FakeStream(terminal=True)
        line = LiveLine(out)

        line.rewrite("a long first line")
        line.rewrite("short")

        first, second = out.getvalue().split("\r")[1:]
        assert len(second) == len(first) and second.startswith("short")

    def test_ending_the_line_twice_writes_one_newline(self):
        out = FakeStream(terminal=True)
        line = LiveLine(out)
        line.rewrite("progress")

        line.end()
        line.end()

        assert out.getvalue().count("\n") == 1

    def test_a_message_ends_the_open_line_first_and_leaves_no_blank_line_behind(self):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        line = LiveLine(out)
        line.rewrite("progress")

        line.message("skip queued", file=err)
        line.end()

        assert out.getvalue().endswith("progress\n") and out.getvalue().count("\n") == 1
        assert err.getvalue() == "skip queued\n"

    def test_a_printed_line_starts_on_a_fresh_row(self):
        out = FakeStream(terminal=True)
        line = LiveLine(out)
        line.rewrite("progress")

        line.print_line("Saved x.flac")

        assert out.getvalue().endswith("progress\nSaved x.flac\n")


class TestElsewhere:
    def test_nothing_is_rewritten_and_lines_stay_plain(self):
        out, err = FakeStream(terminal=False), FakeStream(terminal=False)
        line = LiveLine(out)

        line.rewrite("progress")
        line.print_line("one")
        line.message("two", file=err)
        line.end()

        assert out.getvalue() == "one\n" and err.getvalue() == "two\n"


class TestKeyMessages:
    def test_a_key_pressed_during_a_live_line_prints_its_message_on_its_own_row(self):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        line = LiveLine(out)
        signal = SkipSignal(line)
        line.rewrite("  Synthesis 3/10")

        signal._on_key("n")
        line.end()

        assert out.getvalue().endswith("Synthesis 3/10\n") and out.getvalue().count("\n") == 1
        assert signal.pending() == "skip"
        assert err.getvalue() == ""  # the message went to the real stderr, not the line's stream

    def test_the_default_signal_writes_through_the_shared_line(self):
        assert SkipSignal()._live_line is shared_line()


class TestErrorStream:
    def test_a_write_ends_the_open_line_first_and_leaves_no_blank_row(self):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        line = LiveLine(out)
        stream = LiveLineStream(err, line)
        line.rewrite("  Plan...")

        stream.write("WARNING: something\n")
        line.end()

        assert out.getvalue().endswith("Plan...\n") and out.getvalue().count("\n") == 1
        assert err.getvalue() == "WARNING: something\n"

    def test_a_write_with_no_open_line_adds_nothing_to_the_live_stream(self):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        stream = LiveLineStream(err, LiveLine(out))

        stream.write("plain\n")

        assert out.getvalue() == "" and err.getvalue() == "plain\n"

    def test_a_write_that_holds_muted_text_is_dropped_and_still_counts_as_written(self):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        line = LiveLine(out)
        stream = LiveLineStream(err, line, mute=("is deprecated",))
        line.rewrite("  Plan...")

        written = stream.write("[lib] `x` is deprecated!\n")

        assert written == len("[lib] `x` is deprecated!\n")
        assert err.getvalue() == "" and out.getvalue() == "\r  Plan..."  # the line stays open: nothing interrupted it

    def test_the_rest_of_the_stream_comes_from_the_wrapped_one(self):
        err = FakeStream(terminal=True)

        stream = LiveLineStream(err, LiveLine(FakeStream(terminal=True)))

        assert stream.isatty() is True
        stream.flush()

    def test_logging_and_warnings_end_the_line_before_they_print(self, monkeypatch):
        out, err = FakeStream(terminal=True), FakeStream(terminal=True)
        line = LiveLine(out)
        monkeypatch.setattr(sys, "stderr", err)
        route_stderr_through_live_line(line=line)
        handler = logging.StreamHandler(sys.stderr)  # as the console handler and a library's own handler are made
        logger = logging.getLogger("test_live_line.library")
        logger.propagate = False
        logger.addHandler(handler)
        line.rewrite("  Synthesis 3/10")
        try:
            logger.warning("[lib] a message")
            line.rewrite("  Synthesis 4/10")
            print("a library's print to the error stream", file=sys.stderr)
        finally:
            logger.removeHandler(handler)

        assert out.getvalue().count("\n") == 2  # each message closed the row it interrupted
        assert err.getvalue() == "[lib] a message\na library's print to the error stream\n"

    def test_routing_twice_wraps_once(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", FakeStream(terminal=True))

        route_stderr_through_live_line(line=LiveLine(FakeStream(terminal=True)))
        wrapped = sys.stderr
        route_stderr_through_live_line(line=LiveLine(FakeStream(terminal=True)))

        assert isinstance(wrapped, LiveLineStream) and sys.stderr is wrapped
