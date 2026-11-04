"""Tests for the console download line: its text, the live line on a terminal and the periodic lines elsewhere."""

from __future__ import annotations

import pytest

from tests._streams import FakeStream
from zvisiongenerator.downloads import CliDownloadReporter, DownloadProgress, DownloadReport
from zvisiongenerator.downloads.cli_report import progress_line
from zvisiongenerator.utils.live_line import LiveLine

GB = 10**9


def _report(done: int, total: int | None, *, part: int = 1, parts: int = 2, speed: float | None = 45e6, eta: float | None = 120.0, repo: str = "org/repo") -> DownloadReport:
    return DownloadReport(repo, part, parts, DownloadProgress(done, total, speed, eta))


class TestLine:
    def test_it_names_the_repo_the_part_the_sizes_the_speed_and_the_time_left(self):
        assert progress_line(_report(int(2.1 * GB), 7 * GB)) == "Downloading org/repo (1 of 2): 2.1/7.0 GB, 45 MB/s, about 2 min left"

    def test_a_small_download_is_counted_in_megabytes(self):
        assert progress_line(_report(10 * 10**6, 90 * 10**6, speed=None, eta=None)) == "Downloading org/repo (1 of 2): 10/90 MB"

    def test_an_unknown_total_shows_only_what_is_done(self):
        assert progress_line(_report(3 * GB, None, speed=None, eta=None)).endswith(": 3.0 GB")

    @pytest.mark.parametrize(("eta", "text"), [(20.0, "less than a minute left"), (3000.0, "about 50 min left"), (4800.0, "about 1 h 20 min left")])
    def test_the_time_left_is_in_minutes_or_hours(self, eta, text):
        assert progress_line(_report(GB, 7 * GB, eta=eta)).endswith(text)

    def test_a_finished_part_has_no_time_left(self):
        assert "left" not in progress_line(_report(7 * GB, 7 * GB, eta=0.0))


def _reporter(*, terminal: bool, **kwargs) -> tuple[CliDownloadReporter, FakeStream]:
    out = FakeStream(terminal=terminal)
    return CliDownloadReporter(line=LiveLine(out), **kwargs), out


class TestOnATerminal:
    def test_one_line_is_rewritten_in_place_and_ends_when_its_part_is_done(self):
        show, out = _reporter(terminal=True)

        show(_report(GB, 7 * GB))
        show(_report(2 * GB, 7 * GB))
        show(_report(7 * GB, 7 * GB))

        assert out.getvalue().count("\r") == 3 and out.getvalue().count("\n") == 1 and out.getvalue().endswith("\n")

    def test_the_next_part_starts_a_new_line(self):
        show, out = _reporter(terminal=True)

        show(_report(GB, 7 * GB, part=1))
        show(_report(GB, 2 * GB, part=2))
        show.close()

        assert out.getvalue().count("\n") == 2

    def test_a_shorter_line_blanks_the_rest_of_the_longer_one_before_it(self):
        show, out = _reporter(terminal=True)

        show(_report(GB, 7 * GB))
        show(_report(GB, 7 * GB, speed=None, eta=None))

        first, second = out.getvalue().split("\r")[1:]
        assert len(second) == len(first)

    def test_closing_with_no_open_line_prints_nothing(self):
        show, out = _reporter(terminal=True)

        show.close()

        assert out.getvalue() == ""


class TestElsewhere:
    def test_a_line_is_printed_for_the_first_report_then_once_per_interval_and_for_the_last(self):
        now = [0.0]
        show, out = _reporter(terminal=False, interval_secs=30.0, clock=lambda: now[0])

        for at, done in ((0.0, 0), (1.0, GB), (29.0, 2 * GB), (31.0, 3 * GB), (40.0, 4 * GB), (65.0, 7 * GB)):
            now[0] = at
            show(_report(done, 7 * GB))

        lines = out.getvalue().splitlines()
        assert [line.split(": ")[1].split(",")[0] for line in lines] == ["0.0/7.0 GB", "3.0/7.0 GB", "7.0/7.0 GB"]
        assert "\r" not in "".join(lines)

    def test_each_part_prints_its_first_report(self):
        show, out = _reporter(terminal=False, clock=lambda: 0.0)

        show(_report(0, 7 * GB, part=1))
        show(_report(0, 2 * GB, part=2))

        assert len(out.getvalue().splitlines()) == 2
