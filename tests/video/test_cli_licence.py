"""Tests for the ziv-video licence notice: it comes before the preflight and any download, and only for a model that has a record."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from tests._factories import video_config
from tests._licences import use_licences
from tests.video._helpers import patch_video_cli_for_preflight
from zvisiongenerator.downloads import DownloadTarget
from zvisiongenerator.utils.licences import NOTICE_HEADER

_TARGET = DownloadTarget("owner/ltx", None, None)
_CONFIG = {
    "video_generation": {"default_ratio": "16:9", "default_size": "m"},
    "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
    "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
    "model_aliases": {"ltx-4": "owner/ltx", "ltx25-4": "owner/ltx25"},
    **video_config(),
}
_LICENCES = {
    "ltx-2-community": {"name": "Test Licence", "supplier": "Test Lab", "summary": "Free under $10M revenue", "url": f"https://example.com/{'a' * 40}/LICENSE", "models": ["ltx-4"]},
}


@pytest.fixture(autouse=True)
def _licence_records(monkeypatch, tmp_path):
    use_licences(monkeypatch, tmp_path, _LICENCES)


@pytest.fixture
def run(monkeypatch, capsys):
    """Return ``run(argv, present=False)``: run ``main()`` and return what had been printed when the preflight started."""
    seen: list[str] = []
    plans = MagicMock(return_value=(_TARGET,))

    def preflight(*args, **kwargs):
        seen.append(capsys.readouterr().out)
        return MagicMock(cancelled=True)

    def start(argv: list[str], *, present: bool = False) -> str:
        video_cli, _, _ = patch_video_cli_for_preflight(monkeypatch, [])
        monkeypatch.setattr(video_cli, "load_config", lambda: _CONFIG)
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **kwargs: _CONFIG["model_aliases"].get(model, model))
        monkeypatch.setattr(video_cli, "video_plan", plans)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: present)
        monkeypatch.setattr(video_cli, "run_preflight", preflight)
        monkeypatch.setattr(sys, "argv", ["ziv-video", "--prompt", "a dog", *argv])
        video_cli.main()
        return seen[-1]

    start.plans = plans
    return start


class TestVideoCliLicenceNotice:
    def test_the_notice_comes_before_the_preflight(self, run):
        out = run(["-m", "ltx-4"])

        assert NOTICE_HEADER in out
        assert "Test Licence (Test Lab): Free under $10M revenue" in out

    def test_the_model_is_looked_up_by_its_name_and_planned_by_its_resolved_path(self, run):
        run(["-m", "ltx-4"])

        assert run.plans.call_args.args == ("owner/ltx",)
        assert run.plans.call_args.kwargs["model_name"] == "ltx-4"

    def test_a_model_without_a_record_builds_no_plan_and_prints_nothing(self, run):
        out = run(["-m", "ltx25-4"])

        run.plans.assert_not_called()
        assert NOTICE_HEADER not in out

    def test_a_model_that_is_present_prints_nothing(self, run):
        assert NOTICE_HEADER not in run(["-m", "ltx-4"], present=True)

    def test_a_plan_that_cannot_be_made_counts_as_missing(self, run):
        run.plans.return_value = None

        assert NOTICE_HEADER in run(["-m", "ltx-4"], present=True)
