"""Tests for the ziv-image licence notice: it comes before the downloads start, and only for a model that has a record."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from tests._factories import _patch_image_main_dependencies
from tests._licences import use_licences
from zvisiongenerator import image_cli
from zvisiongenerator.downloads import DownloadTarget
from zvisiongenerator.utils.licences import NOTICE_HEADER

_TARGET = DownloadTarget("owner/klein9b", None, None)
_CONFIG = {
    "generation": {"default_ratio": "2:3", "default_size": "m"},
    "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
    "schedulers": {},
    "model_aliases": {"klein9b": "owner/klein9b", "zit": "owner/zit"},
}
_LICENCES = {
    "flux-nc": {"name": "Test Licence", "supplier": "Test Lab", "summary": "Non-commercial", "url": f"https://example.com/{'a' * 40}/LICENSE", "models": ["klein9b"]},
    "matte": {"name": "Matte Licence", "supplier": "Matte Lab", "summary": "Research only", "url": f"https://example.com/{'b' * 40}/README.md", "models": ["matte"]},
}


@pytest.fixture(autouse=True)
def _licence_records(monkeypatch, tmp_path):
    use_licences(monkeypatch, tmp_path, _LICENCES)


@pytest.fixture
def run(monkeypatch, capsys):
    """Return ``run(argv, present=False)``: run ``main()`` and return the steps in order, each notice as a ``("notice", text)`` entry."""
    steps: list[tuple[str, str]] = []
    plans = MagicMock(return_value=(_TARGET,))
    matte_plans = MagicMock(return_value=(_TARGET,))

    def stage(name: str):
        def step(*args, **kwargs):
            steps.append((name, capsys.readouterr().out))
            return True

        return step

    def start(argv: list[str], *, present: bool = False) -> list[tuple[str, str]]:
        _patch_image_main_dependencies(monkeypatch)
        start.batch = image_cli.run_batch
        monkeypatch.setattr(image_cli, "load_config", lambda: _CONFIG)
        monkeypatch.setattr(image_cli, "resolve_model_path", lambda model, **kwargs: _CONFIG["model_aliases"].get(model, model))
        monkeypatch.setattr(image_cli, "image_plan", plans)
        monkeypatch.setattr(image_cli, "matte_plan", matte_plans)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: present)
        monkeypatch.setattr(image_cli, "ensure_built_model", stage("ensure_built_model"))
        monkeypatch.setattr(image_cli, "_fetch_matte_models", stage("fetch_matte_models"))
        monkeypatch.setattr(sys, "argv", ["ziv-image", "--prompt", "a cat", *argv])
        image_cli.main()
        return steps

    start.plans = plans
    start.matte_plans = matte_plans
    return start


class TestImageCliLicenceNotice:
    def test_the_notice_comes_before_the_model_is_prepared(self, run):
        steps = run(["-m", "klein9b"])

        assert [name for name, _ in steps] == ["ensure_built_model"]
        assert NOTICE_HEADER in steps[0][1]
        assert "Test Licence (Test Lab): Non-commercial" in steps[0][1]

    def test_the_model_is_looked_up_by_the_alias_that_was_picked_and_planned_by_its_resolved_path(self, run):
        steps = run(["-m", "klein9b"])

        assert run.plans.call_args.args == ("owner/klein9b",)
        assert NOTICE_HEADER in steps[0][1]

    def test_a_repo_id_that_an_alias_resolves_to_gets_that_alias_s_notice(self, run):
        steps = run(["-m", "owner/klein9b"])

        assert run.plans.call_args.args == ("owner/klein9b",)
        assert NOTICE_HEADER in steps[0][1]

    def test_a_repo_id_that_no_alias_resolves_to_gets_no_notice(self, run):
        steps = run(["-m", "owner/other"])

        run.plans.assert_not_called()
        assert NOTICE_HEADER not in steps[0][1]

    def test_a_model_without_a_record_builds_no_plan_and_prints_nothing(self, run):
        steps = run(["-m", "zit"])

        run.plans.assert_not_called()
        assert NOTICE_HEADER not in steps[0][1]

    def test_a_model_that_is_present_prints_nothing(self, run):
        steps = run(["-m", "klein9b"], present=True)

        assert NOTICE_HEADER not in steps[0][1]

    def test_a_plan_that_cannot_be_made_counts_as_missing(self, run):
        run.plans.return_value = None

        steps = run(["-m", "klein9b"], present=True)

        assert NOTICE_HEADER in steps[0][1]

    def test_the_background_removal_notice_comes_before_its_download(self, run):
        steps = run(["-m", "zit", "--transparent"])

        assert [name for name, _ in steps] == ["ensure_built_model", "fetch_matte_models"]
        assert NOTICE_HEADER not in steps[0][1]
        assert "Matte Licence (Matte Lab): Research only" in steps[1][1]

    def test_the_background_removal_notice_is_skipped_once_the_models_are_present(self, run):
        steps = run(["-m", "zit", "--transparent"], present=True)

        assert NOTICE_HEADER not in steps[1][1]


class TestImageCliRecordedModelName:
    @pytest.mark.parametrize("model", ["klein9b", "owner/klein9b"])
    def test_a_model_is_recorded_under_its_alias_whether_picked_by_alias_or_repo_id(self, run, model):
        run(["-m", model])

        args = run.batch.call_args.args[4]
        assert args.model_name == "klein9b"
        assert args.model == "owner/klein9b"
