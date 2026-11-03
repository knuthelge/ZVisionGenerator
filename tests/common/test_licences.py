"""Tests for the licence records: the loader, the packaged file's contract, the lookups and the download notice."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from tests._licences import use_licences
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL, LicenceRecord, licence_notice, licence_payload, licence_text, licences_for, load_licences

_SHA = "455f1273c2f08f1fc93d725c2e6f1d7fc681cd94"
_URL = f"https://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md"
_ATTRIBUTION = "Powered by Acme"
_MODEL_PAGE = re.compile(r"^/[^/]+/[^/]+$")
_PINNED_PATH = re.compile(r"(?:^|[/@])[0-9a-f]{40}(?:[/?#]|$)")


def _record(**overrides: Any) -> dict[str, Any]:
    record = {"name": "Test License", "supplier": "Acme", "summary": "Non-commercial", "url": _URL, "models": ["alpha"]}
    record.update(overrides)
    return record


def _records(**records: dict[str, Any]) -> tuple[LicenceRecord, ...]:
    """Build records the way the loader does, without a file."""
    raw = records or {"test": _record()}
    return tuple(
        LicenceRecord(
            id=key, name=r["name"], supplier=r["supplier"], summary=r["summary"], url=r["url"], models=tuple(r["models"]), url_label=r.get("url_label", "Licence"), attribution=r.get("attribution")
        )
        for key, r in raw.items()
    )


def _plan(targets: list[str] | None) -> Any:
    return lambda: targets


def test_loads_records_with_defaults_and_optional_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_licences(monkeypatch, tmp_path, {"plain": _record(), "full": _record(attribution=_ATTRIBUTION, url_label="Model card")})

    records = {record.id: record for record in load_licences()}
    plain, full = records["plain"], records["full"]

    assert plain == LicenceRecord(id="plain", name="Test License", supplier="Acme", summary="Non-commercial", url=_URL, models=("alpha",))
    assert (full.id, full.attribution, full.url_label) == ("full", _ATTRIBUTION, "Model card")


def test_a_model_may_be_in_two_records() -> None:
    records = _records(first=_record(name="First"), second=_record(name="Second", models=["alpha", "beta"]))

    assert [record.name for record in licences_for(records, "alpha")] == ["First", "Second"]
    assert [record.name for record in licences_for(records, "beta")] == ["Second"]
    assert licences_for(records, "gamma") == ()


def test_the_payload_is_the_json_form_of_the_records_of_a_model() -> None:
    records = _records(test=_record(attribution=_ATTRIBUTION, url_label="Model card"))

    assert licence_payload(records, "alpha") == [{"name": "Test License", "supplier": "Acme", "summary": "Non-commercial", "url": _URL, "url_label": "Model card", "attribution": _ATTRIBUTION}]
    assert licence_payload(records, "gamma") == []


def test_licence_text_names_the_licence_its_supplier_and_summary() -> None:
    (record,) = _records()

    assert licence_text(record) == "Test License (Acme): Non-commercial"


def test_notice_lists_every_record_of_a_missing_model() -> None:
    records = _records(first=_record(name="First"), second=_record(name="Second", attribution=_ATTRIBUTION))

    lines = licence_notice(records, "alpha", _plan(["target"]), lambda target: False)

    assert lines == (
        "By downloading you agree to these terms:",
        "First (Acme): Non-commercial",
        f"  {_URL}",
        "Second (Acme): Non-commercial",
        f"  {_URL}",
        f"  {_ATTRIBUTION}",
    )


def test_notice_for_unplannable_targets_does_not_call_present() -> None:
    present = MagicMock(return_value=True)

    assert licence_notice(_records(), "alpha", _plan(None), present)
    present.assert_not_called()


def test_notice_is_empty_when_every_target_is_present_and_shown_when_one_is_missing() -> None:
    assert licence_notice(_records(), "alpha", _plan(["a", "b"]), lambda target: True) == ()
    assert licence_notice(_records(), "alpha", _plan(["a", "b"]), lambda target: target == "a")


def test_a_model_without_a_record_is_never_planned() -> None:
    plan = MagicMock()

    assert licence_notice(_records(), "gamma", plan) == ()
    plan.assert_not_called()


class TestPackagedRecords:
    """The records the packaged ``licences.yaml`` ships: pinned, naming known models."""

    def test_every_url_is_https_and_pinned_to_a_commit_or_a_model_page(self) -> None:
        records = load_licences()

        assert records
        for url in [r.url for r in records]:
            parsed = urlparse(url)
            model_page = parsed.netloc == "huggingface.co" and _MODEL_PAGE.match(parsed.path)
            assert parsed.scheme == "https" and (_PINNED_PATH.search(parsed.path) or model_page), url

    def test_no_link_points_to_a_mirror_the_app_downloads_from(self) -> None:
        mirrors = ("unsloth/", "cocktailpeanut/", "Comfy-Org/", "mlx-community/", "Hippotes/")

        assert [r.id for r in load_licences() if any(mirror in r.url for mirror in mirrors)] == []

    def test_only_the_stability_and_higgs_audio_records_carry_an_attribution(self) -> None:
        carrying = {r.id: r.attribution for r in load_licences() if r.attribution}

        assert set(carrying) == {"stability-community", "higgs-audio2-community"}
        assert carrying["stability-community"] == "Powered by Stability AI"

    def test_every_record_names_known_models(self, packaged_config_shared: dict[str, Any]) -> None:
        known = {MATTE_LICENCE_MODEL, *packaged_config_shared["model_aliases"], *packaged_config_shared["audio_model_presets"]}

        for record in load_licences():
            assert record.models and set(record.models) <= known, record.id

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("klein9b", ["flux-nc"]),
            ("krea2", ["krea-2-community"]),
            ("ideo", ["ideogram-4-nc"]),
            ("ltx-4", ["ltx-2-community", "gemma"]),
            ("ltx-8", ["ltx-2-community", "gemma"]),
            ("ltx25-4", ["ltx-2x-community"]),
            ("h3", ["minimax-h3-community"]),
            ("yue2", ["yue2-creator-nc"]),
            ("higgs3", ["higgs3-nc", "higgs-audio2-community"]),
            ("sa3", ["stability-community", "gemma"]),
            ("matte", ["matte"]),
            ("zit", []),
            ("klein4b", []),
            ("moss", []),
        ],
    )
    def test_the_records_of_each_model(self, model: str, expected: list[str]) -> None:
        assert [record.id for record in licences_for(load_licences(), model)] == expected


def test_a_licences_key_in_the_user_config_has_no_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.config import load_config

    (tmp_path / "config.yaml").write_text(f"licences:\n  flux-nc:\n    name: Mine\n    supplier: Me\n    summary: Free\n    url: {_URL}\n    models: [zit]\n", encoding="utf-8")
    monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: tmp_path)
    before = load_licences()

    load_config()

    assert load_licences() == before
    assert licences_for(load_licences(), "zit") == ()
