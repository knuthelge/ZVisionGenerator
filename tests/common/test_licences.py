"""Tests for the licence records: parsing, the known-model check ``load_config`` runs, the packaged file and the download notice."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests._licences import use_licences
from zvisiongenerator.utils.licences import (
    LicenceRecord,
    check_licence_models,
    known_model_ids,
    licence_notice,
    licence_payload,
    licence_text,
    licences_for,
    load_licences,
    parse_licences,
)

_SHA = "455f1273c2f08f1fc93d725c2e6f1d7fc681cd94"
_URL = f"https://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md"
_NOTICE_URL = f"https://huggingface.co/owner/repo/blob/{_SHA}/NOTICE"


def _record(**overrides: Any) -> dict[str, Any]:
    record = {"name": "Test License", "supplier": "Acme", "summary": "Non-commercial", "url": _URL, "models": ["alpha"]}
    record.update(overrides)
    return record


def _records(**records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return records or {"test": _record()}


def _parsed(**records: dict[str, Any]) -> tuple[LicenceRecord, ...]:
    return parse_licences(_records(**records))


def test_parses_a_record_with_defaults() -> None:
    (record,) = _parsed()

    assert record == LicenceRecord(id="test", name="Test License", supplier="Acme", summary="Non-commercial", url=_URL, models=("alpha",), notice_url=None, url_label="Licence")


def test_parses_the_optional_keys() -> None:
    (record,) = _parsed(test=_record(notice_url=_NOTICE_URL, url_label="Model card"))

    assert record.notice_url == _NOTICE_URL
    assert record.url_label == "Model card"


def test_records_keep_the_order_they_are_written_in() -> None:
    assert [record.id for record in _parsed(b=_record(), a=_record(), c=_record())] == ["b", "a", "c"]


@pytest.mark.parametrize("key", ["name", "supplier", "summary", "url", "models"])
def test_refuses_a_missing_required_key(key: str) -> None:
    record = _record()
    del record[key]

    with pytest.raises(ValueError, match=key):
        _parsed(test=record)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/owner/repo/blob/main/LICENSE.md",
        "https://cdn.jsdelivr.net/gh/owner/repo@main/LICENSE.pdf",
        "https://github.com/owner/repo/blob/master/LICENSE",
        f"http://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md",
    ],
)
@pytest.mark.parametrize("key", ["url", "notice_url"])
def test_refuses_a_url_without_a_revision(key: str, url: str) -> None:
    with pytest.raises(ValueError, match=key):
        _parsed(test=_record(**{key: url}))


@pytest.mark.parametrize(
    "url",
    [
        f"https://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md",
        f"https://github.com/owner/repo/blob/{_SHA}/LICENSE",
        f"https://raw.githubusercontent.com/owner/repo/{_SHA}/LICENSE",
        f"https://cdn.jsdelivr.net/gh/owner/repo@{_SHA}/assets/LICENSE.pdf",
    ],
)
def test_accepts_a_pinned_url(url: str) -> None:
    (record,) = _parsed(test=_record(url=url))

    assert record.url == url


@pytest.mark.parametrize("models", [[], "alpha", None, [3]])
def test_refuses_an_empty_or_malformed_models_list(models: object) -> None:
    with pytest.raises(ValueError, match="models"):
        _parsed(test=_record(models=models))


_CONFIG = {"model_aliases": {"alpha": "o/alpha", "beta": "o/beta"}, "audio_model_presets": {"song": {}}}


def test_the_known_models_are_the_aliases_the_audio_presets_and_matte() -> None:
    assert known_model_ids(_CONFIG) == {"alpha", "beta", "song", "matte"}


def test_a_config_without_model_sections_knows_only_matte() -> None:
    assert known_model_ids({}) == {"matte"}


def test_a_record_may_name_aliases_audio_presets_and_matte() -> None:
    records = _parsed(test=_record(models=["alpha", "matte", "song"]))

    check_licence_models(records, known_model_ids(_CONFIG))


def test_refuses_an_unknown_model() -> None:
    records = _parsed(test=_record(models=["alpha", "nope"]))

    with pytest.raises(ValueError, match="'test' names an unknown model 'nope'"):
        check_licence_models(records, known_model_ids(_CONFIG))


def test_an_audio_preset_is_unknown_while_the_config_has_no_audio_section() -> None:
    records = _parsed(test=_record(models=["song"]))

    with pytest.raises(ValueError, match="unknown model 'song'"):
        check_licence_models(records, known_model_ids({"model_aliases": {"alpha": "o/alpha"}}))


def test_a_model_may_be_in_two_records() -> None:
    records = _parsed(first=_record(name="First"), second=_record(name="Second", models=["alpha", "beta"]))

    assert [record.name for record in licences_for(records, "alpha")] == ["First", "Second"]
    assert [record.name for record in licences_for(records, "beta")] == ["Second"]


def test_licences_for_a_model_without_a_record_is_empty() -> None:
    assert licences_for(_parsed(), "gamma") == ()


def test_the_payload_is_the_json_form_of_the_records_of_a_model() -> None:
    records = _parsed(test=_record(notice_url=_NOTICE_URL, url_label="Model card"))

    assert licence_payload(records, "alpha") == [{"name": "Test License", "supplier": "Acme", "summary": "Non-commercial", "url": _URL, "url_label": "Model card", "notice_url": _NOTICE_URL}]
    assert licence_payload(records, "gamma") == []


def test_licence_text_names_the_licence_its_supplier_and_summary() -> None:
    (record,) = _parsed()

    assert licence_text(record) == "Test License (Acme): Non-commercial"


def test_notice_for_a_missing_target() -> None:
    records = _parsed(test=_record(notice_url=_NOTICE_URL))

    lines = licence_notice(records, "alpha", ["target"], lambda target: False)  # type: ignore[list-item]

    assert lines == ("By downloading you agree to these terms:", "Test License (Acme): Non-commercial", f"  {_URL}", f"  NOTICE: {_NOTICE_URL}")


def test_notice_lists_every_record_of_a_model() -> None:
    records = _parsed(first=_record(name="First"), second=_record(name="Second", notice_url=_NOTICE_URL))

    lines = licence_notice(records, "alpha", None, MagicMock())

    assert lines == (
        "By downloading you agree to these terms:",
        "First (Acme): Non-commercial",
        f"  {_URL}",
        "Second (Acme): Non-commercial",
        f"  {_URL}",
        f"  NOTICE: {_NOTICE_URL}",
    )


def test_notice_for_unplannable_targets_does_not_call_present() -> None:
    present = MagicMock(return_value=True)

    lines = licence_notice(_parsed(), "alpha", None, present)

    assert lines
    present.assert_not_called()


def test_notice_is_empty_when_every_target_is_present() -> None:
    present = MagicMock(return_value=True)

    assert licence_notice(_parsed(), "alpha", ["a", "b"], present) == ()  # type: ignore[list-item]
    assert present.call_count == 2


def test_notice_appears_when_any_target_is_missing() -> None:
    assert licence_notice(_parsed(), "alpha", ["a", "b"], lambda target: target == "a")  # type: ignore[list-item]


def test_notice_is_empty_for_a_model_without_a_record_and_never_calls_present() -> None:
    present = MagicMock()

    assert licence_notice(_parsed(), "gamma", ["a"], present) == ()  # type: ignore[list-item]
    assert licence_notice(_parsed(), "gamma", None, present) == ()
    present.assert_not_called()


class TestPackagedRecords:
    """The records the packaged ``licences.yaml`` ships."""

    def test_every_record_parses_is_pinned_and_names_known_models(self, packaged_config_shared: dict[str, Any]) -> None:
        records = load_licences()

        assert records
        check_licence_models(records, known_model_ids(packaged_config_shared))

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

    def test_only_stable_audio_links_a_notice(self) -> None:
        records = load_licences()
        notices = {model: [record.notice_url is not None for record in licences_for(records, model) if record.id != "gemma"] for model in ("yue2", "higgs3", "sa3")}

        assert notices == {"yue2": [False], "higgs3": [False, False], "sa3": [True]}


class TestLoadConfig:
    """``load_config`` checks the records against the merged config's models and ignores any ``licences`` key of the user's."""

    @staticmethod
    def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> dict[str, Any]:
        from zvisiongenerator.utils.config import load_config

        (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
        monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: tmp_path)
        return load_config()

    def test_a_licences_key_in_the_user_config_has_no_effect(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = f"licences:\n  flux-nc:\n    name: Mine\n    supplier: Me\n    summary: Free\n    url: {_URL}\n    models: [zit]\n"
        before = load_licences()

        self._load(tmp_path, monkeypatch, text)

        assert load_licences() == before
        assert licences_for(load_licences(), "zit") == ()

    def test_a_user_config_that_removes_a_licensed_model_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        use_licences(monkeypatch, tmp_path, _records(mine=_record(models=["not-a-model"])))

        with pytest.raises(ValueError, match="unknown model 'not-a-model'"):
            self._load(tmp_path, monkeypatch, "generation: {}\n")
