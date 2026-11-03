"""Tests for the licence records: parsing the config section, the refusals ``load_config`` relies on, and the download notice."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.utils.licences import LicenceRecord, licence_notice, licence_text, licences_for, parse_licences

_SHA = "455f1273c2f08f1fc93d725c2e6f1d7fc681cd94"
_URL = f"https://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md"
_NOTICE_URL = f"https://huggingface.co/owner/repo/blob/{_SHA}/NOTICE"


def _record(**overrides: Any) -> dict[str, Any]:
    record = {"name": "Test License", "supplier": "Acme", "summary": "Non-commercial", "url": _URL, "models": ["alpha"]}
    record.update(overrides)
    return record


def _config(**records: dict[str, Any]) -> dict[str, Any]:
    return {"model_aliases": {"alpha": "o/alpha", "beta": "o/beta", "gamma": "o/gamma"}, "licences": records or {"test": _record()}}


def test_parses_a_record_with_defaults() -> None:
    (record,) = parse_licences(_config())

    assert record == LicenceRecord(id="test", name="Test License", supplier="Acme", summary="Non-commercial", url=_URL, models=("alpha",), notice_url=None, url_label="Licence")


def test_parses_the_optional_keys() -> None:
    (record,) = parse_licences(_config(test=_record(notice_url=_NOTICE_URL, url_label="Model card")))

    assert record.notice_url == _NOTICE_URL
    assert record.url_label == "Model card"


def test_a_config_without_the_section_has_no_records() -> None:
    assert parse_licences({"model_aliases": {}}) == ()


def test_the_record_is_frozen() -> None:
    (record,) = parse_licences(_config())

    with pytest.raises(AttributeError):
        record.name = "Other"  # type: ignore[misc]


@pytest.mark.parametrize("key", ["name", "supplier", "summary", "url", "models"])
def test_refuses_a_missing_required_key(key: str) -> None:
    record = _record()
    del record[key]

    with pytest.raises(ValueError, match=key):
        parse_licences(_config(test=record))


@pytest.mark.parametrize("section", [["a"], "text", 3])
def test_refuses_a_section_that_is_not_a_mapping(section: object) -> None:
    with pytest.raises(ValueError, match="licences"):
        parse_licences({"licences": section})


def test_refuses_a_record_that_is_not_a_mapping() -> None:
    with pytest.raises(ValueError, match="licences.test"):
        parse_licences({"licences": {"test": "text"}})


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/owner/repo/blob/main/LICENSE.md",
        "https://huggingface.co/owner/repo/blob/455f1273/LICENSE.md",
        f"https://huggingface.co/owner/repo/blob/{_SHA}0/LICENSE.md",
        "https://cdn.jsdelivr.net/gh/owner/repo@main/LICENSE.pdf",
        "https://cdn.jsdelivr.net/gh/owner/repo/LICENSE.pdf",
        "https://github.com/owner/repo/blob/master/LICENSE",
        f"http://huggingface.co/owner/repo/blob/{_SHA}/LICENSE.md",
        f"https://huggingface.co/owner/repo/blob/main/LICENSE.md?ref={_SHA}",
    ],
)
@pytest.mark.parametrize("key", ["url", "notice_url"])
def test_refuses_a_url_without_a_revision(key: str, url: str) -> None:
    with pytest.raises(ValueError, match=key):
        parse_licences(_config(test=_record(**{key: url})))


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
    (record,) = parse_licences(_config(test=_record(url=url)))

    assert record.url == url


@pytest.mark.parametrize("models", [[], "alpha", None])
def test_refuses_an_empty_or_malformed_models_list(models: object) -> None:
    with pytest.raises(ValueError, match="models"):
        parse_licences(_config(test=_record(models=models)))


def test_refuses_an_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        parse_licences(_config(test=_record(models=["alpha", "nope"])))


def test_accepts_matte_and_audio_presets_and_aliases() -> None:
    config = _config(test=_record(models=["alpha", "matte", "song"]))
    config["audio_model_presets"] = {"song": {}}

    (record,) = parse_licences(config)

    assert record.models == ("alpha", "matte", "song")


def test_an_audio_preset_is_unknown_while_the_config_has_no_audio_section() -> None:
    with pytest.raises(ValueError, match="unknown model 'song'"):
        parse_licences(_config(test=_record(models=["song"])))


def test_a_model_may_be_in_two_records() -> None:
    config = _config(first=_record(name="First"), second=_record(name="Second", models=["alpha", "beta"]))

    assert [record.name for record in licences_for(config, "alpha")] == ["First", "Second"]
    assert [record.name for record in licences_for(config, "beta")] == ["Second"]


def test_licences_for_a_model_without_a_record_is_empty() -> None:
    assert licences_for(_config(), "gamma") == ()


def test_licence_text_names_the_licence_its_supplier_and_summary() -> None:
    (record,) = parse_licences(_config())

    assert licence_text(record) == "Test License (Acme): Non-commercial"


def test_notice_for_a_missing_target() -> None:
    config = _config(test=_record(notice_url=_NOTICE_URL))

    lines = licence_notice(config, "alpha", ["target"], lambda target: False)  # type: ignore[list-item]

    assert lines == ("By downloading you agree to these terms:", "Test License (Acme): Non-commercial", f"  {_URL}", f"  NOTICE: {_NOTICE_URL}")


def test_notice_has_no_notice_line_without_a_notice_url() -> None:
    lines = licence_notice(_config(), "alpha", None, MagicMock())

    assert lines[-1] == f"  {_URL}"
    assert not any(line.startswith("  NOTICE:") for line in lines)


def test_notice_lists_every_record_of_a_model() -> None:
    config = _config(first=_record(name="First"), second=_record(name="Second", notice_url=_NOTICE_URL))

    lines = licence_notice(config, "alpha", None, MagicMock())

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

    lines = licence_notice(_config(), "alpha", None, present)

    assert lines
    present.assert_not_called()


def test_notice_is_empty_when_every_target_is_present() -> None:
    present = MagicMock(return_value=True)

    assert licence_notice(_config(), "alpha", ["a", "b"], present) == ()  # type: ignore[list-item]
    assert present.call_count == 2


def test_notice_appears_when_any_target_is_missing() -> None:
    assert licence_notice(_config(), "alpha", ["a", "b"], lambda target: target == "a")  # type: ignore[list-item]


def test_notice_is_empty_for_a_model_without_a_record_and_never_calls_present() -> None:
    present = MagicMock()

    assert licence_notice(_config(), "gamma", ["a"], present) == ()  # type: ignore[list-item]
    assert licence_notice(_config(), "gamma", None, present) == ()
    present.assert_not_called()


class TestPackagedConfig:
    """The records the packaged ``config.yaml`` ships."""

    def test_every_record_parses_and_is_pinned(self, packaged_config_shared: dict[str, Any]) -> None:
        records = parse_licences(packaged_config_shared)

        assert records
        assert all(record.models for record in records)

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
    def test_the_records_of_each_model(self, packaged_config_shared: dict[str, Any], model: str, expected: list[str]) -> None:
        assert [record.id for record in licences_for(packaged_config_shared, model)] == expected

    def test_the_audio_records_are_the_ones_the_models_page_names(self, packaged_config_shared: dict[str, Any]) -> None:
        summaries = {model: [(record.name, record.supplier, record.summary) for record in licences_for(packaged_config_shared, model)] for model in ("yue2", "higgs3")}

        assert summaries == {
            "yue2": [("CC BY-NC 4.0 with individual creator permission", "M-A-P", "Non-commercial; individual creators may monetise their outputs, companies need a licence")],
            "higgs3": [
                (
                    "Boson Higgs TTS 3 Research and Non-Commercial License",
                    "Boson AI",
                    'Non-commercial; creators may publish with credit; the app\'s Higgs code is "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc."',
                ),
                (
                    "Boson Higgs Audio 2 Community License Agreement",
                    "Boson AI",
                    "Free up to 100,000 annual active users; products credit Boson AI and Meta Llama 3",
                ),
            ],
        }

    def test_only_stable_audio_links_a_notice(self, packaged_config_shared: dict[str, Any]) -> None:
        notices = {model: [record.notice_url is not None for record in licences_for(packaged_config_shared, model) if record.id != "gemma"] for model in ("yue2", "higgs3", "sa3")}

        assert notices == {"yue2": [False], "higgs3": [False, False], "sa3": [True]}

    def test_the_cli_lines_of_an_audio_model_name_its_terms(self, packaged_config_shared: dict[str, Any]) -> None:
        lines = licence_notice(packaged_config_shared, "sa3", None, MagicMock())

        assert [line for line in lines if "(" in line] == ["Stability AI Community License (Stability AI): Free under $1M revenue", "Gemma Terms of Use (Google): Gemma's use policy applies"]

    def test_the_cli_lines_of_speech_carry_the_boson_credit_and_the_codec_terms(self, packaged_config_shared: dict[str, Any]) -> None:
        lines = licence_notice(packaged_config_shared, "higgs3", None, MagicMock())

        assert [line for line in lines if "(" in line] == [
            'Boson Higgs TTS 3 Research and Non-Commercial License (Boson AI): Non-commercial; creators may publish with credit; the app\'s Higgs code is "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc."',
            "Boson Higgs Audio 2 Community License Agreement (Boson AI): Free up to 100,000 annual active users; products credit Boson AI and Meta Llama 3",
        ]

    def test_the_matte_record_links_a_model_card(self, packaged_config_shared: dict[str, Any]) -> None:
        (record,) = licences_for(packaged_config_shared, "matte")

        assert record.url_label == "Model card"


class TestLoadConfigRefusals:
    """``load_config`` runs the parser on the merged config, user overrides included."""

    @staticmethod
    def _load(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, text: str) -> dict[str, Any]:
        from zvisiongenerator.utils.config import load_config

        (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
        monkeypatch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: tmp_path)
        return load_config()

    def test_a_valid_user_record_is_merged(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        text = f"licences:\n  mine:\n    name: Mine\n    supplier: Me\n    summary: Free\n    url: {_URL}\n    models: [zit]\n"

        config = self._load(tmp_path, monkeypatch, text)

        assert [record.id for record in licences_for(config, "zit")] == ["mine"]

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("licences: [a]\n", "licences"),
            ("licences:\n  mine: {name: Mine}\n", "missing the required key"),
            ("licences:\n  flux-nc:\n    url: https://example.com/LICENSE\n", "pinned"),
            ("licences:\n  flux-nc:\n    models: []\n", "models"),
            ("licences:\n  flux-nc:\n    models: [nope]\n", "unknown model"),
        ],
    )
    def test_a_bad_record_is_refused(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, text: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            self._load(tmp_path, monkeypatch, text)
