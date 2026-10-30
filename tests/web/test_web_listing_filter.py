"""Tests for the content filter, with made-up terms standing in for the packaged ones."""

from __future__ import annotations

import base64

import pytest

from zvisiongenerator.web import listing_filter
from zvisiongenerator.web.listing_filter import ListingFilter, content_filter_level, is_filtered, listing_filter_for, name_words, packaged_listing_filter, parse_listing_filter

_RULES = ListingFilter(frozenset({"zap"}), ("zzpix",))


def _encoded(*terms: str) -> str:
    return base64.b64encode(" ".join(terms).encode()).decode()


class TestNameWords:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("o/Mystic-Zap_style", {"mystic", "zap", "style"}),
            ("MysticZAP", {"mystic", "zap"}),
            ("ZAPStyle", {"zap", "style"}),
            ("ZAP-v2", {"zap", "v2"}),
            ("ZAPv2", {"zap"}),
            ("krea2zap", {"krea", "2", "zap"}),
        ],
    )
    def test_splits_at_separators_case_changes_and_digits(self, text, expected):
        assert expected <= name_words(text)


class TestIsFiltered:
    @pytest.mark.parametrize("name", ["zap", "o/zap-style", "MysticZap", "ZAP_v2", "ZAPv2", "ZAPxl", "ZAPStyle", "zap2", "krea2zap", "zzpix", "o/MyZZPIXMix", "glossyzzpix_v2"])
    def test_a_word_or_a_stem_is_found(self, name):
        assert is_filtered([name], _RULES)

    @pytest.mark.parametrize("name", ["o/clean-style", "zapper", "o/mysticzap", "lazap", "zz-pix", "SDKzap", "AIKzap"])
    def test_a_word_inside_another_word_is_not_found(self, name):
        assert not is_filtered([name], _RULES)

    def test_one_name_is_enough(self):
        assert is_filtered(["o/clean", "images/zap.png"], _RULES)


class TestParseListingFilter:
    def test_reads_both_encoded_lists_lower_cased(self):
        rules = parse_listing_filter({"words": _encoded("Zap", "qq"), "stems": _encoded("zzpix")})

        assert rules == ListingFilter(frozenset({"zap", "qq"}), ("zzpix",))

    @pytest.mark.parametrize("data", [{}, {"words": _encoded("a")}, {"words": "not base64!", "stems": _encoded("a")}, {"words": 1, "stems": _encoded("a")}])
    def test_malformed_data_is_refused(self, data):
        with pytest.raises(ValueError):
            parse_listing_filter(data)


class TestPackagedListingFilter:
    def test_the_packaged_terms_are_lower_case_single_words(self):
        rules = packaged_listing_filter()

        assert rules.words and rules.stems
        assert all(term == term.lower() and term.isalnum() for term in (*rules.words, *rules.stems))

    @pytest.mark.parametrize(
        "name",
        [
            "analog",
            "AnalogFilm",
            "documentary",
            "accumulate",
            "cumulus",
            "Essex",
            "Middlesex",
            "sextant",
            "cocktail",
            "Hitchcock",
            "Dickinson",
            "manuscript",
            "Anushka",
            "SDJanus",
            "AIManus",
            "classic",
            "title",
            "hololive",
            "Siriporn",
            "Pornchanart",
            "vr180",
            "umt5-xxl",
            "fluxxx",
            "sexy",
            "bikini",
            "lingerie",
            "young-woman",
            "adult-swim",
        ],
    )
    def test_the_packaged_terms_leave_harmless_names_alone(self, name):
        assert not is_filtered([name], packaged_listing_filter())

    def test_a_missing_packaged_file_is_a_server_fault(self, tmp_path, monkeypatch):
        monkeypatch.setattr(listing_filter, "DATA_FILE", tmp_path / "missing.json")
        packaged_listing_filter.cache_clear()
        try:
            with pytest.raises(RuntimeError):
                packaged_listing_filter()
        finally:
            packaged_listing_filter.cache_clear()


class TestContentFilterLevel:
    @pytest.mark.parametrize(
        ("setting", "level"),
        [(None, "on"), (True, "on"), ("on", "on"), (" ON ", "on"), (False, "off"), ("off", "off"), ("explicit", "explicit"), ("Explicit", "explicit")],
    )
    def test_names_a_level(self, setting, level):
        assert content_filter_level(setting) == level

    @pytest.mark.parametrize("setting", ["none", 0, ["off"]])
    def test_an_unknown_setting_warns_and_is_on(self, setting):
        with pytest.warns(UserWarning):
            assert content_filter_level(setting) == "on"


class TestListingFilterFor:
    def test_on_uses_the_terms_and_the_adult_tag(self):
        assert listing_filter_for("on", _RULES) == _RULES

    def test_off_hides_only_the_adult_tag(self):
        rules = listing_filter_for("off", _RULES)

        assert (rules.words, rules.stems, rules.hide_adult_tagged) == (frozenset(), (), True)
        assert not is_filtered(["zap", "zzpix"], rules)

    def test_explicit_hides_neither(self):
        rules = listing_filter_for("explicit", _RULES)

        assert (rules.words, rules.stems, rules.hide_adult_tagged) == (frozenset(), (), False)
