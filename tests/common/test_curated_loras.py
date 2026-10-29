"""Tests for the curated-list loader: parsing, lookups and the packaged list."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from zvisiongenerator.utils.curated_loras import (
    CuratedLora,
    parse_curated_loras,
    curated_by_id,
    curated_by_name,
    curated_for_family,
)


def _raw(**overrides: Any) -> dict[str, Any]:
    """Return one JSON entry with every field, overridden by *overrides*."""
    entry: dict[str, Any] = {
        "id": "owner--repo",
        "name": "some_style",
        "family": "zimage",
        "size_variant": None,
        "version": None,
        "display_name": "Some Style",
        "author": "owner",
        "description": "A look.",
        "section": "style",
        "what_it_does": None,
        "workflow": None,
        "repo": "owner/repo",
        "revision": "a" * 40,
        "file": "some.safetensors",
        "size_bytes": 1024,
        "sha256": "b" * 64,
        "trigger": None,
        "weight": 1.0,
        "licence": "apache-2.0",
        "non_commercial": False,
        "downloads": 12,
        "platforms": ["darwin", "win32", "linux"],
        "note": None,
        "previews": ["https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/a.png", "https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/b.png"],
        "preview_videos": [],
    }
    entry.update(overrides)
    return entry


def _entries() -> tuple[CuratedLora, ...]:
    return parse_curated_loras(
        {
            "schema": 1,
            "entries": [
                _raw(id="z1", name="z_one"),
                _raw(id="k9", name="k_nine", family="flux2_klein", size_variant="9b"),
                _raw(id="k4", name="k_four", family="flux2_klein", size_variant="4b"),
                _raw(id="l23", name="l_two_three", family="ltx", version="2.3"),
                _raw(id="l25", name="l_two_five", family="ltx", version="2.5"),
            ],
        }
    )


class TestParse:
    def test_entries_keep_list_order_and_become_frozen_values(self):
        entries = _entries()
        assert [entry.id for entry in entries] == ["z1", "k9", "k4", "l23", "l25"]
        with pytest.raises(AttributeError):
            entries[0].name = "other"  # type: ignore[misc]

    def test_unknown_schema_is_refused(self):
        with pytest.raises(ValueError, match="schema"):
            parse_curated_loras({"schema": 2, "entries": []})

    def test_missing_field_names_the_entry(self):
        raw = _raw(id="broken")
        del raw["sha256"]
        with pytest.raises(ValueError, match="broken"):
            parse_curated_loras({"schema": 1, "entries": [raw]})

    def test_does_not_change_its_input(self):
        data = {"schema": 1, "entries": [_raw()]}
        before = copy.deepcopy(data)
        parse_curated_loras(data)
        assert data == before


class TestLookups:
    def test_by_family_returns_all_sizes_without_a_size(self):
        assert [entry.id for entry in curated_for_family("flux2_klein", entries=_entries())] == ["k9", "k4"]

    def test_by_family_and_size(self):
        assert [entry.id for entry in curated_for_family("flux2_klein", size_variant="4b", entries=_entries())] == ["k4"]

    def test_by_family_and_version_does_not_offer_the_other_version(self):
        assert [entry.id for entry in curated_for_family("ltx", version="2.5", entries=_entries())] == ["l25"]
        assert [entry.id for entry in curated_for_family("ltx", version="2.3", entries=_entries())] == ["l23"]

    def test_unknown_family_is_empty(self):
        assert curated_for_family("flux1", entries=_entries()) == ()

    def test_by_id_and_by_name(self):
        entries = _entries()
        assert curated_by_id("k9", entries=entries) is entries[1]
        assert curated_by_name("l_two_three", entries=entries) is entries[3]
        assert curated_by_id("missing", entries=entries) is None
        assert curated_by_name("missing", entries=entries) is None
