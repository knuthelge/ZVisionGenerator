"""Tests for the model metadata reader: the cached parse, the entry lookups and the per-platform values."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.utils.model_metadata import (
    METADATA_FILE,
    for_platform,
    load_model_metadata,
    model_entry,
    model_ids,
)


def test_the_packaged_file_is_read_once_and_shared() -> None:
    assert METADATA_FILE.is_file()
    assert load_model_metadata() is load_model_metadata()


def test_a_copy_edited_and_served_replaces_the_packaged_data_for_one_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = metadata_copy()
    data["models"]["zit"]["licences"] = ["flux-nc"]
    use_model_metadata(monkeypatch, tmp_path, data)

    assert model_entry("zit")["licences"] == ["flux-nc"]
    assert data is not load_model_metadata()


def test_the_packaged_metadata_is_not_changed_by_a_served_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = metadata_copy()
    data["models"].clear()
    use_model_metadata(monkeypatch, tmp_path, data)

    assert model_ids() == ()
    monkeypatch.undo()
    assert "zit" in model_ids()


def test_an_unknown_model_has_no_entry() -> None:
    assert model_entry("no-such-model") is None


def test_model_ids_are_in_file_order_and_can_be_limited_to_a_kind() -> None:
    assert model_ids("upscaler") == ("seedvr2-7b", "seedvr2-3b")
    assert model_ids("matte") == ("matte",)
    assert model_ids()[:2] == ("ltx-8", "ltx-4")
    assert set(model_ids("video")) < set(model_ids())


@pytest.mark.parametrize(
    ("value", "platform_key", "expected"),
    [
        ({"darwin": 1, "win32": 2, "linux": 3}, "win32", 2),
        ({"darwin": 1}, "linux", None),
        ({"weights_gb": 1.6}, "linux", {"weights_gb": 1.6}),
        ("19 GB", "darwin", "19 GB"),
        ([1, 2], "darwin", [1, 2]),
        ({}, "darwin", {}),
    ],
)
def test_for_platform_picks_a_per_platform_value_and_passes_a_plain_one_through(value: object, platform_key: str, expected: object) -> None:
    assert for_platform(value, platform_key) == expected
