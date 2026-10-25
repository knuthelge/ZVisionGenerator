"""Tests for the pinned-pack lookup by Hugging Face reference."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.hf_snapshot import pack_for_reference

PIN = "a1" * 20
PACK = HubFiles("Owner/Pack", PIN, ("a.safetensors",))
PACKS = {f"Owner/Pack@{PIN}": PACK}


@pytest.mark.parametrize("reference", ["Owner/Pack", "owner/pack", "OWNER/PACK", f"Owner/Pack@{PIN}", f"owner/PACK@{PIN}"])
def test_a_pack_is_named_by_its_repo_id_in_any_case_with_no_revision_or_its_own(reference: str) -> None:
    assert pack_for_reference(PACKS, reference) is PACK


@pytest.mark.parametrize("reference", ["Owner/Pack@" + "b2" * 20, "Owner/Other", "ltx-4", "/Owner/Pack", "~/Owner/Pack", "./Owner/Pack", ""])
def test_another_revision_repo_or_a_local_path_names_no_pack(reference: str) -> None:
    assert pack_for_reference(PACKS, reference) is None


def test_no_packs_name_nothing() -> None:
    assert pack_for_reference({}, "Owner/Pack") is None


def test_the_first_pack_of_a_repo_answers_an_unpinned_reference_and_each_pin_its_own() -> None:
    other = HubFiles("Owner/Pack", "c3" * 20, ("b.safetensors",))
    packs = {**PACKS, f"Owner/Pack@{'c3' * 20}": other}

    assert pack_for_reference(packs, "Owner/Pack") is PACK
    assert pack_for_reference(packs, f"Owner/Pack@{'c3' * 20}") is other
