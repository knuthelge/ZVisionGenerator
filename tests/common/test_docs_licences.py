"""Tests that the Licences section of the models page names every licence record of the packaged licence file."""

from __future__ import annotations

import re

from tests._paths import REPO_ROOT
from zvisiongenerator.utils.licences import load_licences

_MODELS_PAGE = REPO_ROOT / "docs" / "concepts" / "models.md"


def _licences_section() -> str:
    """Return the text of the ``## Licences`` section of the models page, up to the next ``##`` heading."""
    text = _MODELS_PAGE.read_text(encoding="utf-8")
    match = re.search(r"^## Licences\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, "docs/concepts/models.md has no '## Licences' section"
    return match.group(1)


def test_every_licence_record_is_listed_in_the_licences_section() -> None:
    section = _licences_section()
    records = load_licences()

    assert records
    assert [record.name for record in records if record.name not in section] == []


def test_every_licence_record_links_its_licence() -> None:
    section = _licences_section()
    records = load_licences()

    assert [record.id for record in records if record.url not in section] == []


def test_every_attribution_is_in_the_licences_section() -> None:
    section = _licences_section()

    assert [record.id for record in load_licences() if record.attribution and record.attribution not in section] == []
