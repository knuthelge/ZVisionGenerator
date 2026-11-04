"""Tests for the song-or-instrumental rule of the Lyrics box, driven by the fixture the TypeScript mirror also reads."""

from __future__ import annotations

import json

import pytest

from tests._paths import REPO_ROOT
from zvisiongenerator.utils.music_mode import music_mode

CASES = json.loads((REPO_ROOT / "tests" / "audio" / "music_mode_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_shared_cases(case):
    lyrics, kind, count = music_mode(case["lyrics"], case["default_sections"])

    assert {"lyrics": lyrics, "kind": kind, "section_count": count} == {key: case["expected"][key] for key in ("lyrics", "kind", "section_count")}
