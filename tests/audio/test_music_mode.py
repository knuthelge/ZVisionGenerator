"""Tests for the song-or-instrumental rule of the Lyrics box, driven by the fixture the TypeScript mirror also reads."""

from __future__ import annotations

import json

import pytest

from tests._paths import REPO_ROOT
from zvisiongenerator.utils.music_mode import music_mode, section_tags

CASES = json.loads((REPO_ROOT / "tests" / "audio" / "music_mode_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_shared_cases(case):
    cot, lyrics, kind, count = music_mode(case["lyrics"], case["default_sections"])

    assert {"cot": cot, "lyrics": lyrics, "kind": kind, "section_count": count} == case["expected"]


def test_the_presets_planning_modes_are_used():
    assert music_mode("words", "[intro]", cot_song="off", cot_instrumental="full")[0] == "off"
    assert music_mode("", "[intro]", cot_song="off", cot_instrumental="full")[0] == "full"


def test_a_default_list_without_tags_is_refused_only_when_it_is_needed():
    assert music_mode("words", "no tags here")[2] == "song"
    assert music_mode("[Verse]", "no tags here")[2] == "instrumental"
    with pytest.raises(ValueError, match="no \\[section\\] tag"):
        music_mode("", "no tags here")


def test_section_tags_keep_their_order_and_spelling():
    assert section_tags("[Intro] la [Verse 2]\n[x]") == ["[Intro]", "[Verse 2]", "[x]"]
