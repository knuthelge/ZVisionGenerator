"""Tests for the Ideogram 4 JSON caption mode of the prompt enhancer."""

from __future__ import annotations

import json
import re

import pytest

from zvisiongenerator.utils.prompt_enhance import STYLE_AXIS, EnhanceSettings
from zvisiongenerator.utils.prompt_enhance_caption import CAPTION_NO_USABLE_MESSAGE, enhance_caption

_PHOTO_KEYS = ["aesthetics", "lighting", "photo", "medium", "color_palette"]
_ART_KEYS = ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
_HEX = re.compile(r"^#[0-9A-F]{6}$")

_REPLY = """SUMMARY: A red fox sits in fresh snow at the edge of a forest.
AESTHETICS: calm, natural
LIGHTING: soft overcast light
CAMERA: eye-level, 200mm lens, shallow depth of field
COLORS: rust orange, white, #3b3f46
BACKGROUND: Snowy pine trunks fading into mist.
OBJECT: A red fox sitting upright, thick russet fur.
OBJECT: Smooth snow in the foreground.
"""


class _FakeEnhancer:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seeds: list[int] = []
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.seeds.append(seed)
        self.messages.append(messages)
        yield self.replies[len(self.seeds) - 1]


def _caption(prompt: str, *replies: str, settings: EnhanceSettings | None = None) -> dict:
    result = enhance_caption(_FakeEnhancer(*replies), prompt, settings or EnhanceSettings(), seed=1, ceiling=300)
    assert "\n" not in result.prompt
    return json.loads(result.prompt)


def _reply(*items: str, look: str = "CAMERA: eye-level, 50mm") -> str:
    return "\n".join(["SUMMARY: s", "AESTHETICS: a", "LIGHTING: l", look, "COLORS: red", "BACKGROUND: b", *items])


def _texts(caption: dict) -> list[str]:
    return [element["text"] for element in caption["compositional_deconstruction"]["elements"] if element["type"] == "text"]


class TestCaptionShape:
    @pytest.mark.parametrize("style", STYLE_AXIS.slugs())
    def test_every_style_follows_the_caption_key_order(self, style: str) -> None:
        caption = _caption("a red fox in snow", _REPLY, settings=EnhanceSettings(style=style))
        assert list(caption) == ["high_level_description", "style_description", "compositional_deconstruction"]
        expected = [key for key in (_PHOTO_KEYS if "photo" in caption["style_description"] else _ART_KEYS) if key in caption["style_description"]]
        assert list(caption["style_description"]) == expected
        assert all(_HEX.match(color) for color in caption["style_description"].get("color_palette", []))
        for element in caption["compositional_deconstruction"]["elements"]:
            assert list(element) == (["type", "text", "desc"] if element["type"] == "text" else ["type", "desc"])

    def test_art_style_picks_the_art_branch(self) -> None:
        style = _caption("a red fox", _REPLY, settings=EnhanceSettings(style="anime"))["style_description"]
        assert style["medium"] == "illustration"
        assert "photo" not in style

    def test_black_and_white_has_no_palette(self) -> None:
        assert "color_palette" not in _caption("a red fox", _REPLY, settings=EnhanceSettings(style="bw"))["style_description"]

    def test_color_names_and_hex_become_uppercase_hex(self) -> None:
        assert _caption("a red fox", _REPLY)["style_description"]["color_palette"] == ["#EF6C00", "#F5F5F5", "#3B3F46"]

    def test_a_word_of_hex_letters_is_not_a_color(self) -> None:
        palette = _caption("a red fox", _reply("OBJECT: a fox").replace("COLORS: red", "COLORS: beaded cream, facade grey"))["style_description"]["color_palette"]
        assert palette == ["#F3E9D2", "#808080"]

    def test_camera_line_holding_an_art_style_is_repaired(self) -> None:
        caption = _caption("a vintage travel poster for Lisbon", _reply("OBJECT: a tram", look="CAMERA: ART STYLE: flat retro poster"))
        assert caption["style_description"]["medium"] == "graphic_design"
        assert caption["style_description"]["art_style"] == "flat retro poster"


class TestMediumDetection:
    @pytest.mark.parametrize(
        ("prompt", "medium"),
        [
            ("a bottle of olive oil on a wooden table", "photograph"),
            ("a canvas tent in the forest", "photograph"),
            ("a kitchen blender on the counter", "photograph"),
            ("an oil portrait of a king", "painting"),
            ("a 3D render shot from above", "3d_render"),
            ("a watercolor of a harbor, photo of the original", "photograph"),
        ],
    )
    def test_the_medium_comes_from_medium_phrases_only(self, prompt: str, medium: str) -> None:
        assert _caption(prompt, _REPLY)["style_description"]["medium"] == medium


class TestOptionsInCaptions:
    @pytest.mark.parametrize(("style", "medium"), [("watercolor", "painting"), ("claymation", "3d_render"), ("graphic", "graphic_design"), ("ukiyo-e", "illustration"), ("candid", "photograph")])
    def test_every_style_has_a_medium(self, style: str, medium: str) -> None:
        assert _caption("a fox in snow", _REPLY, settings=EnhanceSettings(style=style))["style_description"]["medium"] == medium

    def test_theme_is_asked_for_in_the_caption(self) -> None:
        enhancer = _FakeEnhancer(_REPLY)
        enhance_caption(enhancer, "a fox in snow", EnhanceSettings(theme="noir"), seed=1, ceiling=300)
        assert "Theme: noir." in enhancer.messages[0][0]["content"]


class TestDesignDetection:
    @pytest.mark.parametrize(
        "prompt",
        ["a cozy bedroom with a band poster on the wall", "a wine bottle with a gold label on a table", "Two men playing cards in a pub", "A girl reading a menu at a cafe"],
    )
    def test_design_inside_a_scene_stays_a_photo(self, prompt: str) -> None:
        enhancer = _FakeEnhancer(_REPLY)
        caption = json.loads(enhance_caption(enhancer, prompt, EnhanceSettings(), seed=1, ceiling=300).prompt)
        assert caption["style_description"]["medium"] == "photograph"
        assert "filling the whole frame" not in enhancer.messages[0][0]["content"]

    @pytest.mark.parametrize("prompt", ["a vintage travel poster for Lisbon", "an elegant wedding invitation with roses", "birthday card with flowers"])
    def test_a_design_opening_the_prompt_fills_the_frame(self, prompt: str) -> None:
        enhancer = _FakeEnhancer(_REPLY)
        caption = json.loads(enhance_caption(enhancer, prompt, EnhanceSettings(), seed=1, ceiling=300).prompt)
        assert caption["style_description"]["medium"] == "graphic_design"
        assert "filling the whole frame" in enhancer.messages[0][0]["content"]

    def test_photo_style_never_gets_the_design_rule(self) -> None:
        enhancer = _FakeEnhancer(_REPLY)
        enhance_caption(enhancer, "a poster for a jazz night", EnhanceSettings(style="photo"), seed=1, ceiling=300)
        assert "filling the whole frame" not in enhancer.messages[0][0]["content"]


class TestLettering:
    def test_quoted_text_is_restored_and_other_text_dropped(self) -> None:
        caption = _caption("a vintage poster with the text 'LISBOA' in bold letters", _reply('TEXT: "Lisboa" | big', 'TEXT: "PORTUGAL" | small', "OBJECT: a tram"))
        assert _texts(caption) == ["LISBOA"]

    def test_unquoted_text_line_with_its_look_matches_the_quote(self) -> None:
        caption = _caption("a poster with the text 'LISBOA'", _reply("OBJECT: a tram", "TEXT: LISBOA in bold serif, centered"))
        element = caption["compositional_deconstruction"]["elements"][-1]
        assert (element["text"], element["desc"]) == ("LISBOA", "in bold serif, centered")

    def test_missing_quoted_text_is_added(self) -> None:
        assert _texts(_caption('a poster saying "HELLO"', _reply("OBJECT: a poster"))) == ["HELLO"]

    def test_object_line_repeating_quoted_lettering_is_dropped(self) -> None:
        caption = _caption("a poster with the text 'LISBOA'", _reply('OBJECT: A bold serif font text "LISBOA" in gold', 'TEXT: "LISBOA" | gold', "OBJECT: a tram"))
        assert [element["type"] for element in caption["compositional_deconstruction"]["elements"]] == ["text", "obj"]

    def test_apostrophes_are_not_quotes(self) -> None:
        assert _texts(_caption("rock 'n' roll guitarist at Moe's bar", _reply("OBJECT: a guitarist", 'TEXT: "n" | small'))) == []

    def test_no_text_without_signs_or_quotes(self) -> None:
        assert _texts(_caption("a cup of coffee on a table", _reply("OBJECT: a cup", 'TEXT: "Coffee" | label'))) == []

    def test_called_name_becomes_text(self) -> None:
        assert _texts(_caption("logo for a coffee shop called Moonbean, minimal, flat", _reply("OBJECT: a bean", 'TEXT: "Moonbean" | below'))) == ["Moonbean"]

    def test_the_thing_carrying_quoted_text_is_kept(self) -> None:
        caption = _caption('a white coffee mug with the text "BEST DAD"', _reply('OBJECT: A white ceramic coffee mug with the text "BEST DAD" in red', 'TEXT: "BEST DAD" | red'))
        kinds = [element["type"] for element in caption["compositional_deconstruction"]["elements"]]
        assert kinds == ["obj", "text"]

    def test_a_line_that_only_describes_the_lettering_is_dropped(self) -> None:
        caption = _caption("a poster with the text 'LISBOA'", _reply('OBJECT: A bold serif font text "LISBOA" in gold', 'TEXT: "LISBOA" | gold'))
        assert [element["type"] for element in caption["compositional_deconstruction"]["elements"]] == ["text"]

    @pytest.mark.parametrize(
        ("prompt", "text"), [("a diner sign with the text OPEN 24 HOURS", "OPEN 24 HOURS"), ("a poster with the text PLAN B", "PLAN B"), ("a neon sign saying 'Joe's Diner'", "Joe's Diner")]
    )
    def test_unquoted_and_single_quoted_lettering_is_read_whole(self, prompt: str, text: str) -> None:
        assert _texts(_caption(prompt, _reply("OBJECT: a sign"))) == [text]

    def test_a_text_line_matching_a_longer_quote_is_not_split_by_a_shorter_one(self) -> None:
        caption = _caption('a street with signs "OPEN" and "OPEN LATE"', _reply("OBJECT: a street", 'TEXT: "OPEN LATE"', 'TEXT: "OPEN" | red neon'))
        elements = [(element["text"], element["desc"]) for element in caption["compositional_deconstruction"]["elements"] if element["type"] == "text"]
        assert elements == [("OPEN LATE", "The words OPEN LATE."), ("OPEN", "red neon")]

    def test_a_name_in_a_scene_is_not_lettering(self) -> None:
        caption = _caption("a golden retriever named Max running on a beach", _reply("OBJECT: a golden retriever", 'TEXT: "Max" | collar'))
        assert _texts(caption) == []

    def test_card_greeting_from_the_prompt_is_kept_and_promoted(self) -> None:
        prompt = "birthday card for my grandma with flowers, Happy 80th Birthday Grandma"
        caption = _caption(prompt, _reply("OBJECT: flowers", 'OBJECT: The words "Happy 80th Birthday Grandma" in script', 'TEXT: "Love you" | small'))
        assert _texts(caption) == ["Happy 80th Birthday Grandma"]

    def test_card_greeting_written_unquoted_with_its_look(self) -> None:
        prompt = "birthday card for my grandma with flowers, Happy 80th Birthday Grandma"
        caption = _caption(prompt, _reply("OBJECT: flowers", "TEXT: Happy 80th Birthday Grandma displayed at the top in a cursive font"))
        element = caption["compositional_deconstruction"]["elements"][-1]
        assert (element["text"], element["desc"]) == ("Happy 80th Birthday Grandma", "displayed at the top in a cursive font")

    def test_card_greeting_written_twice_appears_once(self) -> None:
        prompt = "birthday card for my grandma with flowers, Happy 80th Birthday Grandma"
        caption = _caption(prompt, _reply('OBJECT: The words "Happy 80th Birthday Grandma" in script', 'TEXT: "Happy 80th Birthday Grandma" | gold'))
        assert _texts(caption) == ["Happy 80th Birthday Grandma"]

    def test_signs_in_a_named_language_keep_only_that_script(self) -> None:
        caption = _caption("a street market at night, neon signs in Japanese", _reply("OBJECT: a crowd", 'TEXT: "NEON NIGHT" | pink', 'TEXT: "ラーメン" | red'))
        assert _texts(caption) == ["ラーメン"]

    def test_a_language_name_inside_a_word_picks_no_script(self) -> None:
        caption = _caption("a cafe sign advertising arabica coffee", _reply("OBJECT: a cafe", 'TEXT: "COFFEE" | chalk'))
        assert _texts(caption) == ["COFFEE"]

    def test_invented_sign_text_is_capped(self) -> None:
        caption = _caption("a street with neon signs", _reply("OBJECT: a street", *(f'TEXT: "W{index}" | neon' for index in range(5))))
        assert len(_texts(caption)) == 3


class TestRun:
    def test_streams_labelled_lines_and_retries_once(self) -> None:
        enhancer = _FakeEnhancer("I can't do that.", _REPLY)
        streamed: list[str] = []
        result = enhance_caption(enhancer, "a red fox in snow", EnhanceSettings(), seed=7, ceiling=300, on_text=streamed.append)
        assert enhancer.seeds == [7, 8]
        assert json.loads(result.prompt)["high_level_description"].startswith("A red fox")
        assert any(text.startswith("SUMMARY:") for text in streamed)

    def test_prompt_is_labelled_as_text_to_rewrite(self) -> None:
        enhancer = _FakeEnhancer(_REPLY)
        enhance_caption(enhancer, "make a poster of a fox", EnhanceSettings(), seed=1, ceiling=300)
        system, user = enhancer.messages[0]
        assert user["content"] == "Prompt: make a poster of a fox"
        assert "- The user gives a Prompt." in system["content"]

    def test_two_unusable_answers_raise(self) -> None:
        with pytest.raises(RuntimeError, match=re.escape(CAPTION_NO_USABLE_MESSAGE)):
            enhance_caption(_FakeEnhancer("", "nope"), "a red fox", EnhanceSettings(), seed=1, ceiling=300)

    def test_empty_prompt(self) -> None:
        with pytest.raises(ValueError):
            enhance_caption(_FakeEnhancer(), "  ", EnhanceSettings(), seed=1, ceiling=300)
