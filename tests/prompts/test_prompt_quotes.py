"""Tests for finding quoted texts and checking that a rewrite keeps them, and for the enhancer's quote rule and retry."""

from __future__ import annotations

import warnings

import pytest

from zvisiongenerator.utils.prompt_enhance import QUOTES_CHANGED_WARNING, EnhanceContext, EnhanceSettings, enforced_quotes, enhance_prompt
from zvisiongenerator.utils.prompt_enhance_video import enhance_video_prompt
from zvisiongenerator.utils.prompt_quotes import find_text, lettering_spans, missing_quotes, quoted_texts


class _FakeEnhancer:
    """Return scripted outputs, one per generate() call, and keep each call's messages."""

    repo = "fake/repo"
    revision = None

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.messages.append(messages)
        yield self.outputs.pop(0)

    def close(self) -> None:
        pass


class TestQuotedTexts:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('a sign that says "OPEN 24 HOURS"', ("OPEN 24 HOURS",)),
            ("a mug with “Best Dad Ever” printed on it", ("Best Dad Ever",)),
            ("a bookshop called 'The Quiet Page'", ("The Quiet Page",)),
            ("Moe's diner and rock 'n' roll", ()),
            ('"A" then "B" then "A" again', ("A", "B")),
            ("a banner with the text GRAND OPENING", ("GRAND OPENING",)),
        ],
    )
    def test_finds_quotes_in_order_without_repeats(self, text, expected):
        assert quoted_texts(text) == expected

    def test_lettering_leaves_out_speech(self):
        text = 'a cafe sign reading "Café Nord"; a man says "Late again." and she asks: "Coming?"'
        assert lettering_spans(text) == ("Café Nord",)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('Hikers pass; a man says "Hi"', ()),
            ('she says "Hi" under a sign reading "Hi"', ("Hi",)),
            ('A hello there sign; she says "hello there"', ()),
            ('a sign reading "EXIT" and he shouts "EXIT"', ("EXIT",)),
            ('he says   \n  "Run!" past a sign reading "Run"', ("Run",)),
        ],
    )
    def test_speech_is_judged_where_each_quote_stands(self, text, expected):
        assert lettering_spans(text) == expected

    @pytest.mark.parametrize(
        ("spans", "output", "missing"),
        [
            (("OPEN",), "A sign reading OPENING SOON.", ("OPEN",)),
            (("OPEN",), 'A sign reading "OPEN", lit red.', ()),
            (("Hi",), "Hikers pass by.", ("Hi",)),
            (("Café Nord",), "Café Nord's awning.", ()),
            (("营业中",), "门上写着营业中三个字。", ()),
            (("24",), "Open 24/7.", ()),
        ],
    )
    def test_a_quote_inside_a_longer_word_is_not_kept(self, spans, output, missing):
        assert missing_quotes(spans, output) == missing

    def test_find_text_skips_a_match_inside_a_longer_word(self):
        assert find_text('Hikers pass a sign reading "Hi".', "Hi") == 28
        assert find_text("Hikers pass.", "Hi") == -1

    def test_missing_quotes_needs_exact_case_and_ignores_spacing(self):
        spans = ("OPEN 24 HOURS", "Café Nord")
        assert missing_quotes(spans, 'Neon reads "OPEN   24 HOURS" over "Café Nord".') == ()
        assert missing_quotes(spans, "Neon reads open 24 hours over Café Nord.") == ("OPEN 24 HOURS",)

    def test_enforced_quotes_for_silent_video_skip_speech(self):
        text = 'a sign reading "EXIT"; he says "Run!"'
        assert enforced_quotes(text, mode="video", context=EnhanceContext(audio=False)) == ("EXIT",)
        assert enforced_quotes(text, mode="video", context=EnhanceContext(audio=True)) == ("EXIT", "Run!")
        assert enforced_quotes(text, mode="image", context=EnhanceContext()) == ("EXIT", "Run!")


class TestImageRewrite:
    _PROMPT = 'a diner with a neon sign that says "OPEN 24 HOURS"'

    def test_rule_lists_the_quotes(self):
        enhancer = _FakeEnhancer(['A diner at night under a neon sign reading "OPEN 24 HOURS".'])
        enhance_prompt(enhancer, self._PROMPT, EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert '"OPEN 24 HOURS"' in enhancer.messages[0][0]["content"].split("Text in quotes", 1)[1]

    def test_no_rule_without_quotes(self):
        enhancer = _FakeEnhancer(["A red fox curls up in deep snow under soft light."])
        enhance_prompt(enhancer, "a red fox in snow", EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert "Text in quotes" not in enhancer.messages[0][0]["content"]

    def test_retries_when_a_quote_changes(self):
        enhancer = _FakeEnhancer(["A diner under a neon open 24 hours sign.", 'A diner under a neon sign reading "OPEN 24 HOURS".'])
        result = enhance_prompt(enhancer, self._PROMPT, EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert len(enhancer.messages) == 2
        assert result.prompt.endswith('"OPEN 24 HOURS".')
        assert result.warning is None

    def test_keeps_the_last_usable_rewrite_with_a_warning(self):
        enhancer = _FakeEnhancer(["A diner under an open 24 hours sign.", "A diner under a neon Open 24 Hours sign."])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = enhance_prompt(enhancer, self._PROMPT, EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert result.prompt == "A diner under a neon Open 24 Hours sign."
        assert result.warning == QUOTES_CHANGED_WARNING
        assert any(str(warning.message) == QUOTES_CHANGED_WARNING for warning in caught)

    def test_a_changed_quote_beats_an_unusable_retry(self):
        enhancer = _FakeEnhancer(["A diner under an open 24 hours sign.", ""])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = enhance_prompt(enhancer, self._PROMPT, EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert result.prompt == "A diner under an open 24 hours sign."
        assert result.warning == QUOTES_CHANGED_WARNING


class TestVideoRewrite:
    _CONTEXT = EnhanceContext(audio=True)

    def test_pair_retries_when_a_quote_changes(self):
        bad = 'SCENE: Two friends at a bus stop in the rain.\nACTION: They meet and one says "you made it".'
        good = 'SCENE: Two friends at a bus stop in the rain.\nACTION: They meet and one says "You made it!"'
        enhancer = _FakeEnhancer([bad, good])
        result = enhance_video_prompt(enhancer, "", 'two friends meet and one says "You made it!"', EnhanceSettings(), seed=1, ceiling=300, context=self._CONTEXT)
        assert result.prompt.endswith('"You made it!"')
        assert result.warning is None
        assert "Text in quotes" in enhancer.messages[0][0]["content"]

    def test_action_only_keeps_last_rewrite_with_a_warning(self):
        enhancer = _FakeEnhancer(['She waves and calls "see you"', 'She turns, waves and calls "See You"'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = enhance_video_prompt(enhancer, "A woman on a snowy bridge.", 'she waves and calls "See you!"', EnhanceSettings(parts="action"), seed=1, ceiling=300, context=self._CONTEXT)
        assert result.prompt == 'She turns, waves and calls "See You"'
        assert result.warning == QUOTES_CHANGED_WARNING

    def test_silent_video_does_not_retry_for_speech(self):
        enhancer = _FakeEnhancer(["She waves goodbye and walks away across the bridge."])
        result = enhance_video_prompt(
            enhancer, "A woman on a snowy bridge.", 'she waves and says "See you!"', EnhanceSettings(parts="action"), seed=1, ceiling=300, context=EnhanceContext(audio=False)
        )
        assert len(enhancer.messages) == 1
        assert result.warning is None
