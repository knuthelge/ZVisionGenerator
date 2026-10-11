"""Tests for zvisiongenerator.utils.prompt_compose — flatten_value and resolve_snippets."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.prompt_compose import (
    compose_video_prompt,
    expand_random_choices,
    flatten_value,
    is_original_prompt,
    normalize_prompt_text,
    resolve_snippets,
    strip_quoted_lines,
    strip_quotes_while_original,
)


# ── flatten_value ───────────────────────────────────────────────────────────


class TestFlattenValue:
    def test_string_passthrough(self):
        assert flatten_value("hello world") == "hello world"

    def test_string_strips_whitespace(self):
        assert flatten_value("  padded  ") == "padded"

    def test_none_returns_empty(self):
        assert flatten_value(None) == ""

    def test_bool_true(self):
        assert flatten_value(True) == "true"

    def test_bool_false(self):
        assert flatten_value(False) == "false"

    def test_int(self):
        assert flatten_value(42) == "42"

    def test_float(self):
        assert flatten_value(3.14) == "3.14"

    def test_dict_key_value_pairs(self):
        result = flatten_value({"style": "anime", "quality": "high"})
        assert "style: anime" in result
        assert "quality: high" in result

    def test_dict_with_empty_value(self):
        result = flatten_value({"mood": None})
        # When value flattens to "", should just show key
        assert "mood" in result

    def test_list_joins_items(self):
        result = flatten_value(["red", "blue", "green"])
        assert "red" in result
        assert "blue" in result
        assert "green" in result

    def test_list_filters_empty(self):
        result = flatten_value(["hello", None, "world"])
        # None flattens to "" and is filtered
        assert result == "hello. world"

    def test_unknown_type_uses_str(self):
        assert flatten_value(object) != ""


# ── resolve_snippets ────────────────────────────────────────────────────────


class TestResolveSnippets:
    def test_standalone_reference(self):
        snippets = {"color": "vivid red"}
        result = resolve_snippets("$color", snippets)
        assert result == "vivid red"

    def test_inline_reference(self):
        snippets = {"style": "watercolor"}
        result = resolve_snippets("a $style painting", snippets)
        assert result == "a watercolor painting"

    def test_multiple_inline_references(self):
        snippets = {"adj": "dark", "noun": "forest"}
        result = resolve_snippets("a $adj $noun", snippets)
        assert result == "a dark forest"

    def test_undefined_snippet_raises(self):
        with pytest.raises(ValueError, match="Undefined snippet"):
            resolve_snippets("$missing", {})

    def test_undefined_inline_snippet_raises(self):
        with pytest.raises(ValueError, match="Undefined snippet"):
            resolve_snippets("a $missing photo", {})

    def test_circular_reference_raises(self):
        snippets = {"a": "$b", "b": "$a"}
        with pytest.raises(ValueError, match="Circular snippet reference"):
            resolve_snippets("$a", snippets)

    def test_nested_snippets(self):
        snippets = {"inner": "deep blue", "outer": "$inner sky"}
        result = resolve_snippets("$outer", snippets)
        assert result == "deep blue sky"

    def test_standalone_returns_non_string_type(self):
        snippets = {"nums": [1, 2, 3]}
        result = resolve_snippets("$nums", snippets)
        assert result == [1, 2, 3]

    def test_dict_values_resolved(self):
        snippets = {"color": "red"}
        result = resolve_snippets({"fg": "$color"}, snippets)
        assert result == {"fg": "red"}

    def test_list_values_resolved(self):
        snippets = {"x": "hello"}
        result = resolve_snippets(["$x", "world"], snippets)
        assert result == ["hello", "world"]

    def test_none_passthrough(self):
        assert resolve_snippets(None, {}) is None

    def test_int_passthrough(self):
        assert resolve_snippets(42, {}) == 42

    def test_bool_passthrough(self):
        assert resolve_snippets(True, {}) is True


# ── compose_video_prompt ────────────────────────────────────────────────────


class TestComposeVideoPrompt:
    @pytest.mark.parametrize("scene", [None, "", "   ", "\n\t"])
    def test_blank_scene_returns_the_prompt_byte_identical(self, scene):
        prompt = "  A dog runs.\n  "
        assert compose_video_prompt(scene, prompt) is prompt

    def test_none_scene_golden(self):
        assert compose_video_prompt(None, "A dog runs.") == "A dog runs."

    def test_blank_scene_keeps_whitespace_of_a_blank_prompt(self):
        assert compose_video_prompt("", " x ") == " x "

    def test_scene_comes_before_the_prompt(self):
        assert compose_video_prompt("A castle at dusk.", "A knight rides in.") == "A castle at dusk. A knight rides in."

    def test_scene_only_returns_the_scene(self):
        assert compose_video_prompt("Night.", "") == "Night."

    def test_scene_with_whitespace_only_prompt_drops_the_prompt(self):
        assert compose_video_prompt("Night.", "   ") == "Night."

    def test_parts_are_stripped_and_joined_by_one_space(self):
        assert compose_video_prompt("  A castle.\n", "\n A knight rides in.  ") == "A castle. A knight rides in."


# ── Extend helpers ──────────────────────────────────────────────────────────

_TALK_PROMPT = 'A woman in a red raincoat stands on a rainy city street at night, neon reflections on the wet pavement. She says "I told you so." Then she sighs.'


class TestStripQuotedLines:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('She looks at the camera and says "I told you…". Then she sighs.', "Then she sighs."),
            (_TALK_PROMPT, "A woman in a red raincoat stands on a rainy city street at night, neon reflections on the wet pavement. Then she sighs."),
            ("He shouts \u201cRun!\u201d and sprints off. The camera follows.", "The camera follows."),
            ('"Hello." She waves.', "She waves."),
            ("It's raining. Nobody speaks.", "It's raining. Nobody speaks."),
            ('"Hi."', ""),
            ("", ""),
            ('Dr. Smith says "Hi". Next.', "Next."),
            ('Mr. and Mrs. Jones wave. Prof. Lee says "Hello". Ms. Ray nods.', "Mr. and Mrs. Jones wave. Ms. Ray nods."),
            ('St. Paul sings "La". She bows, e.g. low. Jr. waves vs. Sr. now.', "She bows, e.g. low. Jr. waves vs. Sr. now."),
            ('He says "Hi" to Dr. Who, i.e. the doctor. Then he leaves.', "Then he leaves."),
            ("Dr. Smith waves. Next.", "Dr. Smith waves. Next."),
        ],
    )
    def test_goldens(self, text, expected):
        assert strip_quoted_lines(text) == expected

    def test_a_quote_at_the_end_is_dropped(self):
        assert strip_quoted_lines('She waves. She says "Bye"') == "She waves."

    def test_nested_punctuation_inside_a_quote_does_not_end_the_sentence(self):
        assert strip_quoted_lines('She asks "Wait... what? Why!" quietly. Then she leaves.') == "Then she leaves."

    def test_a_lowercase_continuation_after_a_closed_quote_stays_in_the_sentence(self):
        assert strip_quoted_lines('"Wait!" he says. The door shuts.') == "The door shuts."

    def test_curly_and_straight_quotes_are_both_recognised(self):
        assert strip_quoted_lines('A "x" b. C \u201cy\u201d d. E.') == "E."

    def test_text_without_quotes_is_returned_untouched(self):
        text = "  One.\n\nTwo   words.  "
        assert strip_quoted_lines(text) == text

    def test_a_question_mark_and_exclamation_end_sentences(self):
        assert strip_quoted_lines('Who is there? She says "Me". Nobody moves!') == "Who is there? Nobody moves!"

    def test_an_unclosed_quote_drops_the_rest(self):
        assert strip_quoted_lines('She waves. She says "Bye. Then she leaves.') == "She waves."

    def test_apostrophes_are_ignored(self):
        assert strip_quoted_lines("Moe's dog runs. It's fast.") == "Moe's dog runs. It's fast."

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('She says "Hello." and leaves. He nods.', "He nods."),
            ('She says "Hello" to him. He nods.', "He nods."),
            ('She says "Hi. Bye" and waves. He nods.', "He nods."),
            ('She says "a" and "b". Next one.', "Next one."),
            ('He says "Hello..." She nods. Last.', "She nods. Last."),
            ("He says \u201cHello\u2026 \u201d then waves. End.", "End."),
            ('She says "Hello.". Next.', "Next."),
            ('She says "Wow!"\nHe nods.', "He nods."),
            ('"Yes." "No." Then silence.', "Then silence."),
            ('One. "Two" and "three. Four. Five.', "One."),
        ],
    )
    def test_quote_edge_cases_drop_exactly_the_quoted_sentences(self, text, expected):
        assert strip_quoted_lines(text) == expected


class TestIsOriginalPrompt:
    @pytest.mark.parametrize(
        ("template", "original", "expected"),
        [
            ("A  dog\nruns.", "A dog runs.", True),
            ("  A dog runs.  ", "A dog runs.", True),
            ("A dog\truns.", "A dog  runs.", True),
            ("A dog runs!", "A dog runs.", False),
            ("a dog runs.", "A dog runs.", False),
            ("", "", True),
            ("", "A dog runs.", False),
            ("A {big|small} dog runs.", "A {big|small}  dog runs.", True),
            ("A {big|small} dog runs.", "A big dog runs.", False),
        ],
    )
    def test_compares_with_whitespace_normalised(self, template, original, expected):
        assert is_original_prompt(template, original) is expected

    def test_normalisation_is_shared(self):
        assert normalize_prompt_text("  a \n b\t c ") == "a b c"
        assert is_original_prompt(" a  b ", "a b") == (normalize_prompt_text(" a  b ") == normalize_prompt_text("a b"))


class TestStripQuotesWhileOriginal:
    def test_unchanged_template_strips_the_resolved_text(self):
        assert strip_quotes_while_original('She says "Hi". She sighs.', template='She says "Hi".  She sighs.', recorded_prompt='She says "Hi". She sighs.') == "She sighs."

    def test_edited_template_is_sent_as_resolved(self):
        resolved = 'She says "Hi". She sighs.'
        assert strip_quotes_while_original(resolved, template="She sighs.", recorded_prompt=resolved) == resolved

    def test_the_template_is_compared_before_choice_expansion(self):
        template = 'She says "{Hi|Yo}". She sighs.'
        resolved = expand_random_choices(template)
        assert strip_quotes_while_original(resolved, template=template, recorded_prompt=template) == "She sighs."
