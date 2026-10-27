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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello world", "hello world"),
        ("  padded  ", "padded"),
        (None, ""),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        (3.14, "3.14"),
        ({"style": "anime", "quality": "high"}, "style: anime. quality: high"),
        ({"mood": None}, "mood"),  # a value that flattens to nothing shows just the key
        (["red", "blue", "green"], "red. blue. green"),
        (["hello", None, "world"], "hello. world"),
        (object, str(object)),
    ],
    ids=["str", "str-stripped", "none", "true", "false", "int", "float", "dict", "dict-empty-value", "list", "list-drops-empty", "other-type-uses-str"],
)
def test_flatten_value(value, expected):
    assert flatten_value(value) == expected


# ── resolve_snippets ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "snippets", "expected"),
    [
        ("$color", {"color": "vivid red"}, "vivid red"),
        ("a $style painting", {"style": "watercolor"}, "a watercolor painting"),
        ("a $adj $noun", {"adj": "dark", "noun": "forest"}, "a dark forest"),
        ("$outer", {"inner": "deep blue", "outer": "$inner sky"}, "deep blue sky"),
        ("$nums", {"nums": [1, 2, 3]}, [1, 2, 3]),  # a standalone reference keeps the snippet's type
        ({"fg": "$color"}, {"color": "red"}, {"fg": "red"}),
        (["$x", "world"], {"x": "hello"}, ["hello", "world"]),
        (None, {}, None),
        (42, {}, 42),
        (True, {}, True),
    ],
    ids=["standalone", "inline", "several-inline", "nested", "standalone-non-string", "dict-values", "list-values", "none", "int", "bool"],
)
def test_resolve_snippets(value, snippets, expected):
    result = resolve_snippets(value, snippets)

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("value", "snippets", "match"),
    [
        ("$missing", {}, "Undefined snippet"),
        ("a $missing photo", {}, "Undefined snippet"),
        ("$a", {"a": "$b", "b": "$a"}, "Circular snippet reference"),
    ],
    ids=["undefined", "undefined-inline", "circular"],
)
def test_resolve_snippets_refuses_bad_references(value, snippets, match):
    with pytest.raises(ValueError, match=match):
        resolve_snippets(value, snippets)


# ── compose_video_prompt ────────────────────────────────────────────────────


class TestComposeVideoPrompt:
    @pytest.mark.parametrize("scene", [None, "", "   ", "\n\t"])
    def test_blank_scene_returns_the_prompt_byte_identical(self, scene):
        prompt = "  A dog runs.\n  "
        assert compose_video_prompt(scene, prompt) is prompt

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

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('She waves. She says "Bye"', "She waves."),  # a quote at the end is dropped
            ('She asks "Wait... what? Why!" quietly. Then she leaves.', "Then she leaves."),  # punctuation inside a quote does not end the sentence
            ('"Wait!" he says. The door shuts.', "The door shuts."),  # a lowercase continuation after a closed quote stays in the sentence
            ('A "x" b. C \u201cy\u201d d. E.', "E."),  # curly and straight quotes are both recognised
            ("  One.\n\nTwo   words.  ", "  One.\n\nTwo   words.  "),  # text without quotes is returned untouched
            ('Who is there? She says "Me". Nobody moves!', "Who is there? Nobody moves!"),
            ('She waves. She says "Bye. Then she leaves.', "She waves."),  # an unclosed quote drops the rest
            ("Moe's dog runs. It's fast.", "Moe's dog runs. It's fast."),  # apostrophes are ignored
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
