"""Tests for prompt loading — filtering and negative prompt extraction."""

from __future__ import annotations

import textwrap

import pytest

from zvisiongenerator.utils.prompts import inspect_prompts_file, inspect_prompts_text, load_prompts_file, scene_by_set


def test_load_prompts_filters_inactive(tmp_path):
    """Inactive entries (active: false) are excluded."""
    yaml_content = textwrap.dedent("""\
        my_set:
          - prompt: "active prompt"
          - prompt: "inactive prompt"
            active: false
          - prompt: "another active"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    result = load_prompts_file(str(f))
    assert len(result["my_set"]) == 2


def test_load_prompts_extracts_negative(tmp_path):
    """Negative prompts are extracted as tuple second element."""
    yaml_content = textwrap.dedent("""\
        set1:
          - prompt: "a cat"
            negative: "blurry"
          - prompt: "a dog"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    result = load_prompts_file(str(f))
    assert result["set1"][0] == ("a cat", "blurry")
    assert result["set1"][1] == ("a dog", None)


@pytest.mark.parametrize(
    ("yaml_content", "match"),
    [
        ("snippets:\n  - item1\n  - item2\nmy_set:\n  - prompt: hello\n", "snippets.*must be a mapping"),
        ('snippets: "not a dict"\nmy_set:\n  - prompt: hello\n', "snippets.*must be a mapping"),
        ("my_set:\n  - prompt: null\n", "empty prompt"),
        ('my_set:\n  - prompt: ""\n', "empty prompt"),
    ],
    ids=["snippets-list", "snippets-scalar", "null-prompt", "empty-prompt"],
)
def test_an_invalid_prompts_file_is_refused(tmp_path, yaml_content, match):
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)

    with pytest.raises(ValueError, match=match):
        load_prompts_file(str(f))


def test_malformed_yaml_prompts_raises_valueerror(tmp_path):
    """Malformed YAML in a prompts file must raise ValueError."""
    bad = tmp_path / "bad_prompts.yaml"
    bad.write_text("prompts:\n  - prompt: hello\n  bad_indent", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse prompts file"):
        load_prompts_file(str(bad))


def test_inspect_prompts_preserves_stable_source_indexes_for_active_options(tmp_path):
    """Active prompt options keep stable ids based on original source positions."""
    yaml_content = textwrap.dedent("""\
        snippets:
          mood: cinematic lighting
        portrait:
          - prompt: ["hero", $mood]
          - prompt: "inactive"
            active: false
          - prompt: "second active"
            negative: "blurry"
    """)
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(yaml_content, encoding="utf-8")

    inspection = inspect_prompts_file(str(prompt_file))

    assert inspection.prompts_data == {
        "portrait": [
            ("hero. cinematic lighting", None),
            ("second active", "blurry"),
        ]
    }
    assert [(option.id, option.prompt, option.negative_prompt) for option in inspection.options] == [
        ("portrait:0", "hero. cinematic lighting", None),
        ("portrait:2", "second active", "blurry"),
    ]


def test_set_names_are_kept_exactly_as_written(tmp_path):
    """Unquoted YAML keys like 2025:, true: or 1.10: must not be renamed (True, 1.1) or crash filename generation."""
    yaml_content = textwrap.dedent("""\
        2025:
          - prompt: "year prompt"
        true:
          - prompt: "bool prompt"
        1.10:
          - prompt: "float prompt"
        null:
          - prompt: "null prompt"
    """)
    f = tmp_path / "prompts.yaml"
    f.write_text(yaml_content)
    result = load_prompts_file(str(f))
    assert set(result) == {"2025", "true", "1.10", "null"}


def test_duplicate_set_names_warn_and_keep_last(tmp_path):
    """Duplicate names (e.g. 1: and "1":) keep YAML's last-wins behaviour so existing files load, with a warning."""
    f = tmp_path / "prompts.yaml"
    f.write_text('1:\n  - prompt: "a"\n"1":\n  - prompt: "b"\n')
    with pytest.warns(UserWarning, match="Duplicate prompt set name '1'"):
        result = load_prompts_file(str(f))
    assert [prompt for prompt, _ in result["1"]] == ["b"]


def test_empty_set_name_raises(tmp_path):
    f = tmp_path / "prompts.yaml"
    f.write_text('"":\n  - prompt: "a"\n')
    with pytest.raises(ValueError, match="Prompt set name .* is empty"):
        load_prompts_file(str(f))


class TestScene:
    """Entry ``scene:`` loading."""

    def test_entry_scene_is_exposed_per_entry_aligned_with_prompts(self):
        text = textwrap.dedent("""\
            s:
              - prompt: "a knight rides in"
                scene: "A castle at dusk."
              - prompt: "a dog barks"
              - prompt: "skipped"
                active: false
                scene: "never loaded"
              - prompt: "a cat sleeps"
                scene: "A tidy room."
        """)

        inspection = inspect_prompts_text(text, source_name="t")

        assert inspection.scene_by_set == {"s": ["A castle at dusk.", None, "A tidy room."]}
        assert [option.scene for option in inspection.options] == ["A castle at dusk.", None, "A tidy room."]
        assert len(inspection.scene_by_set["s"]) == len(inspection.prompts_data["s"])

    def test_scene_does_not_change_prompt_pairs(self):
        with_scene = inspect_prompts_text("s:\n  - prompt: a\n    negative: b\n    scene: c\n", source_name="t")
        without = inspect_prompts_text("s:\n  - prompt: a\n    negative: b\n", source_name="t")

        assert with_scene.prompts_data == without.prompts_data == {"s": [("a", "b")]}

    def test_file_without_scene_has_none_for_every_entry(self):
        inspection = inspect_prompts_text("s:\n  - prompt: a\n  - prompt: b\nt:\n  - prompt: c\n", source_name="t")

        assert inspection.scene_by_set == {"s": [None, None], "t": [None]}

    def test_scene_resolves_snippets_and_keeps_choices(self):
        text = "snippets:\n  place: a {harbor|market}\n  look:\n    Style: noir\ns:\n  - prompt: p\n    scene: 'Set in $place. $look'\n"

        scene = inspect_prompts_text(text, source_name="t").options[0].scene

        assert scene == "Set in a {harbor|market}. Style: noir"

    def test_one_snippet_scene_is_shared_by_entries(self):
        text = "snippets:\n  castle: A castle at dusk.\ns:\n  - prompt: one\n    scene: $castle\n  - prompt: two\n    scene: $castle\n"

        assert inspect_prompts_text(text, source_name="t").scene_by_set == {"s": ["A castle at dusk.", "A castle at dusk."]}

    @pytest.mark.parametrize("scene", ['""', '"   "', "null"])
    def test_blank_scene_means_no_scene(self, scene):
        inspection = inspect_prompts_text(f"s:\n  - prompt: a\n    scene: {scene}\n", source_name="t")

        assert inspection.scene_by_set == {"s": [None]}

    def test_undefined_snippet_in_scene_raises(self):
        with pytest.raises(ValueError):
            inspect_prompts_text("s:\n  - prompt: a\n    scene: $missing\n", source_name="t")

    def test_entry_still_needs_a_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            inspect_prompts_text("s:\n  - scene: only a scene\n", source_name="t")

    def test_scene_by_set_groups_options_in_order(self):
        options = inspect_prompts_text("a:\n  - prompt: x\n    scene: one\nb:\n  - prompt: y\n", source_name="t").options

        assert scene_by_set(options) == {"a": ["one"], "b": [None]}

    def test_circular_snippet_in_scene_raises(self):
        with pytest.raises(ValueError, match="Circular"):
            inspect_prompts_text("snippets:\n  a: $b\n  b: $a\ns:\n  - prompt: p\n    scene: $a\n", source_name="t")
