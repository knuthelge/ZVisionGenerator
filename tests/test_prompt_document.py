"""Tests for the prompt-builder document: YAML load, comment-preserving render, and option-id remapping."""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from zvisiongenerator.utils.prompt_document import (
    DocumentEntry,
    DocumentSet,
    DocumentSnippet,
    PromptDocument,
    PromptValue,
    append_entry,
    is_active,
    load_prompt_document,
    minimal_enhance,
    option_id_map,
    render_prompt_document,
)
from zvisiongenerator.utils.prompts import inspect_prompts_text

SAMPLE = """\
# Header comment
snippets:
  nina:
    Hair: Tidy
  diner: a cozy diner

# Portrait section
portrait:
  - prompt: "A woman at $diner"  # quoted on purpose
    negative: blurry
  - active: False
    prompt: |
      Two lines
      of text

# Structured section
scene:
  - prompt:
      Subjects:
        - Nina: $nina
      Style: warm
    custom_key: kept
"""


def _text(text: str) -> PromptValue:
    return PromptValue(kind="text", text=text)


class TestLoad:
    def test_loads_snippets_sets_and_entries_with_positional_ids(self):
        document = load_prompt_document(SAMPLE)

        assert [(snippet.id, snippet.name) for snippet in document.snippets] == [("n0", "nina"), ("n1", "diner")]
        assert [(doc_set.id, doc_set.name) for doc_set in document.sets] == [("s0", "portrait"), ("s1", "scene")]
        assert [entry.id for entry in document.sets[0].entries] == ["s0.e0", "s0.e1"]

    def test_value_kinds(self):
        document = load_prompt_document(SAMPLE)

        assert document.snippets[0].value == PromptValue(kind="fields", fields=(("Hair", "Tidy"),))
        assert document.snippets[1].value == _text("a cozy diner")
        assert document.sets[0].entries[1].prompt == _text("Two lines\nof text\n")
        structured = document.sets[1].entries[0].prompt
        assert structured.kind == "structured"
        assert structured.data == {"Subjects": [{"Nina": "$nina"}], "Style": "warm"}

    def test_active_uses_yaml_1_1_booleans(self):
        assert load_prompt_document("s:\n  - prompt: a\n    active: no\n").sets[0].entries[0].active is False
        assert load_prompt_document("s:\n  - prompt: a\n    active: yes\n").sets[0].entries[0].active is True
        assert load_prompt_document("s:\n  - prompt: a\n").sets[0].entries[0].active is True

    def test_active_agrees_with_the_prompt_loader_for_quoted_values(self):
        text = 's:\n  - prompt: a\n    active: "no"\n  - prompt: b\n    active: ""\n'
        document = load_prompt_document(text)

        loader_ids = [option.id for option in inspect_prompts_text(text, source_name="t").options]
        builder_ids = [f"s:{index}" for index, entry in enumerate(document.sets[0].entries) if entry.active]
        assert builder_ids == loader_ids == ["s:0"]

    @pytest.mark.parametrize("value", ["no", "Off", "FALSE", "false"])
    def test_is_active_false_values(self, value):
        assert is_active(value) is False

    def test_enhance_kept_as_written(self):
        document = load_prompt_document("s:\n  - prompt: a\n    enhance: true\n  - prompt: b\n    enhance:\n      style: cinematic\n")

        assert [entry.enhance for entry in document.sets[0].entries] == [True, {"style": "cinematic"}]

    def test_empty_file_is_an_empty_document(self):
        assert load_prompt_document("") == PromptDocument()

    @pytest.mark.parametrize(
        "text",
        [
            "s: [\n",  # invalid YAML
            "- a\n- b\n",  # not a mapping
            "s: a prompt\n",  # set is not a list
            "s:\n  - just text\n",  # entry is not a mapping
            "snippets: [a]\ns:\n  - prompt: x\n",  # snippets not a mapping
        ],
    )
    def test_rejects_invalid_structure(self, text):
        with pytest.raises(ValueError):
            load_prompt_document(text)


class TestRender:
    def test_unchanged_document_keeps_comments_quotes_and_blocks(self):
        rendered = render_prompt_document(SAMPLE, load_prompt_document(SAMPLE))

        assert rendered == SAMPLE

    def test_capitalized_booleans_are_kept(self):
        rendered = render_prompt_document(SAMPLE, load_prompt_document(SAMPLE))

        assert "active: False" in rendered

    def test_reordering_sets_moves_their_section_comments(self):
        document = load_prompt_document(SAMPLE)
        reordered = replace(document, sets=(document.sets[1], document.sets[0]))

        rendered = render_prompt_document(SAMPLE, reordered)

        assert rendered.index("# Structured section") < rendered.index("scene:") < rendered.index("# Portrait section") < rendered.index("portrait:")
        assert rendered.startswith("# Header comment\n")
        assert "# quoted on purpose" in rendered
        assert "custom_key: kept" in rendered

    def test_moving_an_entry_between_sets(self):
        document = load_prompt_document(SAMPLE)
        portrait, scene = document.sets
        moved = portrait.entries[0]
        edited = replace(document, sets=(replace(portrait, entries=portrait.entries[1:]), replace(scene, entries=(*scene.entries, moved))))

        data = inspect_prompts_text(render_prompt_document(SAMPLE, edited), source_name="t")

        assert [option.id for option in data.options] == ["scene:0", "scene:1"]
        assert data.options[1].prompt == "A woman at a cozy diner"

    def test_renaming_a_set_and_a_snippet(self):
        document = load_prompt_document(SAMPLE)
        diner = replace(document.snippets[1], name="place")
        portrait = document.sets[0]
        first = replace(portrait.entries[0], prompt=_text("A woman at $place"))
        edited = replace(document, snippets=(document.snippets[0], diner), sets=(replace(portrait, name="woman", entries=(first, portrait.entries[1])), document.sets[1]))

        rendered = render_prompt_document(SAMPLE, edited)
        data = inspect_prompts_text(rendered, source_name="t")

        assert "woman:" in rendered and "portrait:" not in rendered
        assert data.options[0].id == "woman:0"
        assert data.options[0].prompt == "A woman at a cozy diner"

    def test_changed_text_is_written_and_multiline_uses_a_block(self):
        document = load_prompt_document(SAMPLE)
        portrait = document.sets[0]
        edited_entry = replace(portrait.entries[0], prompt=_text("line one\nline two"))
        edited = replace(document, sets=(replace(portrait, entries=(edited_entry, portrait.entries[1])), document.sets[1]))

        rendered = render_prompt_document(SAMPLE, edited)

        assert "prompt: |-\n      line one\n      line two\n" in rendered
        assert "negative: blurry" in rendered

    def test_new_entry_writes_only_needed_keys(self):
        document = load_prompt_document(SAMPLE)
        new = DocumentEntry(id="new-1", prompt=_text("fresh"), active=False, enhance={"style": "cinematic", "mood": "keep", "details": [], "length": "same", "motion": []})
        edited = replace(document, sets=(*document.sets, DocumentSet(id="new-2", name="extra", entries=(new,))))

        rendered = render_prompt_document(SAMPLE, edited)

        assert rendered.endswith("extra:\n  - active: False\n    prompt: fresh\n    enhance:\n      style: cinematic\n")

    def test_turning_off_enhance_negative_and_active(self):
        text = "s:\n  - prompt: a\n    negative: b\n    enhance: true\n"
        document = load_prompt_document(text)
        entry = replace(document.sets[0].entries[0], negative=None, enhance=None, active=False)

        rendered = render_prompt_document(text, replace(document, sets=(replace(document.sets[0], entries=(entry,)),)))

        assert rendered == "s:\n  - active: false\n    prompt: a\n"

    def test_structured_values_are_never_rewritten(self):
        document = load_prompt_document(SAMPLE)

        rendered = render_prompt_document(SAMPLE, document)

        assert "        - Nina: $nina\n      Style: warm\n" in rendered

    def test_new_snippets_section_in_a_file_without_one(self):
        text = "s:\n  - prompt: a $x\n"
        document = replace(load_prompt_document(text), snippets=(DocumentSnippet(id="new-1", name="x", value=_text("thing")),))

        rendered = render_prompt_document(text, document)

        assert rendered == "snippets:\n  x: thing\n\ns:\n  - prompt: a $x\n"

    def test_empty_set_renders_as_empty_list(self):
        document = PromptDocument(sets=(DocumentSet(id="new-1", name="empty"),))

        assert render_prompt_document("", document) == "empty: []\n"

    def test_comment_above_a_set_stays_there_when_the_set_before_it_grows(self):
        text = "a:\n  - prompt: one\n\n# Set B comment\nb:\n  - prompt: two\n"
        document = load_prompt_document(text)
        first, second = document.sets
        grown = replace(document, sets=(replace(first, entries=(*first.entries, DocumentEntry(id="new-1", prompt=_text("three")))), second))

        rendered = render_prompt_document(text, grown)

        assert rendered == "a:\n  - prompt: one\n  - prompt: three\n\n# Set B comment\nb:\n  - prompt: two\n"

    def test_comment_above_a_set_stays_there_when_the_last_entry_moves_up(self):
        text = "a:\n  - prompt: one\n  - prompt: two\n\n# Set B comment\nb:\n  - prompt: three\n"
        document = load_prompt_document(text)
        first, second = document.sets
        swapped = replace(document, sets=(replace(first, entries=(first.entries[1], first.entries[0])), second))

        rendered = render_prompt_document(text, swapped)

        assert rendered.count("# Set B comment") == 1
        assert rendered.startswith("a:\n  - prompt: two\n  - prompt: one\n\n# Set B comment\nb:")

    def test_words_the_cli_reads_as_booleans_or_numbers_stay_strings(self):
        entries = (
            DocumentEntry(id="new-1", prompt=_text("off")),
            DocumentEntry(id="new-2", prompt=PromptValue(kind="fields", fields=(("on", "yes"), ("Year", "2025")))),
        )
        snippets = (DocumentSnippet(id="new-3", name="no", value=_text("1.10")),)
        document = PromptDocument(snippets=snippets, sets=(DocumentSet(id="new-4", name="yes", entries=entries),))

        data = inspect_prompts_text(render_prompt_document("", document), source_name="t")

        assert [option.id for option in data.options] == ["yes:0", "yes:1"]
        assert data.options[0].prompt == "off"
        assert data.options[1].prompt == "on: yes. Year: 2025"

    def test_quoted_snippet_names_still_resolve(self):
        document = PromptDocument(
            snippets=(DocumentSnippet(id="new-1", name="off", value=_text("soft light")),),
            sets=(DocumentSet(id="new-2", name="s", entries=(DocumentEntry(id="new-3", prompt=_text("a cat, $off")),)),),
        )

        data = inspect_prompts_text(render_prompt_document("", document), source_name="t")

        assert data.options[0].prompt == "a cat, soft light"

    @pytest.mark.parametrize(
        "change",
        [
            {"enhance": {"style": "cinematic"}},
            {"enhance": {"style": "cinematic", "details": []}},
            {"negative": PromptValue(kind="fields", fields=(("A", "x"), ("B", "y")))},
        ],
    )
    def test_comment_under_an_entry_survives_adding_a_mapping_key(self, change):
        text = "s:\n  - prompt: a\n  # keep this note\n  - prompt: b\n"
        document = load_prompt_document(text)
        first, second = document.sets[0].entries
        edited = replace(document, sets=(replace(document.sets[0], entries=(replace(first, **change), second)),))

        rendered = render_prompt_document(text, edited)

        assert rendered.count("  # keep this note\n") == 1
        assert rendered.index("# keep this note") < rendered.index("prompt: b")

    def test_indented_comment_after_the_last_entry_is_kept(self):
        text = "a:\n  - prompt: one\n  # - prompt: disabled\nb:\n  - prompt: two\n"

        rendered = render_prompt_document(text, load_prompt_document(text))

        assert "  - prompt: one\n  # - prompt: disabled\n" in rendered
        assert rendered.endswith("b:\n  - prompt: two\n")

    def test_comment_between_entries_survives_a_reorder(self):
        text = "a:\n  - prompt: one\n# col0 between\n  - prompt: three\nb:\n  - prompt: x\n"
        document = load_prompt_document(text)
        first, second = document.sets
        swapped = replace(document, sets=(replace(first, entries=(first.entries[1], first.entries[0])), second))

        rendered = render_prompt_document(text, swapped)

        assert rendered.count("# col0 between") == 1
        assert rendered.index("prompt: three") < rendered.index("prompt: one") < rendered.index("b:")

    def test_end_of_line_comment_on_the_last_entry_is_kept(self):
        text = "a:\n  - prompt: one  # keep me\n\n# Set B\nb:\n  - prompt: two\n"

        rendered = render_prompt_document(text, load_prompt_document(text))

        assert "prompt: one  # keep me\n" in rendered
        assert rendered.count("# Set B") == 1


class TestMinimalEnhance:
    def test_keeps_only_changed_axes(self):
        assert minimal_enhance({"style": "photo", "mood": "keep", "theme": "keep", "details": [], "length": "same", "camera": "keep", "motion": []}) == {"style": "photo"}

    def test_defaults_become_true(self):
        assert minimal_enhance({"style": "keep", "details": []}) is True

    def test_picked_details_are_written(self):
        assert minimal_enhance({"style": "photo", "details": ["lighting"]}) == {"style": "photo", "details": ["lighting"]}


class TestOptionIdMap:
    def test_maps_moved_and_renamed_active_entries(self):
        before = load_prompt_document(SAMPLE)
        portrait, scene = before.sets
        after = replace(before, sets=(replace(scene, entries=(*scene.entries, portrait.entries[0])), replace(portrait, name="woman", entries=portrait.entries[1:])))

        assert option_id_map(before, after) == {"portrait:0": "scene:1", "scene:0": "scene:0"}

    def test_drops_deleted_and_deactivated_entries(self):
        before = load_prompt_document(SAMPLE)
        portrait, scene = before.sets
        after = replace(before, sets=(replace(portrait, entries=(replace(portrait.entries[0], active=False),)),))

        assert option_id_map(before, after) == {}


SCENE_SAMPLE = """\
snippets:
  castle: A castle at dusk.

# Knight section
knights:
  - prompt: A knight rides in.  # the action
    scene: $castle  # shared look
    negative: blurry
    custom_key: kept
  - active: false
    prompt: Second
"""


def _with_entries(document: PromptDocument, *entries: DocumentEntry) -> PromptDocument:
    return replace(document, sets=(replace(document.sets[0], entries=entries), *document.sets[1:]))


class TestScene:
    def test_load_reads_scene_like_negative(self):
        document = load_prompt_document(SCENE_SAMPLE)
        with_scene, without = document.sets[0].entries

        assert with_scene.scene == _text("$castle")
        assert without.scene is None

    def test_null_or_missing_scene_loads_as_none(self):
        document = load_prompt_document("s:\n  - prompt: a\n    scene:\n  - prompt: b\n")

        assert [entry.scene for entry in document.sets[0].entries] == [None, None]

    def test_unchanged_document_round_trips_exactly(self):
        assert render_prompt_document(SCENE_SAMPLE, load_prompt_document(SCENE_SAMPLE)) == SCENE_SAMPLE

    def test_editing_a_scene_keeps_comments_order_and_other_keys(self):
        document = load_prompt_document(SCENE_SAMPLE)
        first, second = document.sets[0].entries

        rendered = render_prompt_document(SCENE_SAMPLE, _with_entries(document, replace(first, scene=_text("A harbour at night.")), second))

        assert "scene: A harbour at night." in rendered
        assert "# shared look" in rendered
        assert rendered.index("prompt:") < rendered.index("scene:") < rendered.index("negative:") < rendered.index("custom_key:")
        assert "# Knight section" in rendered
        assert "# the action" in rendered

    def test_adding_a_scene_puts_it_before_the_prompt_and_after_active(self):
        text = "s:\n  - active: false\n    prompt: a\n    negative: b\n"
        document = load_prompt_document(text)

        rendered = render_prompt_document(text, _with_entries(document, replace(document.sets[0].entries[0], scene=_text("A castle."))))

        assert rendered == "s:\n  - active: false\n    scene: A castle.\n    prompt: a\n    negative: b\n"

    def test_adding_a_scene_keeps_a_trailing_comment_last(self):
        text = "s:\n  - prompt: a\n    negative: b\n    # note\n"
        document = load_prompt_document(text)

        rendered = render_prompt_document(text, _with_entries(document, replace(document.sets[0].entries[0], scene=_text("A castle."))))

        assert rendered == "s:\n  - scene: A castle.\n    prompt: a\n    negative: b\n    # note\n"

    def test_removing_a_scene_deletes_the_key(self):
        document = load_prompt_document(SCENE_SAMPLE)
        first, second = document.sets[0].entries

        rendered = render_prompt_document(SCENE_SAMPLE, _with_entries(document, replace(first, scene=None), second))

        assert "scene:" not in rendered
        assert "negative: blurry" in rendered
        assert load_prompt_document(rendered).sets[0].entries[0].scene is None

    def test_new_entry_with_a_scene_writes_scene_before_prompt(self):
        text = "s:\n  - prompt: a\n"
        document = load_prompt_document(text)
        new = DocumentEntry(id="new-1", prompt=_text("fresh"), scene=_text("A castle."))

        rendered = render_prompt_document(text, _with_entries(document, *document.sets[0].entries, new))

        assert rendered.endswith("  - scene: A castle.\n    prompt: fresh\n")

    def test_fields_scene_round_trips_through_both_loaders(self):
        text = "s:\n  - prompt: a\n"
        document = load_prompt_document(text)
        scene = PromptValue(kind="fields", fields=(("Place", "a harbour"), ("Look", "noir")))

        rendered = render_prompt_document(text, _with_entries(document, replace(document.sets[0].entries[0], scene=scene)))

        assert load_prompt_document(rendered).sets[0].entries[0].scene == scene
        assert inspect_prompts_text(rendered, source_name="t").options[0].scene == "Place: a harbour. Look: noir"

    def test_existing_scene_in_an_unusual_position_keeps_its_place_when_edited_or_removed(self):
        text = "# top\ns:\n  - negative: n  # nn\n    scene: old  # keep\n    prompt: p\n    active: true\n\n  - prompt: q\n"
        document = load_prompt_document(text)
        first, second = document.sets[0].entries

        edited = render_prompt_document(text, _with_entries(document, replace(first, scene=_text("new")), second))
        removed = render_prompt_document(text, _with_entries(document, replace(first, scene=None), second))

        assert edited == text.replace("scene: old", "scene: new")
        assert removed == "# top\ns:\n  - negative: n  # nn\n    prompt: p\n    active: true\n\n  - prompt: q\n"


COMMENTED = """\
# Header comment
snippets:
  diner: a cozy diner  # x

# Above portrait
portrait:
  - prompt: first  # inline
    negative: blurry

# Above prompts
prompts:
  - prompt: existing

# Footer comment
"""


class TestAppendEntry:
    def test_appends_to_an_existing_set(self):
        document = load_prompt_document(SAMPLE)
        before = document.sets[0].entries

        updated, entry_id = append_entry(document, "portrait", "new one")

        assert entry_id == f"{document.sets[0].id}.e{len(before)}"
        assert updated.sets[0].entries[:-1] == before
        assert updated.sets[0].entries[-1].id == entry_id
        assert updated.sets[0].entries[-1].prompt == _text("new one")
        assert updated.sets[1:] == document.sets[1:]
        assert updated.snippets == document.snippets

    def test_adds_a_missing_set_at_the_end(self):
        document = load_prompt_document(SAMPLE)

        updated, entry_id = append_entry(document, "prompts", "new")

        assert entry_id == f"s{len(document.sets)}.e0"
        assert updated.sets[:-1] == document.sets
        assert updated.sets[-1].id == f"s{len(document.sets)}"
        assert updated.sets[-1].name == "prompts"
        assert len(updated.sets[-1].entries) == 1

    def test_empty_file_gets_one_set_with_one_entry(self):
        updated, entry_id = append_entry(load_prompt_document(""), "prompts", "new")

        assert [doc_set.name for doc_set in updated.sets] == ["prompts"]
        assert len(updated.sets[0].entries) == 1
        assert entry_id == "s0.e0"

    def test_scene_and_negative_are_included_and_stripped(self):
        updated, _ = append_entry(load_prompt_document(""), "prompts", "  new \n", scene="  a room ", negative=" ugly\n")

        entry = updated.sets[0].entries[0]
        assert (entry.prompt, entry.scene, entry.negative) == (_text("new"), _text("a room"), _text("ugly"))

    @pytest.mark.parametrize("blank", [None, "", "  ", "\n"])
    def test_blank_scene_and_negative_are_left_out(self, blank):
        updated, _ = append_entry(load_prompt_document(""), "prompts", "new", scene=blank, negative=blank)

        entry = updated.sets[0].entries[0]
        assert entry.scene is None
        assert entry.negative is None

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_prompt_raises(self, blank):
        with pytest.raises(ValueError):
            append_entry(load_prompt_document(SAMPLE), "prompts", blank)

    def test_new_entry_is_active_without_enhance(self):
        updated, _ = append_entry(load_prompt_document(""), "prompts", "new")

        entry = updated.sets[0].entries[0]
        assert entry.active is True
        assert entry.enhance is None

    def test_uses_the_first_of_duplicate_set_names(self):
        document = load_prompt_document("a:\n  - prompt: x\nb:\n  - prompt: y\n")
        document = replace(document, sets=(document.sets[0], replace(document.sets[1], name="a")))

        updated, entry_id = append_entry(document, "a", "new")

        assert entry_id == "s0.e1"
        assert len(updated.sets[1].entries) == 1


class TestAppendRender:
    def _render(self, text: str, **kwargs) -> tuple[str, str]:
        updated, entry_id = append_entry(load_prompt_document(text), "prompts", "new", **kwargs)
        return render_prompt_document(text, updated), entry_id

    def test_new_entry_is_last_with_scene_prompt_negative_order(self):
        rendered, entry_id = self._render(COMMENTED, scene="a room", negative="ugly")

        reloaded = load_prompt_document(rendered)
        assert reloaded.sets[-1].entries[-1].id == entry_id
        assert reloaded.sets[-1].entries[-1].prompt == _text("new")
        tail = rendered[rendered.index("  - prompt: existing") :]
        assert tail.index("scene: a room") < tail.index("prompt: new") < tail.index("negative: ugly")

    def test_no_active_or_enhance_key_is_written(self):
        rendered, _ = self._render(COMMENTED, scene="a room", negative="ugly")

        assert yaml.safe_load(rendered)["prompts"][-1] == {"scene": "a room", "prompt": "new", "negative": "ugly"}

    def test_comments_and_other_sets_survive(self):
        rendered, _ = self._render(COMMENTED)

        for line in ("# Header comment", "diner: a cozy diner  # x", "# Above portrait", "prompt: first  # inline", "# Above prompts"):
            assert line in rendered
        assert rendered.rstrip().endswith("# Footer comment")
        before, after = load_prompt_document(COMMENTED), load_prompt_document(rendered)
        assert after.sets[0] == before.sets[0]
        assert after.snippets == before.snippets

    def test_missing_set_is_created_before_the_footer(self):
        text = "portrait:\n  - prompt: a\n# footer\n"

        rendered, entry_id = self._render(text)

        assert entry_id == "s1.e0"
        assert rendered.index("prompts:") < rendered.index("# footer")
        assert rendered.rstrip().endswith("# footer")

    def test_indented_trailing_comment_stays_before_the_new_entry(self):
        text = "prompts:\n  - prompt: a\n  # - prompt: idea\n\n# above other\nother:\n  - prompt: b\n# footer\n"

        rendered, _ = self._render(text)

        assert rendered.index("# - prompt: idea") < rendered.index("prompt: new")
        assert "# above other\nother:" in rendered
        assert rendered.rstrip().endswith("# footer")
        inspect_prompts_text(rendered, source_name="test")

    def test_multi_line_prompt_renders_as_a_block_and_reloads_equal(self):
        updated, _ = append_entry(load_prompt_document(SAMPLE), "prompts", "line one\nline two\n")

        rendered = render_prompt_document(SAMPLE, updated)

        assert "prompt: |-\n" in rendered
        assert load_prompt_document(rendered).sets[-1].entries[-1].prompt == _text("line one\nline two")

    def test_yaml11_ambiguous_text_is_quoted(self):
        rendered, _ = self._render("", negative="no")

        assert yaml.safe_load(rendered)["prompts"][0]["negative"] == "no"
