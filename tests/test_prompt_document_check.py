"""Tests for prompt-builder previews and problems."""

from __future__ import annotations

from dataclasses import replace

from zvisiongenerator.utils.prompt_document import DocumentEntry, DocumentSet, DocumentSnippet, PromptDocument, PromptValue
from zvisiongenerator.utils.prompt_document_check import has_errors, preview_document, roll_prompt, snippet_uses


def _text(text: str) -> PromptValue:
    return PromptValue(kind="text", text=text)


def _document(*entries: DocumentEntry, snippets: tuple[DocumentSnippet, ...] = (), name: str = "s") -> PromptDocument:
    return PromptDocument(snippets=snippets, sets=(DocumentSet(id="s0", name=name, entries=entries),))


def _problems(document: PromptDocument) -> list[tuple[str, str]]:
    return [(problem.target, problem.severity) for problem in preview_document(document)[1]]


class TestPreview:
    def test_resolves_snippets_and_flattens_fields(self):
        snippets = (DocumentSnippet(id="n0", name="light", value=_text("soft light")),)
        entry = DocumentEntry(id="e", prompt=PromptValue(kind="fields", fields=(("Subject", "a fox"), ("Style", "$light"))), negative=_text("blur"))

        previews, problems = preview_document(_document(entry, snippets=snippets))

        assert previews["e"].prompt == "Subject: a fox. Style: soft light"
        assert previews["e"].negative == "blur"
        assert problems == []

    def test_structured_values_preview_like_the_loader(self):
        entry = DocumentEntry(id="e", prompt=PromptValue(kind="structured", data={"Subjects": ["Lisa", {"Nina": "red dress"}]}))

        previews, _ = preview_document(_document(entry))

        assert previews["e"].prompt == "Subjects: Lisa. Nina: red dress"

    def test_roll_picks_one_option_per_choice(self):
        previews, _ = preview_document(_document(DocumentEntry(id="e", prompt=_text("a {red|red} car"))))

        assert roll_prompt(previews["e"]) == "a red car"


class TestProblems:
    def test_undefined_snippet_is_an_error_for_active_entries(self):
        assert _problems(_document(DocumentEntry(id="e", prompt=_text("a $missing")))) == [("e", "error")]

    def test_problems_in_inactive_entries_are_warnings(self):
        assert _problems(_document(DocumentEntry(id="e", prompt=_text(""), active=False))) == [("e", "warning")]

    def test_empty_prompt_is_an_error(self):
        assert _problems(_document(DocumentEntry(id="e", prompt=_text("  ")))) == [("e", "error")]

    def test_invalid_enhance_is_a_warning(self):
        entry = DocumentEntry(id="e", prompt=_text("a"), enhance={"style": "nope"})

        problems = preview_document(_document(entry))[1]

        assert [(problem.target, problem.severity) for problem in problems] == [("e", "warning")]
        assert not has_errors(problems)

    def test_no_op_enhance_is_a_warning(self):
        entry = DocumentEntry(id="e", prompt=_text("a"), enhance={"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": []})

        assert _problems(_document(entry)) == [("e", "warning")]

    def test_set_names(self):
        entry = DocumentEntry(id="e", prompt=_text("a"))
        document = PromptDocument(sets=(DocumentSet(id="a", name="x", entries=(entry,)), DocumentSet(id="b", name="x"), DocumentSet(id="c", name="snippets"), DocumentSet(id="d", name=" ")))

        assert _problems(document) == [("a", "error"), ("b", "error"), ("c", "error"), ("d", "error")]

    def test_snippet_names(self):
        snippets = (DocumentSnippet(id="n0", name="2tone", value=_text("x")), DocumentSnippet(id="n1", name="a", value=_text("x")), DocumentSnippet(id="n2", name="a", value=_text("y")))

        assert _problems(_document(DocumentEntry(id="e", prompt=_text("p")), snippets=snippets)) == [("n0", "error"), ("n1", "error"), ("n2", "error")]

    def test_field_names(self):
        blank = DocumentEntry(id="e", prompt=PromptValue(kind="fields", fields=(("", "x"),)))
        twice = DocumentEntry(id="f", prompt=PromptValue(kind="fields", fields=(("A", "x"), ("A", "y"))))

        assert ("e", "error") in _problems(_document(blank))
        assert ("f", "error") in _problems(_document(twice))


class TestSnippetUses:
    def test_counts_references_in_entries_negatives_and_snippets(self):
        snippets = (DocumentSnippet(id="n0", name="a", value=_text("$b")), DocumentSnippet(id="n1", name="b", value=_text("x")), DocumentSnippet(id="n2", name="c", value=_text("y")))
        entry = DocumentEntry(id="e", prompt=PromptValue(kind="fields", fields=(("K", "$a and $a"),)), negative=_text("$b"))

        assert snippet_uses(_document(entry, snippets=snippets)) == {"n0": 2, "n1": 2, "n2": 0}

    def test_renamed_snippet_counts_by_its_new_name(self):
        snippets = (DocumentSnippet(id="n0", name="a", value=_text("x")),)
        document = _document(DocumentEntry(id="e", prompt=_text("$b")), snippets=snippets)

        assert snippet_uses(replace(document, snippets=(replace(snippets[0], name="b"),))) == {"n0": 1}
