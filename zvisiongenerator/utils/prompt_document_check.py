"""Preview and check a prompt-builder document with the same rules the prompt loader uses.

Previews resolve snippets and flatten structure with :mod:`zvisiongenerator.utils.prompt_compose`, exactly as
a run would. Problems use the loader's severities: what makes the CLI reject a file is an ``error`` (it blocks a
save), what the CLI only warns about or skips is a ``warning``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Literal

from zvisiongenerator.utils.prompt_compose import expand_random_choices, flatten_value, resolve_snippets, snippet_references
from zvisiongenerator.utils.prompt_document import SNIPPETS_KEY, DocumentEntry, PromptDocument
from zvisiongenerator.utils.prompt_enhance import parse_enhance_entry, validate_settings

Severity = Literal["error", "warning"]

_SNIPPET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class DocumentProblem:
    """One problem, tied to the id of the snippet, set or entry it is about."""

    target: str
    severity: Severity
    message: str


@dataclass(frozen=True)
class EntryPreview:
    """An entry's prompt and negative prompt as the model receives them (``{a|b}`` choices not yet picked)."""

    prompt: str
    negative: str | None


def _snippet_values(document: PromptDocument) -> dict[str, Any]:
    """Return the document's snippets as the loader's ``snippets`` mapping."""
    return {snippet.name: snippet.value.to_python() for snippet in document.snippets}


def _preview_entry(entry: DocumentEntry, snippets: dict[str, Any]) -> EntryPreview:
    """Resolve and flatten one entry.

    Raises:
        ValueError: For an undefined or circular snippet reference.
    """
    prompt = flatten_value(resolve_snippets(entry.prompt.to_python(), snippets))
    negative = flatten_value(resolve_snippets(entry.negative.to_python(), snippets)) if entry.negative is not None else ""
    return EntryPreview(prompt=prompt, negative=negative or None)


def preview_document(document: PromptDocument) -> tuple[dict[str, EntryPreview], list[DocumentProblem]]:
    """Return a preview for every entry that resolves, plus every problem in the document."""
    snippets = _snippet_values(document)
    previews: dict[str, EntryPreview] = {}
    problems = _name_problems(document)
    for doc_set in document.sets:
        for entry in doc_set.entries:
            # The loader skips inactive entries, so their problems never stop a run.
            severity: Severity = "error" if entry.active else "warning"
            problems.extend(_field_problems(entry))
            try:
                preview = _preview_entry(entry, snippets)
            except ValueError as exc:
                problems.append(DocumentProblem(entry.id, severity, f"{exc}."))
                continue
            previews[entry.id] = preview
            if not preview.prompt.strip():
                problems.append(DocumentProblem(entry.id, severity, "The prompt is empty."))
            problems.extend(_enhance_problems(entry))
    return previews, problems


def roll_prompt(preview: EntryPreview) -> str:
    """Pick one option from every ``{a|b|c}`` choice in *preview*'s prompt, as a run would."""
    return expand_random_choices(preview.prompt)


def snippet_uses(document: PromptDocument) -> dict[str, int]:
    """Count the ``$name`` references to each snippet, by snippet id, across entries and other snippets."""
    counts: Counter[str] = Counter()
    for snippet in document.snippets:
        counts.update(snippet_references(snippet.value.to_python()))
    for doc_set in document.sets:
        for entry in doc_set.entries:
            counts.update(snippet_references(entry.prompt.to_python()))
            if entry.negative is not None:
                counts.update(snippet_references(entry.negative.to_python()))
    return {snippet.id: counts[snippet.name] for snippet in document.snippets}


def has_errors(problems: list[DocumentProblem]) -> bool:
    """Return whether any problem would stop the file from loading."""
    return any(problem.severity == "error" for problem in problems)


def _name_problems(document: PromptDocument) -> list[DocumentProblem]:
    problems: list[DocumentProblem] = []
    set_counts = Counter(doc_set.name for doc_set in document.sets)
    for doc_set in document.sets:
        if not doc_set.name.strip():
            problems.append(DocumentProblem(doc_set.id, "error", "A prompt set needs a name."))
        elif doc_set.name == SNIPPETS_KEY:
            problems.append(DocumentProblem(doc_set.id, "error", f"'{SNIPPETS_KEY}' is reserved for snippets; give the set another name."))
        elif set_counts[doc_set.name] > 1:
            problems.append(DocumentProblem(doc_set.id, "error", f"There is more than one set called '{doc_set.name}'."))
    snippet_counts = Counter(snippet.name for snippet in document.snippets)
    for snippet in document.snippets:
        if not _SNIPPET_NAME_RE.match(snippet.name):
            problems.append(DocumentProblem(snippet.id, "error", f"'{snippet.name}' can't be used as ${snippet.name}: start with a letter or _ and use only letters, digits and _."))
        elif snippet_counts[snippet.name] > 1:
            problems.append(DocumentProblem(snippet.id, "error", f"There is more than one snippet called '{snippet.name}'."))
    return problems


def _field_problems(entry: DocumentEntry) -> list[DocumentProblem]:
    problems: list[DocumentProblem] = []
    for value in (entry.prompt, entry.negative):
        if value is None or value.kind != "fields":
            continue
        keys = [key.strip() for key, _ in value.fields]
        if not all(keys):
            problems.append(DocumentProblem(entry.id, "error", "Every field needs a name."))
        elif len(set(keys)) != len(keys):
            problems.append(DocumentProblem(entry.id, "error", "Two fields have the same name."))
    return problems


def _enhance_problems(entry: DocumentEntry) -> list[DocumentProblem]:
    if entry.enhance is None:
        return []
    try:
        settings = parse_enhance_entry(entry.enhance, mode="video", where="this entry")
        if settings is not None:
            validate_settings(settings, mode="video")
    except ValueError as exc:
        return [DocumentProblem(entry.id, "warning", f"{exc} The entry is not enhanced.")]
    return []
