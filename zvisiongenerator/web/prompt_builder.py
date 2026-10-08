"""Serve the Web UI prompt builder: load, preview, save and create prompt files as editable documents."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any
import warnings

from zvisiongenerator.utils.prompt_document import (
    DocumentEntry,
    DocumentSet,
    DocumentSnippet,
    PromptDocument,
    PromptValue,
    load_prompt_document,
    option_id_map,
    render_prompt_document,
)
from zvisiongenerator.utils.prompt_enhance import matrix_contract
from zvisiongenerator.utils.prompt_document_check import has_errors, preview_document, roll_prompt, snippet_uses
from zvisiongenerator.utils.prompts import inspect_prompts_text
from zvisiongenerator.utils.atomic_write import write_text_atomic
from zvisiongenerator.web.prompt_files import host_local_path, normalize_prompt_file_path


class PromptFileChangedError(Exception):
    """The prompt file changed on disk after the builder loaded it."""


def file_revision(data: bytes) -> str:
    """Return the revision id of a prompt file's bytes (SHA-256, hex)."""
    return hashlib.sha256(data).hexdigest()


def load_document_payload(path: str, *, accepted_extensions: tuple[str, ...]) -> dict[str, Any]:
    """Load a prompt file for the builder.

    Returns ``{path, revision, raw_text, enhance_matrix, document}``, or ``problem`` instead of ``document`` when
    the file is not valid YAML or not shaped like a prompt file, for the repair view.
    """
    normalized = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    data = normalized.read_bytes()
    text = data.decode("utf-8")
    payload: dict[str, Any] = {"path": str(normalized), "revision": file_revision(data), "raw_text": text, "enhance_matrix": matrix_contract()}
    try:
        payload["document"] = _document_to_payload(load_prompt_document(text))
    except ValueError as exc:
        payload["problem"] = str(exc)
    return payload


def save_document_payload(
    path: str,
    revision: str,
    document_payload: Any,
    *,
    base_text: str | None = None,
    force: bool = False,
    accepted_extensions: tuple[str, ...],
) -> dict[str, Any]:
    """Validate a builder document, apply it onto the file it was loaded from, and write it atomically.

    Args:
        path: The prompt file.
        revision: The revision the builder loaded.
        document_payload: The edited document (see :func:`_document_from_payload`).
        base_text: The text the builder loaded; required with *force*, so edits apply to the version they were made on.
        force: Overwrite even when the file changed on disk since *revision*.
        accepted_extensions: Allowed prompt-file extensions.

    Raises:
        PromptFileChangedError: When the file no longer matches *revision* and *force* is not set.
        ValueError: When the document is malformed or has errors, or the result would not load.
    """
    normalized = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    if force:
        if base_text is None or file_revision(base_text.encode("utf-8")) != revision:
            raise ValueError("Overwriting needs the text the builder loaded.")
        original_text = base_text
    else:
        current = normalized.read_bytes()
        if file_revision(current) != revision:
            raise PromptFileChangedError("The prompt file changed on disk since it was opened.")
        original_text = current.decode("utf-8")

    document = _document_from_payload(document_payload)
    _previews, problems = preview_document(document)
    if has_errors(problems):
        messages = "; ".join(dict.fromkeys(problem.message for problem in problems if problem.severity == "error"))
        raise ValueError(f"Fix these problems before saving: {messages}")

    before = load_prompt_document(original_text)
    new_text = render_prompt_document(original_text, document)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inspect_prompts_text(new_text, source_name=str(normalized))
    write_text_atomic(normalized, new_text)

    return {
        "path": str(normalized),
        "revision": file_revision(new_text.encode("utf-8")),
        "raw_text": new_text,
        "document": _document_to_payload(load_prompt_document(new_text)),
        "ids": _positional_ids(document),
        "option_id_map": option_id_map(before, document),
        "warnings": list(dict.fromkeys(str(warning.message) for warning in caught)),
    }


def preview_document_payload(document_payload: Any, *, roll_entry_id: str | None = None) -> dict[str, Any]:
    """Return entry previews, problems and snippet use counts for a builder document; optionally roll one entry's choices."""
    document = _document_from_payload(document_payload)
    previews, problems = preview_document(document)
    rolled = None
    if roll_entry_id is not None and roll_entry_id in previews:
        rolled = {"entry_id": roll_entry_id, "prompt": roll_prompt(previews[roll_entry_id])}
    return {
        "entries": {entry_id: asdict(preview) for entry_id, preview in previews.items()},
        "problems": [asdict(problem) for problem in problems],
        "snippet_uses": snippet_uses(document),
        "rolled": rolled,
    }


def create_prompt_file(directory: str, name: str, *, accepted_extensions: tuple[str, ...]) -> dict[str, Any]:
    """Create a new, empty prompt file in an existing folder.

    Raises:
        ValueError: When the folder is missing, the name is empty or contains a path separator, or the file exists.
    """
    folder = host_local_path(directory)
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    stem = name.strip()
    if not stem or any(separator in stem for separator in ("/", "\\")) or stem in {".", ".."}:
        raise ValueError("Give the prompt file a name without '/' or '\\'.")
    if Path(stem).suffix.lower() not in accepted_extensions:
        stem = f"{stem}{accepted_extensions[0]}"
    target = folder / stem
    try:
        with target.open("x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        raise ValueError(f"A file called {stem} already exists in {folder}.") from exc
    return {"path": str(target.resolve())}


def _document_to_payload(document: PromptDocument) -> dict[str, Any]:
    """Serialize a document for the Web UI."""
    return {
        "snippets": [{"id": snippet.id, "name": snippet.name, "value": _value_to_payload(snippet.value)} for snippet in document.snippets],
        "sets": [
            {
                "id": doc_set.id,
                "name": doc_set.name,
                "entries": [
                    {
                        "id": entry.id,
                        "prompt": _value_to_payload(entry.prompt),
                        "negative": _value_to_payload(entry.negative) if entry.negative is not None else None,
                        "active": entry.active,
                        "enhance": entry.enhance,
                    }
                    for entry in doc_set.entries
                ],
            }
            for doc_set in document.sets
        ],
    }


def _document_from_payload(payload: Any) -> PromptDocument:
    """Parse a document sent by the Web UI.

    Raises:
        ValueError: When the payload is not shaped like a document.
    """
    data = _mapping(payload, "document")
    snippets = tuple(
        DocumentSnippet(id=_string(item, "id"), name=_string(item, "name", allow_empty=True), value=_value_from_payload(item.get("value"), "snippet value"))
        for item in (_mapping(raw, "snippet") for raw in _list(data, "snippets"))
    )
    sets: list[DocumentSet] = []
    for raw_set in _list(data, "sets"):
        set_data = _mapping(raw_set, "set")
        entries: list[DocumentEntry] = []
        for raw_entry in _list(set_data, "entries"):
            entry = _mapping(raw_entry, "entry")
            active = entry.get("active", True)
            if not isinstance(active, bool):
                raise ValueError("Entry 'active' must be true or false.")
            negative = entry.get("negative")
            entries.append(
                DocumentEntry(
                    id=_string(entry, "id"),
                    prompt=_value_from_payload(entry.get("prompt"), "prompt"),
                    negative=None if negative is None else _value_from_payload(negative, "negative"),
                    active=active,
                    enhance=entry.get("enhance"),
                )
            )
        sets.append(DocumentSet(id=_string(set_data, "id"), name=_string(set_data, "name", allow_empty=True), entries=tuple(entries)))
    document = PromptDocument(snippets=snippets, sets=tuple(sets))
    _require_unique_ids(document)
    return document


def _require_unique_ids(document: PromptDocument) -> None:
    """Reject a document where two items share an id: both would be written onto the same original node."""
    ids = [snippet.id for snippet in document.snippets]
    ids += [doc_set.id for doc_set in document.sets]
    ids += [entry.id for doc_set in document.sets for entry in doc_set.entries]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"The document has duplicate ids: {', '.join(duplicates)}.")


def _positional_ids(document: PromptDocument) -> dict[str, str]:
    """Map each document id to the positional id the saved file gives it on the next load."""
    ids = {snippet.id: f"n{index}" for index, snippet in enumerate(document.snippets)}
    for set_index, doc_set in enumerate(document.sets):
        ids[doc_set.id] = f"s{set_index}"
        for entry_index, entry in enumerate(doc_set.entries):
            ids[entry.id] = f"s{set_index}.e{entry_index}"
    return ids


def _value_to_payload(value: PromptValue) -> dict[str, Any]:
    if value.kind == "text":
        return {"kind": "text", "text": value.text}
    if value.kind == "fields":
        return {"kind": "fields", "fields": [{"key": key, "value": text} for key, text in value.fields]}
    return {"kind": "structured", "data": value.data}


def _value_from_payload(raw: Any, what: str) -> PromptValue:
    data = _mapping(raw, what)
    kind = data.get("kind")
    if kind == "text":
        return PromptValue(kind="text", text=_string(data, "text", allow_empty=True))
    if kind == "fields":
        fields = tuple((_string(item, "key", allow_empty=True), _string(item, "value", allow_empty=True)) for item in (_mapping(raw_field, "field") for raw_field in _list(data, "fields")))
        return PromptValue(kind="fields", fields=fields)
    if kind == "structured":
        return PromptValue(kind="structured", data=data.get("data"))
    raise ValueError(f"Unknown {what} kind: {kind!r}.")


def _mapping(raw: Any, what: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Each {what} must be an object.")
    return raw


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be a list.")
    return value


def _string(data: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"'{key}' must be a non-empty string." if not allow_empty else f"'{key}' must be a string.")
    return value
