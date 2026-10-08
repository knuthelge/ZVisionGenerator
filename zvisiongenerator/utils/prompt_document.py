"""Convert prompt-file YAML to an editable document and back, keeping comments and layout.

The Web UI's prompt builder edits a :class:`PromptDocument`. Loading and rendering use ruamel.yaml's round-trip
mode so a save keeps the file's comments, quoting and block scalars; :func:`render_prompt_document` reuses the
original nodes for every item that came from the file and only writes keys that changed. What a file *means*
is still decided by :mod:`zvisiongenerator.utils.prompts` (PyYAML), which every save is checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import re
from typing import Any, Literal

from ruamel.yaml import YAML
import yaml as pyyaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, LiteralScalarString, ScalarString

from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, settings_to_mapping

SNIPPETS_KEY = "snippets"
ENTRY_KEY_ORDER = ("active", "prompt", "negative", "enhance")

# YAML 1.1 booleans that PyYAML (and so the CLI) reads as false; ruamel reads YAML 1.2 and keeps them as strings.
_YAML11_FALSE = frozenset({"no", "No", "NO", "false", "False", "FALSE", "off", "Off", "OFF"})
# The CLI reads prompt files with PyYAML (YAML 1.1); its resolver says what a plain scalar would become.
_PYYAML_RESOLVER = pyyaml.resolver.Resolver()
_PYYAML_STR_TAG = "tag:yaml.org,2002:str"
_CAPITAL_BOOL_RE = re.compile(r":[ \t]+(True|False)[ \t]*(?:#|$)", re.MULTILINE)
_LOWER_BOOL_RE = re.compile(r":[ \t]+(true|false)[ \t]*(?:#|$)", re.MULTILINE)

ValueKind = Literal["text", "fields", "structured"]


@dataclass(frozen=True)
class PromptValue:
    """Hold a ``prompt``, ``negative`` or snippet value in the form the builder edits it.

    ``text`` is a plain string, ``fields`` a flat mapping of strings to strings, and ``structured`` anything
    else (lists, nested mappings, numbers), carried as plain data and never rewritten.
    """

    kind: ValueKind
    text: str = ""
    fields: tuple[tuple[str, str], ...] = ()
    data: Any = None

    def to_python(self) -> Any:
        """Return the value as the prompt loader would see it."""
        if self.kind == "text":
            return self.text
        if self.kind == "fields":
            return dict(self.fields)
        return self.data


@dataclass(frozen=True)
class DocumentEntry:
    """One prompt entry of a set."""

    id: str
    prompt: PromptValue
    negative: PromptValue | None = None
    active: bool = True
    enhance: Any = None  # None (off), True (default options), or the mapping as written


@dataclass(frozen=True)
class DocumentSet:
    """One prompt set: a name and its entries in file order."""

    id: str
    name: str
    entries: tuple[DocumentEntry, ...] = ()


@dataclass(frozen=True)
class DocumentSnippet:
    """One reusable snippet under the ``snippets`` key."""

    id: str
    name: str
    value: PromptValue


@dataclass(frozen=True)
class PromptDocument:
    """A prompt file as the builder edits it. Ids from a load are positional (``n0``, ``s0``, ``s0.e1``)."""

    snippets: tuple[DocumentSnippet, ...] = ()
    sets: tuple[DocumentSet, ...] = ()


def load_prompt_document(text: str) -> PromptDocument:
    """Build the editable document for prompt-file *text*.

    Raises:
        ValueError: When the text is not valid YAML, or its structure is not a mapping of sets that are lists
            of entry mappings (with an optional ``snippets`` mapping).
    """
    return _document_from_root(_load_round_trip(text))


def _document_from_root(root: Any) -> PromptDocument:
    """Build the document from a loaded round-trip tree, checking its structure (see :func:`load_prompt_document`)."""
    if root is None:
        return PromptDocument()
    if not isinstance(root, dict):
        raise ValueError("A prompt file must be a mapping of prompt set names to lists of entries.")

    raw_snippets = root.get(SNIPPETS_KEY)
    if raw_snippets is not None and not isinstance(raw_snippets, dict):
        raise ValueError(f"'{SNIPPETS_KEY}' must be a mapping of snippet names to values.")
    snippets = tuple(DocumentSnippet(id=f"n{index}", name=str(name), value=_value_from_node(value)) for index, (name, value) in enumerate((raw_snippets or {}).items()))

    sets: list[DocumentSet] = []
    for set_id, key, node in _iter_sets(root):
        name = str(key)
        if not isinstance(node, list):
            raise ValueError(f"Prompt set '{name}' must be a list of entries.")
        entries: list[DocumentEntry] = []
        for index, entry in enumerate(node):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index + 1} of prompt set '{name}' must be a mapping with a 'prompt' key.")
            entries.append(
                DocumentEntry(
                    id=f"{set_id}.e{index}",
                    prompt=_value_from_node(entry.get("prompt")),
                    negative=None if entry.get("negative") is None else _value_from_node(entry["negative"]),
                    active=is_active(entry.get("active", True)),
                    enhance=_enhance_from_node(entry.get("enhance")),
                )
            )
        sets.append(DocumentSet(id=set_id, name=name, entries=tuple(entries)))
    return PromptDocument(snippets=snippets, sets=tuple(sets))


def render_prompt_document(original_text: str, document: PromptDocument) -> str:
    """Return YAML text for *document*, applied onto *original_text* so its comments and layout survive.

    Items whose ids came from loading *original_text* reuse their original nodes, so their comments, key order
    and unknown keys are kept and only changed keys are written. ``snippets`` is written first. Column-0 comment
    lines above a top-level key travel with that key; the file's header and footer comments stay in place; top-level
    items are separated by one blank line.

    Raises:
        ValueError: When *original_text* cannot be loaded (see :func:`load_prompt_document`).
    """
    yaml = _round_trip_yaml(capital_bools=_prefers_capital_bools(original_text))
    root = _load_round_trip(original_text)
    _document_from_root(root)  # same structural checks as a load
    if not isinstance(root, CommentedMap):
        root = CommentedMap()
    lines = original_text.splitlines()

    items: list[tuple[Any, Any, Any]] = []  # (original key or None, key, value)
    if document.snippets:
        items.append((SNIPPETS_KEY if SNIPPETS_KEY in root else None, SNIPPETS_KEY, _render_snippets(root, document.snippets)))
    for key in root:
        _drop_trailing_comments(root[key])
    old_sets = {set_id: key for set_id, key, _ in _iter_sets(root)}
    old_entries = _index_entries(root)
    for doc_set in document.sets:
        old_key = old_sets.get(doc_set.id)
        key = old_key if old_key is not None and str(old_key) == doc_set.name else _yaml11_safe(doc_set.name)
        items.append((old_key, key, _render_entries(doc_set, old_entries)))

    blocks = [_dump_item(yaml, root, old_key, key, value, _leading_comments(root, old_key, lines)) for old_key, key, value in items]
    header, footer = _header_comments(root, lines), _footer_comments(root, lines)
    body = "\n".join(blocks + (["\n".join(footer) + "\n"] if footer else []))
    if not header:
        return body
    # Keep the header's own spacing: a blank line before the first key only when the file had one.
    gap = "\n" if root and lines[_key_line(root, next(iter(root))) - 1].strip() == "" else ""
    return "\n".join(header) + "\n" + gap + body


def option_id_map(before: PromptDocument, after: PromptDocument) -> dict[str, str]:
    """Map each active prompt id (``set:index``) in *before* to its id in *after*, for entries still active.

    Entries are matched by document id, so a moved, reordered or renamed entry keeps its selection.
    """
    after_ids = {entry.id: f"{doc_set.name}:{index}" for doc_set in after.sets for index, entry in enumerate(doc_set.entries) if entry.active}
    return {f"{doc_set.name}:{index}": after_ids[entry.id] for doc_set in before.sets for index, entry in enumerate(doc_set.entries) if entry.active and entry.id in after_ids}


def is_active(value: Any) -> bool:
    """Return whether an entry's ``active`` value runs the entry, the way PyYAML (and so the CLI) reads it.

    Plain ``no`` / ``off`` / ``false`` are YAML 1.1 booleans; quoted or block scalars stay strings, so they run the
    entry when non-empty.
    """
    if isinstance(value, ScalarString):
        return bool(str(value))
    if isinstance(value, str):
        return value not in _YAML11_FALSE
    return bool(value)


def minimal_enhance(settings: dict[str, Any]) -> dict[str, Any] | bool:
    """Return the axes of an ``enhance:`` mapping that differ from the defaults, or ``True`` when none do."""
    defaults = settings_to_mapping(EnhanceSettings())
    changed = {key: value for key, value in settings.items() if key in defaults and _as_comparable(value) != _as_comparable(defaults[key])}
    return changed or True


def _plain_data(node: Any) -> Any:
    """Convert round-trip YAML data to plain dicts, lists and scalars (mapping keys as strings)."""
    if isinstance(node, dict):
        return {str(key): _plain_data(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_plain_data(item) for item in node]
    if isinstance(node, bool) or node is None:
        return node
    if isinstance(node, str):
        return str(node)
    if isinstance(node, int):
        return int(node)
    if isinstance(node, float):
        return float(node)
    return node


def _as_comparable(value: Any) -> Any:
    return sorted(value) if isinstance(value, (list, tuple)) else value


def _load_round_trip(text: str) -> Any:
    try:
        return _round_trip_yaml().load(text)
    except YAMLError as exc:
        raise ValueError(f"The file is not valid YAML: {exc}") from exc


def _round_trip_yaml(*, capital_bools: bool = False) -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    if capital_bools:
        yaml.Representer = _CapitalBoolRepresenter
    return yaml


class _CapitalBoolRepresenter(RoundTripRepresenter):
    """Write booleans as ``True`` / ``False``, for files written that way."""


def _represent_capital_bool(representer: RoundTripRepresenter, data: bool) -> Any:
    return representer.represent_scalar("tag:yaml.org,2002:bool", "True" if data else "False")


_CapitalBoolRepresenter.add_representer(bool, _represent_capital_bool)


def _prefers_capital_bools(text: str) -> bool:
    return len(_CAPITAL_BOOL_RE.findall(text)) > len(_LOWER_BOOL_RE.findall(text))


def _iter_sets(root: Any) -> list[tuple[str, Any, Any]]:
    """Return ``(set_id, key, node)`` for every top-level key except ``snippets``, in file order."""
    if not isinstance(root, dict):
        return []
    keys = [key for key in root if key != SNIPPETS_KEY]
    return [(f"s{index}", key, root[key]) for index, key in enumerate(keys)]


def _index_entries(root: Any) -> dict[str, tuple[CommentedMap, CommentedSeq, int]]:
    """Map each entry id to ``(node, parent_seq, index)`` in the original tree."""
    index: dict[str, tuple[CommentedMap, CommentedSeq, int]] = {}
    for set_id, _key, node in _iter_sets(root):
        if isinstance(node, CommentedSeq):
            for position, entry in enumerate(node):
                if isinstance(entry, CommentedMap):
                    index[f"{set_id}.e{position}"] = (entry, node, position)
    return index


def _drop_trailing_comments(node: Any) -> None:
    """Remove the blank lines and column-0 comments ruamel attaches after the last item of a top-level value.

    Column-0 comments there sit above the next top-level key, and :func:`_leading_comments` writes them above that
    key; left in place they would follow the last item when it moves, or end up between it and a new item.
    End-of-line and indented comments (e.g. a commented-out entry) belong to the item and are kept.
    """
    while True:
        if isinstance(node, CommentedMap) and node:
            last = next(reversed(node))
            slot_owner, slot_key, index = node.ca.items, last, 2
            child = node[last]
        elif isinstance(node, CommentedSeq) and node:
            last = len(node) - 1
            slot_owner, slot_key, index = node.ca.items, last, 0
            child = node[last]
        else:
            return
        slot = slot_owner.get(slot_key)
        token = slot[index] if slot and len(slot) > index else None
        if token is not None:
            kept = _item_comment_lines(token.value, token.column)
            if kept is None:
                slot[index] = None
            else:
                token.value = kept
        node = child


def _item_comment_lines(value: str, first_column: int) -> str | None:
    """Return a trailing comment token's value without blank lines and column-0 comments, or None when nothing is left.

    The value's first line starts where the token starts (``first_column``) unless the value begins with a newline;
    every later line carries its own indentation.
    """
    starts_on_new_line = value.startswith("\n")
    lines = (value[1:] if starts_on_new_line else value).split("\n")
    kept = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        column = first_column if index == 0 and not starts_on_new_line else len(line) - len(line.lstrip())
        if column > 0:
            kept.append(line)
    if not kept:
        return None
    return ("\n" if starts_on_new_line else "") + "\n".join(kept) + "\n"


def _render_entries(doc_set: DocumentSet, old_entries: dict[str, tuple[CommentedMap, CommentedSeq, int]]) -> CommentedSeq:
    seq = CommentedSeq()
    for entry in doc_set.entries:
        old = old_entries.get(entry.id)
        node = old[0] if old is not None else CommentedMap()
        _apply_entry(node, entry)
        if old is not None and old[2] in old[1].ca.items:
            seq.ca.items[len(seq)] = old[1].ca.items[old[2]]
        seq.append(node)
    return seq


def _dump_item(yaml: YAML, root: CommentedMap, old_key: Any, key: Any, value: Any, leading: list[str]) -> str:
    """Dump one top-level item with its leading comments; trailing blank lines are dropped (items are joined with one)."""
    single = CommentedMap()
    single[key] = value
    if old_key is not None:
        _carry_item_comment(root, old_key, single, key)
    buffer = io.StringIO()
    yaml.dump(single, buffer)
    lines = buffer.getvalue().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(leading + lines) + "\n"


def _key_line(root: CommentedMap, key: Any) -> int:
    return root.lc.key(key)[0]


def _leading_comments(root: CommentedMap, key: Any, lines: list[str]) -> list[str]:
    """Return the column-0 comment lines directly above *key* (none for the first key: those are the header)."""
    if key is None or key not in root or next(iter(root)) == key:
        return []
    comments: list[str] = []
    index = _key_line(root, key) - 1
    while index >= 0 and (not lines[index].strip() or lines[index].startswith("#")):
        if lines[index].startswith("#"):
            comments.insert(0, lines[index])
        index -= 1
    return comments


def _header_comments(root: CommentedMap, lines: list[str]) -> list[str]:
    end = _key_line(root, next(iter(root))) if root else len(lines)
    return [line for line in lines[:end] if line.startswith("#")]


def _footer_comments(root: CommentedMap, lines: list[str]) -> list[str]:
    if not root:
        return []
    footer: list[str] = []
    index = len(lines) - 1
    while index > _key_line(root, list(root)[-1]) and (not lines[index].strip() or lines[index].startswith("#")):
        if lines[index].startswith("#"):
            footer.insert(0, lines[index])
        index -= 1
    return footer


def _carry_item_comment(source: CommentedMap, source_key: Any, target: CommentedMap, target_key: Any) -> None:
    if source_key in source.ca.items:
        target.ca.items[target_key] = source.ca.items[source_key]


def _render_snippets(root: CommentedMap, snippets: tuple[DocumentSnippet, ...]) -> CommentedMap:
    old = root.get(SNIPPETS_KEY)
    old_map = old if isinstance(old, CommentedMap) else CommentedMap()
    old_keys = {f"n{index}": key for index, key in enumerate(old_map)}
    out = CommentedMap()
    out.ca.comment = old_map.ca.comment
    for snippet in snippets:
        old_key = old_keys.get(snippet.id)
        key = old_key if old_key is not None and str(old_key) == snippet.name else _yaml11_safe(snippet.name)
        out[key] = _value_node(snippet.value, old_map[old_key] if old_key is not None else None)
        if old_key is not None:
            _carry_item_comment(old_map, old_key, out, key)
    return out


def _value_from_node(node: Any) -> PromptValue:
    if node is None:
        return PromptValue(kind="text", text="")
    if isinstance(node, str):
        return PromptValue(kind="text", text=str(node))
    if isinstance(node, dict) and all(isinstance(key, (str, int)) and not isinstance(key, bool) for key in node) and all(isinstance(value, str) for value in node.values()):
        return PromptValue(kind="fields", fields=tuple((str(key), str(value)) for key, value in node.items()))
    return PromptValue(kind="structured", data=_plain_data(node))


def _enhance_from_node(node: Any) -> Any:
    if node is None or node is False:
        return None
    return _plain_data(node)


def _value_node(value: PromptValue, old: Any) -> Any:
    """Return the node to write for *value*: the original *old* node when it already holds the same value."""
    if old is not None and (value.kind == "structured" or _plain_data(old) == value.to_python()):
        return old
    if value.kind == "text":
        return _text_node(value.text)
    if value.kind == "fields":
        mapping = CommentedMap()
        for key, text in value.fields:
            mapping[_yaml11_safe(key)] = _text_node(text)
        if isinstance(old, CommentedMap):
            for key in mapping:
                _carry_item_comment(old, key, mapping, key)
        return mapping
    return value.data


def _text_node(text: str) -> Any:
    if "\n" in text.rstrip("\n"):
        return LiteralScalarString(text)
    return _yaml11_safe(text.rstrip("\n"))


def _yaml11_safe(text: str) -> str:
    """Quote *text* when PyYAML would read it unquoted as something else (``off`` as a bool, ``1.10`` as a float).

    ruamel writes YAML 1.2, where ``yes``/``no``/``on``/``off`` are plain strings; the CLI reads YAML 1.1.
    """
    if _PYYAML_RESOLVER.resolve(pyyaml.ScalarNode, text, (True, False)) != _PYYAML_STR_TAG:
        return DoubleQuotedScalarString(text)
    return text


def _apply_entry(node: CommentedMap, entry: DocumentEntry) -> None:
    """Write *entry*'s keys onto *node*, leaving unchanged keys (and their comments) as they are."""
    _set_value(node, "prompt", entry.prompt)
    if entry.negative is None:
        if "negative" in node and node["negative"] is not None:
            del node["negative"]
    else:
        _set_value(node, "negative", entry.negative)

    if "active" in node:
        if is_active(node["active"]) != entry.active:
            node["active"] = entry.active
    elif not entry.active:
        _put(node, "active", False)

    current = node.get("enhance")
    if entry.enhance is None:
        if "enhance" in node and _enhance_from_node(current) is not None:
            del node["enhance"]
    elif _plain_data(current) != entry.enhance:
        wanted = minimal_enhance(entry.enhance) if isinstance(entry.enhance, dict) else entry.enhance
        _put(node, "enhance", _enhance_node(wanted))


def _set_value(node: CommentedMap, key: str, value: PromptValue) -> None:
    _put(node, key, _value_node(value, node.get(key)))


def _put(node: CommentedMap, key: str, value: Any) -> None:
    """Set *key*, inserting a new key in :data:`ENTRY_KEY_ORDER` position.

    A blank line or comment after the entry's old last key is moved below a key appended after it.
    """
    if key in node:
        node[key] = value
        return
    last = next(reversed(node), None)
    position = _insert_position(node, key)
    node.insert(position, key, value)
    if last is not None and position == len(node) - 1:
        _move_trailing_comment(node, last, key)


def _move_trailing_comment(node: CommentedMap, from_key: Any, to_key: Any) -> None:
    """Move the comment after *from_key*'s value to after the last line of *to_key*'s value."""
    slot = node.ca.items.get(from_key)
    if not slot or len(slot) < 3 or slot[2] is None:
        return
    token, slot[2] = slot[2], None
    # A block mapping value ends with its own last key; descend until the last value is a scalar or flow list.
    owner, key = node, to_key
    while isinstance(owner[key], CommentedMap) and owner[key]:
        owner, key = owner[key], next(reversed(owner[key]))
    owner.ca.items[key] = [None, None, token, None]


def _insert_position(node: CommentedMap, key: str) -> int:
    """Place a new key after the keys that come before it in :data:`ENTRY_KEY_ORDER`."""
    earlier = ENTRY_KEY_ORDER[: ENTRY_KEY_ORDER.index(key)]
    positions = [index for index, existing in enumerate(node) if existing in earlier]
    return positions[-1] + 1 if positions else 0


def _enhance_node(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    mapping = CommentedMap()
    for key, item in value.items():
        if isinstance(item, list):
            seq = CommentedSeq(item)
            seq.fa.set_flow_style()
            mapping[key] = seq
        else:
            mapping[key] = item
    return mapping
