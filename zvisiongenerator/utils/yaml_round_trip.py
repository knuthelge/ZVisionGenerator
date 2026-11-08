"""Load, edit and dump YAML with ruamel.yaml's round-trip mode, keeping a file's comments and layout.

The prompt builder (:mod:`zvisiongenerator.utils.prompt_document`) and the Web UI's ``config.yaml`` writer share
these helpers. What a file *means* is still decided by PyYAML (YAML 1.1), which the CLI reads with; ruamel only
carries the text through an edit, so strings PyYAML would read as another type are quoted (:func:`yaml11_safe`).

In a loaded tree, the comment after a value is one token holding the end-of-line comment and every comment or
blank line below it, up to the next key. A mapping or list value keeps the token on its *last* descendant. The
helpers split such a token into the lines that belong to the item and the lines that follow it, so an edit moves
or drops only the right ones.
"""

from __future__ import annotations

from typing import Any

from ruamel.yaml import YAML
import yaml as pyyaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark, YAMLError
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from ruamel.yaml.tokens import CommentToken

__all__ = [
    "attach_following",
    "delete_key",
    "load_round_trip",
    "move_trailing_comment",
    "round_trip_yaml",
    "set_key",
    "split_comment_lines",
    "yaml11_safe",
]

# The CLI reads YAML with PyYAML (YAML 1.1); its resolver says what a plain scalar would become.
_PYYAML_RESOLVER = pyyaml.resolver.Resolver()
_PYYAML_STR_TAG = "tag:yaml.org,2002:str"
# A nested mapping is indented by this many columns when a new key has no position in a loaded file.
_INDENT = 2
# The comment slot of a mapping key is the third entry; a list item has it first.
_MAP_SLOT = 2
_SEQ_SLOT = 0


def round_trip_yaml(*, capital_bools: bool = False) -> YAML:
    """Return a round-trip YAML instance that keeps quotes and comments and does not wrap long lines.

    Args:
        capital_bools: Write booleans as ``True`` / ``False``, for files written that way.
    """
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


def load_round_trip(text: str) -> Any:
    """Load *text* into a round-trip tree.

    Raises:
        ValueError: When the text is not valid YAML (a repeated mapping key counts as invalid).
    """
    try:
        return round_trip_yaml().load(text)
    except YAMLError as exc:
        raise ValueError(f"The file is not valid YAML: {exc}") from exc


def yaml11_safe(text: str) -> str:
    """Quote *text* when PyYAML would read it unquoted as something else (``off`` as a bool, ``1.10`` as a float).

    ruamel writes YAML 1.2, where ``yes``/``no``/``on``/``off`` are plain strings; the CLI reads YAML 1.1.
    """
    if _PYYAML_RESOLVER.resolve(pyyaml.ScalarNode, text, (True, False)) != _PYYAML_STR_TAG:
        return DoubleQuotedScalarString(text)
    return text


def split_comment_lines(value: str, first_column: int, key_column: int = 0) -> tuple[str | None, str | None]:
    """Split a trailing comment token's value into the item's own lines and the lines that follow it.

    Lines indented deeper than *key_column* belong to the item. Blank lines and lines at or above *key_column*
    follow it (a heading for the next key, a commented-out sibling). The value's first line is the end-of-line
    comment and starts at *first_column*; it is empty when the value begins with a newline. The default
    *key_column* of 0 treats every column-0 comment and blank line as following.

    Args:
        value: The token's text, as ruamel stores it (``"# note\\n  # more\\n\\n# Heading\\n"``).
        first_column: The column where the value's first line starts.
        key_column: The column of the key the comment trails.

    Returns:
        The own lines as a token value (a leading newline when there is no end-of-line comment), or None when
        there are none; and the following lines, each ending in a newline (a blank line is ``"\\n"``), or None.
    """
    lines = value.split("\n")
    if lines[-1] == "":
        lines.pop()
    first, rest = (lines[0], lines[1:]) if lines else ("", [])
    own_first = first if first.strip() and first_column > key_column else ""
    following = [first] if first.strip() and not own_first else []
    own_rest: list[str] = []
    for line in rest:
        if line.strip() and len(line) - len(line.lstrip()) > key_column:
            own_rest.append(line)
        else:
            following.append(line)
    own = (own_first + "\n" + "".join(line + "\n" for line in own_rest)) if own_first or own_rest else None
    return own, "".join(line + "\n" for line in following) or None


def move_trailing_comment(node: CommentedMap, from_key: Any, to_key: Any) -> None:
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


def set_key(node: CommentedMap, key: str, value: Any) -> None:
    """Set *key* on *node*, keeping its comments; a new key goes last, below the comments that followed the old last key.

    An existing key keeps its position and its comments. A new key is appended; the lines after the old last
    key's value (a blank line, the next section's heading, commented-out keys at the end) move below the new
    key, and the old last key keeps its own end-of-line comment.

    Args:
        node: The mapping to edit.
        key: The key to assign.
        value: The new value; a new :class:`CommentedMap` is written in block style.
    """
    if key in node:
        node[key] = value
        _register_children(node, key, _key_column(node, key))
        return
    column = _sibling_column(node)
    following = None
    last = next(reversed(node), None)
    if last is None:
        node.fa.set_block_style()
    else:
        following = _detach_following(node, last)
    node[key] = value
    node.lc.add_kv_line_col(key, [0, column, 0, column])
    _register_children(node, key, column)
    if following:
        attach_following(node, key, following)


def delete_key(node: CommentedMap, key: Any) -> str | None:
    """Remove *key* from *node* with its own comments, and return the lines that followed it.

    The key's own comments are its end-of-line comment and the lines indented deeper than its column. The
    returned lines (see :func:`split_comment_lines`) stay in the file: pass them to :func:`attach_following`.
    """
    leaf_owner, leaf_key = _trailing_owner(node, key)
    token = _slot_token(leaf_owner, leaf_key)
    following = split_comment_lines(token.value, token.column, _key_column(node, key))[1] if token is not None else None
    del node[key]
    node.ca.items.pop(key, None)
    return following


def attach_following(node: CommentedMap | CommentedSeq, key: Any, lines: str, *, at_key: bool = False) -> None:
    """Add *lines* (as :func:`delete_key` returns them) below *key*'s value in *node*.

    Args:
        node: The container that holds *key*.
        key: The key (or list index) the lines go below.
        lines: Lines each ending in a newline.
        at_key: Put the lines directly after ``key:`` instead of after its last nested value, for lines that
            lead a mapping whose first key was removed.
    """
    owner, leaf = (node, key) if at_key else _trailing_owner(node, key)
    index = _MAP_SLOT if isinstance(owner, CommentedMap) else _SEQ_SLOT
    slot = owner.ca.items.setdefault(leaf, [None, None, None, None])
    slot.extend([None] * (index + 1 - len(slot)))
    if slot[index] is None:
        slot[index] = CommentToken("\n" + lines, CommentMark(0))
    else:
        slot[index].value += lines


def _is_block(value: Any) -> bool:
    """Return whether *value* is a non-empty block mapping or list, whose trailing comment sits on its last item."""
    return isinstance(value, (CommentedMap, CommentedSeq)) and len(value) > 0 and value.fa.flow_style() is not True


def _trailing_owner(node: CommentedMap | CommentedSeq, key: Any) -> tuple[CommentedMap | CommentedSeq, Any]:
    """Return the container and key that hold the comment after *key*'s whole value (its last descendant)."""
    owner, leaf = node, key
    while _is_block(owner[leaf]):
        owner = owner[leaf]
        leaf = next(reversed(owner)) if isinstance(owner, CommentedMap) else len(owner) - 1
    return owner, leaf


def _slot_token(owner: CommentedMap | CommentedSeq, key: Any) -> Any:
    """Return the trailing comment token stored for *key* in *owner*, or None."""
    index = _MAP_SLOT if isinstance(owner, CommentedMap) else _SEQ_SLOT
    slot = owner.ca.items.get(key)
    return slot[index] if slot and len(slot) > index else None


def _detach_following(node: CommentedMap, key: Any) -> str | None:
    """Cut the lines after *key*'s value that are not its own from its trailing comment, and return them."""
    owner, leaf = _trailing_owner(node, key)
    token = _slot_token(owner, leaf)
    if token is None:
        return None
    own, following = split_comment_lines(token.value, token.column, _key_column(node, key))
    index = _MAP_SLOT if isinstance(owner, CommentedMap) else _SEQ_SLOT
    if own is None:
        owner.ca.items[leaf][index] = None
    else:
        token.value = own
    return following


def _key_column(node: CommentedMap, key: Any) -> int:
    """Return the column where *key* starts in the loaded file (or where it was registered when added)."""
    position = node.lc.key(key)
    return position[1] if position else 0


def _sibling_column(node: CommentedMap) -> int:
    """Return the column the keys of *node* start at, taken from the first one with a known position."""
    for existing in node:
        position = node.lc.key(existing)
        if position:
            return position[1]
    return 0


def _register_children(node: CommentedMap, key: Any, column: int) -> None:
    """Record the columns of the keys in a mapping added under *key* (at *column*) in this run."""
    child = node[key]
    if isinstance(child, CommentedMap):
        for name in child:
            if not child.lc.key(name):
                child.lc.add_kv_line_col(name, [0, column + _INDENT, 0, column + _INDENT])
            _register_children(child, name, _key_column(child, name))
