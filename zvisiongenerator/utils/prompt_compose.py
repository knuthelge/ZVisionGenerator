"""Compose prompts from YAML snippets and expand random choices."""

from __future__ import annotations

import random
import re
from typing import Any

_SNIPPET_REF_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_STANDALONE_REF_RE = re.compile(r"^\s*\$([a-zA-Z_][a-zA-Z0-9_]*)\s*$")
SEPARATOR = ". "


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = flatten_value(v)
            parts.append(f"{k}: {flat}" if flat else str(k))
        return SEPARATOR.join(parts)
    if isinstance(value, list):
        parts = [flatten_value(item) for item in value]
        return SEPARATOR.join(p for p in parts if p)
    return str(value)


def resolve_snippets(
    value: Any,
    snippets: dict[str, Any],
    _seen: frozenset[str] = frozenset(),
) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        standalone = _STANDALONE_REF_RE.match(value)
        if standalone:
            name = standalone.group(1)
            if name not in snippets:
                raise ValueError(f"Undefined snippet: '{name}'")
            if name in _seen:
                raise ValueError(f"Circular snippet reference: '{name}' already seen in {_seen}")
            return resolve_snippets(snippets[name], snippets, _seen | {name})

        def _replace_inline(match: re.Match) -> str:
            name = match.group(1)
            if name not in snippets:
                raise ValueError(f"Undefined snippet: '{name}'")
            if name in _seen:
                raise ValueError(f"Circular snippet reference: '{name}' already seen in {_seen}")
            resolved = resolve_snippets(snippets[name], snippets, _seen | {name})
            return flatten_value(resolved)

        return _SNIPPET_REF_RE.sub(_replace_inline, value)

    if isinstance(value, dict):
        return {k: resolve_snippets(v, snippets, _seen) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_snippets(item, snippets, _seen) for item in value]

    return value


_INNERMOST_RE = re.compile(r"\{([^{}]+)\}")


def expand_random_choices(text: str) -> str:
    """Expand ``{a|b|c}`` random choice blocks in *text*, supporting nesting.

    Innermost blocks are resolved first; the process repeats until no more
    blocks remain.

    Args:
        text: Input text containing zero or more ``{a|b|c}`` blocks.

    Returns:
        Text with all choice blocks resolved and whitespace stripped.
    """
    rng = random.Random()
    while _INNERMOST_RE.search(text):
        text = _INNERMOST_RE.sub(
            lambda m: rng.choice(m.group(1).split("|")),
            text,
        )
    return text.strip()
