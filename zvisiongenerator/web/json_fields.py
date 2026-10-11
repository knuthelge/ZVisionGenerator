"""Typed reads of optional fields in a JSON request body."""

from __future__ import annotations

from typing import Any


def json_field(payload: dict[str, Any], key: str, expected: type | tuple[type, ...], noun: str, *, name: str | None = None, minimum: int | None = None) -> Any:
    """Return ``payload[key]`` when it is an *expected* type (a boolean is not a number) and at least *minimum*, else ``None`` when absent.

    Raises:
        ValueError: ``"<name> must be <noun>."`` when the value is present but not valid; *name* defaults to *key*.
    """
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool) or (minimum is not None and value < minimum):
        raise ValueError(f"{name or key} must be {noun}.")
    return value
