"""The environment for Python child processes whose output is piped or discarded."""

from __future__ import annotations

import os

__all__ = ["utf8_child_env"]


def utf8_child_env(**extra: str) -> dict[str, str]:
    """Return this process's environment with the child's standard streams in UTF-8, plus *extra*.

    A child whose output goes to a pipe or nowhere encodes it with the locale's code page (cp1252 on Windows),
    where printing a non-ASCII name raises. UTF-8 output can encode anything.
    """
    return {**os.environ, **extra, "PYTHONIOENCODING": "utf-8"}
