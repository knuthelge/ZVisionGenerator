"""A text stream that says whether it is a terminal, for the tests of the console's live line."""

from __future__ import annotations

import io

__all__ = ["FakeStream"]


class FakeStream(io.StringIO):
    """A ``StringIO`` that reports *terminal* from ``isatty``."""

    def __init__(self, *, terminal: bool) -> None:
        super().__init__()
        self._terminal = terminal

    def isatty(self) -> bool:
        return self._terminal
