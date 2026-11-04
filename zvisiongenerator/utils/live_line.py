"""Own the console's live line: one line rewritten in place on a terminal, plain lines anywhere else."""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from typing import Any, TextIO

__all__ = ["LiveLine", "LiveLineStream", "route_stderr_through_live_line", "shared_line"]


class LiveLine:
    """Write a line that is rewritten in place, and end it before anything else is printed.

    Progress reporters and the key listener share one instance, so a message never lands on the row of an unfinished line and
    ending the line twice never leaves a blank one. On a stream that is not a terminal (a pipe, a log file) nothing is rewritten:
    :meth:`rewrite` does nothing there and callers print whole lines with :meth:`print_line`.

    Args:
        stream: Where the line is written; ``sys.stdout`` (looked up at each call) when ``None``.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._lock = threading.RLock()
        self._open_width: int | None = None  # the width of the live line while it is unfinished

    @property
    def stream(self) -> TextIO:
        """The stream the line is written to."""
        return self._stream or sys.stdout

    def is_terminal(self) -> bool:
        """Return whether the stream is a terminal, where a line can be rewritten."""
        return self.stream.isatty()

    def rewrite(self, text: str) -> None:
        """Replace the live line with *text*, blanking what is left of a longer one; do nothing off a terminal."""
        with self._lock:
            stream = self.stream
            if not stream.isatty():
                return
            padding = " " * max(0, (self._open_width or 0) - len(text))
            stream.write(f"\r{text}{padding}")
            stream.flush()
            self._open_width = len(text)

    def end(self) -> None:
        """End the live line so the next output starts on a fresh row; do nothing when none is open."""
        with self._lock:
            if self._open_width is None:
                return
            stream = self.stream
            stream.write("\n")
            stream.flush()
            self._open_width = None

    def print_line(self, text: str) -> None:
        """Print *text* as a whole line, after ending the live line."""
        with self._lock:
            self.end()
            stream = self.stream
            stream.write(f"{text}\n")
            stream.flush()

    def message(self, text: str, *, file: TextIO | None = None) -> None:
        """Print *text* as a whole line to *file* (``sys.stderr`` when ``None``), after ending the live line."""
        with self._lock:
            self.end()
            target = file or sys.stderr
            target.write(f"{text}\n")
            target.flush()

    def write_through(self, text: str, file: TextIO) -> None:
        """Write *text* to *file* as it is, after ending the live line; the caller supplies any newline."""
        with self._lock:
            self.end()
            file.write(text)


_SHARED = LiveLine()


def shared_line() -> LiveLine:
    """Return the live line of the process console, which every reporter and the key listener write through."""
    return _SHARED


class LiveLineStream:
    """Wrap a stream so that whatever is written to it ends the live line first.

    Library warnings, log records and ``print`` calls to an error stream land on the row of an unfinished progress line
    otherwise. Attributes other than ``write`` (``isatty``, ``fileno``, ``encoding``) come from the wrapped stream.

    Args:
        stream: The stream to wrap, normally ``sys.stderr``.
        line: The live line to end before each write.
        mute: Text that marks a write to drop: a write that contains any of these is swallowed whole.
    """

    def __init__(self, stream: TextIO, line: LiveLine, *, mute: Sequence[str] = ()) -> None:
        self._stream = stream
        self._line = line
        self._mute = tuple(mute)

    def write(self, text: str) -> int:
        """Write *text* after ending the live line, unless it holds muted text; return its length."""
        if text and not any(muted in text for muted in self._mute):
            self._line.write_through(text, self._stream)
        return len(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def route_stderr_through_live_line(*, line: LiveLine | None = None, mute: Sequence[str] = ()) -> None:
    """Make ``sys.stderr`` end the live line before it writes; do nothing when it already does.

    Call it before the logging handlers are made, so they write through the wrapper.

    Args:
        line: The live line to end; the process console's when ``None``.
        mute: Text that marks a write to drop (see :class:`LiveLineStream`).
    """
    if isinstance(sys.stderr, LiveLineStream):
        return
    sys.stderr = LiveLineStream(sys.stderr, line or shared_line(), mute=mute)  # type: ignore[assignment]
