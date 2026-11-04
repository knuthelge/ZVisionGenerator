"""The live progress line of ``ziv audio``: one line rewritten in place on a terminal, a line per phase anywhere else."""

from __future__ import annotations

from zvisiongenerator.core.audio_backend import DECODE_PHASE
from zvisiongenerator.downloads.cli_report import eta_text
from zvisiongenerator.utils.live_line import LiveLine, shared_line

__all__ = ["ConsoleProgress", "progress_text"]

_FINISHING = "Finishing"


def _part_label(phase: str, part: int, n_parts: int) -> str:
    """Return the name of what is being made, such as ``Speech tokens (part 2 of 3)``; the decode phase reads ``Finishing``."""
    name = _FINISHING if phase == DECODE_PHASE else phase
    return f"{name} (part {part} of {n_parts})" if n_parts > 1 else name


def progress_text(phase: str, step: int, total: int | None, part: int, n_parts: int, eta_secs: float | None) -> str:
    """Return the line for one progress report, such as ``  Synthesis 12/64 · about 2 min left in the batch`` or ``  Song tokens 1,840``.

    Args:
        phase: What the model is doing (``Synthesis``, ``Song tokens``). The decode phase has no count and reads ``Finishing...``.
        step: The units done in the phase.
        total: How many there will be, or ``None`` when the phase ends on its own.
        part: The part of a multi-part speech take being made.
        n_parts: How many parts the take has.
        eta_secs: The time left in the whole batch (the takes still to make, this one included), or ``None`` before a take has finished.
    """
    label = _part_label(phase, part, n_parts)
    if phase == DECODE_PHASE:
        progress = f"{label}..."
    elif total:
        progress = f"{label} {step}/{total}"
    else:
        progress = f"{label} {step:,}" if step else f"{label}..."
    return f"  {progress}" if eta_secs is None else f"  {progress} · {eta_text(eta_secs)} in the batch"


class ConsoleProgress:
    """Show the progress of one take on the console.

    On a terminal one line is rewritten in place and a new phase (or part) starts a new line. Anywhere else (a pipe, a log
    file) one line is printed when the phase or part changes, so a log is not flooded. Call :meth:`close` when the take
    ends, so the cursor is not left on the line.

    The total shown for a phase (or part) is the first one reported, so it stays put while the count runs. A model that
    cannot know its length in advance reports a moving estimate; once the count passes the first total, only the count is shown.

    Args:
        line: The console line to write through; the process console's when ``None``.
    """

    def __init__(self, line: LiveLine | None = None) -> None:
        self._line = line or shared_line()
        self._marker: tuple[str, int] | None = None
        self._total: int | None = None  # the first total reported for the phase and part in progress

    def __call__(self, phase: str, step: int, total: int | None, part: int, n_parts: int, eta_secs: float | None) -> None:
        """Show one progress report."""
        changed = self._marker != (phase, part)
        self._marker = (phase, part)
        if changed:
            self._total = None
        if self._total is None:
            self._total = total
        total = self._total if self._total and step <= self._total else None
        if not self._line.is_terminal():
            if changed:
                self._line.print_line(f"  {_part_label(phase, part, n_parts)}...")
            return
        if changed:
            self._line.end()
        self._line.rewrite(progress_text(phase, step, total, part, n_parts, eta_secs))

    def close(self) -> None:
        """End the live line, if one is open."""
        self._line.end()
