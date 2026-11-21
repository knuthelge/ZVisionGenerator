"""Word a memory verdict: the sentence a status reads as, the heading of what was judged, the figures behind it, the other choices of the job, and the warning a command line prints.

Everything here is a pure function of a :class:`~zvisiongenerator.utils.memory_verdict.MemoryVerdict` and the
:class:`~zvisiongenerator.utils.model_memory.MemoryBudget` it was judged against. The command lines, the audio out-of-memory message and the Web UI's
memory answer (:mod:`zvisiongenerator.web.memory_api`) all use these functions, so every word a verdict is read in comes from here; the Web UI's
client lays the words out and holds none of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from zvisiongenerator.utils.memory_grade import FITS, TIGHT, TOO_LARGE
from zvisiongenerator.utils.memory_rows import NONE_KEY
from zvisiongenerator.utils.memory_verdict import MemoryVerdict
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

__all__ = [
    "control_title",
    "judged_details",
    "judged_heading",
    "judged_title",
    "level_label",
    "memory_figures",
    "memory_rows_line",
    "memory_sentence",
    "memory_warning",
    "status_word",
    "table_heading",
]

_FITS_SENTENCE = "Fits this computer."
_TIGHT_SENTENCES = {
    "gpu": "Fits, but close to the GPU's limit. It may run out of memory.",
    "ram": "Fits, but close to the system memory limit. It may run slowly.",
    "unified": "Fits, but macOS will compress or swap memory to run it. It may run slowly.",
}
_TOO_LARGE_SENTENCES = {
    "gpu": "Needs more GPU memory than this card has.",
    "ram": "Needs more RAM than this computer has.",
    "unified": "Needs more memory than this Mac has.",
}
_STATUS_WORDS = {FITS: "Fits", TIGHT: "Tight", TOO_LARGE: "Too large"}


def memory_sentence(status: str, resource: str) -> str:
    """Return the sentence that says what a *status* decided by *resource* (``"gpu"``, ``"ram"`` or ``"unified"``) means for this computer."""
    if status == FITS:
        return _FITS_SENTENCE
    if status == TIGHT:
        return _TIGHT_SENTENCES[resource]
    return _TOO_LARGE_SENTENCES[resource]


def status_word(status: str) -> str:
    """Return the word a status is shown as: ``Fits``, ``Tight`` or ``Too large``."""
    return _STATUS_WORDS[status]


def level_label(control: str | None, key: str) -> str:
    """Return the name of a row's level as its control shows it: ``None``, ``q8`` or ``q4`` for ``"quantize"``, ``On`` or ``Off`` for ``"low_memory"``, else empty."""
    if control == "quantize":
        return "None" if key == NONE_KEY else f"q{key}"
    if control == "low_memory":
        return "On" if key == "on" else "Off"
    return ""


def control_title(control: str | None) -> str | None:
    """Return the title of the level column of a job's rows: ``Quant`` or ``Low memory``; ``None`` for a job with one row."""
    return {"quantize": "Quant", "low_memory": "Low memory"}.get(control or "")


def table_heading(control: str | None, frame: str) -> str | None:
    """Return the small heading over a job's table of levels, or ``None`` when the job has one row.

    The table lists every level, the current one included: ``Quants for this image``, ``Low memory on or off for this clip``; a reference job (a listed model) reads ``at this size``.
    """
    tail = "at this size" if frame == "reference" else f"for this {frame}"
    if control == "quantize":
        return f"Quants {tail}"
    return f"Low memory on or off {tail}" if control == "low_memory" else None


def judged_title(frame: str, *, width: int | None = None, height: int | None = None, seconds: float | None = None) -> str:
    """Return what a memory answer is about: ``Memory for this image``, or ``Memory at 5 s · 768×448`` for the reference job of a listed model, which names its size.

    A reference job with no size (an audio model's measured peak) is about a track.
    """
    if frame == "reference":
        size = _size(width, height, seconds)
        return f"Memory at {size}" if size else "Memory for this track"
    return f"Memory for this {frame}"


def judged_details(
    frame: str,
    *,
    width: int | None = None,
    height: int | None = None,
    seconds: float | None = None,
    quantize: int | None = None,
    low_memory: bool | None = None,
    references: int = 0,
    loras: int = 0,
    label: str | None = None,
) -> str:
    """Return what was judged besides the title, in one order everywhere: length and size, the Quant level, Low memory, the reference and LoRA counts, then *label*.

    ``1152×1728 · q4 · 3 references · 2 LoRAs``. A reference job's size is in its title, so the details leave it out; an unquantized job names no level. *label* names what a
    job has no setting for, such as the audio model.
    """
    parts = [
        "" if frame == "reference" else _size(width, height, seconds),
        "" if quantize is None else f"q{quantize}",
        "" if low_memory is None else f"Low memory {'on' if low_memory else 'off'}",
        _count(references, "reference"),
        _count(loras, "LoRA"),
        label or "",
    ]
    return " · ".join(part for part in parts if part)


def judged_heading(
    frame: str,
    *,
    width: int | None = None,
    height: int | None = None,
    seconds: float | None = None,
    quantize: int | None = None,
    low_memory: bool | None = None,
    references: int = 0,
    loras: int = 0,
    label: str | None = None,
) -> str:
    """Return the heading of a command's memory warning: the title with the details in brackets, ``Memory for this image (1152×1728 · q4)``.

    The arguments are those of :func:`judged_title` and :func:`judged_details`.
    """
    title = judged_title(frame, width=width, height=height, seconds=seconds)
    details = judged_details(frame, width=width, height=height, seconds=seconds, quantize=quantize, low_memory=low_memory, references=references, loras=loras, label=label)
    return f"{title} ({details})" if details else title


def _size(width: int | None, height: int | None, seconds: float | None) -> str:
    """Return a job's length and size, such as ``5 s · 768×448`` or ``1152×1728``; empty when it has none."""
    length = "" if seconds is None else f"{round(seconds, 1):g} s"
    size = "" if width is None or height is None else f"{width}×{height}"
    return " · ".join(part for part in (length, size) if part)


def _count(number: int, noun: str) -> str:
    return "" if number <= 0 else f"{number} {noun}{'' if number == 1 else 's'}"


def memory_figures(verdict: MemoryVerdict, machine: MemoryBudget) -> tuple[float, float] | None:
    """Return what *verdict*'s deciding resource needs and what *machine* has of it, in GB to one decimal; ``None`` when the verdict holds no figure for it."""
    if verdict.resource == "ram":
        if verdict.ram_bytes is None or machine.system_bytes is None:
            return None
        need, have = verdict.ram_bytes, machine.system_bytes
    elif verdict.resource == "unified":
        if machine.unified_bytes is None:
            return None
        need, have = verdict.gpu_bytes, machine.unified_bytes
    else:
        need, have = verdict.gpu_bytes, machine.gpu_bytes
    return round(need / GIB, 1), round(have / GIB, 1)


def memory_rows_line(control: str, rows: Sequence[tuple[str, MemoryVerdict]], current: str, machine: MemoryBudget) -> str | None:
    """Return one line naming the status and need of each row of a job but the *current* one, or ``None`` when there is no other row.

    Args:
        control: What the rows vary: ``"quantize"`` (``Other quants: q8 Fits (8.1 GB) · q4 Fits (6 GB)``) or ``"low_memory"`` (``Low memory on: Fits (8.1 GB)``).
        rows: The job's ``(key, verdict)`` rows, as :func:`~zvisiongenerator.utils.memory_rows.judge_rows` returns them.
        current: The key of the row the warning is about, which the line leaves out.
        machine: The memory the rows were judged against; each need is the figure of the resource that decided its row.
    """
    others = [(key, verdict) for key, verdict in rows if key != current]
    if not others:
        return None
    if control == "quantize":
        return "Other quants: " + " · ".join(f"{level_label('quantize', key)} {_row_status(verdict, machine)}" for key, verdict in others)
    return " · ".join(f"Low memory {key}: {_row_status(verdict, machine)}" for key, verdict in others)


def memory_warning(heading: str, verdict: MemoryVerdict, machine: MemoryBudget, *, other_choices: str | None = None) -> str | None:
    """Return the warning a command prints for a job that is Tight or Too large, or ``None`` when it fits.

    Args:
        heading: What was judged, for example ``"Memory for this image (1152×1728 · q4)"``.
        verdict: The job's verdict.
        machine: The memory the verdict was judged against.
        other_choices: The job's other choices (:func:`memory_rows_line`), added after the sentence when there are any.
    """
    if verdict.status == FITS:
        return None
    figures = memory_figures(verdict, machine)
    detail = "" if figures is None else f" (about {figures[0]:g} of {figures[1]:g} GB)"
    warning = f"{heading}: {memory_sentence(verdict.status, verdict.resource).rstrip('.')}{detail}. Continuing."
    return warning if other_choices is None else f"{warning} {other_choices}"


def _row_status(verdict: MemoryVerdict, machine: MemoryBudget) -> str:
    """Return a row's status and the need of its deciding resource: ``Fits (8.1 GB)``."""
    figures = memory_figures(verdict, machine)
    return status_word(verdict.status) if figures is None else f"{status_word(verdict.status)} ({figures[0]:g} GB)"
