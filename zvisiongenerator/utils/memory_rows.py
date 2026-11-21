"""Judge one job at every choice of the setting that moves its placement: the rows of the Web UI's popover and of the CLI warnings.

An image job has a row for each Quant level, an LTX clip one for Low memory on and one for off; everything else (a
stored copy, MiniMax H3, audio) has a single row. Each row is the same job with only that setting changed, judged by
:func:`~zvisiongenerator.utils.memory_verdict.judge`. A row's key is what the Web UI sends and shows: ``"none"``, ``"8"``
and ``"4"`` for Quant, ``"on"`` and ``"off"`` for Low memory, ``"only"`` for a single row.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict, MemoryWeights, judge
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.utils.model_metadata import model_capabilities

__all__ = ["NONE_KEY", "ONLY_KEY", "judge_rows", "level_key", "low_memory_key", "low_memory_rows", "quantize_choices", "quantize_rows", "supports_quantize"]

NONE_KEY, ONLY_KEY = "none", "only"
_LOW_MEMORY_SETTINGS = (True, False)

type JobRows = list[tuple[str, MemoryJob]]
type VerdictRows = list[tuple[str, MemoryVerdict]]


def level_key(level: int | None) -> str:
    """Return the row key of a Quant level: ``"none"`` for unquantized, else the bits."""
    return NONE_KEY if level is None else str(level)


def low_memory_key(low_memory: bool) -> str:
    """Return the row key of a Low memory setting: ``"on"`` or ``"off"``."""
    return "on" if low_memory else "off"


def supports_quantize(family: str | None) -> bool:
    """Return whether the image family can be quantized; a family the metadata does not know can."""
    return family is None or bool(model_capabilities(family)["supports_quantize"])


def quantize_choices(stored_bits: int | None, family: str | None, levels: Iterable[int]) -> tuple[int | None, ...]:
    """Return the Quant levels an image job is judged at: its stored level alone, else unquantized then *levels* from the highest down.

    A family that cannot be quantized has only the unquantized level.
    """
    if stored_bits is not None:
        return (stored_bits,)
    offered = tuple(sorted(set(levels), reverse=True)) if supports_quantize(family) else ()
    return (None, *offered)


def quantize_rows(job: MemoryJob, choices: Iterable[int | None]) -> JobRows:
    """Return *job* once for each Quant level in *choices*, keyed by :func:`level_key`."""
    return [(level_key(level), replace(job, quantize=level)) for level in choices]


def low_memory_rows(job: MemoryJob) -> JobRows:
    """Return *job* with Low memory on, then off."""
    return [(low_memory_key(value), replace(job, low_memory=value)) for value in _LOW_MEMORY_SETTINGS]


def judge_rows(rows: JobRows, budget: MemoryBudget, weights: MemoryWeights) -> VerdictRows | None:
    """Judge each row's job once, leaving out a row the engine cannot judge (a level it has no estimate for, such as q6).

    Returns ``None`` when it can judge none of them (missing weights, a canvas it does not take).
    """
    judged: VerdictRows = []
    for key, job in rows:
        try:
            judged.append((key, judge(job, budget, weights)))
        except ValueError:
            continue
    return judged or None
