"""The memory verdict of an audio model's measured GPU peak, and what it needs of this machine in the words the out-of-memory message of a take uses.

The peak is judged by the memory verdict engine; the functions are pure: the caller passes the memory budget.
"""

from __future__ import annotations

import sys

from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict, MemoryWeights, judge
from zvisiongenerator.utils.memory_wording import memory_figures
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

__all__ = ["audio_memory_need", "audio_verdict"]


def audio_verdict(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None) -> MemoryVerdict | None:
    """Return the verdict of the preset's recorded GPU peak on *platform_key* against *budget*, or ``None`` without a budget or a measured peak.

    A Mac budget without the machine's total unified memory cannot be judged (:func:`judge` raises ``ValueError``), so it gives ``None`` as well.
    On Windows the preset's ``windows`` figure, when it has one, is read in place of its ``cuda`` one.
    """
    peak_gb = _peak_gb(preset, platform_key)
    if budget is None or not budget.gpu_bytes or peak_gb is None:
        return None
    try:
        return judge(MemoryJob(media="audio"), budget, MemoryWeights(audio_peak_bytes=round(peak_gb * GIB)))
    except ValueError:
        return None


def _peak_gb(preset: AudioPreset, platform_key: str) -> float | None:
    """Return the measured peak to judge on this machine: the ``windows`` figure on a Windows CUDA machine that has one, else *platform_key*'s."""
    if platform_key == "cuda" and sys.platform == "win32":
        windows_gb = preset.memory_gb_for("windows")
        if windows_gb is not None:
            return windows_gb
    return preset.memory_gb_for(platform_key)


def audio_memory_need(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None) -> str | None:
    """Return what the preset's recorded GPU peak on *platform_key* needs against what *budget* has, or ``None`` without a budget or a measured peak.

    The text reads like ``it needs about 8.2 GB and the GPU has 8 GB``; on a Mac, which has one memory, ``it needs about 8.2 GB and this Mac has 24 GB``
    (its total memory).
    """
    verdict = audio_verdict(preset, platform_key, budget)
    if verdict is None or budget is None:
        return None
    figures = memory_figures(verdict, budget)
    if figures is None:
        return None
    holder = "this Mac" if verdict.resource == "unified" else "the GPU"
    return f"it needs about {figures[0]:g} GB and {holder} has {figures[1]:g} GB"
