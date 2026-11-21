"""Read this machine's live memory and shape the readings the Web UI shows beside a running job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zvisiongenerator.backends import accelerator_memory_budget_or_none, memory_usage
from zvisiongenerator.core.memory_sampler import MemoryUsage
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

__all__ = ["MemoryProbe", "machine_probe", "reading_payload"]


@dataclass(frozen=True)
class MemoryProbe:
    """The calls the runner uses to watch a job's memory.

    Attributes:
        read: Return the current usage, or ``None`` when it cannot be read.
        budget: Return the machine's memory budget, or ``None`` when unknown; readings are still sampled, and each is published when its own totals are known.
    """

    read: Callable[[], MemoryUsage | None]
    budget: Callable[[], MemoryBudget | None]


def machine_probe() -> MemoryProbe:
    """Return the probe that reads this machine's memory load and budget: the whole machine's memory on a Mac, the whole machine's RAM and the memory of the card CUDA uses elsewhere."""
    return MemoryProbe(read=memory_usage, budget=accelerator_memory_budget_or_none)


def reading_payload(usage: MemoryUsage, budget: MemoryBudget | None) -> dict[str, Any] | None:
    """Return the ``memory`` payload of a ``memory_usage`` event: the usage in GB over the totals it is measured against, or ``None`` when a total is unknown.

    The figures are plain: the payload carries no status, and they are the machine's load, not this app's share. On unified memory (no
    separate GPU) the whole machine's memory in use is shown over the machine's total unified memory (``kind`` ``unified``: ``used_gb``,
    ``budget_gb``); that needs *budget*. On a discrete GPU, the card's memory in use and the machine's RAM in use are reported apart
    (``kind`` ``discrete``: ``gpu_gb``, ``gpu_budget_gb``, ``system_gb``, ``system_budget_gb``), each over the total the reading carries,
    else over *budget*'s. ``gpu_gb`` is ``None`` while the GPU figure is not known yet (this process has not started CUDA).
    """
    if usage.gpu_used_bytes is None and not usage.gpu_unknown:
        if budget is None:
            return None
        total = budget.unified_bytes or budget.gpu_bytes  # a Mac verdict is graded against the total memory
        return {"kind": "unified", "used_gb": _gb(usage.system_used_bytes), "budget_gb": _gb(total)}
    gpu_total = usage.gpu_total_bytes or (budget.gpu_bytes if budget is not None else None)
    system_total = usage.system_total_bytes or (budget.system_bytes if budget is not None else None)
    if not gpu_total or not system_total:
        return None
    return {
        "kind": "discrete",
        "gpu_gb": None if usage.gpu_used_bytes is None else _gb(usage.gpu_used_bytes),
        "gpu_budget_gb": _gb(gpu_total),
        "system_gb": _gb(usage.system_used_bytes),
        "system_budget_gb": _gb(system_total),
    }


def _gb(value: int) -> float:
    """Return *value* bytes as GiB rounded to one decimal."""
    return round(value / GIB, 1)
