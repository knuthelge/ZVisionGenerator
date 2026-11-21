"""The machine a case's estimate is judged against, and the estimate row the three media record: the setup the image, video and audio estimates share."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from bench.memory.measure import GIB, platform_label

if TYPE_CHECKING:
    from zvisiongenerator.utils.memory_verdict import MemoryVerdict
    from zvisiongenerator.utils.model_memory import MemoryBudget


@dataclass(frozen=True)
class CaseMachine:
    """The memory a case is judged against.

    ``foreign_mac`` is set for a Mac row judged on another machine (``fit --total-gib``): its memory is the given total and its weights come from the shipped table, because this
    machine's downloads are the other platform's files.
    """

    budget: MemoryBudget
    mac: bool
    foreign_mac: bool


def case_machine(case: Mapping[str, Any]) -> CaseMachine | None:
    """Return the machine *case* is judged against, or ``None`` when this machine has no budget or is not the kind of machine the case ran on.

    A case's ``budget_gib`` makes the card that many GiB (CUDA), and its ``total_gib`` makes a Mac's unified memory that many GiB; the other one means nothing on that kind of machine.
    """
    from zvisiongenerator.backends import accelerator_memory_budget_or_none
    from zvisiongenerator.utils.model_memory import MemoryBudget

    mac = (case.get("platform") or platform_label()) == "mac"
    foreign_mac = mac and bool(case.get("mac_unified_bytes"))
    if foreign_mac:
        # Only the total grades a Mac verdict; the working set is not used.
        machine = MemoryBudget(gpu_bytes=case["mac_unified_bytes"], unified_bytes=case["mac_unified_bytes"])
    else:
        machine = accelerator_memory_budget_or_none()
    if machine is None or mac != (machine.system_bytes is None):
        return None
    if mac and case.get("total_gib"):
        machine = replace(machine, unified_bytes=int(case["total_gib"] * GIB))
    elif not mac and case.get("budget_gib"):
        machine = replace(machine, gpu_bytes=int(case["budget_gib"] * GIB))
    return CaseMachine(machine, mac, foreign_mac)


def verdict_row(verdict: MemoryVerdict, machine: MemoryBudget) -> dict[str, Any]:
    """Return the estimate a row records: the verdict's figures and status and the budget it was judged against.

    ``gpu_bytes`` is the verdict's GPU figure, the allocated peak plus the CUDA offset (a Mac: the whole unified-memory need, fitted to the peak footprint), ``system_bytes`` the RAM
    figure (``None`` where there is none), ``budget_bytes`` the card or working set it was judged against, and ``status``, ``resource`` and ``placement`` say what the tag and the loader would do.
    """
    return {
        "gpu_bytes": verdict.gpu_bytes,
        "budget_bytes": machine.gpu_bytes,
        "system_bytes": verdict.ram_bytes,
        "status": verdict.status,
        "resource": verdict.resource,
        "placement": verdict.placement,
        "unified_bytes": machine.unified_bytes,
    }
