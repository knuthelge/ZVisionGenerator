"""Where a CUDA image model's weights sit at load, chosen from the memory the card has free.

The memory verdict names the placement the tag shows (:func:`~zvisiongenerator.utils.memory_verdict.cuda_image_placement`:
the first candidate that fits the card, else the one that needs the least). The loader also checks the memory free *now*, which
other programs and the driver already take from: a candidate is taken when its figure, the estimate plus the CUDA offset, is not
above that memory (not Too large against it: the Tight line of :func:`~zvisiongenerator.utils.memory_grade.grade`). The Fits line
is not applied again, because free memory already excludes other programs. When the tag's pick is above it, the loader steps down to
the next candidate and says so; when no candidate is below it, it takes the one that needs the least and warns that it may run out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS, TOO_LARGE, grade
from zvisiongenerator.utils.memory_verdict import (
    MemoryJob,
    capped_reference_megapixels,
    cuda_image_gpu_bytes,
    cuda_image_placement,
    image_placement_candidates,
)
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.model_memory import GIB, WeightTotals

__all__ = ["PlacementChoice", "choose_placement", "model_label"]

_FAMILY_LABELS = {"zimage": "Z-Image", "krea2": "Krea 2", "flux1": "FLUX.1", "flux2_klein": "FLUX.2 Klein"}


@dataclass(frozen=True)
class PlacementChoice:
    """Where the weights go, and what to tell the user about it.

    Attributes:
        placement: ``"resident"``, ``"turns"`` or ``"streamed"``.
        warning: The one line to log when the loader stepped down from what the tag assumed or has too little free memory for any candidate; ``None`` otherwise.
        note: The short sentence the running job's card shows for the same two cases, with the same free figure; ``None`` otherwise.
    """

    placement: str
    warning: str | None = None
    note: str | None = None


def model_label(family: str | None, model_size: str | None) -> str:
    """Return the name a placement warning calls the model: ``FLUX.2 Klein 4B``, or ``this model`` for a family with no name."""
    label = _FAMILY_LABELS.get(family or "")
    if label is None:
        return "this model"
    return f"{label} {model_size.upper()}" if family == "flux2_klein" and model_size else label


def choose_placement(
    components: Mapping[str, WeightTotals],
    quantize: int | None,
    *,
    job: MemoryJob,
    family: str | None,
    model_size: str | None,
    card_bytes: int,
    free_bytes: int,
) -> PlacementChoice:
    """Return the placement a CUDA image model loads in.

    Args:
        components: Each component's totals.
        quantize: The level the model loads at (a stored copy's own level).
        job: The job the model is loaded for.
        family: The image family.
        model_size: Klein's ``"4b"`` or ``"9b"``.
        card_bytes: The card's budget (its total less the driver's reserve), which the tag judges against.
        free_bytes: The memory the job can have now: the card's free memory, which other programs and the driver do not hold.

    Returns:
        The first candidate from the tag's pick on whose figure (the estimate plus the CUDA offset) is at most *free_bytes*; when none is, the one that needs the least, with a warning.
    """
    sizing = {
        "family": family,
        "model_size": model_size,
        "lora_bytes": job.lora_bytes,
        "job_megapixels": job.megapixels,
        "reference_megapixels": capped_reference_megapixels(job.reference_megapixels),
    }
    tagged, _ = cuda_image_placement(components, quantize, card_bytes, **sizing)
    candidates = image_placement_candidates(quantize)
    # Candidates before the tag's pick do not fit the card; the ones from it on are the tag's pick and the placements that need less.
    remaining = candidates[candidates.index(tagged) :]
    needs = {placement: cuda_image_gpu_bytes(placement, components, quantize, **sizing) for placement in remaining}
    chosen = next((placement for placement in remaining if grade(needs[placement] + CUDA_OFFSET_BYTES, free_bytes, CUDA_GPU_RATIOS) != TOO_LARGE), None)
    label = model_label(family, model_size)
    if chosen == tagged:
        return PlacementChoice(chosen)
    if chosen is None:
        last = min(remaining, key=needs.__getitem__)
        return PlacementChoice(
            last,
            f"{_step_down_text(label, free_bytes) if last != tagged else _too_little_text(label, free_bytes)} It may run out of GPU memory.",
            f"{_free_text(free_bytes)}. This may run out of memory.",
        )
    return PlacementChoice(chosen, _step_down_text(label, free_bytes), f"{_free_text(free_bytes)}, so this runs slower.")


def _free_text(free_bytes: int) -> str:
    """Return the start of every sentence about too little free memory: ``Only 7.6 GB of GPU memory is free``."""
    return f"Only {free_bytes / GIB:.1f} GB of GPU memory is free"


def _step_down_text(label: str, free_bytes: int) -> str:
    """Return the sentence that says the text encoder and transformer of *label* take turns on the GPU because only *free_bytes* of it is free."""
    return f"{_free_text(free_bytes)}, so the text encoder and transformer of {label} take turns on the GPU. This is slower."


def _too_little_text(label: str, free_bytes: int) -> str:
    return f"{_free_text(free_bytes)}, which is less than {label} needs."
