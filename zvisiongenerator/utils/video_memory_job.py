"""Build the video :class:`~zvisiongenerator.utils.memory_verdict.MemoryJob` of a clip or an extension window, for the command line and the Web UI alike."""

from __future__ import annotations

from zvisiongenerator.utils.memory_verdict import MemoryJob

__all__ = ["clip_memory_job"]

_DEFAULT_FORWARDS = 8


def clip_memory_job(
    family: str,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool = False,
    mode: str = "t2v",
    references: int = 0,
    upscale: bool = False,
    review: bool = False,
    steps: int | None = None,
    lora_bytes: int = 0,
) -> MemoryJob:
    """Return the memory job of a clip of *family* (``"ltx"`` or ``"h3"``) that generates *frames* frames at *width* x *height*.

    Args:
        family: The video family.
        width: The width the model draws (an upscale's enlarged size, when the clip is upscaled).
        height: The height the model draws.
        frames: The frames the model generates: the clip's, or an extension's window.
        low_memory: Whether the clip runs with Low memory on (LTX).
        mode: ``"t2v"``, ``"i2v"`` or ``"extend"``.
        references: How many reference images the clip holds.
        upscale: Whether the clip is upscaled 2x.
        review: Whether the upscale pauses to review the small render first; only an upscale has one.
        steps: The steps the run takes, which set MiniMax H3's expected time; the default count when ``None``.
        lora_bytes: The total size of the LoRA files the clip applies.
    """
    return MemoryJob(
        media="video",
        family=family,
        width=width,
        height=height,
        frames=frames,
        low_memory=low_memory,
        mode=mode,
        references=references,
        upscale=upscale,
        review=upscale and review,
        forwards=steps or _DEFAULT_FORWARDS,
        lora_bytes=lora_bytes,
    )
