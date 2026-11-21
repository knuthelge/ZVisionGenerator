"""Grade a memory need against a limit: the three statuses and the thresholds every memory verdict shares.

The thresholds live only here. The verdict engine (:mod:`zvisiongenerator.utils.memory_verdict`) grades every job with them, and
a decision the estimators make about the card (whether an H3 upscale keeps its transformer loaded) asks the same question of the
same line, so the tag and the backend cannot disagree.
"""

from __future__ import annotations

__all__ = ["CUDA_GPU_RATIOS", "CUDA_RAM_RATIOS", "FITS", "MAC_RATIOS", "TIGHT", "TOO_LARGE", "grade"]

FITS = "fits"
TIGHT = "tight"
TOO_LARGE = "too_large"

# Thresholds, as a share of the limit: a need up to the first ratio fits, up to the second is tight, beyond is too large.
CUDA_GPU_RATIOS = (0.9, 1.0)
CUDA_RAM_RATIOS = (0.8, 1.0)
# A Mac figure is the process's peak footprint, graded against the machine's total unified memory like system memory on CUDA: past
# the first ratio macOS compresses and swaps, past the whole memory the job cannot run.
MAC_RATIOS = (0.75, 1.0)


def grade(need: float, limit: float, ratios: tuple[float, float]) -> str:
    """Return ``"fits"``, ``"tight"`` or ``"too_large"`` for a *need* against a *limit*, by the share of it each ratio allows."""
    fits, tight = ratios
    if need <= limit * fits:
        return FITS
    return TIGHT if need <= limit * tight else TOO_LARGE
