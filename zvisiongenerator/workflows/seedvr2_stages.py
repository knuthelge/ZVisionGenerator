"""SeedVR2 workflow stage: upscale the loaded source image with the SeedVR2 engine.

The stage is named ``upscale_stage`` so its progress label is ``upscale``, like the refining upscale stage.
"""

from __future__ import annotations

import time

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.upscale import seedvr2_output_size
from zvisiongenerator.workflows.image_stages import emit_step_progress


def upscale_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Upscale ``artifacts.image`` by ``request.upscale_factor`` with SeedVR2; a quit during the call discards the result.

    SeedVR2 reports no per-step progress, so the stage sends one complete step when the call returns.

    Raises:
        ValueError: If the source path, the factor or the loaded image is missing.
    """
    if request.upscale_source is None:
        raise ValueError("upscale_source must be set to upscale with SeedVR2")
    if request.upscale_factor is None:
        raise ValueError("upscale_factor must be set to upscale with SeedVR2")
    if artifacts.image is None:
        raise ValueError("the source image must be loaded before the SeedVR2 upscale")

    width, height = seedvr2_output_size(artifacts.image.width, artifacts.image.height, request.upscale_factor)
    start = time.perf_counter()
    image = request.backend.upscale_image(
        request.model,
        request.upscale_source,
        width=width,
        height=height,
        seed=request.seed,
        softness=request.upscale_softness or 0.0,
    )
    elapsed = time.perf_counter() - start

    if request.skip_signal is not None and request.skip_signal.pending() == "quit":
        print("Upscale skipped.")
        return StageOutcome.skipped

    artifacts.image = image
    artifacts.was_upscaled = True
    artifacts.generation_time += elapsed
    callback = emit_step_progress(request.step_callback, phase="image_upscale", total_steps=1)
    if callback is not None:
        callback({"current_step": 1})
    print(f"SeedVR2 upscale done in {elapsed:.2f} seconds.")
    return StageOutcome.success
