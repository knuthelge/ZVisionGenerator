"""Dynamic workflow builder — assembles stage list from CLI flags."""

from __future__ import annotations

import argparse

from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.review import review_enabled
from zvisiongenerator.workflows import seedvr2_stages
from zvisiongenerator.workflows.image_stages import (
    enhance_prompt_stage,
    resolve_prompt_stage,
    suppress_negative_stage,
    load_reference_stage,
    load_references_stage,
    load_source_stage,
    review_stage,
    text_to_image_stage,
    upscale_stage,
    contrast_stage,
    saturation_stage,
    sharpen_stage,
    save_image_stage,
)

__all__ = ["build_workflow", "build_upscale_workflow", "build_seedvr2_upscale_workflow", "build_video_workflow"]


def build_workflow(args: argparse.Namespace, *, enhance: bool = False) -> GenerationWorkflow:
    """Build a dynamic workflow from CLI flags; *enhance* adds the auto prompt-enhancement stage."""
    stages = [resolve_prompt_stage]
    if enhance:
        stages.append(enhance_prompt_stage)
    stages += [suppress_negative_stage, load_reference_stage, load_references_stage, text_to_image_stage]
    if args.upscale:
        if review_enabled(args):
            stages.append(review_stage)
        stages.append(upscale_stage)
    if args.contrast is not False:
        stages.append(contrast_stage)
    if args.saturation is not False:
        stages.append(saturation_stage)
    if args.sharpen is not False:
        stages.append(sharpen_stage)
    stages.append(save_image_stage)
    return GenerationWorkflow(name="dynamic", stages=stages)


def build_upscale_workflow(*, sharpen: bool = True) -> GenerationWorkflow:
    """Build the workflow that upscales an existing image: load it, upscale and refine, sharpen, save.

    Contrast and saturation are left out: the source already had them applied, and the refinement keeps its colours.
    """
    stages = [resolve_prompt_stage, suppress_negative_stage, load_source_stage, upscale_stage]
    if sharpen:
        stages.append(sharpen_stage)
    stages.append(save_image_stage)
    return GenerationWorkflow(name="upscale", stages=stages)


def build_seedvr2_upscale_workflow() -> GenerationWorkflow:
    """Build the workflow that upscales an existing image with SeedVR2: resolve the prompt, load, upscale, save.

    The prompt stage puts the rendered prompt into the saved file. There is no sharpen stage: SeedVR2 output is saved as is.
    """
    return GenerationWorkflow(name="upscale", stages=[resolve_prompt_stage, load_source_stage, seedvr2_stages.upscale_stage, save_image_stage])


def build_video_workflow(args: argparse.Namespace, *, enhance: bool = False) -> GenerationWorkflow:
    """Build video generation workflow based on CLI args.

    Args:
        args: Parsed video CLI arguments. Must have ``image_path`` attribute; ``extend_source`` selects extending a clip.
        enhance: Add the auto prompt-enhancement stage after prompt resolution.

    Returns:
        GenerationWorkflow with stages for T2V, I2V or extending a clip. Extending ends with the joined clip, which
        already has the right audio, so it has no audio-stripping stage.
    """
    from zvisiongenerator.workflows.video_stages import (
        enhance_prompt_stage as video_enhance_prompt,
        resolve_prompt_stage as video_resolve_prompt,
        generate_filename_stage,
        extend_video_stage,
        join_clips_stage,
        text_to_video_stage,
        image_to_video_stage,
        strip_audio_stage,
        log_video_stage,
    )

    stages = [video_resolve_prompt]
    if enhance:
        stages.append(video_enhance_prompt)
    stages.append(generate_filename_stage)
    if getattr(args, "extend_source", None) is not None:
        stages += [extend_video_stage, join_clips_stage]
    else:
        stages.append(image_to_video_stage if args.image_path else text_to_video_stage)
        if getattr(args, "no_audio", False):
            stages.append(strip_audio_stage)
    stages.append(log_video_stage)
    return GenerationWorkflow(name="video", stages=stages)
