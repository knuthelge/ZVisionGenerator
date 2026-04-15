"""Dynamic workflow builder — assembles stage list from CLI flags."""

from __future__ import annotations

import argparse

from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.workflows.image_stages import (
    resolve_prompt_stage,
    suppress_negative_stage,
    load_reference_stage,
    text_to_image_stage,
    upscale_stage,
    contrast_stage,
    saturation_stage,
    sharpen_stage,
    save_image_stage,
)

__all__ = ["build_workflow", "build_video_workflow"]


def build_workflow(args: argparse.Namespace) -> GenerationWorkflow:
    """Build a dynamic workflow from CLI flags."""
    stages = [resolve_prompt_stage, suppress_negative_stage, load_reference_stage, text_to_image_stage]
    if args.upscale:
        stages.append(upscale_stage)
    if args.contrast is not False:
        stages.append(contrast_stage)
    if args.saturation is not False:
        stages.append(saturation_stage)
    if args.sharpen is not False:
        stages.append(sharpen_stage)
    stages.append(save_image_stage)
    return GenerationWorkflow(name="dynamic", stages=stages)


def build_video_workflow(args: argparse.Namespace) -> GenerationWorkflow:
    """Build video generation workflow based on CLI args.

    Args:
        args: Parsed video CLI arguments. Must have ``image_path`` attribute.

    Returns:
        GenerationWorkflow with stages for T2V or I2V.
    """
    from zvisiongenerator.workflows.video_stages import (
        resolve_prompt_stage as video_resolve_prompt,
        generate_filename_stage,
        text_to_video_stage,
        image_to_video_stage,
        strip_audio_stage,
        log_video_stage,
    )

    stages = [video_resolve_prompt, generate_filename_stage]
    if args.image_path:
        stages.append(image_to_video_stage)
    else:
        stages.append(text_to_video_stage)
    if getattr(args, "no_audio", False):
        stages.append(strip_audio_stage)
    stages.append(log_video_stage)
    return GenerationWorkflow(name="video", stages=stages)
