"""Reusable image workflow stage functions.

Each stage has signature:
    (request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome

Stages are pure functions operating on the immutable request and mutable artifacts.
"""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.alignment import round_to_alignment
from zvisiongenerator.utils.prompt_compose import expand_random_choices

_EXIF_IMAGE_DESCRIPTION = 0x010E


def resolve_prompt_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Replace {a|b|c} random choice blocks in prompt, supporting nesting."""
    artifacts.resolved_prompt = expand_random_choices(request.prompt)
    return StageOutcome.success


def suppress_negative_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Single point of truth: suppress negative prompt if model doesn't support it."""
    if not request.supports_negative_prompt and request.negative_prompt:
        artifacts.metadata["negative_suppressed"] = True
    return StageOutcome.success


def load_reference_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Load, resize (scale-to-fill), and center-crop reference image.

    Sets artifacts.image so that text_to_image_stage branches into img2img.
    No-op if request.image_path is None.
    """
    if request.image_path is None:
        return StageOutcome.success

    if not os.path.isfile(request.image_path):
        raise FileNotFoundError(f"Reference image not found: {request.image_path}")

    with Image.open(request.image_path) as ref_img:
        ref_img = ref_img.convert("RGB")

        # Scale-to-fill then center-crop to target dimensions
        src_w, src_h = ref_img.size
        scale = max(request.width / src_w, request.height / src_h)
        scaled_w = round(src_w * scale)
        scaled_h = round(src_h * scale)
        ref_img = ref_img.resize((scaled_w, scaled_h), Image.LANCZOS)
        left = (scaled_w - request.width) // 2
        top = (scaled_h - request.height) // 2
        ref_img = ref_img.crop((left, top, left + request.width, top + request.height))

        artifacts.image = ref_img.copy()
    return StageOutcome.success


def text_to_image_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Text-to-image or image-to-image generation via backend.

    If a previous stage (load_reference_stage) set artifacts.image,
    this becomes an img2img call with effective steps derived from
    request.image_strength. Otherwise, standard text-to-image.
    """
    prompt = artifacts.resolved_prompt or request.prompt
    neg = None if artifacts.metadata.get("negative_suppressed") else request.negative_prompt

    prompt_one_line = " ".join(prompt.split())
    print(f"Prompt: \n{prompt_one_line}\n")

    start = time.perf_counter()
    if artifacts.image is not None:
        # Reference image was loaded by load_reference_stage — do img2img
        print(f"Reference image: {request.image_path} (strength={request.image_strength})")
        effective_steps = math.floor(request.steps / request.image_strength) if request.image_strength > 0 else request.steps
        image = request.backend.image_to_image(
            model=request.model,
            image=artifacts.image,
            prompt=prompt,
            strength=request.image_strength,
            steps=effective_steps,
            seed=request.seed,
            guidance=request.guidance,
            scheduler=request.scheduler,
            negative_prompt=neg,
            skip_signal=request.skip_signal,
        )
    else:
        image = request.backend.text_to_image(
            model=request.model,
            prompt=prompt,
            width=request.width,
            height=request.height,
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            scheduler=request.scheduler,
            negative_prompt=neg,
            skip_signal=request.skip_signal,
        )
    elapsed = time.perf_counter() - start
    artifacts.generation_time += elapsed

    if image is None:
        print("Generation skipped.")
        return StageOutcome.skipped
    artifacts.image = image

    print(f"Image generated in {elapsed:.2f} seconds.")

    # Build filepath with generation time
    if artifacts.filename:
        os.makedirs(request.output_dir, exist_ok=True)
        artifacts.filepath = os.path.join(
            request.output_dir,
            f"{artifacts.filename}_time{int(artifacts.generation_time)}s.png",
        )

    return StageOutcome.success


def upscale_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Lanczos resize + optional CAS pre-sharpen + i2i refinement.

    No-op if upscale_factor is None or no image exists.
    """
    if request.upscale_factor is None:
        return StageOutcome.success
    if artifacts.image is None:
        return StageOutcome.success

    if request.upscale_factor not in (2, 4):
        raise ValueError(f"upscale must be 2 or 4, got {request.upscale_factor}")

    # Optional: save pre-upscale image for reference
    if request.upscale_save_pre and artifacts.filepath:
        p = Path(artifacts.filepath)
        pre_path = str(p.with_name(p.stem + "_preupscale" + p.suffix))
        artifacts.image.save(pre_path)

    # Denoise strength — already resolved by runner from config or explicit CLI arg
    if request.upscale_denoise is None:
        raise ValueError("upscale_denoise must be set before reaching upscale_stage")
    if not (0.0 <= request.upscale_denoise <= 1.0):
        raise ValueError(f"upscale_denoise must be between 0.0 and 1.0, got {request.upscale_denoise}")
    strength = request.upscale_denoise

    # Lanczos resize with 16-pixel alignment
    raw_width = artifacts.image.width * request.upscale_factor
    raw_height = artifacts.image.height * request.upscale_factor
    new_width = round_to_alignment(raw_width)
    new_height = round_to_alignment(raw_height)
    if new_width != raw_width or new_height != raw_height:
        warnings.warn(f"Upscaled dimensions adjusted from {raw_width}x{raw_height} to {new_width}x{new_height} for model compatibility (16-pixel alignment)")
    artifacts.image = artifacts.image.resize((new_width, new_height), Image.LANCZOS)

    # Optional pre-CAS sharpening to give img2img refinement crisper input
    if request.upscale_sharpen:
        from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening

        artifacts.image = contrast_adaptive_sharpening(
            artifacts.image,
            amount=request.sharpen_amount_pre_upscale,
        )

    # img2img refinement pass
    # Inflate num_inference_steps so effective step count matches user intent
    if request.upscale_steps is not None:
        effective_steps = math.floor(request.upscale_steps / strength) if strength > 0 else request.upscale_steps
    else:
        effective_steps = request.steps

    prompt = artifacts.resolved_prompt or request.prompt
    neg = None if artifacts.metadata.get("negative_suppressed") else request.negative_prompt

    start = time.perf_counter()
    image = request.backend.image_to_image(
        model=request.model,
        image=artifacts.image,
        prompt=prompt,
        strength=strength,
        steps=effective_steps,
        seed=request.seed,
        guidance=(request.upscale_guidance if request.upscale_guidance is not None else request.guidance),
        scheduler=request.scheduler,
        negative_prompt=neg,
        skip_signal=request.skip_signal,
    )
    elapsed = time.perf_counter() - start
    artifacts.generation_time += elapsed

    if image is None:
        print("Upscale skipped.")
        return StageOutcome.skipped

    artifacts.image = image
    artifacts.was_upscaled = True

    # Update filepath with total time and upscale info
    total_time = artifacts.generation_time
    print(f"Upscale refinement done. Total time: {total_time:.2f} seconds.")

    if artifacts.filepath:
        p = Path(artifacts.filepath)
        stem = p.stem
        # Replace generation-only time with total time
        stem = re.sub(r"_time\d+s$", f"_time{int(total_time)}s", stem)
        # Add upscale suffix
        stem += f"_u{request.upscale_factor}x_s{int(strength * 100)}p"
        # Update image dimensions in filename
        stem = re.sub(r"_(\d+)x(\d+)_", f"_{new_width}x{new_height}_", stem)
        artifacts.filepath = str(p.with_name(stem + p.suffix))

    return StageOutcome.success


def sharpen_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Apply CAS sharpening to current image."""
    if artifacts.image is None:
        return StageOutcome.success
    if request.sharpen_amount_override is not None:
        amount = request.sharpen_amount_override
    else:
        amount = request.sharpen_amount_upscaled if artifacts.was_upscaled else request.sharpen_amount_normal
    from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening

    artifacts.image = contrast_adaptive_sharpening(artifacts.image, amount=amount)
    return StageOutcome.success


def contrast_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Apply contrast adjustment to current image."""
    if artifacts.image is None:
        return StageOutcome.success
    from zvisiongenerator.processing.contrast import adjust_contrast

    artifacts.image = adjust_contrast(artifacts.image, amount=request.contrast_amount)
    return StageOutcome.success


def saturation_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Apply saturation adjustment to current image."""
    if artifacts.image is None:
        return StageOutcome.success
    from zvisiongenerator.processing.saturation import adjust_saturation

    artifacts.image = adjust_saturation(artifacts.image, amount=request.saturation_amount)
    return StageOutcome.success


def save_image_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Save image to disk with PNG metadata and EXIF."""
    if artifacts.image is None:
        return StageOutcome.success

    if not artifacts.filepath:
        return StageOutcome.success

    os.makedirs(os.path.dirname(artifacts.filepath) or ".", exist_ok=True)

    prompt = artifacts.resolved_prompt or request.prompt

    # PNG metadata
    metadata = PngInfo()
    metadata.add_text("Description", prompt)

    # EXIF ImageDescription tag
    exif = artifacts.image.getexif()
    exif[_EXIF_IMAGE_DESCRIPTION] = prompt

    artifacts.image.save(artifacts.filepath, pnginfo=metadata, exif=exif.tobytes())
    return StageOutcome.success
