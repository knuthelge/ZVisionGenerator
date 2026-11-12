"""Reusable image workflow stage functions.

Each stage has signature:
    (request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome

Stages are pure functions operating on the immutable request and mutable artifacts.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import EditWork, ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.processing.edit_region import PAINT_FADE, Plan, extend_canvas, extend_reference, extend_draw_size, grid_size, paste_paint, paste_strip, plan_region
from zvisiongenerator.processing.matte import on_backdrop
from zvisiongenerator.utils.alignment import round_to_alignment
from zvisiongenerator.utils.edit_request import image_is_cutout
from zvisiongenerator.utils.filename import unique_output_path
from zvisiongenerator.utils.provenance import EXIF_IMAGE_DESCRIPTION, build_image_config_payload, embed_png_config
from zvisiongenerator.utils.prompt_compose import expand_random_choices
from zvisiongenerator.utils.input_images import open_image_upright, to_srgb
from zvisiongenerator.utils.references import check_reference_count, load_reference_image
from zvisiongenerator.workflows.enhance_stage import apply_planned_enhancement
from zvisiongenerator.workflows.generation_prompt import conditioning_prompt, generation_negative, generation_prompt


def emit_step_progress(
    step_callback,
    *,
    phase: str,
    total_steps: int,
):
    """Wrap backend step callbacks with a stable phase label."""
    if step_callback is None:
        return None

    def _callback(event: dict[str, Any]) -> None:
        # The backend's total counts the steps that actually run (img2img skips the start of the schedule).
        steps_total = max(event.get("total_steps") or total_steps, 1)
        payload: dict[str, Any] = {
            "phase": phase,
            "current_step": min(event.get("current_step", 0), steps_total),
            "total_steps": steps_total,
        }
        if "preview" in event:
            payload["preview"] = event["preview"]
        step_callback(payload)

    return _callback


def resolve_prompt_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Replace {a|b|c} random choice blocks in prompt, supporting nesting.

    A ``request.resolved_prompt`` planned during preflight is used as is, so a retry or
    repeat keeps the same choices. When ``request.json_prompt`` is set, the prompt is a
    literal structured JSON caption and is passed through verbatim without random-choice expansion.
    """
    if request.resolved_prompt is not None:
        artifacts.resolved_prompt = request.resolved_prompt
    elif request.json_prompt:
        artifacts.resolved_prompt = request.prompt
    else:
        artifacts.resolved_prompt = expand_random_choices(request.prompt)
    return StageOutcome.success


def enhance_prompt_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Apply the prompt rewrite planned during preflight; never fails the generation."""
    apply_planned_enhancement(request, artifacts)
    return StageOutcome.success


def suppress_negative_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Record that the model's lack of negative-prompt support dropped the request's negative prompt."""
    if request.negative_prompt and generation_negative(request.negative_prompt, supports_negative=request.supports_negative_prompt) is None:
        artifacts.metadata["negative_suppressed"] = True
    return StageOutcome.success


def load_reference_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Load the start image upright and on white, resize (scale-to-fill), and center-crop it.

    Sets artifacts.image so that text_to_image_stage branches into img2img.
    No-op if request.image_path is None.
    """
    if request.image_path is None:
        return StageOutcome.success

    if not os.path.isfile(request.image_path):
        raise FileNotFoundError(f"Start image not found: {request.image_path}")

    ref_img = open_image_upright(request.image_path)

    # Scale-to-fill then center-crop to target dimensions
    src_w, src_h = ref_img.size
    scale = max(request.width / src_w, request.height / src_h)
    scaled_w = round(src_w * scale)
    scaled_h = round(src_h * scale)
    ref_img = ref_img.resize((scaled_w, scaled_h), Image.LANCZOS)
    left = (scaled_w - request.width) // 2
    top = (scaled_h - request.height) // 2
    artifacts.image = ref_img.crop((left, top, left + request.width, top + request.height))
    return StageOutcome.success


def load_references_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Put the references into ``artifacts.reference_images``, each upright, RGB, at most 1 MP and at its own aspect.

    Unlike the start image, a reference is not resized or cropped to the output size; the model sizes it.
    No-op if request.reference_paths is empty.

    Raises:
        ValueError: When the model takes no references, there are more than it takes, or a file is not a readable image.
        FileNotFoundError: When a reference file does not exist.
    """
    if not request.reference_paths:
        return StageOutcome.success
    check_reference_count(request.model_family, len(request.reference_paths), request.max_reference_images)
    artifacts.reference_images = [load_reference_image(path) for path in request.reference_paths]
    return StageOutcome.success


def load_source_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Load the image an upscale job refines, at its native size, and name the output after it.

    With ``request.transparent`` the source is composited onto the grey matte backdrop, because the upscalers take
    opaque input; the matte stage cuts the result out again.

    The output is ``{source stem}_{factor}x.png`` in ``request.output_dir``, with a counter when taken.
    No-op if request.upscale_source is None.
    """
    if request.upscale_source is None:
        return StageOutcome.success
    if request.upscale_factor is None:
        raise ValueError("upscale_factor must be set to upscale an existing image")
    source = Path(request.upscale_source)
    if not source.is_file():
        raise FileNotFoundError(f"Image to upscale not found: {source}")

    with Image.open(source) as source_image:
        if request.transparent:
            from zvisiongenerator.processing.matte import on_backdrop

            artifacts.image = on_backdrop(source_image)
        else:
            artifacts.image = source_image.convert("RGB")
    artifacts.metadata["upscale_source"] = {"path": str(source), "width": artifacts.image.width, "height": artifacts.image.height}

    os.makedirs(request.output_dir, exist_ok=True)
    artifacts.filename = f"{source.stem}_{request.upscale_factor}x"
    artifacts.filepath = str(unique_output_path(request.output_dir, artifacts.filename, ".png"))
    return StageOutcome.success


def load_edit_source_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Load the image an edit job changes, upright, and name the output after it.

    A source with a colour profile is converted to sRGB first. A source with a transparent background (a cut-out) goes on the grey matte backdrop for the model, and its alpha and
    own colours are kept, so the result gets them back. The output is ``artifacts.filename`` (the batch names it after the
    source) in ``request.output_dir``, with a counter when taken.

    Raises:
        ValueError: ``request.edit_source`` is not set.
        FileNotFoundError: The source does not exist.
    """
    if request.edit_source is None:
        raise ValueError("edit_source must be set to edit an image")
    source = Path(request.edit_source)
    if not source.is_file():
        raise FileNotFoundError(f"Image to edit not found: {source}")

    with Image.open(source) as opened:
        upright = to_srgb(ImageOps.exif_transpose(opened))
        upright.load()
    if image_is_cutout(upright):
        rgba = upright.convert("RGBA")
        work = EditWork(on_backdrop(rgba), cutout=rgba)
    else:
        work = EditWork(upright.convert("RGB"))
    artifacts.edit = work
    artifacts.metadata["edit_source"] = {"path": str(source), "width": work.canvas.width, "height": work.canvas.height}

    os.makedirs(request.output_dir, exist_ok=True)
    artifacts.filepath = str(unique_output_path(request.output_dir, artifacts.filename, ".png"))
    return StageOutcome.success


def _crop_for_model(canvas: Image.Image, plan: Plan) -> Image.Image:
    """Return the part of *canvas* the plan covers, scaled to the size the model draws at."""
    return canvas.crop(plan.box).resize(plan.target, Image.Resampling.LANCZOS)


def _resize_paint(paint: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return *paint* scaled to *size* (bilinear, then above half)."""
    scaled = Image.fromarray(paint.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(scaled) > 127


def _prepare_extend(edit: EditWork, extend: tuple[str, int]) -> None:
    """Set up an Extend: the canvas the result is pasted onto, and what the model draws.

    The model draws the source scaled first (:func:`extend_draw_size`) and padded after, so the new strip and the source
    both sit on whole 16 px cells. Its reference is the same canvas with the strip one flat colour (:func:`extend_reference`). The
    canvas is the source at its real size with the strip at the real amount; the result is pasted onto it strip and
    source apart.
    """
    side, amount = extend
    source = edit.canvas
    drawn_size, strip = extend_draw_size(source.size, side, amount)
    edit.canvas, _, offset = extend_canvas(source, side, amount)
    edit.source_box = (offset[0], offset[1], offset[0] + source.width, offset[1] + source.height)
    scaled = source.resize(drawn_size, Image.Resampling.LANCZOS)
    edit.crop, edit.crop_paint, drawn_offset = extend_canvas(scaled, side, strip)
    edit.drawn_source_box = (drawn_offset[0], drawn_offset[1], drawn_offset[0] + drawn_size[0], drawn_offset[1] + drawn_size[1])
    edit.reference = extend_reference(scaled, side, strip)


def prepare_edit_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Turn the source, the painted mask and the Extend request into what the model works on.

    The mask is read at the source's size before any Extend. With paint the model gets a crop around it and the paint at
    the crop's size; with none (whole picture) it gets the picture at its own size on the 16 px grid. An Extend is drawn on the
    scaled source with a new strip beside it (see :func:`_prepare_extend`). The model's
    reference is the crop, or for an Extend the drawn canvas with a flat strip, so the strip's edge-pixel fill is never
    shown to it.

    The inputs were checked where the job was made (the CLI and the Web server), before it was queued: the mask is the
    source's size and the edit can run on this model.

    Raises:
        ValueError: The source image was not loaded.
    """
    edit = artifacts.edit
    if edit is None:
        raise ValueError("The source image must be loaded before the edit is prepared")
    if request.edit_extend is not None:
        _prepare_extend(edit, request.edit_extend)
        return StageOutcome.success
    edit.paint = request.edit_paint
    width, height = edit.canvas.size
    if edit.paint is None:
        edit.plan = Plan((0, 0, width, height), grid_size(width, height))
        edit.crop = _crop_for_model(edit.canvas, edit.plan)
    else:
        edit.plan = plan_region(edit.paint)
        x0, y0, x1, y1 = edit.plan.box
        edit.crop = _crop_for_model(edit.canvas, edit.plan)
        edit.crop_paint = _resize_paint(edit.paint[y0:y1, x0:x1], edit.plan.target)
    edit.reference = edit.crop
    return StageOutcome.success


def edit_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Have the model redraw the painted area of the crop, or, with no paint, redraw the whole picture from the crop as its reference.

    The prompt is the one the generation stage would send; an Extend sends it behind the Extend instruction (:func:`conditioning_prompt`), the same text the batch pre-encoded. Returns ``skipped`` when the backend was stopped by Next or Stop.
    """
    edit = artifacts.edit
    if edit is None or edit.crop is None or edit.reference is None:
        raise ValueError("The edit must be prepared before the model runs")
    prompt = generation_prompt(request.prompt, artifacts.resolved_prompt)
    print(f"Prompt: \n{' '.join(prompt.split())}\n")
    # Every edit path sends the text from this one function, as the batch pre-encoded it (`_job_prompts`).
    extend = request.edit_extend is not None
    sent = conditioning_prompt(prompt, extend=extend)
    step_callback = emit_step_progress(request.step_callback, phase="image_generate", total_steps=request.steps)

    start = time.perf_counter()
    if edit.crop_paint is not None:
        image = request.backend.edit_region(
            model=request.model,
            image=edit.crop,
            reference=edit.reference,
            paint=edit.crop_paint,
            prompt=sent,
            fix_join=extend,
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            scheduler=request.scheduler,
            skip_signal=request.skip_signal,
            step_callback=step_callback,
        )
    else:
        width, height = edit.crop.size
        image = request.backend.text_to_image(
            model=request.model,
            prompt=sent,
            width=width,
            height=height,
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            scheduler=request.scheduler,
            skip_signal=request.skip_signal,
            step_callback=step_callback,
            steps_explicit=request.steps_explicit,
            guidance_explicit=request.guidance_explicit,
            reference_images=[edit.crop],
        )
    elapsed = time.perf_counter() - start
    artifacts.generation_time += elapsed
    if image is None:
        print("Edit skipped.")
        return StageOutcome.skipped
    artifacts.image = image
    print(f"Image edited in {elapsed:.2f} seconds.")
    return StageOutcome.success


def composite_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Paste the model's result back over the canvas through the paint, so every pixel beyond the fade around it stays the source's own.

    The paint is replaced whole and fades into the picture around it; an Extend's new area is replaced whole too.

    A cut-out is pasted onto its own colours, not the grey backdrop the model saw, and gets its alpha back everywhere. A
    whole-picture edit has no paint: its picture is scaled back to the source's size when the 16 px grid changed it.
    """
    edit = artifacts.edit
    if edit is None or artifacts.image is None:
        return StageOutcome.success
    if edit.paint is None and edit.source_box is None:
        if artifacts.image.size != edit.canvas.size:
            artifacts.image = artifacts.image.resize(edit.canvas.size, Image.Resampling.LANCZOS)
        return StageOutcome.success
    base = edit.canvas if edit.cutout is None else edit.cutout.convert("RGB")
    if edit.source_box is not None:
        image = paste_strip(base, artifacts.image, edit.source_box, edit.drawn_source_box, edit.crop.size)
    else:
        image = paste_paint(base, artifacts.image, edit.plan, edit.paint, PAINT_FADE)
    if edit.cutout is not None:
        image.putalpha(edit.cutout.getchannel("A"))
    artifacts.image = image
    return StageOutcome.success


def text_to_image_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Text-to-image or image-to-image generation via backend.

    If a previous stage (load_reference_stage) set artifacts.image,
    this becomes an img2img call with effective steps derived from
    request.image_strength. Otherwise, standard text-to-image. References
    loaded by load_references_stage go to either call.
    """
    prompt = generation_prompt(request.prompt, artifacts.resolved_prompt)
    neg = generation_negative(request.negative_prompt, supports_negative=request.supports_negative_prompt)

    prompt_one_line = " ".join(prompt.split())
    print(f"Prompt: \n{prompt_one_line}\n")
    prompt = conditioning_prompt(prompt, transparent=request.transparent)

    reference_images = artifacts.reference_images or None
    if reference_images:
        print(f"References: {len(reference_images)} image(s)")

    start = time.perf_counter()
    if artifacts.image is not None:
        # Start image was loaded by load_reference_stage — do img2img
        print(f"Start image: {request.image_path} (strength={request.image_strength})")
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
            step_callback=emit_step_progress(request.step_callback, phase="image_refine", total_steps=effective_steps),
            reference_images=reference_images,
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
            step_callback=emit_step_progress(request.step_callback, phase="image_generate", total_steps=request.steps),
            steps_explicit=request.steps_explicit,
            guidance_explicit=request.guidance_explicit,
            first_sigma=request.first_sigma,
            reference_images=reference_images,
        )
    elapsed = time.perf_counter() - start
    artifacts.generation_time += elapsed

    if image is None:
        print("Generation skipped.")
        return StageOutcome.skipped
    artifacts.image = image

    print(f"Image generated in {elapsed:.2f} seconds.")

    if artifacts.filename:
        os.makedirs(request.output_dir, exist_ok=True)
        artifacts.filepath = str(unique_output_path(request.output_dir, artifacts.filename, ".png"))

    return StageOutcome.success


def review_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Show the small render for review before the upscale; a discard skips the generation.

    No-op if request.review is None or no image exists.
    """
    if request.review is None or artifacts.image is None:
        return StageOutcome.success
    if not request.review(artifacts.image):
        return StageOutcome.skipped
    return StageOutcome.success


def upscale_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Lanczos resize + optional CAS pre-sharpen + i2i refinement, which sees the references the render used.

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

    prompt = generation_prompt(request.prompt, artifacts.resolved_prompt)
    prompt = conditioning_prompt(prompt, transparent=request.transparent)
    neg = generation_negative(request.negative_prompt, supports_negative=request.supports_negative_prompt)

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
        step_callback=emit_step_progress(request.step_callback, phase="image_upscale", total_steps=effective_steps),
        # The refinement keeps the references, so the people and objects they fix stay the same in the larger image.
        reference_images=artifacts.reference_images or None,
    )
    elapsed = time.perf_counter() - start
    artifacts.generation_time += elapsed

    if image is None:
        print("Upscale skipped.")
        return StageOutcome.skipped

    artifacts.image = image
    artifacts.was_upscaled = True
    print(f"Upscale refinement done. Total time: {artifacts.generation_time:.2f} seconds.")
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


def matte_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Cut the subject out of the image, leaving the background transparent.

    Runs last, after every post-processing stage, because sharpening drops the alpha channel and contrast and
    saturation would recolour the pixels under it. The chain loads on its first cut-out; ``make_room`` is asked to
    free memory for it, given the image's pixel count, only while it is not loaded.
    """
    if artifacts.image is None or request.matte is None:
        return StageOutcome.success
    if not request.matte.loaded and request.make_room is not None:
        request.make_room(artifacts.image.width * artifacts.image.height)
    artifacts.image = request.matte.cut_out(artifacts.image.convert("RGB"))
    callback = emit_step_progress(request.step_callback, phase="image_matte", total_steps=1)
    if callback is not None:
        callback({"current_step": 1})
    return StageOutcome.success


def save_image_stage(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> StageOutcome:
    """Save image to disk with PNG metadata and EXIF."""
    if artifacts.image is None:
        return StageOutcome.success

    if not artifacts.filepath:
        return StageOutcome.success

    os.makedirs(os.path.dirname(artifacts.filepath) or ".", exist_ok=True)

    prompt = generation_prompt(request.prompt, artifacts.resolved_prompt)

    # PNG metadata
    metadata = PngInfo()
    metadata.add_text("Description", prompt)
    embed_png_config(metadata, build_image_config_payload(request, artifacts))

    # EXIF ImageDescription tag
    exif = artifacts.image.getexif()
    exif[EXIF_IMAGE_DESCRIPTION] = prompt

    # Create the file exclusively so a name taken since it was chosen (another job, same second) is never overwritten.
    path = Path(artifacts.filepath)
    while True:
        try:
            handle = open(path, "xb")
        except FileExistsError:
            path = unique_output_path(path.parent, artifacts.filename or path.stem, path.suffix)
            continue
        try:
            with handle:
                artifacts.image.save(handle, format="PNG", pnginfo=metadata, exif=exif.tobytes())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        break
    artifacts.filepath = str(path)
    return StageOutcome.success
