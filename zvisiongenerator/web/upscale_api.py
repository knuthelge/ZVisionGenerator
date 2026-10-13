"""Plan a Web UI upscale job for an existing image: pick the refinement model and settings, enforce the size limits."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.upscale_runner import UpscaleSource
from zvisiongenerator.utils.config import resolve_defaults, resolve_scheduler_class, resolve_upscale_steps, sharpening_amounts
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_CREATIVE, UPSCALE_METHOD_SEEDVR2, RecordedSettings
from zvisiongenerator.utils.seedvr2 import SEEDVR2_BITS, SEEDVR2_REPO, SeedVR2Setup
from zvisiongenerator.utils.upscale import UPSCALE_FACTORS, UpscaleOption, existing_image_denoise, seedvr2_options, upscale_options
from zvisiongenerator.web.config import WebUiConfig, preferred_option

NOTICE_SETTINGS_UNKNOWN = "original settings unknown"
NOTICE_MODEL_NOT_CONFIGURED = "original model not configured, using the default"
NOTICE_LORA_MISSING = "missing LoRAs left out"
_UPSCALE_METHODS = (UPSCALE_METHOD_CREATIVE, UPSCALE_METHOD_SEEDVR2)


@dataclass(frozen=True)
class UpscalePlan:
    """Everything needed to submit one upscale job."""

    request: ImageGenerationRequest
    model_ref: str
    option: UpscaleOption
    # Job-panel notes on how the job differs from the original, e.g. a model fallback.
    notices: tuple[str, ...] = field(default_factory=tuple)


def plan_upscale(source: UpscaleSource, factor: int, web_config: WebUiConfig, *, backend_name: str) -> UpscalePlan:
    """Build the upscale request for *source* at *factor*.

    The refinement reuses the image's recorded settings. When its model is missing or not configured, the
    default image model refines it with that model's own defaults (the recorded steps, guidance, scheduler and
    LoRAs belong to the other model).

    Raises:
        ValueError: If *factor* is not offered, no image model is configured, or the output is not allowed.
    """
    if factor not in UPSCALE_FACTORS:
        raise ValueError(f"Upscale factor must be one of {list(UPSCALE_FACTORS)}, got {factor}.")
    app_config = web_config.app_config
    recorded = source.settings or RecordedSettings()
    notices: list[str] = []
    if source.settings is None:
        notices.append(NOTICE_SETTINGS_UNKNOWN)

    model_name = recorded.model if recorded.model in web_config.image_model_options else None
    if model_name is None:
        if recorded.model is not None:
            notices.append(NOTICE_MODEL_NOT_CONFIGURED)
        model_name = preferred_option(web_config.default_models.image, web_config.image_model_options)
    if model_name is None:
        raise ValueError("No image model is configured to refine the upscale.")
    keeps_recorded = model_name == recorded.model

    model_ref = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
    model_info = detect_image_model(model_ref)
    defaults = resolve_defaults(model_info, app_config, {}, backend_name)

    option = next(option for option in upscale_options(source.width, source.height, defaults, app_config) if option.factor == factor)
    if not option.allowed:
        raise ValueError(option.reason or f"Upscaling this image {factor}× is not allowed.")

    steps = recorded.steps if keeps_recorded and recorded.steps else defaults["steps"]
    guidance = recorded.guidance if keeps_recorded and recorded.guidance is not None else defaults["guidance"]
    scheduler = recorded.scheduler if keeps_recorded and recorded.scheduler in app_config.get("schedulers", {}) else defaults["scheduler"]
    lora_paths, lora_weights, dropped_loras = _existing_loras(recorded.lora) if keeps_recorded else (None, None, False)
    if dropped_loras:
        notices.append(NOTICE_LORA_MISSING)
    recorded_quantize = recorded.generation.get("quantize")
    quantize = recorded_quantize if keeps_recorded and defaults.get("supports_quantize", True) and recorded_quantize in web_config.quantize_options else None
    supports_negative = defaults.get("supports_negative_prompt", False)
    sharpening = sharpening_amounts(app_config)
    pre_sharpen = sharpening["existing_pre_upscale"]
    prompt = recorded.prompt or ""

    request = ImageGenerationRequest(
        backend=None,
        model=None,
        prompt=prompt,
        resolved_prompt=source.rendered_prompt or prompt or None,
        model_name=model_name,
        model_family=model_info.family,
        supports_negative_prompt=supports_negative,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        negative_prompt=recorded.negative_prompt if supports_negative else None,
        ratio=recorded.ratio,
        size=recorded.size,
        width=source.width,
        height=source.height,
        seed=recorded.seed or 0,
        steps=steps,
        guidance=guidance,
        scheduler=resolve_scheduler_class(scheduler, app_config, backend_name),
        scheduler_name=scheduler,
        quantize=quantize,
        upscale_factor=factor,
        upscale_denoise=existing_image_denoise(app_config, option.width, option.height),
        upscale_steps=resolve_upscale_steps(defaults, steps),
        # The source is already sharpened, so the pass before refining is off unless existing_pre_upscale sets one.
        upscale_sharpen=pre_sharpen > 0,
        sharpen=_sharpens_after_refining(recorded),
        sharpen_amount_upscaled=sharpening["existing_upscaled"],
        sharpen_amount_pre_upscale=pre_sharpen,
        # Recorded only: the upscale workflow never loads the reference, but Reuse settings needs it.
        image_path=recorded.image_path,
        image_strength=recorded.image_strength if recorded.image_strength is not None else ImageGenerationRequest.image_strength,
        output_dir=str(Path(source.path).parent),
        upscale_source=source.path,
    )
    return UpscalePlan(request=request, model_ref=model_ref, option=option, notices=tuple(notices))


def plan_seedvr2_upscale(source: UpscaleSource, factor: int, *, setup: SeedVR2Setup) -> UpscalePlan:
    """Build the SeedVR2 upscale request for *source* at *factor*.

    No image model is involved, so nothing is detected or defaulted from one: the request carries the source's
    recorded settings so the new file stays reusable. When the source records no seed, steps or guidance, the
    request holds class defaults and ``source_settings_known`` is False, so they are not saved as the original's.

    Raises:
        ValueError: If *factor* is not offered.
    """
    if factor not in UPSCALE_FACTORS:
        raise ValueError(f"Upscale factor must be one of {list(UPSCALE_FACTORS)}, got {factor}.")
    option = next(option for option in seedvr2_options(source.width, source.height, setup.tier, budget_bytes=setup.budget_bytes) if option.factor == factor)
    recorded = source.settings or RecordedSettings()
    notices = (NOTICE_SETTINGS_UNKNOWN,) if source.settings is None else ()

    prompt = recorded.prompt or ""
    lora_paths, lora_weights = _recorded_loras(recorded.lora)
    settings_known = recorded.seed is not None and recorded.steps is not None and recorded.guidance is not None
    request = ImageGenerationRequest(
        backend=None,
        model=None,
        prompt=prompt,
        resolved_prompt=source.rendered_prompt or prompt or None,
        model_name=recorded.model,
        model_family=recorded.model_family or "unknown",
        supports_negative_prompt=recorded.negative_prompt is not None,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        negative_prompt=recorded.negative_prompt,
        ratio=recorded.ratio,
        size=recorded.size,
        width=source.width,
        height=source.height,
        seed=recorded.seed if settings_known else ImageGenerationRequest.seed,
        steps=recorded.steps if settings_known else ImageGenerationRequest.steps,
        guidance=recorded.guidance if settings_known else ImageGenerationRequest.guidance,
        scheduler=None,
        scheduler_name=recorded.scheduler,
        quantize=recorded.generation.get("quantize"),
        source_settings_known=settings_known,
        upscale_factor=factor,
        upscale_method=UPSCALE_METHOD_SEEDVR2,
        upscale_model=setup.tier.model,
        upscale_quantize=SEEDVR2_BITS,
        upscale_softness=setup.settings.softness,
        sharpen=False,
        image_path=recorded.image_path,
        image_strength=recorded.image_strength if recorded.image_strength is not None else ImageGenerationRequest.image_strength,
        output_dir=str(Path(source.path).parent),
        upscale_source=source.path,
    )
    return UpscalePlan(request=request, model_ref=SEEDVR2_REPO, option=option, notices=notices)


def _recorded_loras(lora: str | None) -> tuple[list[str] | None, list[float] | None]:
    """Parse recorded LoRAs for the new file's record, keeping entries whose files are gone; unparseable text gives none."""
    if lora is None:
        return None, None
    try:
        entries = parse_lora_arg(lora)
    except ValueError:
        return None, None
    return [path for path, _ in entries], [weight for _, weight in entries]


def _sharpens_after_refining(recorded: RecordedSettings) -> bool:
    """Return whether to end with a sharpen pass: yes, unless the source was recorded without sharpening.

    Files that predate the post-processing record (no ``generation`` block) are sharpened.
    """
    return not recorded.generation or "sharpen" in recorded.generation


def _existing_loras(lora: str | None) -> tuple[list[str] | None, list[float] | None, bool]:
    """Parse recorded LoRAs and drop local files that no longer exist.

    Returns:
        ``(paths, weights, dropped)``; references that are not local files are kept.
    """
    if lora is None:
        return None, None, False
    try:
        entries = parse_lora_arg(lora)
    except ValueError:
        return None, None, False
    kept = [(path, weight) for path, weight in entries if not _is_missing_local_file(path)]
    if not kept:
        return None, None, bool(entries)
    return [path for path, _ in kept], [weight for _, weight in kept], len(kept) != len(entries)


def _is_missing_local_file(reference: str) -> bool:
    path = Path(reference).expanduser()
    return path.is_absolute() and not path.exists()


def upscale_size_label(plan: UpscalePlan) -> str:
    """Return the job's size line, e.g. ``2× → 1664×2432``, or ``2× creative → 1664×2432``."""
    creative = " creative" if plan.request.upscale_method == UPSCALE_METHOD_CREATIVE else ""
    return f"{plan.option.factor}×{creative} → {plan.option.width}×{plan.option.height}"


def upscale_json_request(body: Any) -> tuple[str, int, str]:
    """Validate a ``POST /api/upscale`` body and return ``(asset_id, factor, method)``.

    ``method`` is optional and defaults to ``creative``.

    Raises:
        ValueError: If the asset ID, factor or method is missing or malformed.
    """
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    asset_id = body.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("asset_id is required.")
    factor = body.get("factor")
    if isinstance(factor, bool) or not isinstance(factor, int) or factor not in UPSCALE_FACTORS:
        raise ValueError(f"factor must be one of {list(UPSCALE_FACTORS)}.")
    method = body.get("method", UPSCALE_METHOD_CREATIVE)
    if method not in _UPSCALE_METHODS:
        raise ValueError(f"method must be one of {list(_UPSCALE_METHODS)}.")
    return asset_id, factor, method
