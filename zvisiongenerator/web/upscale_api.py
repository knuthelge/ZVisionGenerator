"""Plan a Web UI upscale job for an existing image: pick the refinement model and settings, enforce the size limits."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import ExtendSource, LtxVideoUpscaleRequest, VideoFileUpscaleRequest
from zvisiongenerator.downloads.declaration import declared_repo_ids
from zvisiongenerator.upscale_runner import UpscaleSource
from zvisiongenerator.utils.model_metadata import alias_map
from zvisiongenerator.schedulers import SCHEDULER_CLASSES
from zvisiongenerator.utils.ltx_facts import ltx_version
from zvisiongenerator.utils.config import resolve_defaults, resolve_scheduler_class, resolve_upscale_steps, sharpening_amounts
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import is_existing_file, is_explicit_local_path, resolve_lora_path, resolve_model_path
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_CREATIVE, UPSCALE_METHOD_LTX, UPSCALE_METHOD_SEEDVR2, RecordedSettings, optional_int, original_video_workflow
from zvisiongenerator.utils.seedvr2 import SEEDVR2_BITS, SeedVR2Setup
from zvisiongenerator.utils.upscale import (
    UPSCALE_FACTORS,
    VIDEO_UPSCALE_FACTOR,
    VIDEO_UPSCALE_SEED,
    UpscaleOption,
    existing_image_denoise,
    seedvr2_options,
    seedvr2_video_option,
    upscale_options,
    video_upscale_batch_size,
)
from zvisiongenerator.utils.video_model_detect import detect_video_model, known_ltx_repo_version
from zvisiongenerator.utils.video_upscale import ltx_upscale_problem, ltx_upscale_prompt, ltx_upscale_steps
from zvisiongenerator.web.config import WebUiConfig, preferred_option

NOTICE_SETTINGS_UNKNOWN = "original settings unknown"
NOTICE_MODEL_NOT_CONFIGURED = "original model not configured, using the default"
NOTICE_LORA_MISSING = "missing LoRAs left out"
NOTICE_MODEL_NOT_LTX23 = "original model can't refine, using LTX-2.3"
NOTICE_LORA_OTHER_MODEL = "LoRAs of the original model left out"
_UPSCALE_METHODS = (UPSCALE_METHOD_CREATIVE, UPSCALE_METHOD_SEEDVR2, UPSCALE_METHOD_LTX)


@dataclass(frozen=True)
class UpscalePlan:
    """Everything needed to submit one upscale job."""

    request: ImageGenerationRequest
    model_ref: str
    option: UpscaleOption
    # Job-panel notes on how the job differs from the original, e.g. a model fallback.
    notices: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VideoUpscalePlan:
    """Everything needed to submit one video upscale job."""

    request: VideoFileUpscaleRequest | LtxVideoUpscaleRequest
    # Job-panel notes on how the job differs from the original, e.g. settings that were not recorded.
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

    model_ref = resolve_model_path(model_name, aliases=alias_map(app_config), platform_key=sys.platform)
    model_info = detect_image_model(model_ref)
    defaults = resolve_defaults(model_info, app_config, {}, backend_name)

    option = next(option for option in upscale_options(source.width, source.height, defaults, app_config) if option.factor == factor)
    if not option.allowed:
        raise ValueError(option.reason or f"Upscaling this image {factor}× is not allowed.")

    steps = recorded.steps if keeps_recorded and recorded.steps else defaults["steps"]
    guidance = recorded.guidance if keeps_recorded and recorded.guidance is not None else defaults["guidance"]
    scheduler = recorded.scheduler if keeps_recorded and recorded.scheduler in SCHEDULER_CLASSES else defaults["scheduler"]
    lora_paths, lora_weights, dropped_loras = _existing_loras(recorded.lora) if keeps_recorded else (None, None, False)
    if dropped_loras:
        notices.append(NOTICE_LORA_MISSING)
    recorded_quantize = recorded.generation.get("quantize")
    quantize = recorded_quantize if keeps_recorded and defaults.get("supports_quantize", True) and recorded_quantize in web_config.quantize_options else None
    supports_negative = defaults.get("supports_negative_prompt", False)
    sharpening = sharpening_amounts(app_config)
    pre_sharpen = sharpening["existing_pre_upscale"]

    request = ImageGenerationRequest(
        **_source_request_fields(source, recorded),
        model_name=model_name,
        model_family=model_info.family,
        supports_negative_prompt=supports_negative,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        negative_prompt=recorded.negative_prompt if supports_negative else None,
        seed=recorded.seed or 0,
        steps=steps,
        guidance=guidance,
        scheduler=resolve_scheduler_class(scheduler, backend_name),
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
    )
    return UpscalePlan(request=request, model_ref=model_ref, option=option, notices=tuple(notices))


def plan_seedvr2_upscale(source: UpscaleSource, factor: int, *, setup: SeedVR2Setup, platform_key: str) -> UpscalePlan:
    """Build the SeedVR2 upscale request for *source* at *factor*.

    No image model is involved, so nothing is detected or defaulted from one: the request carries the source's
    recorded settings so the new file stays reusable. When the source records no seed, steps or guidance, the
    request holds class defaults and ``source_settings_known`` is False, so they are not saved as the original's.
    The plan's ``model_ref`` is the first repo the tier's weights come from on *platform_key*.

    Raises:
        ValueError: If *factor* is not offered.
    """
    if factor not in UPSCALE_FACTORS:
        raise ValueError(f"Upscale factor must be one of {list(UPSCALE_FACTORS)}, got {factor}.")
    option = next(option for option in seedvr2_options(source.width, source.height, setup.tier, budget_bytes=setup.budget_bytes) if option.factor == factor)
    recorded = source.settings or RecordedSettings()
    notices = (NOTICE_SETTINGS_UNKNOWN,) if source.settings is None else ()

    lora_paths, lora_weights, _ = _existing_loras(recorded.lora, keep_missing=True)
    settings_known = recorded.seed is not None and recorded.steps is not None and recorded.guidance is not None
    request = ImageGenerationRequest(
        **_source_request_fields(source, recorded),
        model_name=recorded.model,
        model_family=recorded.model_family or "unknown",
        supports_negative_prompt=recorded.negative_prompt is not None,
        lora_paths=lora_paths,
        lora_weights=lora_weights,
        negative_prompt=recorded.negative_prompt,
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
    )
    return UpscalePlan(request=request, model_ref=declared_repo_ids(setup.tier.model, platform_key)[0], option=option, notices=notices)


def plan_video_upscale(source: ExtendSource, recorded_config: Mapping[str, Any] | None, *, setup: SeedVR2Setup) -> VideoUpscalePlan:
    """Build the SeedVR2 upscale request for the checked video *source*, always 2×.

    The output size, batch size and memory warning follow the size rules of ``utils.upscale``; the pre-blur is the
    configured ``seedvr2.video_softness``. *recorded_config* is the settings embedded in the clip, carried on to
    the result; without it the job says the original settings are unknown.
    """
    option = seedvr2_video_option(source.width, source.height, setup.tier, budget_bytes=setup.budget_bytes)
    request = VideoFileUpscaleRequest(
        source=source,
        recorded_config=recorded_config,
        output_dir=str(source.path.parent),
        width=option.width,
        height=option.height,
        upscale_model=setup.tier.model,
        softness=setup.settings.video_softness,
        batch_size=video_upscale_batch_size(option.width, option.height, setup.tier, budget_bytes=setup.budget_bytes),
    )
    notices = () if recorded_config else (NOTICE_SETTINGS_UNKNOWN,)
    return VideoUpscalePlan(request=request, notices=notices)


def ltx_upscale_model(recorded_model: str | None, web_config: WebUiConfig) -> tuple[str, bool]:
    """Return the LTX-2.3 model that refines a video upscale, and whether it is the one the clip was made with.

    The clip's own model when it is a configured LTX-2.3 model, else the default video model when it is one, else the
    first configured LTX-2.3 model. LTX-2.5 and other families are passed over: the upscale is tuned on LTX-2.3's refine.

    Raises:
        ValueError: If no LTX-2.3 model is configured.
    """
    candidates = [name for name in web_config.video_model_options if _is_ltx23(name, web_config.app_config)]
    if recorded_model in candidates:
        return recorded_model, True
    choice = preferred_option(web_config.default_models.video, tuple(candidates))
    if choice is None:
        raise ValueError("Upscaling a video on this computer needs an LTX-2.3 video model, and none is configured.")
    return choice, False


def _is_ltx23(model_name: str, app_config: dict[str, Any]) -> bool:
    """Return whether *model_name* is an LTX model outside every version block and not a known LTX-2.5 repo, which makes it LTX-2.3."""
    model_ref = resolve_model_path(model_name, aliases=alias_map(app_config), platform_key=sys.platform)
    return detect_video_model(model_ref).family == "ltx" and ltx_version(model_name) is None and known_ltx_repo_version(model_ref) != "2.5"


def _recorded_ltx23(model_name: str, app_config: dict[str, Any]) -> bool:
    """Return whether a clip's recorded *model_name*, which may no longer be configured, is surely LTX-2.3.

    Unlike ``_is_ltx23``, a name that resolves to neither a known LTX repo nor a local folder is not trusted, since an
    ``ltx`` in the name alone does not tell the version.
    """
    model_ref = resolve_model_path(model_name, aliases=alias_map(app_config), platform_key=sys.platform)
    known = known_ltx_repo_version(model_ref)
    if known is not None:
        return known == "2.3" and ltx_version(model_name) is None
    return Path(model_ref).expanduser().is_dir() and _is_ltx23(model_name, app_config)


def plan_ltx_video_upscale(source: ExtendSource, recorded_config: Mapping[str, Any] | None, web_config: WebUiConfig) -> VideoUpscalePlan:
    """Build the LTX upscale request for the checked video *source*, always 2× (macOS).

    The refine runs on the clip's own LTX-2.3 model when it is configured (``ltx_upscale_model``), with the clip's
    recorded scene and prompt (empty when none was recorded; the scene alone for an extended clip), its seed and the
    configured refine steps. When the clip was made with LTX-2.3 it also keeps the clip's LoRAs; any that no longer
    exist locally, or all of them for a clip of another model, are left out with a notice.
    *recorded_config* is carried on to the result; without it the job says the original settings are unknown.

    Raises:
        ValueError: If the clip is off the LTX grid or no LTX-2.3 model is configured.
    """
    problem = ltx_upscale_problem(source.width, source.height, source.frame_count)
    if problem is not None:
        raise ValueError(f"{source.path.name} can't be upscaled: {problem}")
    app_config = web_config.app_config
    model_name, own_model = ltx_upscale_model(source.model, web_config)
    notices: list[str] = []
    if not recorded_config:
        notices.append(NOTICE_SETTINGS_UNKNOWN)
    elif not own_model and source.model is not None:
        notices.append(NOTICE_MODEL_NOT_LTX23 if source.model in web_config.video_model_options else NOTICE_MODEL_NOT_CONFIGURED)
    loras: tuple[tuple[str, float], ...] = ()
    if own_model or (source.model is not None and _recorded_ltx23(source.model, app_config)):
        loras, dropped_loras = _existing_video_loras(source.lora)
        if dropped_loras:
            notices.append(NOTICE_LORA_MISSING)
    elif source.lora:
        notices.append(NOTICE_LORA_OTHER_MODEL)
    recorded_seed = optional_int((recorded_config or {}).get("seed"))
    request = LtxVideoUpscaleRequest(
        source=source,
        recorded_config=recorded_config,
        output_dir=str(source.path.parent),
        width=source.width * VIDEO_UPSCALE_FACTOR,
        height=source.height * VIDEO_UPSCALE_FACTOR,
        model_name=model_name,
        model_ref=resolve_model_path(model_name, aliases=alias_map(app_config), platform_key=sys.platform),
        prompt=ltx_upscale_prompt(source.scene, source.prompt, extended=original_video_workflow(recorded_config) == "extend"),
        seed=recorded_seed if recorded_seed is not None else VIDEO_UPSCALE_SEED,
        steps=ltx_upscale_steps(app_config),
        loras=loras,
    )
    return VideoUpscalePlan(request=request, notices=tuple(notices))


def _source_request_fields(source: UpscaleSource, recorded: RecordedSettings) -> dict[str, Any]:
    """Return the request fields both upscale methods take from *source* and its recorded settings."""
    # An edit's prompt says what to change, not what the picture shows: refining it with that text would redraw the change.
    prompt = "" if recorded.workflow == "edit" else recorded.prompt or ""
    return {
        "backend": None,
        "model": None,
        "prompt": prompt,
        "resolved_prompt": (source.rendered_prompt if recorded.workflow != "edit" else None) or prompt or None,
        "ratio": recorded.ratio,
        "size": recorded.size,
        "width": source.width,
        "height": source.height,
        # Recorded only: the upscale workflow never loads the reference, but Reuse settings needs it.
        "image_path": recorded.image_path,
        "image_strength": recorded.image_strength if recorded.image_strength is not None else ImageGenerationRequest.image_strength,
        # Recorded only, like the start image: the upscale refines without references, but Reuse settings brings them back.
        "reference_paths": recorded.reference_images,
        "output_dir": str(Path(source.path).parent),
        "upscale_source": source.path,
        # A cut-out source stays a cut-out: the upscale works on a grey backdrop and cuts the result out again.
        "transparent": recorded.generation.get("transparent") is True,
        # An enhanced source keeps the user's own prompt and how it was enhanced, so the upscale still shows and reuses them.
        "prompt_input": recorded.prompt_input,
        "enhance_record": recorded.enhance,
    }


def _sharpens_after_refining(recorded: RecordedSettings) -> bool:
    """Return whether to end with a sharpen pass: yes, unless the source was recorded without sharpening.

    Files that predate the post-processing record (no ``generation`` block) are sharpened.
    """
    return not recorded.generation or "sharpen" in recorded.generation


def _existing_loras(lora: str | None, *, keep_missing: bool = False) -> tuple[list[str] | None, list[float] | None, bool]:
    """Parse recorded LoRAs and drop local files that no longer exist, unless *keep_missing*.

    Returns:
        ``(paths, weights, dropped)``; references that are not local files are kept, and unparseable text gives none.
    """
    if lora is None:
        return None, None, False
    try:
        entries = parse_lora_arg(lora)
    except ValueError:
        return None, None, False
    kept = entries if keep_missing else [(path, weight) for path, weight in entries if not _is_missing_local_file(path)]
    if not kept:
        return None, None, bool(entries)
    return [path for path, _ in kept], [weight for _, weight in kept], len(kept) != len(entries)


def _existing_video_loras(lora: str | None) -> tuple[tuple[tuple[str, float], ...], bool]:
    """Resolve a clip's recorded LoRAs to the local files that still exist.

    Returns:
        ``(loras, dropped)``: each kept LoRA as ``(path, weight)``, and whether any were left out (missing, remote or unreadable).
    """
    if not lora:
        return (), False
    try:
        entries = parse_lora_arg(lora)
    except ValueError:
        return (), True
    resolved = [(_lora_file(name), weight) for name, weight in entries]
    kept = tuple((path, weight) for path, weight in resolved if path is not None)
    return kept, len(kept) != len(entries)


def _lora_file(name: str) -> str | None:
    """Return the existing LoRA file *name* resolves to, or ``None``; a bare name not in the LoRA folder resolves to nothing."""
    path = resolve_lora_path(name)
    if path == name.strip() and not is_explicit_local_path(path):
        return None
    return path if is_existing_file(path) else None


def _is_missing_local_file(reference: str) -> bool:
    path = Path(reference).expanduser()
    return path.is_absolute() and not path.exists()


def upscale_size_label(plan: UpscalePlan) -> str:
    """Return the job's size line, e.g. ``2× → 1664×2432``, or ``2× creative → 1664×2432``."""
    creative = " creative" if plan.request.upscale_method == UPSCALE_METHOD_CREATIVE else ""
    return f"{plan.option.factor}×{creative} → {plan.option.width}×{plan.option.height}"


def video_upscale_size_label(plan: VideoUpscalePlan) -> str:
    """Return the video job's size line, e.g. ``2× → 3840×2160``."""
    return f"{VIDEO_UPSCALE_FACTOR}× → {plan.request.width}×{plan.request.height}"


def upscale_json_request(body: Any) -> tuple[str, int, str]:
    """Validate a ``POST /api/upscale`` body and return ``(asset_id, factor, method)``.

    ``method`` is optional and defaults to ``creative``; ``ltx`` is for videos on macOS.

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
