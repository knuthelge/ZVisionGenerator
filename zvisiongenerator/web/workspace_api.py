"""Build shared SPA payloads for workspace and models routes."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_backend_name, supports_stored_quants
from zvisiongenerator.converters.list_assets import list_loras
from zvisiongenerator.backends.prompt_enhancer_session import is_model_downloaded
from zvisiongenerator.utils.config import resolve_defaults, resolve_enhancer_model, resolve_video_defaults
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.prompt_enhance import matrix_contract, resolve_enhance_ceiling
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.config import WebUiConfig, preferred_option
from zvisiongenerator.web.defaults import resolve_image_ratio_size_defaults, resolve_video_ratio_size_defaults
from zvisiongenerator.web.model_delete import installed_models_linking_to, model_delete_target
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry, declared_image_family, stored_quant_of
from zvisiongenerator.web.model_status import describe_model_status, memory_budget_bytes


_UNKNOWN_STATUS: dict[str, Any] = {"downloaded": None, "memory_fit": None}
_IMAGE_BOOTSTRAP_STRENGTH = 0.5
_IMAGE_BOOTSTRAP_POSTPROCESS = {
    # On with no amount: the backend uses the config's sharpening amounts unless the user sets one.
    "sharpen": True,
    "contrast": False,
    "saturation": False,
}
_IMAGE_BOOTSTRAP_UPSCALE = {
    "enabled": False,
    "factor": None,
    "denoise": None,
    "steps": None,
    "guidance": None,
    "sharpen": True,
    "save_pre": False,
}
_VIDEO_BOOTSTRAP_UPSCALE = {
    "enabled": False,
    "factor": 2,
    "steps": None,
}


def build_workspace_bootstrap_view(web_config: WebUiConfig) -> dict[str, Any]:
    """Resolve per-model bootstrap defaults using shared backend config authority."""
    image_default_model = preferred_option(web_config.default_models.image, web_config.image_model_options)
    video_default_model = preferred_option(web_config.default_models.video, web_config.video_model_options)
    image_defaults = {model_name: _build_image_bootstrap_defaults(model_name, web_config) for model_name in web_config.image_model_options}
    video_defaults = {model_name: _build_video_bootstrap_defaults(model_name, web_config) for model_name in web_config.video_model_options}
    return {
        "image_default_model": image_default_model,
        "video_default_model": video_default_model,
        "image_model_defaults": image_defaults,
        "video_model_defaults": video_defaults,
    }


def build_workspace_response(
    web_config: WebUiConfig,
    history_assets: list[dict[str, Any]],
    *,
    active_job: dict[str, Any] | None,
    queued_jobs: list[dict[str, Any]] | None = None,
    prompt_sources: list[str],
    default_prompt_source: str,
    prompt_file_contract: dict[str, Any],
    workflow_contract: dict[str, Any],
    build_bootstrap_view: Any = build_workspace_bootstrap_view,
) -> dict[str, Any]:
    """Build the workspace bootstrap payload consumed by the SPA."""
    form_view = build_bootstrap_view(web_config)
    image_default_model = form_view["image_default_model"]
    video_default_model = form_view["video_default_model"]
    image_model_defaults_map = form_view["image_model_defaults"]
    video_model_defaults_map = form_view["video_model_defaults"]

    status = _status_resolver(web_config, image_model_defaults_map)
    image_entries = {entry.name: entry for entry in web_config.image_inventory}
    video_entries = {entry.name: entry for entry in web_config.video_inventory}
    image_models = [{"id": name, "label": name, "type": "image", **status(image_entries.get(name), "image")} for name in web_config.image_model_options]
    video_models = [{"id": name, "label": name, "type": "video", **status(video_entries.get(name), "video")} for name in web_config.video_model_options]
    loras = [{"name": name, "path": str(Path(web_config.loras_dir) / f"{name}.safetensors")} for name in web_config.lora_options]
    image_defaults = image_model_defaults_map.get(image_default_model) or _build_image_bootstrap_defaults(image_default_model or "", web_config)
    video_defaults = video_model_defaults_map.get(video_default_model) or _build_video_bootstrap_defaults(video_default_model or "", web_config)

    return {
        "image_models": image_models,
        "video_models": video_models,
        "loras": loras,
        "history_assets": history_assets,
        "active_job": active_job,
        "queued_jobs": queued_jobs or [],
        "defaults": image_defaults,
        "video_defaults": video_defaults,
        "image_model_defaults": image_model_defaults_map,
        "video_model_defaults": video_model_defaults_map,
        "current_image_model": image_default_model,
        "current_video_model": video_default_model,
        "output_dir": web_config.output_dir,
        "quantize_options": list(web_config.quantize_options),
        "image_ratios": list(web_config.image_ratios),
        "video_ratios": list(web_config.video_ratios),
        "image_size_options": {ratio: list(options) for ratio, options in web_config.image_size_options.items()},
        "video_size_options": {ratio: list(options) for ratio, options in web_config.video_size_options.items()},
        "image_size_dimensions": {ratio: {size: list(wh) for size, wh in sizes.items()} for ratio, sizes in web_config.image_size_dimensions.items()},
        "video_size_dimensions": {ratio: {size: list(wh) for size, wh in sizes.items()} for ratio, sizes in web_config.video_size_dimensions.items()},
        "scheduler_options": list(web_config.scheduler_options),
        "prompt_sources": prompt_sources,
        "default_prompt_source": default_prompt_source,
        "prompt_file": prompt_file_contract,
        "workflow_contract": workflow_contract,
        "prompt_enhancer": build_prompt_enhancer_contract(web_config),
        "config": {
            "gallery_page_size": web_config.gallery_page_size,
            "startup_view": web_config.startup_view,
            "output_dir": web_config.output_dir,
            "default_models": {
                "image": web_config.default_models.image,
                "video": web_config.default_models.video,
            },
        },
    }


def build_prompt_enhancer_contract(web_config: WebUiConfig, *, downloaded: Callable[[str, str | None], bool] = is_model_downloaded) -> dict[str, Any]:
    """Describe the prompt enhancer for the SPA: option matrix, effective model, and download state."""
    app_config = web_config.app_config
    section = app_config.get("prompt_enhancer") or {}
    sizes = section.get("download_size_label") or {}
    size_label = sizes.get(sys.platform) if isinstance(sizes, dict) else None
    contract: dict[str, Any] = {
        "matrix": matrix_contract(),
        "model": None,
        "revision": None,
        "downloaded": False,
        "download_size_label": size_label if isinstance(size_label, str) else None,
        "default_max_words": resolve_enhance_ceiling(app_config, family=None, mode="image"),
        "error": None,
    }
    try:
        repo, revision = resolve_enhancer_model(app_config, platform_key=sys.platform)
    except ValueError as exc:
        contract["error"] = str(exc)
        return contract
    contract["model"], contract["revision"] = repo, revision
    contract["downloaded"] = downloaded(repo, revision)
    # The size label describes the built-in default; a custom model's size is unknown.
    if contract["downloaded"] or section.get("user_model"):
        contract["download_size_label"] = None
    return contract


def build_models_response(
    web_config: WebUiConfig,
    *,
    token_var: str | None,
    image_defaults_for: Callable[[str, WebUiConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the models inventory payload from the inventory discovered with the Web UI config.

    Args:
        web_config: The Web UI config, including its discovered model inventory.
        token_var: The HuggingFace token environment variable in use, if any.
        image_defaults_for: Resolves one image model's bootstrap defaults; only its ``supports_quantize`` is used.
    """
    data_dir = Path(web_config.data_dir)
    image_defaults_for = image_defaults_for or _build_image_bootstrap_defaults
    # Quantize levels only shape memory estimates, so skip resolving them where there is no budget (off macOS).
    image_defaults = {entry.name: image_defaults_for(entry.name, web_config) for entry in web_config.image_inventory} if memory_budget_bytes() else {}
    find_local_dir = functools.cache(find_local_model_dir)
    status = _status_resolver(web_config, image_defaults, find_local_dir)
    delete_info = _delete_info_resolver(data_dir / "models", find_local_dir)
    loras = [{"name": lora.name, "file_size_mb": lora.file_size_mb, "size_label": f"{lora.file_size_mb} MB"} for lora in list_loras(data_dir)]
    return {
        "models_dir": web_config.models_dir,
        "loras_dir": web_config.loras_dir,
        "image_models": [
            {
                "name": entry.name,
                "family": entry.family,
                "size_label": entry.size or "Unknown",
                "source": entry.source,
                "stored_quant": _stored_quant_payload(entry),
                **status(entry, "image"),
                "delete": delete_info(entry),
            }
            for entry in web_config.image_inventory
        ],
        "video_models": [
            {"name": entry.name, "family": entry.family, "supports_i2v": entry.supports_i2v, "source": entry.source, **status(entry, "video"), "delete": delete_info(entry)}
            for entry in web_config.video_inventory
        ],
        "loras": loras,
        "stored_quants_supported": supports_stored_quants(),
        "huggingface_configured": token_var is not None,
        "huggingface_token_env_var": token_var,
    }


def _stored_quant_payload(entry: ImageInventoryEntry) -> dict[str, Any] | None:
    """Describe an installed stored quant (its base model and bits), or ``None`` for other models."""
    stored = stored_quant_of(entry)
    return None if stored is None else {"base_model": stored[0], "bits": stored[1]}


def _status_resolver(
    web_config: WebUiConfig,
    image_model_defaults: dict[str, dict[str, Any]],
    find_local_dir: Callable[[str], Path | None] | None = None,
) -> Callable[[ImageInventoryEntry | VideoInventoryEntry | None, str], dict[str, Any]]:
    """Return a per-request resolver for each model's ``downloaded``/``memory_fit`` fields.

    Image quantize levels come from the same bootstrap defaults that drive the workspace quantize picker, and
    download lookups are memoised for the request (LTX MLX models share one Gemma text-encoder lookup).
    """
    budget = memory_budget_bytes()
    find_local_dir = find_local_dir or functools.cache(find_local_model_dir)

    def _status(entry: ImageInventoryEntry | VideoInventoryEntry | None, kind: str) -> dict[str, Any]:
        if entry is None:
            return dict(_UNKNOWN_STATUS)
        quantize_options = _image_quantize_levels(web_config, image_model_defaults.get(entry.name, {})) if kind == "image" else ()
        try:
            return describe_model_status(entry.resolved_path, kind=kind, quantize_options=quantize_options, budget_bytes=budget, find_local_dir=find_local_dir)
        except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
            warnings.warn(f"Could not determine status for model '{entry.name}': {exc}", stacklevel=2)
            return dict(_UNKNOWN_STATUS)

    return _status


def _delete_info_resolver(models_dir: Path, find_local_dir: Callable[[str], Path | None] = find_local_model_dir) -> Callable[[ImageInventoryEntry | VideoInventoryEntry], dict[str, Any] | None]:
    """Return a resolver for each model's ``delete`` field: what deleting it removes, or ``None`` when it cannot be deleted.

    HuggingFace downloads list the installed models linking into them (``linked_by``), since deleting the download
    breaks those models.
    """

    def _delete_info(entry: ImageInventoryEntry | VideoInventoryEntry) -> dict[str, Any] | None:
        target = model_delete_target(entry, models_dir, find_local_dir=find_local_dir)
        if target is None or not (target.path.exists() or target.path.is_symlink()):
            return None
        linked_by = list(installed_models_linking_to(target.path, models_dir)) if target.kind == "huggingface" else []
        stored_quants = [path.name for path in target.stored_quants]
        return {"kind": target.kind, "repo_id": target.repo_id, "linked_by": linked_by, "stored_quants": stored_quants}

    return _delete_info


def _image_quantize_levels(web_config: WebUiConfig, capabilities: dict[str, Any]) -> tuple[int, ...]:
    """Return the quantize levels offered for an image model: one rule for the picker and the memory estimates."""
    return tuple(web_config.quantize_options) if capabilities.get("supports_quantize", True) else ()


def _resolve_image_bootstrap_dimensions(app_config: dict[str, Any], ratio: str, size: str) -> dict[str, int]:
    dims = app_config.get("sizes", {}).get(ratio, {}).get(size, {})
    return {
        "width": dims.get("width", 1024),
        "height": dims.get("height", 1024),
    }


def _dimension_is_supported(value: int, *, minimum: int, maximum: int | None, step: int) -> bool:
    return value >= minimum and (maximum is None or value <= maximum) and value % step == 0


def _is_supported_image_preset(app_config: dict[str, Any], ratio: str, size: str, defaults: dict[str, Any]) -> bool:
    dims = _resolve_image_bootstrap_dimensions(app_config, ratio, size)
    minimum = int(defaults.get("dimension_min", 16))
    maximum = defaults.get("dimension_max", None)
    step = int(defaults.get("dimension_step", 16))
    return _dimension_is_supported(dims["width"], minimum=minimum, maximum=maximum, step=step) and _dimension_is_supported(dims["height"], minimum=minimum, maximum=maximum, step=step)


def _pick_supported_image_bootstrap_preset(web_config: WebUiConfig, defaults: dict[str, Any], preferred_ratio: str, preferred_size: str) -> tuple[str, str]:
    app_config = web_config.app_config
    ratios = list(getattr(web_config, "image_ratios", ()))
    if preferred_ratio in ratios:
        ratio_candidates = [preferred_ratio, *[ratio for ratio in ratios if ratio != preferred_ratio]]
    else:
        ratio_candidates = ratios

    for ratio in ratio_candidates:
        size_options = list(getattr(web_config, "image_size_options", {}).get(ratio, ()))
        if not size_options:
            continue
        if ratio == preferred_ratio and preferred_size in size_options:
            preferred_index = size_options.index(preferred_size)
            candidate_indices = [preferred_index]
            for offset in range(1, len(size_options)):
                lower = preferred_index - offset
                upper = preferred_index + offset
                if lower >= 0:
                    candidate_indices.append(lower)
                if upper < len(size_options):
                    candidate_indices.append(upper)
            ordered_sizes = [size_options[index] for index in candidate_indices]
        else:
            ordered_sizes = size_options

        for size in ordered_sizes:
            if _is_supported_image_preset(app_config, ratio, size, defaults):
                return ratio, size

    return preferred_ratio, preferred_size


def _resolve_video_bootstrap_family(app_config: dict[str, Any], family: str | None) -> str:
    if family and family != "unknown":
        return family
    video_presets = app_config.get("video_model_presets", {})
    if "ltx" in video_presets:
        return "ltx"
    return next(iter(video_presets), "ltx")


def _default_video_max_steps(app_config: dict[str, Any], family: str) -> int | None:
    value = app_config.get("video_model_presets", {}).get(family, {}).get("default_steps")
    return value if isinstance(value, int) else None


def _image_bootstrap_postprocess() -> dict[str, Any]:
    return dict(_IMAGE_BOOTSTRAP_POSTPROCESS)


def _image_bootstrap_upscale() -> dict[str, Any]:
    return dict(_IMAGE_BOOTSTRAP_UPSCALE)


def _video_bootstrap_upscale() -> dict[str, Any]:
    return dict(_VIDEO_BOOTSTRAP_UPSCALE)


def _build_image_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    preferred_ratio, preferred_size = resolve_image_ratio_size_defaults(web_config)
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        declared = declared_image_family(app_config, model_name)
        if declared is not None:
            model_info = ImageModelInfo(family=declared, is_distilled=False, size=None)
        else:
            model_info = detect_image_model(resolved_model)
        defaults = resolve_defaults(model_info, app_config, {}, get_backend_name())
        enhance_max_words = resolve_enhance_ceiling(app_config, family=model_info.family, mode="image")
    except Exception:
        enhance_max_words = resolve_enhance_ceiling(app_config, family=None, mode="image")
        defaults = {
            "steps": app_config.get("generation", {}).get("default_steps", 10),
            "guidance": app_config.get("generation", {}).get("default_guidance", 3.5),
            "scheduler": None,
            "supports_negative_prompt": False,
            "supports_img2img": True,
            "supports_upscale": True,
            "supports_json_prompt": False,
            "supports_first_sigma": False,
            "supports_scheduler": True,
            "dimension_min": 16,
            "dimension_max": None,
            "dimension_step": 16,
        }
    ratio, size = _pick_supported_image_bootstrap_preset(web_config, defaults, preferred_ratio, preferred_size)
    dims = _resolve_image_bootstrap_dimensions(app_config, ratio, size)
    return {
        "ratio": ratio,
        "size": size,
        "width": dims["width"],
        "height": dims["height"],
        "steps": defaults["steps"],
        "guidance": defaults["guidance"],
        "scheduler": defaults.get("scheduler"),
        "supports_negative_prompt": bool(defaults.get("supports_negative_prompt", False)),
        "supports_quantize": bool(_image_quantize_levels(web_config, defaults)),
        "quantize": None,
        "image_strength": _IMAGE_BOOTSTRAP_STRENGTH,
        "postprocess": _image_bootstrap_postprocess(),
        "upscale": _image_bootstrap_upscale(),
        "supports_img2img": bool(defaults.get("supports_img2img", True)),
        "supports_upscale": bool(defaults.get("supports_upscale", True)),
        "supports_json_prompt": bool(defaults.get("supports_json_prompt", False)),
        "supports_first_sigma": bool(defaults.get("supports_first_sigma", False)),
        "supports_scheduler": bool(defaults.get("supports_scheduler", True)),
        "dimension_min": int(defaults.get("dimension_min", 16)),
        "dimension_max": defaults.get("dimension_max", None),
        "dimension_step": int(defaults.get("dimension_step", 16)),
        "enhance_max_words": enhance_max_words,
    }


def _build_video_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    ratio, size = resolve_video_ratio_size_defaults(web_config)
    supports_i2v = False
    fps = 24
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        model_info = detect_video_model(resolved_model)
        family = _resolve_video_bootstrap_family(app_config, getattr(model_info, "family", None))
        supports_i2v = bool(getattr(model_info, "supports_i2v", False))
        fps_value = getattr(model_info, "default_fps", 24)
        fps = fps_value if isinstance(fps_value, int) else 24
    except Exception:
        family = _resolve_video_bootstrap_family(app_config, None)
    defaults = resolve_video_defaults(family, app_config, {"ratio": ratio, "size": size})
    return {
        "ratio": defaults.get("ratio", ratio),
        "size": defaults.get("size", size),
        "steps": defaults["steps"],
        "width": defaults["width"],
        "height": defaults["height"],
        "frame_count": defaults["num_frames"],
        "audio": True,
        "low_memory": True,
        "supports_i2v": supports_i2v,
        "supports_quantize": False,
        "quantize": None,
        "max_steps": _default_video_max_steps(app_config, family),
        "fps": fps,
        "upscale": _video_bootstrap_upscale(),
        "enhance_max_words": resolve_enhance_ceiling(app_config, family=family, mode="video"),
    }
