"""Build shared SPA payloads for workspace and models routes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_backend_name
from zvisiongenerator.converters.list_assets import list_loras
from zvisiongenerator.utils.config import resolve_defaults, resolve_video_defaults
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.config import WebUiConfig
from zvisiongenerator.web.defaults import resolve_image_ratio_size_defaults, resolve_video_ratio_size_defaults
from zvisiongenerator.web.model_inventory import declared_image_family, discover_image_inventory, discover_video_inventory


_IMAGE_BOOTSTRAP_STRENGTH = 0.5
_IMAGE_BOOTSTRAP_POSTPROCESS = {
    "sharpen": 0.8,
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
    image_default_model = _preferred_option(web_config.default_models.image, web_config.image_model_options)
    video_default_model = _preferred_option(web_config.default_models.video, web_config.video_model_options)
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
    prompt_sources: list[str],
    default_prompt_source: str,
    prompt_file_contract: dict[str, Any],
    workflow_contract: dict[str, Any],
    build_bootstrap_view: Any = build_workspace_bootstrap_view,
) -> dict[str, Any]:
    """Build the workspace bootstrap payload consumed by the SPA."""
    image_models = [{"id": name, "label": name, "type": "image"} for name in web_config.image_model_options]
    video_models = [{"id": name, "label": name, "type": "video"} for name in web_config.video_model_options]
    loras = [{"name": name, "path": str(Path(web_config.loras_dir) / f"{name}.safetensors")} for name in web_config.lora_options]

    form_view = build_bootstrap_view(web_config)
    image_default_model = form_view["image_default_model"]
    video_default_model = form_view["video_default_model"]
    image_model_defaults_map = form_view["image_model_defaults"]
    video_model_defaults_map = form_view["video_model_defaults"]
    image_defaults = image_model_defaults_map.get(image_default_model) or _build_image_bootstrap_defaults(image_default_model or "", web_config)
    video_defaults = video_model_defaults_map.get(video_default_model) or _build_video_bootstrap_defaults(video_default_model or "", web_config)

    return {
        "image_models": image_models,
        "video_models": video_models,
        "loras": loras,
        "history_assets": history_assets,
        "active_job": active_job,
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
        "scheduler_options": list(web_config.scheduler_options),
        "prompt_sources": prompt_sources,
        "default_prompt_source": default_prompt_source,
        "prompt_file": prompt_file_contract,
        "workflow_contract": workflow_contract,
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


def build_models_response(web_config: WebUiConfig, *, token_var: str | None) -> dict[str, Any]:
    """Build the models inventory payload from the authoritative backend inventory."""
    data_dir = Path(web_config.data_dir)
    image_inventory = discover_image_inventory(web_config.app_config, data_dir)
    video_inventory = discover_video_inventory(web_config.app_config, data_dir)
    loras = [{"name": lora.name, "file_size_mb": lora.file_size_mb, "size_label": f"{lora.file_size_mb} MB"} for lora in list_loras(data_dir)]
    return {
        "models_dir": web_config.models_dir,
        "loras_dir": web_config.loras_dir,
        "image_models": [{"name": entry.name, "family": entry.family, "size_label": entry.size or "Unknown", "source": entry.source} for entry in image_inventory],
        "video_models": [{"name": entry.name, "family": entry.family, "supports_i2v": entry.supports_i2v, "source": entry.source} for entry in video_inventory],
        "loras": loras,
        "huggingface_configured": token_var is not None,
        "huggingface_token_env_var": token_var,
    }


def _preferred_option(preferred: str | None, options: tuple[str, ...]) -> str | None:
    if preferred in options:
        return preferred
    return options[0] if options else None


def _resolve_image_bootstrap_dimensions(app_config: dict[str, Any], ratio: str, size: str) -> dict[str, int]:
    dims = app_config.get("sizes", {}).get(ratio, {}).get(size, {})
    return {
        "width": dims.get("width", 1024),
        "height": dims.get("height", 1024),
    }


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
    ratio, size = resolve_image_ratio_size_defaults(web_config)
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        declared = declared_image_family(app_config, model_name)
        if declared is not None:
            model_info = ImageModelInfo(family=declared, is_distilled=False, size=None)
        else:
            model_info = detect_image_model(resolved_model)
        defaults = resolve_defaults(model_info, app_config, {}, get_backend_name())
    except Exception:
        defaults = {
            "steps": app_config.get("generation", {}).get("default_steps", 10),
            "guidance": app_config.get("generation", {}).get("default_guidance", 3.5),
            "scheduler": None,
            "supports_negative_prompt": False,
            "supports_img2img": True,
            "supports_upscale": True,
            "supports_json_prompt": False,
            "supports_first_sigma": False,
            "dimension_min": 16,
            "dimension_max": None,
            "dimension_step": 16,
        }
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
        "supports_quantize": bool(web_config.quantize_options) and bool(defaults.get("supports_quantize", True)),
        "quantize": None,
        "image_strength": _IMAGE_BOOTSTRAP_STRENGTH,
        "postprocess": _image_bootstrap_postprocess(),
        "upscale": _image_bootstrap_upscale(),
        "supports_img2img": bool(defaults.get("supports_img2img", True)),
        "supports_upscale": bool(defaults.get("supports_upscale", True)),
        "supports_json_prompt": bool(defaults.get("supports_json_prompt", False)),
        "supports_first_sigma": bool(defaults.get("supports_first_sigma", False)),
        "dimension_min": int(defaults.get("dimension_min", 16)),
        "dimension_max": defaults.get("dimension_max", None),
        "dimension_step": int(defaults.get("dimension_step", 16)),
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
    }
