"""Build JSON views for Web UI configuration routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads import token_source
from zvisiongenerator.web.config import WebUiConfig
from zvisiongenerator.web.config_contract import build_writable_config_schema
from zvisiongenerator.web.defaults import build_image_size_labels, resolve_image_ratio_size_defaults


def build_api_config_response(web_config: WebUiConfig) -> dict[str, Any]:
    """Build a JSON-serializable AppConfig payload from backend config state."""
    config_view = build_config_view(web_config)
    return {
        "output_dir": web_config.output_dir,
        "log_level": "info",
        "ui": {
            "startup_view": web_config.startup_view,
            "gallery_page_size": web_config.gallery_page_size,
            "output_dir": web_config.output_dir,
            "default_models": {
                "image": web_config.default_models.image,
                "video": web_config.default_models.video,
            },
            "image_model_options": list(web_config.image_model_options),
            "video_model_options": list(web_config.video_model_options),
            "model_cache_dir": config_view["model_cache_dir"],
            "loras_dir": config_view["lora_dir"],
            "huggingface_token_configured": config_view["huggingface_token_configured"],
            "huggingface_token_source": config_view["huggingface_token_source"],
            "image_size_labels": config_view["image_size_labels"],
            "default_image_size": config_view["default_image_size"],
        },
        "writable_config": build_writable_config_schema(web_config),
        "models": {},
    }


def build_config_view(web_config: WebUiConfig) -> dict[str, Any]:
    """Build derived config fields that are read-only in the Web UI."""
    _default_ratio, default_size = resolve_image_ratio_size_defaults(web_config)
    image_size_labels = build_image_size_labels(web_config)
    source = token_source()
    model_cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface" / "hub")
    return {
        "image_size_labels": image_size_labels,
        "model_cache_dir": model_cache_dir,
        "lora_dir": web_config.loras_dir,
        "huggingface_token_configured": source is not None,
        "huggingface_token_source": source,
        "default_image_size": default_size,
    }
