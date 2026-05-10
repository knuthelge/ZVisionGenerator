"""Load the declarative Web UI configuration for the maintained SPA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zvisiongenerator.converters.list_assets import list_loras, list_models, list_video_models
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.config_contract import resolve_loras_dir, resolve_models_dir, resolve_output_dir
from zvisiongenerator.web.model_inventory import (
    discover_image_inventory,
    discover_video_inventory,
    inventory_names,
    resolve_default_inventory_name,
)


_DEFAULT_QUANTIZE_OPTIONS = (4, 8)
_KNOWN_STARTUP_VIEWS = frozenset({"workspace", "gallery", "config"})


@dataclass(frozen=True)
class WebUiDefaultModels:
    """Hold the default model names surfaced by the Web UI."""

    image: str | None = None
    video: str | None = None


@dataclass(frozen=True)
class WebUiConfig:
    """Typed Web UI settings surfaced to the maintained SPA and API."""

    app_config: dict[str, Any] = field(repr=False)
    startup_view: str
    gallery_page_size: int
    data_dir: str
    output_dir: str
    models_dir: str
    loras_dir: str
    default_models: WebUiDefaultModels
    image_model_options: tuple[str, ...]
    video_model_options: tuple[str, ...]
    lora_options: tuple[str, ...]
    image_ratios: tuple[str, ...]
    image_size_options: dict[str, tuple[str, ...]]
    video_ratios: tuple[str, ...]
    video_size_options: dict[str, tuple[str, ...]]
    scheduler_options: tuple[str, ...]
    quantize_options: tuple[int, ...] = _DEFAULT_QUANTIZE_OPTIONS


def load_web_config() -> WebUiConfig:
    """Load the declarative UI config layered over the package config."""
    app_config = load_config()
    ui_config = app_config.get("ui", {})
    if ui_config and not isinstance(ui_config, dict):
        raise ValueError("config 'ui' must be a mapping.")

    data_dir = get_ziv_data_dir()
    output_dir = resolve_output_dir(ui_config.get("output_dir"), data_dir=data_dir)
    image_inventory = discover_image_inventory(
        app_config,
        data_dir,
        list_installed=list_models,
        resolve_alias_path=resolve_model_path,
        detect_model=detect_image_model,
    )
    video_inventory = discover_video_inventory(
        app_config,
        data_dir,
        list_installed=list_video_models,
        resolve_alias_path=resolve_model_path,
        detect_model=detect_video_model,
    )

    default_models = _resolve_default_models(ui_config.get("default_models"), image_inventory, video_inventory)
    image_model_options = inventory_names(image_inventory)
    video_model_options = inventory_names(video_inventory)
    lora_options = tuple(entry.name for entry in list_loras(data_dir))

    image_sizes = app_config.get("sizes", {})
    video_sizes = app_config.get("video_sizes", {})

    return WebUiConfig(
        app_config=app_config,
        startup_view=_validate_choice(ui_config.get("startup_view", "workspace"), _KNOWN_STARTUP_VIEWS, "ui.startup_view"),
        gallery_page_size=_coerce_positive_int(ui_config.get("gallery_page_size", 12), "ui.gallery_page_size"),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        models_dir=str(resolve_models_dir(data_dir=data_dir)),
        loras_dir=str(resolve_loras_dir(data_dir=data_dir)),
        default_models=default_models,
        image_model_options=image_model_options,
        video_model_options=video_model_options,
        lora_options=lora_options,
        image_ratios=tuple(image_sizes.keys()),
        image_size_options={ratio: tuple(size_map.keys()) for ratio, size_map in image_sizes.items()},
        video_ratios=tuple(video_sizes.keys()),
        video_size_options={ratio: tuple(size_map.keys()) for ratio, size_map in video_sizes.items()},
        scheduler_options=tuple(app_config.get("schedulers", {}).keys()),
        quantize_options=_resolve_quantize_options(ui_config),
    )


def _resolve_quantize_options(ui_config: dict[str, Any]) -> tuple[int, ...]:
    value = ui_config.get("quantize_options")
    if value is None:
        return _DEFAULT_QUANTIZE_OPTIONS
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("config 'ui.quantize_options' must be a list of integers.")
    resolved = tuple(dict.fromkeys(item for item in value if item > 0))
    return resolved or _DEFAULT_QUANTIZE_OPTIONS


def _resolve_default_models(
    value: Any,
    image_inventory: tuple[Any, ...],
    video_inventory: tuple[Any, ...],
) -> WebUiDefaultModels:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("config 'ui.default_models' must be a mapping.")

    image_default = value.get("image") if isinstance(value.get("image"), str) else None
    video_default = value.get("video") if isinstance(value.get("video"), str) else None

    return WebUiDefaultModels(
        image=resolve_default_inventory_name(image_default, image_inventory),
        video=resolve_default_inventory_name(video_default, video_inventory),
    )


def _coerce_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"config '{field_name}' must be a positive integer.")
    return value


def _validate_choice(value: Any, choices: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"config '{field_name}' must be one of {sorted(choices)}.")
    return value
