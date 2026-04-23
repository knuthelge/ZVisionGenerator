"""Load the declarative Web UI configuration for the maintained SPA."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.list_assets import list_loras, list_models, list_video_models
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model


_DEFAULT_VISIBLE_SECTIONS = (
    "image_generation",
    "video_generation",
    "lora_management",
    "gallery_summary",
)
_KNOWN_THEMES = frozenset({"dark", "light"})
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
    port: int
    theme: str
    startup_view: str
    gallery_page_size: int
    output_dir: str
    visible_sections: tuple[str, ...]
    default_models: WebUiDefaultModels
    image_model_options: tuple[str, ...]
    video_model_options: tuple[str, ...]
    lora_options: tuple[str, ...]
    image_ratios: tuple[str, ...]
    image_size_options: dict[str, tuple[str, ...]]
    video_ratios: tuple[str, ...]
    video_size_options: dict[str, tuple[str, ...]]
    scheduler_options: tuple[str, ...]
    quantize_options: tuple[int, ...] = (4, 8)


def load_web_config() -> WebUiConfig:
    """Load the declarative UI config layered over the package config."""
    app_config = load_config()
    ui_config = app_config.get("ui", {})
    if ui_config and not isinstance(ui_config, dict):
        raise ValueError("config 'ui' must be a mapping.")

    data_dir = get_ziv_data_dir()
    output_dir = _resolve_output_dir(ui_config, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_sections = _resolve_visible_sections(ui_config.get("visible_sections"))
    default_models = _resolve_default_models(ui_config.get("default_models"), app_config)
    image_model_options = _discover_model_options(app_config, data_dir, media_kind="image")
    video_model_options = _discover_model_options(app_config, data_dir, media_kind="video")
    lora_options = tuple(entry.name for entry in list_loras(data_dir))

    image_sizes = app_config.get("sizes", {})
    video_sizes = app_config.get("video_sizes", {}).get("ltx", {})

    return WebUiConfig(
        app_config=app_config,
        port=_coerce_positive_int(ui_config.get("port", 8080), "ui.port"),
        theme=_validate_choice(ui_config.get("theme", "dark"), _KNOWN_THEMES, "ui.theme"),
        startup_view=_validate_choice(ui_config.get("startup_view", "workspace"), _KNOWN_STARTUP_VIEWS, "ui.startup_view"),
        gallery_page_size=_coerce_positive_int(ui_config.get("gallery_page_size", 12), "ui.gallery_page_size"),
        output_dir=str(output_dir),
        visible_sections=visible_sections,
        default_models=default_models,
        image_model_options=image_model_options,
        video_model_options=video_model_options,
        lora_options=lora_options,
        image_ratios=tuple(image_sizes.keys()),
        image_size_options={ratio: tuple(size_map.keys()) for ratio, size_map in image_sizes.items()},
        video_ratios=tuple(video_sizes.keys()),
        video_size_options={ratio: tuple(size_map.keys()) for ratio, size_map in video_sizes.items()},
        scheduler_options=tuple(app_config.get("schedulers", {}).keys()),
    )


def _resolve_output_dir(ui_config: dict[str, Any], data_dir: Path) -> Path:
    configured = ui_config.get("output_dir")
    if configured is None:
        return data_dir / "outputs"
    output_dir = Path(str(configured)).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return output_dir


def _resolve_visible_sections(value: Any) -> tuple[str, ...]:
    if value is None:
        return _DEFAULT_VISIBLE_SECTIONS
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("config 'ui.visible_sections' must be a list of strings.")
    resolved = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
    return resolved or _DEFAULT_VISIBLE_SECTIONS


def _resolve_default_models(value: Any, app_config: dict[str, Any]) -> WebUiDefaultModels:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("config 'ui.default_models' must be a mapping.")

    aliases = app_config.get("model_aliases", {})
    image_default = value.get("image") if isinstance(value.get("image"), str) else None
    video_default = value.get("video") if isinstance(value.get("video"), str) else None

    if image_default is None and aliases:
        image_default = next((name for name in aliases if _is_supported_alias(name, aliases, "image")), None)
    if video_default is None and aliases:
        video_default = next((name for name in aliases if _is_supported_alias(name, aliases, "video")), None)

    return WebUiDefaultModels(image=image_default, video=video_default)


def _discover_model_options(app_config: dict[str, Any], data_dir: Path, *, media_kind: str) -> tuple[str, ...]:
    aliases = app_config.get("model_aliases", {})
    names: list[str] = []

    if media_kind == "image":
        names.extend(entry.name for entry in list_models(data_dir))
    else:
        names.extend(entry.name for entry in list_video_models(data_dir))

    for alias_name in aliases:
        if _is_supported_alias(alias_name, aliases, media_kind):
            names.append(alias_name)

    return tuple(dict.fromkeys(sorted(names)))


def _is_supported_alias(alias_name: str, aliases: dict[str, Any], media_kind: str) -> bool:
    try:
        resolved = resolve_model_path(alias_name, aliases=aliases, platform_key=sys.platform)
    except RuntimeError:
        return False

    if media_kind == "image":
        lowered = f"{alias_name} {resolved}".lower()
        if detect_video_model(resolved).family != "unknown":
            return False
        return any(token in lowered for token in ("flux", "klein", "zimage", "z-image", "zit"))

    try:
        return detect_video_model(resolved).family != "unknown"
    except Exception:
        return False


def _coerce_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"config '{field_name}' must be a positive integer.")
    return value


def _validate_choice(value: Any, choices: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"config '{field_name}' must be one of {sorted(choices)}.")
    return value
