"""Config loading, deep-merge, and layered default resolution.

Loads config.yaml from the package via importlib.resources, with optional
user override from ~/.ziv/config.yaml (deep-merged).  Also provides
resolve_defaults() for the config layering precedence:

    CLI explicit flags > model preset variant > model preset family > global defaults
"""

from __future__ import annotations

import importlib.resources
from typing import Any

import yaml

from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.paths import get_ziv_data_dir


def load_config() -> dict[str, Any]:
    """Load config.yaml from package resources, with optional user override.

    Returns:
        Parsed config dict with all defaults and model presets.

    Raises:
        FileNotFoundError: If the bundled config.yaml cannot be located.
    """
    ref = importlib.resources.files("zvisiongenerator").joinpath("config.yaml")
    with importlib.resources.as_file(ref) as path:
        with open(path, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}") from e

    # Optional user override (ZIV_DATA_DIR/config.yaml)
    user_config = get_ziv_data_dir() / "config.yaml"
    if user_config.exists():
        with open(user_config, encoding="utf-8") as f:
            try:
                user = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}") from e
        if user and isinstance(user, dict):
            _deep_merge(config, user)

    # Validate that known sections have the right types after merge
    _EXPECTED_DICTS = (
        "platforms",
        "sizes",
        "generation",
        "sharpening",
        "upscale",
        "contrast",
        "saturation",
        "schedulers",
        "model_aliases",
        "model_presets",
        "video_sizes",
        "video_generation",
        "video_model_presets",
    )
    for section in _EXPECTED_DICTS:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping, got {type(config[section]).__name__}. Check your user config (~/.ziv/config.yaml) for overrides.")

    # Validate nested config values that runner.py indexes directly
    gen = config.get("generation", {})
    if not isinstance(gen.get("default_steps"), int):
        raise ValueError("config 'generation.default_steps' must be an integer.")
    if not isinstance(gen.get("default_guidance"), (int, float)):
        raise ValueError("config 'generation.default_guidance' must be a number.")
    for ratio_key, scales in config.get("sizes", {}).items():
        if not isinstance(scales, dict):
            raise ValueError(f"config 'sizes.{ratio_key}' must be a mapping of size \u2192 dimensions.")
        for size_key, dims in scales.items():
            if not isinstance(dims, dict):
                raise ValueError(f"config 'sizes.{ratio_key}.{size_key}' must be a mapping with 'width' and 'height'.")
            if not isinstance(dims.get("width"), int):
                raise ValueError(f"config 'sizes.{ratio_key}.{size_key}.width' must be an integer.")
            if not isinstance(dims.get("height"), int):
                raise ValueError(f"config 'sizes.{ratio_key}.{size_key}.height' must be an integer.")

    # Validate default_ratio and default_size reference valid entries
    gen = config.get("generation", {})
    sizes = config.get("sizes", {})
    default_ratio = gen.get("default_ratio")
    default_size = gen.get("default_size")

    if default_ratio is not None and default_ratio not in sizes:
        raise ValueError(f"config 'generation.default_ratio' value '{default_ratio}' is not a valid ratio. Valid: {list(sizes.keys())}")
    if default_ratio is not None and default_size is not None:
        if default_size not in sizes.get(default_ratio, {}):
            raise ValueError(f"config 'generation.default_size' value '{default_size}' is not a valid size for ratio '{default_ratio}'. Valid: {list(sizes.get(default_ratio, {}).keys())}")

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* (mutates *base*).

    Dict values are merged recursively; all other types are replaced.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_variant_key(model_info: ImageModelInfo) -> str | None:
    """Determine the variant key for preset lookup.

    Returns:
        ``"distilled"`` or ``"base"`` for flux2_klein, ``None`` otherwise.
    """
    if model_info.family == "flux2_klein":
        return "distilled" if model_info.is_distilled else "base"
    return None


def resolve_defaults(
    model_info: ImageModelInfo,
    config: dict,
    cli_overrides: dict[str, Any],
    backend_name: str,
) -> dict[str, Any]:
    """Resolve effective config using layered precedence.

    Precedence (highest → lowest):
        1. CLI explicit flags  (``cli_overrides``)
        2. Model preset variant
        3. Model preset family
        4. Global defaults

    Args:
        model_info: Detected model metadata.
        config: Loaded config.yaml dict.
        cli_overrides: Only explicitly-provided CLI flags (not argparse defaults).
        backend_name: ``"mflux"`` or ``"diffusers"`` — for scheduler default lookup.

    Returns:
        Dict with resolved ``steps``, ``guidance``, and ``scheduler`` values.
    """
    preset = config.get("model_presets", {}).get(model_info.family, {})

    # Start with global defaults
    effective: dict[str, Any] = {
        "steps": config["generation"]["default_steps"],
        "guidance": config["generation"]["default_guidance"],
        "scheduler": None,
        "supports_negative_prompt": preset.get("supports_negative_prompt", False),
    }

    # Layer family defaults
    if "default_steps" in preset:
        effective["steps"] = preset["default_steps"]
    if "default_guidance" in preset:
        effective["guidance"] = preset["default_guidance"]

    # Layer variant defaults
    variant_key = get_variant_key(model_info)
    if variant_key and "variants" in preset:
        variant = preset["variants"].get(variant_key, {})
        if "default_steps" in variant:
            effective["steps"] = variant["default_steps"]
        if "default_guidance" in variant:
            effective["guidance"] = variant["default_guidance"]

    # Layer scheduler default (keyed by backend name, NOT platform)
    sched_defaults = preset.get("default_scheduler", {})
    if not isinstance(sched_defaults, dict):
        sched_defaults = {}
    effective["scheduler"] = sched_defaults.get(backend_name)

    # CLI explicit flags override everything
    for key, value in cli_overrides.items():
        if value is not None:
            effective[key] = value

    return effective


def validate_scheduler(scheduler_name: str | None, config: dict[str, Any]) -> None:
    """Check that *scheduler_name* is a known scheduler in the config.

    Args:
        scheduler_name: Scheduler name to validate, or ``None`` (no-op).
        config: Loaded config.yaml dict.

    Raises:
        ValueError: If the scheduler name is not ``None`` and not listed
            under ``config["schedulers"]``.
    """
    if scheduler_name is None:
        return
    known = config.get("schedulers", {})
    if scheduler_name not in known:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'. Valid options: {list(known.keys())}")


def resolve_video_defaults(
    model_family: str,
    config: dict,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Resolve effective video config using layered precedence.

    Precedence (highest to lowest):
        1. CLI explicit dimension flags (width/height/num_frames)
        2. Ratio + size preset lookup
        3. Video model preset
        4. Video global defaults

    Args:
        model_family: Detected video model family ("ltx").
        config: Loaded config.yaml dict.
        cli_overrides: Only explicitly-provided CLI flags.  May include
            ``ratio`` and ``size`` to select a preset, plus ``width``,
            ``height``, ``num_frames``, and ``steps`` to override
            individual values.

    Returns:
        Dict with resolved steps, width, height, num_frames, ratio, size.
    """
    vgen = config.get("video_generation", {})
    vsizes = config.get("video_sizes", {})
    vpresets = config.get("video_model_presets", {})

    preset = vpresets.get(model_family, {})

    # Determine ratio and size (CLI override > config default)
    ratio = cli_overrides.get("ratio") or vgen.get("default_ratio", "16:9")
    size = cli_overrides.get("size") or vgen.get("default_size", "m")

    # Look up dimensions from video_sizes[family][ratio][size]
    family_sizes = vsizes.get(model_family, {})
    ratio_sizes = family_sizes.get(ratio, {})
    size_entry = ratio_sizes.get(size, {})

    effective: dict[str, Any] = {
        "steps": preset.get("default_steps", 8),
        "width": size_entry.get("width", 704),
        "height": size_entry.get("height", 448),
        "num_frames": size_entry.get("frames", 49),
        "ratio": ratio,
        "size": size,
    }

    # CLI explicit flags override dimensions (ratio/size already consumed above)
    for key, value in cli_overrides.items():
        if value is not None and key not in ("ratio", "size"):
            effective[key] = value

    return effective
