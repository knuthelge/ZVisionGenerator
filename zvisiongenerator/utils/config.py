"""Config loading, deep-merge, and layered default resolution.

Loads config.yaml from the package via importlib.resources, with optional
user override from ~/.ziv/config.yaml (deep-merged).  Also provides
resolve_defaults() for the config layering precedence:

    CLI explicit flags > model preset variant > model preset family > global defaults
"""

from __future__ import annotations

import importlib.resources
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.upscale import max_megapixels


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

    _resolve_config_references(config)

    # Validate that known sections have the right types after merge
    _EXPECTED_DICTS = (
        "sizes",
        "generation",
        "sharpening",
        "upscale",
        "contrast",
        "saturation",
        "schedulers",
        "platforms",
        "model_aliases",
        "model_presets",
        "video_sizes",
        "video_generation",
        "video_model_presets",
        "prompt_enhancer",
    )
    for section in _EXPECTED_DICTS:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping, got {type(config[section]).__name__}. Check your user config (~/.ziv/config.yaml) for overrides.")

    # Reject a bad upscale.max_megapixels at load time rather than on every gallery page.
    max_megapixels(config)
    for key, amount in sharpening_amounts(config).items():
        validate_sharpen_amount(amount, name=f"config 'sharpening.{key}'")

    platforms = config.get("platforms", {})
    for key, value in platforms.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("config 'platforms' must map platform keys to human-readable labels.")

    aliases = config.get("model_aliases", {})
    for alias_name, alias_value in aliases.items():
        if isinstance(alias_value, str):
            continue
        if not isinstance(alias_value, dict):
            raise ValueError(f"config 'model_aliases.{alias_name}' must be a string or mapping.")
        for platform_key, platform_value in alias_value.items():
            if not isinstance(platform_key, str):
                raise ValueError(f"config 'model_aliases.{alias_name}' platform keys must be strings.")
            if isinstance(platform_value, str):
                continue
            if not isinstance(platform_value, dict):
                raise ValueError(f"config 'model_aliases.{alias_name}.{platform_key}' must be a string or mapping with a message.")
            message = platform_value.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"config 'model_aliases.{alias_name}.{platform_key}.message' must be a non-empty string.")

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

    video_sizes = config.get("video_sizes", {})
    for ratio_key, scales in video_sizes.items():
        if not isinstance(scales, dict):
            raise ValueError(f"config 'video_sizes.{ratio_key}' must be a mapping of size → dimensions.")
        for size_key, dims in scales.items():
            if not isinstance(dims, dict):
                raise ValueError(f"config 'video_sizes.{ratio_key}.{size_key}' must be a mapping with 'width', 'height', and 'frames'.")
            if not isinstance(dims.get("width"), int):
                raise ValueError(f"config 'video_sizes.{ratio_key}.{size_key}.width' must be an integer.")
            if not isinstance(dims.get("height"), int):
                raise ValueError(f"config 'video_sizes.{ratio_key}.{size_key}.height' must be an integer.")
            if not isinstance(dims.get("frames"), int):
                raise ValueError(f"config 'video_sizes.{ratio_key}.{size_key}.frames' must be an integer.")

    vgen = config.get("video_generation", {})
    default_video_ratio = vgen.get("default_ratio")
    default_video_size = vgen.get("default_size")
    if default_video_ratio is not None and default_video_ratio not in video_sizes:
        raise ValueError(f"config 'video_generation.default_ratio' value '{default_video_ratio}' is not a valid ratio. Valid: {list(video_sizes.keys())}")
    if default_video_ratio is not None and default_video_size is not None:
        if default_video_size not in video_sizes.get(default_video_ratio, {}):
            raise ValueError(
                f"config 'video_generation.default_size' value '{default_video_size}' is not a valid size for ratio '{default_video_ratio}'. Valid: {list(video_sizes.get(default_video_ratio, {}).keys())}"
            )

    return config


def _resolve_config_references(config: dict[str, Any]) -> None:
    """Resolve ${path.to.value} references inside the merged config tree."""

    def _resolve_value(value: Any) -> Any:
        if isinstance(value, dict):
            for key, nested in value.items():
                value[key] = _resolve_value(nested)
            return value
        if isinstance(value, list):
            return [_resolve_value(item) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            resolved = _lookup_config_path(config, value[2:-1])
            return deepcopy(resolved)
        return value

    _resolve_value(config)


def _lookup_config_path(config: dict[str, Any], dotted_path: str) -> Any:
    """Look up a dotted config path from the merged config tree."""

    current: Any = config
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Config reference '${{{dotted_path}}}' could not be resolved.")
        current = current[key]
    return current


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


def model_capabilities(config: dict[str, Any], family: str) -> dict[str, Any]:
    """Return a model family's capability flags and dimension limits from its preset, with permissive defaults.

    These are family-level only (variants change steps and guidance), so they can be looked up by family alone.
    """
    preset = config.get("model_presets", {}).get(family, {})
    return {
        "supports_negative_prompt": preset.get("supports_negative_prompt", False),
        "supports_img2img": preset.get("supports_img2img", True),
        "supports_upscale": preset.get("supports_upscale", True),
        "supports_quantize": preset.get("supports_quantize", True),
        "supports_json_prompt": preset.get("supports_json_prompt", False),
        "supports_first_sigma": preset.get("supports_first_sigma", False),
        "dimension_min": preset.get("dimension_min", 16),
        "dimension_max": preset.get("dimension_max", None),
        "dimension_step": preset.get("dimension_step", 16),
    }


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
        Dict with resolved ``steps``, ``guidance``, ``scheduler``, and
        optional ``upscale_steps`` values.
    """
    preset = config.get("model_presets", {}).get(model_info.family, {})

    # Start with global defaults
    effective: dict[str, Any] = {
        "steps": config["generation"]["default_steps"],
        "guidance": config["generation"]["default_guidance"],
        "scheduler": None,
        "upscale_steps": None,
        **model_capabilities(config, model_info.family),
    }

    # Layer family defaults
    if "default_steps" in preset:
        effective["steps"] = preset["default_steps"]
    if "default_guidance" in preset:
        effective["guidance"] = preset["default_guidance"]
    preset_upscale = preset.get("upscale", {})
    if isinstance(preset_upscale, dict) and "default_upscale_steps" in preset_upscale:
        effective["upscale_steps"] = preset_upscale["default_upscale_steps"]

    # Layer variant defaults
    variant_key = get_variant_key(model_info)
    if variant_key and "variants" in preset:
        variant = preset["variants"].get(variant_key, {})
        if "default_steps" in variant:
            effective["steps"] = variant["default_steps"]
        if "default_guidance" in variant:
            effective["guidance"] = variant["default_guidance"]
        variant_upscale = variant.get("upscale", {})
        if isinstance(variant_upscale, dict) and "default_upscale_steps" in variant_upscale:
            effective["upscale_steps"] = variant_upscale["default_upscale_steps"]

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


def resolve_scheduler_class(scheduler: str | None, config: dict[str, Any], backend_name: str) -> str | None:
    """Resolve a scheduler name to its backend-specific class path from config, or the name itself."""
    if scheduler is None:
        return None
    sched_cfg = config.get("schedulers", {}).get(scheduler, {})
    return sched_cfg.get(f"{backend_name}_class", scheduler)


# Above ~1.67 the CAS filter's normaliser (1 + 4w) reaches zero in flat areas and the output breaks down.
MAX_SHARPEN_AMOUNT = 1.5


_DEFAULT_SHARPENING = {"normal": 1.0, "upscaled": 1.2, "pre_upscale": 0.8, "existing_pre_upscale": 0.0}


def sharpening_amounts(config: dict[str, Any]) -> dict[str, float]:
    """Return every ``sharpening`` amount, with defaults; ``existing_upscaled`` falls back to ``upscaled``."""
    amounts = {**_DEFAULT_SHARPENING, **config.get("sharpening", {})}
    amounts.setdefault("existing_upscaled", amounts["upscaled"])
    return amounts


def validate_sharpen_amount(amount: float, *, name: str = "Sharpen amount") -> None:
    """Raise ValueError when *amount* is outside 0..MAX_SHARPEN_AMOUNT."""
    if not 0 <= amount <= MAX_SHARPEN_AMOUNT:
        raise ValueError(f"{name} must be between 0 and {MAX_SHARPEN_AMOUNT:g}, got {amount:g}.")


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


def select_ratio_size_defaults(
    preferred_ratio: str | None,
    ratios: tuple[str, ...],
    size_options_map: dict[str, tuple[str, ...]],
    preferred_size: str | None,
    *,
    fallback_ratio: str,
    fallback_size: str,
) -> tuple[str, str]:
    """Resolve the nearest valid ratio and size from config-backed options."""
    ratio = preferred_ratio if preferred_ratio in ratios else (ratios[0] if ratios else fallback_ratio)
    size_options = size_options_map.get(ratio, ())
    size = preferred_size if preferred_size in size_options else (size_options[0] if size_options else fallback_size)
    return ratio, size


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

    # Look up dimensions from flat video_sizes[ratio][size]. Model family still selects video_model_presets.
    ratio_sizes = vsizes.get(ratio, {})
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


def resolve_upscale_steps(defaults: dict[str, Any], steps: int) -> int:
    """Return the upscale refinement steps when not set explicitly: the preset default, else ``max(1, steps // 2)``."""
    preset_steps = defaults.get("upscale_steps")
    return preset_steps if preset_steps is not None else max(1, steps // 2)


def model_reference(repo: str, revision: str | None) -> str:
    """Return ``repo`` or ``repo@revision`` (the inverse of :func:`split_model_revision`)."""
    return f"{repo}@{revision}" if revision else repo


def split_model_revision(value: str) -> tuple[str, str | None]:
    """Split ``REPO[@REVISION]`` into the repo (or local path) and an optional revision.

    An existing local path is never split, even when it contains ``@``.
    """
    if Path(value.strip()).expanduser().exists():
        return value.strip(), None
    repo, sep, revision = value.strip().rpartition("@")
    if not sep:
        return value.strip(), None
    if not repo.strip() or not revision.strip():
        raise ValueError(f"Invalid enhancer model '{value}'. Use REPO or REPO@REVISION.")
    return repo.strip(), revision.strip()


def resolve_enhancer_model(config: dict[str, Any], *, platform_key: str, cli_model: str | None = None) -> tuple[str, str | None]:
    """Return the prompt-enhancer ``(repo, revision)``: CLI > user override (``REPO[@REVISION]``) > platform default.

    Args:
        config: Loaded config mapping.
        platform_key: Platform key such as ``darwin`` or ``win32``.
        cli_model: Optional ``REPO[@REVISION]`` from ``--enhance-model``.

    Raises:
        ValueError: If no model is configured for the platform or *cli_model* is malformed.
    """
    if cli_model and cli_model.strip():
        return split_model_revision(cli_model)
    section = config.get("prompt_enhancer") or {}
    user_model = section.get("user_model")
    if isinstance(user_model, str) and user_model.strip():
        return split_model_revision(user_model)
    models = section.get("model") or {}
    repo = models.get(platform_key) if isinstance(models, dict) else models
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(f"No prompt enhancer model is configured for platform '{platform_key}'. Set prompt_enhancer.user_model.")
    revisions = section.get("revision") or {}
    revision = revisions.get(platform_key) if isinstance(revisions, dict) else revisions
    return repo.strip(), revision.strip() if isinstance(revision, str) and revision.strip() else None
