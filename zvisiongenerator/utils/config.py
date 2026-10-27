"""Config loading, deep-merge, and layered default resolution.

Loads config.yaml from the package via importlib.resources, with optional
user override from ~/.ziv/config.yaml (deep-merged).  Also provides
resolve_defaults() for the config layering precedence:

    CLI explicit flags > model preset variant > model preset family > global defaults
"""

from __future__ import annotations

import importlib.resources
import warnings
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Final

import yaml

from zvisiongenerator.core.video_family import FrameRule, VideoCapabilities, VideoFamily
from zvisiongenerator.utils.h3_preset import HubFiles, h3_preset
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.ltx_converted import LTX_FAMILY_MODEL_TYPE
from zvisiongenerator.utils.paths import get_ziv_data_dir, parse_huggingface_repo_reference, split_huggingface_revision
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, seedvr2_settings
from zvisiongenerator.utils.upscale import max_megapixels

_YAML_LOADER: Final = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
"""libyaml's safe loader when PyYAML was built with it (about 8x faster on the packaged config), else the pure-Python one."""

EULER_ANCESTRAL: Final = "euler_ancestral"
LTX_SAMPLERS: Final = frozenset({EULER_ANCESTRAL})
"""The LTX denoising samplers a model version may select; the single definition of the names."""


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
                config = yaml.load(f, Loader=_YAML_LOADER)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}") from e

    # Optional user override (ZIV_DATA_DIR/config.yaml)
    user_config = get_ziv_data_dir() / "config.yaml"
    if user_config.exists():
        with open(user_config, encoding="utf-8") as f:
            try:
                user = yaml.load(f, Loader=_YAML_LOADER)
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
        "seedvr2",
        "contrast",
        "saturation",
        "schedulers",
        "platforms",
        "model_aliases",
        "ideogram4_instant",
        "moss_image",
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

    seedvr2_settings(config, MLX_WEIGHTS.tiers_key)
    seedvr2_settings(config, FP8_WEIGHTS.tiers_key)

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


def video_preset(config: dict[str, Any], family: str, model_name: str | None = None) -> dict[str, Any]:
    """Return the ``video_model_presets.<family>`` block with the version that lists *model_name* merged over it.

    A version is ``video_model_presets.<family>.versions.<id>``: an ``aliases`` list plus any keys of the family
    block. Its keys (minus ``aliases``) are deep-merged over the family block (minus ``versions``) the way user
    config is: dicts merge, every other value, ``None`` included, replaces. When no version lists *model_name*
    (or it is ``None``), the result is the family block alone. *config* is never mutated.

    Raises:
        ValueError: If ``versions`` is not a mapping of mappings, a version's ``aliases`` is not a list of
            non-blank strings, or an alias appears in two versions.
    """
    return _merged_preset(config, family, _video_version(config, family, model_name))


def _merged_preset(config: dict[str, Any], family: str, version: str | None) -> dict[str, Any]:
    """Return the ``video_model_presets.<family>`` block with version *version* merged over it (the family block alone for ``None``)."""
    preset = (config.get("video_model_presets") or {}).get(family) or {}
    merged = {key: deepcopy(value) for key, value in preset.items() if key != "versions"}
    if version is not None:
        overlay = {key: deepcopy(value) for key, value in preset["versions"][version].items() if key not in ("aliases", "model_type")}
        _deep_merge(merged, overlay)
    return merged


def ltx_version_for_type(config: dict[str, Any], model_type: str) -> str | None:
    """Return the id of the LTX version block a converter model type belongs to (its ``model_type``); ``None`` for ``ltx-2.3``, the family block.

    Raises:
        ValueError: If no version block names *model_type*.
    """
    if model_type == LTX_FAMILY_MODEL_TYPE:
        return None
    versions = ((config.get("video_model_presets") or {}).get("ltx") or {}).get("versions") or {}
    for version, block in versions.items():
        if isinstance(block, dict) and block.get("model_type") == model_type:
            return version
    raise ValueError(f"No video_model_presets.ltx.versions block has model_type '{model_type}'.")


def _converted_model_type(model_name: str) -> str | None:
    """Return the kind of the converted LTX folder *model_name* names (a folder in the models folder, or a path), or ``None`` when it is not one."""
    from zvisiongenerator.utils.h3_preset import is_folder_name
    from zvisiongenerator.utils.ltx_converted import read_converted_ltx

    folder = get_ziv_data_dir() / "models" / model_name if is_folder_name(model_name) else Path(model_name).expanduser()
    manifest = read_converted_ltx(folder) if folder.is_dir() else None
    return manifest["kind"] if manifest is not None else None


def _video_version(config: dict[str, Any], family: str, model_name: str | None) -> str | None:
    """Return the id of the version of *family* that lists *model_name* in its ``aliases``, or ``None``.

    A converted LTX folder (by name or path) belongs to the version whose ``model_type`` is its manifest's kind.
    Every version is validated, whatever *model_name* is.

    Raises:
        ValueError: If ``versions`` is not a mapping of mappings, a version's ``aliases`` is not a list of
            non-blank strings, or an alias appears in two versions.
    """
    versions = ((config.get("video_model_presets") or {}).get(family) or {}).get("versions")
    if versions is None:
        return None
    base = f"video_model_presets.{family}.versions"
    if not isinstance(versions, dict) or not all(isinstance(block, dict) for block in versions.values()):
        raise ValueError(f"{base} must be a mapping of version ids to mappings.")
    owners: dict[str, str] = {}
    for version, block in versions.items():
        aliases = block.get("aliases")
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
            raise ValueError(f"{base}.{version}.aliases must be a list of non-blank strings.")
        for alias in aliases:
            if alias in owners and owners[alias] != version:
                raise ValueError(f"{base}: alias '{alias}' is listed by both '{owners[alias]}' and '{version}'.")
            owners[alias] = version
    if model_name is None:
        return None
    if model_name in owners or family != "ltx":
        return owners.get(model_name)
    kind = _converted_model_type(model_name)
    if kind is None or kind == LTX_FAMILY_MODEL_TYPE:
        return None
    return next((version for version, block in versions.items() if block.get("model_type") == kind), None)


def ltx_version(config: dict[str, Any], model_name: str | None) -> str | None:
    """Return the id of the LTX version block that lists *model_name* in its ``aliases``, or ``None``.

    ``None`` means the model is not in any version block, which makes it LTX-2.3 (the family block): a ``None``
    model name, an LTX-2.3 alias, a repo id or a local path.

    Raises:
        ValueError: If ``versions`` is not a mapping of mappings, a version's ``aliases`` is not a list of
            non-blank strings, or an alias appears in two versions.
    """
    return _video_version(config, "ltx", model_name)


def video_family(config: dict[str, Any], family: str) -> VideoFamily | None:
    """Return the description of a video family from ``video_model_presets.<family>``, or ``None`` without that block.

    The family block is read as it stands: a version's overrides do not apply.

    Raises:
        ValueError: If a required key (``label``, ``default_steps``, ``frame_rule``, ``capabilities``,
            ``extend_frame_step``, ``carry_frames``) is missing or a value is malformed; the message names the key path.
    """
    preset = (config.get("video_model_presets") or {}).get(family)
    if preset is None:
        return None
    base = f"video_model_presets.{family}"
    if not isinstance(preset, dict):
        raise ValueError(f"{base} must be a mapping.")
    for key in ("label", "default_steps", "frame_rule", "capabilities", "extend_frame_step", "carry_frames"):
        if key not in preset:
            raise ValueError(f"{base}.{key} is required.")
    label = preset["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{base}.label must be a non-blank string.")
    default_steps = _positive_int(preset["default_steps"], f"{base}.default_steps")
    max_steps = None if preset.get("max_steps") is None else _positive_int(preset["max_steps"], f"{base}.max_steps")
    frame_rule = _frame_rule(preset["frame_rule"], f"{base}.frame_rule")
    return VideoFamily(
        name=family,
        label=label,
        frame_rule=frame_rule,
        capabilities=_video_capabilities(preset["capabilities"], f"{base}.capabilities"),
        platform_capabilities=_platform_capabilities(preset.get("platform_capabilities"), f"{base}.platform_capabilities"),
        default_steps=default_steps,
        max_steps=max_steps,
        step_options=_step_options(preset.get("step_options"), max_steps, f"{base}.step_options"),
        default_frames=_default_frames(preset.get("default_frames"), frame_rule, f"{base}.default_frames"),
        extend_frame_step=_positive_int(preset["extend_frame_step"], f"{base}.extend_frame_step"),
        carry_frames=_positive_int(preset["carry_frames"], f"{base}.carry_frames"),
        default_upscale_steps=_default_upscale_steps(preset.get("upscale"), default_steps, f"{base}.upscale"),
        extend_fps=None if preset.get("extend_fps") is None else _positive_int(preset["extend_fps"], f"{base}.extend_fps"),
        platform_carry_frames=_platform_carry_frames(preset.get("platform_carry_frames"), f"{base}.platform_carry_frames"),
    )


def _platform_carry_frames(value: Any, path: str) -> dict[str, int]:
    """Parse ``platform_carry_frames``: per platform, the frames an extension carries there."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    return {str(platform): _positive_int(frames, f"{path}.{platform}") for platform, frames in value.items()}


def _default_upscale_steps(value: Any, default_steps: int, path: str) -> int:
    """Return ``default_upscale_steps`` from a family's ``upscale`` block, or *default_steps* when it has none."""
    if value is None:
        return default_steps
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    if "default_upscale_steps" not in value:
        return default_steps
    return _positive_int(value["default_upscale_steps"], f"{path}.default_upscale_steps")


def _positive_int(value: Any, path: str) -> int:
    """Return *value* when it is an integer of at least 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be an integer of at least 1.")
    return value


def _mapping_with_keys(value: Any, allowed: tuple[str, ...], path: str) -> dict[str, Any]:
    """Return *value* when it is a mapping whose keys are all in *allowed*."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    for key in value:
        if key not in allowed:
            raise ValueError(f"{path}.{key} is not a known key (expected {', '.join(allowed)}).")
    return value


def _frame_rule(value: Any, path: str) -> FrameRule:
    """Parse a ``frame_rule`` mapping."""
    rule = _mapping_with_keys(value, ("alignment", "offset", "round"), path)
    for key in ("alignment", "offset", "round"):
        if key not in rule:
            raise ValueError(f"{path}.{key} is required.")
    alignment, offset, rounding = rule["alignment"], rule["offset"], rule["round"]
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
        raise ValueError(f"{path}.alignment must be an integer of at least 1.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError(f"{path}.offset must be an integer of at least 0.")
    if rounding not in ("nearest", "up"):
        raise ValueError(f"{path}.round must be 'nearest' or 'up'.")
    return FrameRule(alignment, offset, rounding)


def _video_capabilities(value: Any, path: str) -> VideoCapabilities:
    """Parse a ``capabilities`` mapping: every capability, each a boolean."""
    names = tuple(field.name for field in fields(VideoCapabilities))
    caps = _mapping_with_keys(value, names, path)
    for name in names:
        if name not in caps:
            raise ValueError(f"{path}.{name} is required.")
        if not isinstance(caps[name], bool):
            raise ValueError(f"{path}.{name} must be true or false.")
    return VideoCapabilities(**caps)


def _platform_capabilities(value: Any, path: str) -> dict[str, dict[str, bool]]:
    """Parse ``platform_capabilities``: per platform, the capabilities that differ there."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    names = tuple(field.name for field in fields(VideoCapabilities))
    parsed: dict[str, dict[str, bool]] = {}
    for platform, caps in value.items():
        caps = _mapping_with_keys(caps, names, f"{path}.{platform}")
        for name, flag in caps.items():
            if not isinstance(flag, bool):
                raise ValueError(f"{path}.{platform}.{name} must be true or false.")
        parsed[str(platform)] = dict(caps)
    return parsed


def _step_options(value: Any, max_steps: int | None, path: str) -> tuple[int, ...] | None:
    """Parse ``step_options``: ascending positive integers, none above *max_steps*."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list of integers.")
    options = tuple(_positive_int(option, f"{path}[{index}]") for index, option in enumerate(value))
    if any(later <= earlier for earlier, later in zip(options, options[1:])):
        raise ValueError(f"{path} must be in ascending order without repeats.")
    if max_steps is not None and options[-1] > max_steps:
        raise ValueError(f"{path} must not exceed max_steps ({max_steps}).")
    return options


def _default_frames(value: Any, rule: FrameRule, path: str) -> int | None:
    """Parse ``default_frames``: a frame count the family's rule accepts."""
    if value is None:
        return None
    frames = _positive_int(value, path)
    if frames < rule.alignment + rule.offset or (frames - rule.offset) % rule.alignment:
        raise ValueError(f"{path} must be a valid frame count ({rule.alignment}k+{rule.offset}, k >= 1).")
    return frames


def get_variant_key(model_info: ImageModelInfo) -> str | None:
    """Determine the variant key for preset lookup.

    Returns:
        ``"distilled"`` or ``"base"`` for flux2_klein, ``"turbo"`` or ``"sft"`` for moss, ``None`` otherwise.
    """
    if model_info.family == "flux2_klein":
        return "distilled" if model_info.is_distilled else "base"
    if model_info.family == "moss":
        return "turbo" if model_info.is_distilled else "sft"
    return None


def model_display_name(config: dict[str, Any], name: str) -> str:
    """Return how a model alias reads in the Web UI: its configured ``model_labels`` name, else the alias itself."""
    entry = (config.get("model_labels") or {}).get(name)
    label = entry.get("name") if isinstance(entry, dict) else None
    return label.strip() if isinstance(label, str) and label.strip() else name


def model_capabilities(config: dict[str, Any], family: str, variant: str | None = None) -> dict[str, Any]:
    """Return a model family's capability flags, dimension limits and negative prompt hint from its preset, with permissive defaults.

    The family's values apply to every variant; a capability key set in ``variants[variant]`` overrides them. Without
    *variant* the result is the family's own, which is what callers that only know the family look up.
    """
    preset = config.get("model_presets", {}).get(family, {})
    variant_preset = (preset.get("variants") or {}).get(variant, {}) if variant else {}

    def capability(key: str, default: Any) -> Any:
        return variant_preset.get(key, preset.get(key, default))

    return {
        "supports_negative_prompt": capability("supports_negative_prompt", False),
        "supports_img2img": capability("supports_img2img", True),
        "supports_upscale": capability("supports_upscale", True),
        "supports_quantize": capability("supports_quantize", True),
        "supports_scheduler": capability("supports_scheduler", True),
        "supports_json_prompt": capability("supports_json_prompt", False),
        "supports_first_sigma": capability("supports_first_sigma", False),
        "negative_prompt_hint": capability("negative_prompt_hint", None),
        "dimension_min": capability("dimension_min", 16),
        "dimension_max": capability("dimension_max", None),
        "dimension_step": capability("dimension_step", 16),
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

    variant_key = get_variant_key(model_info)

    # Start with global defaults
    effective: dict[str, Any] = {
        "steps": config["generation"]["default_steps"],
        "guidance": config["generation"]["default_guidance"],
        "scheduler": None,
        "upscale_steps": None,
        **model_capabilities(config, model_info.family, variant_key),
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


def validate_video_size_choice(config: dict[str, Any], ratio: str | None, size: str | None) -> None:
    """Check that the config's video size table offers *ratio* and *size*; a config without a table accepts any.

    Raises:
        ValueError: If the ratio or the size is not in the table.
    """
    video_sizes = config.get("video_sizes", {})
    if not video_sizes:
        return
    if ratio not in video_sizes:
        raise ValueError(f"Unknown ratio '{ratio}'. Valid: {list(video_sizes.keys())}")
    if size not in video_sizes.get(ratio, {}):
        raise ValueError(f"Unknown size '{size}' for ratio '{ratio}'. Valid: {list(video_sizes.get(ratio, {}).keys())}")


def resolve_video_defaults(
    model_family: str,
    config: dict,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Resolve effective video config using layered precedence.

    Precedence (highest to lowest):
        1. CLI explicit dimension flags (width/height/num_frames)
        2. Ratio + size preset lookup (the frame count is the family's ``default_frames`` when it sets one)
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
    size = cli_overrides.get("size") or vgen.get("default_size", "s")

    # Look up dimensions from flat video_sizes[ratio][size]. Model family still selects video_model_presets.
    ratio_sizes = vsizes.get(ratio, {})
    size_entry = ratio_sizes.get(size, {})

    # A preset without the key keeps the size table's count, and a partial preset (no label, rule, ...) is not parsed.
    family = video_family(config, model_family) if preset.get("default_frames") is not None else None
    default_frames = family.default_frames if family is not None else None
    effective: dict[str, Any] = {
        "steps": preset.get("default_steps", 8),
        "width": size_entry.get("width", 768),
        "height": size_entry.get("height", 448),
        "num_frames": default_frames if default_frames is not None else size_entry.get("frames", 121),
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


def ltx_latent_upsampler_source(config: dict[str, Any], model_name: str | None = None) -> tuple[str, str] | None:
    """Return the configured LTX latent upsampler as ``(repo_id, filename)``, or ``None`` when not configured.

    The upsampler is read from the version that lists *model_name*, else from the family block. The repo
    string may carry ``@revision``; it is returned as is.

    Raises:
        ValueError: If ``latent_upsampler`` is set but is not a mapping with non-blank string ``repo`` and ``filename``.
    """
    diffusers, path = _ltx_diffusers(config, model_name)
    if "latent_upsampler" not in diffusers:
        return None
    return _ltx_hub_file(diffusers["latent_upsampler"], f"{path}.latent_upsampler")


def ltx_transformer_source(config: dict[str, Any], model_name: str | None = None) -> tuple[str, str] | None:
    """Return the configured single-file LTX transformer checkpoint as ``(repo_id, filename)``, or ``None`` to use the repo's own.

    The transformer is read from the version that lists *model_name*, else from the family block. A ``null``
    entry (a user config turning the packaged one off, or a version using the repo's own) means the repo's own
    transformer.

    Raises:
        ValueError: If ``transformer`` is set but is not a mapping with non-blank string ``repo`` and ``filename``.
    """
    diffusers, path = _ltx_diffusers(config, model_name)
    if diffusers.get("transformer") is None:
        return None
    return _ltx_hub_file(diffusers["transformer"], f"{path}.transformer")


def ltx_companion_repos(config: dict[str, Any], repo_id: str) -> tuple[str, ...]:
    """Return the other Hugging Face repos the diffusers LTX backend downloads beside *repo_id*.

    These are the single-file transformer and the latent upsampler, as repo ids without a revision, when *repo_id*
    is the configured ``video_model_presets.ltx.diffusers.default_repo`` (at any revision); otherwise none.
    """
    diffusers = ((config.get("video_model_presets") or {}).get("ltx") or {}).get("diffusers") or {}
    default_repo = diffusers.get("default_repo")
    own = split_huggingface_revision(repo_id)[0]
    if not isinstance(default_repo, str) or split_huggingface_revision(default_repo)[0] != own:
        return ()
    sources = (ltx_transformer_source(config), ltx_latent_upsampler_source(config))
    companions = (split_huggingface_revision(source[0])[0] for source in sources if source is not None)
    return tuple(dict.fromkeys(companion for companion in companions if companion != own))


def _ltx_path(version: str | None) -> str:
    """Return the config key path of the LTX family block, or of its version *version*."""
    return "video_model_presets.ltx" + (f".versions.{version}" if version is not None else "")


def ltx_distilled_lora_source(config: dict[str, Any], version: str | None = None) -> tuple[str, str] | None:
    """Return Lightricks' distilled LoRA of LTX version *version* (``distilled_lora``) as ``(repo_id, filename)``, or ``None`` when not configured.

    LTX checkpoint conversion merges it into plain-dev checkpoints with ``--distill``. *version* is a version block
    id (:func:`ltx_version_for_type`); ``None`` is the family block, LTX-2.3. A version without its own entry has
    none: another version's LoRA never fits it.

    Raises:
        ValueError: If the entry is set but is not a mapping with non-blank string ``repo`` and ``filename``.
    """
    block = _ltx_block(config, version)
    if block.get("distilled_lora") is None:
        return None
    return _ltx_hub_file(block["distilled_lora"], f"{_ltx_path(version)}.distilled_lora")


def ltx_mlx_base_repo(config: dict[str, Any], bits: int, version: str | None = None) -> str:
    """Return the MLX LTX repo a converted checkpoint of *version* at *bits* takes its other components from on macOS (``mlx_base.q<bits>``).

    Raises:
        ValueError: If it is not set to a non-blank string (a version without its own entry has none).
    """
    repo = (_ltx_block(config, version).get("mlx_base") or {}).get(f"q{bits}")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(f"{_ltx_path(version)}.mlx_base.q{bits} must be a non-blank string.")
    return repo.strip()


def ltx_mlx_packs(config: dict[str, Any]) -> dict[str, HubFiles]:
    """Return the pinned MLX LTX packs of *config*, keyed by their ``owner/repo@revision`` reference, each with the files the app loads from it.

    Every ``mlx_base`` entry of the LTX family block and of its version blocks that carries a revision is a pack, and
    its block's ``mlx_files`` are its files. An entry without a revision is skipped: it has no fixed file set.

    Raises:
        ValueError: If ``mlx_base`` is not a mapping of non-blank strings, a pinned entry's block has no valid ``mlx_files``
            (a non-empty list of non-blank strings), or two blocks pin the same reference to different files.
    """
    ltx = (config.get("video_model_presets") or {}).get("ltx") or {}
    blocks = [(None, ltx), *((version, block) for version, block in (ltx.get("versions") or {}).items() if isinstance(block, dict))]
    packs: dict[str, HubFiles] = {}
    for version, block in blocks:
        path = _ltx_path(version)
        bases = block.get("mlx_base")
        if bases is None:
            continue
        if not isinstance(bases, dict) or not all(isinstance(ref, str) and ref.strip() for ref in bases.values()):
            raise ValueError(f"{path}.mlx_base must be a mapping of quantize levels to non-blank strings.")
        for ref in dict.fromkeys(ref.strip() for ref in bases.values()):
            reference = parse_huggingface_repo_reference(ref)
            if reference is None or reference.revision is None:
                continue
            files = block.get("mlx_files")
            if not isinstance(files, list) or not files or not all(isinstance(name, str) and name.strip() for name in files):
                raise ValueError(f"{path}.mlx_files must be a non-empty list of non-blank strings: the files of the pinned pack {ref}.")
            pack = HubFiles(reference.repo_id, reference.revision, tuple(name.strip() for name in files))
            if packs.setdefault(ref, pack) != pack:
                raise ValueError(f"{path}.mlx_files lists other files for {ref} than another LTX block.")
    return packs


def ltx_mlx_text_encoder(config: dict[str, Any]) -> str:
    """Return the Gemma 3 text encoder of the macOS LTX-2.3 packs (``mlx_text_encoder``) as a ``repo`` or ``repo@revision`` reference.

    Raises:
        ValueError: If it is not set to a Hugging Face repo reference.
    """
    ref = ((config.get("video_model_presets") or {}).get("ltx") or {}).get("mlx_text_encoder")
    if not isinstance(ref, str) or parse_huggingface_repo_reference(ref) is None:
        raise ValueError("video_model_presets.ltx.mlx_text_encoder must be a Hugging Face repo reference (owner/repo or owner/repo@revision).")
    return ref.strip()


def ltx_mlx_text_encoder_repo_id(config: dict[str, Any]) -> str:
    """Return the repo id of the Gemma 3 text encoder of the macOS LTX-2.3 packs, without its revision.

    Raises:
        ValueError: If ``mlx_text_encoder`` is not set to a Hugging Face repo reference.
    """
    return split_huggingface_revision(ltx_mlx_text_encoder(config))[0]


def ltx_diffusers_repo(config: dict[str, Any], version: str | None = None) -> str:
    """Return the diffusers LTX repo of Windows/Linux for *version* (``diffusers.default_repo``, the version's own or else the family's).

    Raises:
        ValueError: If it is not set to a non-blank string.
    """
    repo = (_merged_preset(config, "ltx", version).get("diffusers") or {}).get("default_repo")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(f"{_ltx_path(version)}.diffusers.default_repo must be a non-blank string.")
    return repo.strip()


def ltx_repo_ignore(config: dict[str, Any], version: str | None = None) -> tuple[str, ...]:
    """Return the patterns *version*'s diffusers repo is downloaded without (``diffusers.repo_ignore``, the version's or else the family's), empty when unset.

    Raises:
        ValueError: If it is set but is not a list of non-blank strings.
    """
    value = (_merged_preset(config, "ltx", version).get("diffusers") or {}).get("repo_ignore")
    return () if value is None else _ltx_repo_ignore(value, f"{_ltx_path(version)}.diffusers.repo_ignore")


def _ltx_block(config: dict[str, Any], version: str | None) -> dict[str, Any]:
    """Return the LTX family block (``None``) or version block *version* as written, without merging one over the other."""
    ltx = (config.get("video_model_presets") or {}).get("ltx") or {}
    return ltx if version is None else (ltx.get("versions") or {}).get(version) or {}


def _ltx_diffusers(config: dict[str, Any], model_name: str | None) -> tuple[dict[str, Any], str]:
    """Return the ``diffusers`` block the LTX preset of *model_name* resolves to, and its config key path.

    The path names the version block when one lists *model_name*, so error messages point at the key to fix.
    """
    version = _video_version(config, "ltx", model_name)
    diffusers = video_preset(config, "ltx", model_name).get("diffusers") or {}
    path = "video_model_presets.ltx" + (f".versions.{version}" if version is not None else "") + ".diffusers"
    return diffusers, path


def _ltx_hub_file(node: Any, key: str) -> tuple[str, str]:
    """Return the config entry *node* (found at *key*) as ``(repo_id, filename)``.

    Raises:
        ValueError: If the entry is not a mapping with non-blank string ``repo`` and ``filename``.
    """
    if not isinstance(node, dict):
        raise ValueError(f"{key} must be a mapping with 'repo' and 'filename'.")
    values: list[str] = []
    for field in ("repo", "filename"):
        value = node.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key}.{field} must be a non-blank string.")
        values.append(value.strip())
    return values[0], values[1]


def ltx_diffusers_quantize(config: dict[str, Any], model_name: str | None) -> int | None:
    """Return the quantize level the diffusers LTX backend loads the model picked as *model_name* at, or ``None`` (bfloat16).

    The levels are set per alias in the ``quantize`` entry of the version that lists *model_name*, merged over
    the family block's (e.g. ``ltx-4: 4``).

    Raises:
        ValueError: If that entry is set but is not a mapping of names to 4 or 8.
    """
    diffusers, path = _ltx_diffusers(config, model_name)
    levels = diffusers.get("quantize")
    if levels is None:
        return None
    if not isinstance(levels, dict) or any(bits not in (4, 8) for bits in levels.values()):
        raise ValueError(f"{path}.quantize must map model names to 4 or 8.")
    return levels.get(model_name) if model_name is not None else None


def _ltx_repo_ignore(value: Any, key: str) -> tuple[str, ...]:
    """Return the ``repo_ignore`` entry *value* (found at *key*) as a tuple of stripped patterns.

    Raises:
        ValueError: If it is not a list of non-blank strings.
    """
    if not isinstance(value, list) or not all(isinstance(pattern, str) and pattern.strip() for pattern in value):
        raise ValueError(f"{key} must be a list of non-blank strings.")
    return tuple(pattern.strip() for pattern in value)


def _ltx_sampler(value: Any, key: str) -> str:
    """Return the ``sampler`` entry *value* (found at *key*).

    Raises:
        ValueError: If it is not one of :data:`LTX_SAMPLERS`.
    """
    if value not in LTX_SAMPLERS:
        raise ValueError(f"{key} must be one of {', '.join(sorted(LTX_SAMPLERS))}.")
    return value


def _ltx_image_crf(value: Any, key: str) -> int:
    """Return the ``image_crf`` entry *value* (found at *key*).

    Raises:
        ValueError: If it is not an integer (a bool is not) from 0 to 51.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 51:
        raise ValueError(f"{key} must be an integer from 0 to 51.")
    return value


_LTX_OPTION_VALIDATORS: Final = {"repo_ignore": _ltx_repo_ignore, "sampler": _ltx_sampler, "image_crf": _ltx_image_crf}


def ltx_load_options(config: dict[str, Any], model_name: str | None, *, upscale: bool) -> dict[str, Any]:
    """Return the configured keyword arguments a video backend's ``load_model`` takes for the model picked as *model_name*.

    The diffusers options (``quantize``, ``transformer_source``, ``model_name``, ``latent_upsampler`` and the
    version-only ``repo_ignore``, ``sampler`` and ``image_crf``) are ignored by the MLX backend, and the MLX options
    (``mlx_packs``, and ``mlx_text_encoder`` when configured) by the diffusers backend. The three
    version-only keys are present only when the preset of *model_name* sets them, so the options of a model
    without them are unchanged.

    Raises:
        ValueError: If one of the ``diffusers`` entries is malformed, or the ``sampler`` is ``euler_ancestral``
            for a model that is in no LTX version block (the ancestral sampler is for LTX-2.5 only).
    """
    diffusers, path = _ltx_diffusers(config, model_name)
    options: dict[str, Any] = {
        "quantize": ltx_diffusers_quantize(config, model_name),
        "transformer_source": ltx_transformer_source(config, model_name),
        "model_name": model_name,
        "mlx_packs": ltx_mlx_packs(config),
    }
    if "mlx_text_encoder" in ((config.get("video_model_presets") or {}).get("ltx") or {}):
        options["mlx_text_encoder"] = ltx_mlx_text_encoder(config)
    for name, validate in _LTX_OPTION_VALIDATORS.items():
        if diffusers.get(name) is not None:
            options[name] = validate(diffusers[name], f"{path}.{name}")
    if options.get("sampler") == EULER_ANCESTRAL and ltx_version(config, model_name) is None:
        raise ValueError(
            f"video_model_presets.ltx.diffusers.sampler is '{EULER_ANCESTRAL}', which is for LTX-2.5 version aliases only, "
            f"but {model_name or 'this model'} is not in an LTX version block. "
            "Set the sampler under video_model_presets.ltx.versions.<id>.diffusers."
        )
    if upscale:
        options["upscale"] = True
        options["latent_upsampler"] = ltx_latent_upsampler_source(config, model_name)
    return options


def video_load_options(family: str, config: dict[str, Any], model_name: str | None, *, upscale: bool, quiet: bool = False) -> dict[str, Any]:
    """Return the configured keyword arguments the video backend of *family* takes in ``load_model`` for the model picked as *model_name*.

    *quiet* marks a run with no terminal (the web runner): a backend that prints its own progress turns it off.

    Raises:
        ValueError: If *family* has no load options, or its entries are malformed (see :func:`ltx_load_options`).
    """
    if family == "ltx":
        return ltx_load_options(config, model_name, upscale=upscale)
    if family == "h3":
        from zvisiongenerator.backends import h3_load_options

        return {**h3_load_options(config, model_name, quiet=quiet), **({"upscale": True} if upscale else {})}
    raise ValueError(f"No video load options for model family '{family}'.")


def video_companion_repos(family: str, config: dict[str, Any], repo_id: str) -> tuple[str, ...]:
    """Return the other Hugging Face repos the video backend of *family* downloads beside *repo_id* (none for a family without companions)."""
    if family == "ltx":
        return ltx_companion_repos(config, repo_id)
    if family == "h3":
        return h3_companion_repos(config, repo_id)
    return ()


def h3_companion_repos(config: dict[str, Any], repo_id: str) -> tuple[str, ...]:
    """Return the other Hugging Face repos the H3 backend downloads beside *repo_id*.

    That is the text encoder, the turbo LoRA and the latent upscaler on Windows and Linux, and the INT8 transformer and
    the turbo LoRA on macOS. These are listed when *repo_id* is the configured H3 repository (at any revision), otherwise none.

    Raises:
        ValueError: If the H3 entry of *config* is malformed.
    """
    from zvisiongenerator.backends import h3_companion_repo_ids

    preset = h3_preset(config)
    if split_huggingface_revision(repo_id)[0] != preset.repo:
        return ()
    return tuple(dict.fromkeys(companion for companion in h3_companion_repo_ids(config) if companion != preset.repo))


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


ENHANCER_TIERS = ("fast", "smart")
_GIB = 2**30


def _platform_value(section: dict[str, Any], key: str, platform_key: str) -> Any:
    """Return *section[key]* for *platform_key* (a per-platform mapping or one value for every platform)."""
    values = section.get(key)
    return values.get(platform_key) if isinstance(values, dict) else values


def _tier_model(section: dict[str, Any], platform_key: str) -> tuple[str, str | None] | None:
    """Return the ``(repo, revision)`` a tier section configures for *platform_key*, or None."""
    repo = _platform_value(section, "model", platform_key)
    if not isinstance(repo, str) or not repo.strip():
        return None
    revision = _platform_value(section, "revision", platform_key)
    return repo.strip(), revision.strip() if isinstance(revision, str) and revision.strip() else None


def smart_enhancer_model(config: dict[str, Any], *, platform_key: str) -> tuple[str, str | None] | None:
    """Return the Smart enhancer ``(repo, revision)`` configured for *platform_key*, or None."""
    return _tier_model((config.get("prompt_enhancer") or {}).get("smart") or {}, platform_key)


def enhancer_tier_models(config: dict[str, Any], *, platform_key: str) -> dict[str, tuple[str, str | None]]:
    """Return the configured enhancer ``(repo, revision)`` per tier: ``fast``, ``smart`` and ``custom`` (``user_model``).

    Tiers without a model for *platform_key* are left out, as is a malformed ``user_model``.
    """
    section = config.get("prompt_enhancer") or {}
    models = {"fast": _tier_model(section, platform_key), "smart": smart_enhancer_model(config, platform_key=platform_key)}
    user_model = section.get("user_model")
    if isinstance(user_model, str) and user_model.strip():
        try:
            models["custom"] = split_model_revision(user_model)
        except ValueError:
            pass
    return {tier: model for tier, model in models.items() if model is not None}


def smart_unavailable(config: dict[str, Any], *, platform_key: str, budget_bytes: int | None) -> str | None:
    """Return why the Smart enhancer cannot run on this machine, or None when it can.

    Args:
        budget_bytes: The accelerator memory (VRAM, or the GPU working set on Apple Silicon); ``None`` when unknown,
            which counts as unavailable.
    """
    smart = (config.get("prompt_enhancer") or {}).get("smart") or {}
    if smart_enhancer_model(config, platform_key=platform_key) is None or budget_bytes is None:
        return "Smart is not available on this machine."
    need = _platform_value(smart, "min_memory_gb", platform_key)
    if isinstance(need, (int, float)) and budget_bytes < need * _GIB:
        return f"Needs about {need:g} GB of memory; this machine has {budget_bytes / _GIB:.0f} GB."
    return None


def resolve_enhancer_model(config: dict[str, Any], *, platform_key: str, cli_model: str | None = None, budget_bytes: int | None = None) -> tuple[str, str | None]:
    """Return the prompt-enhancer ``(repo, revision)``: CLI choice > user override (``REPO[@REVISION]``) > Fast.

    The CLI choice is ``fast``, ``smart`` or ``REPO[@REVISION]``. Smart falls back to Fast, with a warning, when
    :func:`smart_unavailable` gives a reason.

    Args:
        config: Loaded config mapping.
        platform_key: Platform key such as ``darwin`` or ``win32``.
        cli_model: Optional ``fast``, ``smart`` or ``REPO[@REVISION]`` from ``--enhance-model``.
        budget_bytes: The machine's accelerator memory, for the Smart check; ``None`` when unknown.

    Raises:
        ValueError: If no model is configured for the platform or *cli_model* is malformed.
    """
    choice = (cli_model or "").strip()
    section = config.get("prompt_enhancer") or {}
    if choice and choice not in ENHANCER_TIERS:
        return split_model_revision(choice)
    user_model = section.get("user_model")
    if not choice and isinstance(user_model, str) and user_model.strip():
        return split_model_revision(user_model)
    if choice == "smart":
        reason = smart_unavailable(config, platform_key=platform_key, budget_bytes=budget_bytes)
        smart = smart_enhancer_model(config, platform_key=platform_key)
        if reason is None and smart is not None:
            return smart
        warnings.warn(f"The Smart enhancer is unavailable: {reason} Using Fast.", stacklevel=2)
    fast = _tier_model(section, platform_key)
    if fast is None:
        raise ValueError(f"No prompt enhancer model is configured for platform '{platform_key}'. Set prompt_enhancer.user_model.")
    return fast
