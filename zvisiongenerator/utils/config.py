"""Config loading, deep-merge, and layered default resolution.

Loads config.yaml from the package via importlib.resources, with optional
user override from ~/.ziv/config.yaml (deep-merged).  Also provides
resolve_defaults() for the config layering precedence:

    CLI explicit flags > model preset variant > model preset family > global defaults
"""

from __future__ import annotations

import importlib.resources
import warnings
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.core.video_family import FrameRule, VideoCapabilities, VideoFamily
from zvisiongenerator.schedulers import SCHEDULER_CLASSES
from zvisiongenerator.utils.enhancer_models import ENHANCER_TIERS, enhancer_tier_models, smart_unavailable
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.ltx_facts import ltx_load_options
from zvisiongenerator.utils.model_metadata import YAML_LOADER, family_block, load_model_metadata, model_capabilities
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.seedvr2 import check_seedvr2_numbers
from zvisiongenerator.utils.upscale import max_megapixels


def load_config() -> dict[str, Any]:
    """Load config.yaml from package resources, with optional user override.

    Returns:
        Parsed config dict with all settings and their defaults.

    Raises:
        FileNotFoundError: If the bundled config.yaml cannot be located.
    """
    ref = importlib.resources.files("zvisiongenerator").joinpath("config.yaml")
    with importlib.resources.as_file(ref) as path:
        with open(path, encoding="utf-8") as f:
            try:
                config = yaml.load(f, Loader=YAML_LOADER)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}") from e

    # Optional user override (ZIV_DATA_DIR/config.yaml)
    user_config = get_ziv_data_dir() / "config.yaml"
    if user_config.exists():
        with open(user_config, encoding="utf-8") as f:
            try:
                user = yaml.load(f, Loader=YAML_LOADER)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}") from e
        if user and isinstance(user, dict):
            _deep_merge(config, user)

    # Validate that known sections have the right types after merge
    _EXPECTED_DICTS = (
        "sizes",
        "generation",
        "sharpening",
        "upscale",
        "seedvr2",
        "contrast",
        "saturation",
        "model_aliases",
        "model_presets",
        "video_sizes",
        "video_generation",
        "video_model_presets",
        "audio_model_presets",
        "prompt_enhancer",
    )
    for section in _EXPECTED_DICTS:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping, got {type(config[section]).__name__}. Check your user config (~/.ziv/config.yaml) for overrides.")

    # Reject a bad upscale.max_megapixels at load time rather than on every gallery page.
    max_megapixels(config)
    for key, amount in sharpening_amounts(config).items():
        validate_sharpen_amount(amount, name=f"config 'sharpening.{key}'")

    aliases = config.get("model_aliases", {})
    for alias_name, alias_value in aliases.items():
        if isinstance(alias_value, str):
            continue
        if not isinstance(alias_value, dict):
            raise ValueError(f"config 'model_aliases.{alias_name}' must be a string or mapping.")
        for platform_key, platform_value in alias_value.items():
            if not isinstance(platform_key, str):
                raise ValueError(f"config 'model_aliases.{alias_name}' platform keys must be strings.")
            if not isinstance(platform_value, str):
                raise ValueError(f"config 'model_aliases.{alias_name}.{platform_key}' must be a string, the repository or path for that platform; a platform with no model is left out.")

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

    check_seedvr2_numbers(config)

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


def video_family(config: dict[str, Any], family: str) -> VideoFamily | None:
    """Return the description of a video family: its facts from the model metadata, its step and frame settings from ``video_model_presets.<family>``, or ``None`` for a family the metadata does not describe.

    The family block is read as it stands: a version's overrides do not apply.

    Raises:
        ValueError: If a setting is missing (``default_steps``) or malformed; the message names the key path.
    """
    if family not in load_model_metadata()["video_families"]:
        return None
    facts = family_block(family, None)
    base = f"video_model_presets.{family}"
    settings = (config.get("video_model_presets") or {}).get(family) or {}
    if not isinstance(settings, dict):
        raise ValueError(f"{base} must be a mapping.")
    if "default_steps" not in settings:
        raise ValueError(f"{base}.default_steps is required.")
    default_steps = _positive_int(settings["default_steps"], f"{base}.default_steps")
    frame_rule = FrameRule(**facts["frame_rule"])
    step_options = facts.get("step_options")
    return VideoFamily(
        name=family,
        label=facts["label"],
        frame_rule=frame_rule,
        capabilities=VideoCapabilities(**facts["capabilities"]),
        platform_capabilities=facts.get("platform_capabilities", {}),
        default_steps=default_steps,
        max_steps=facts.get("max_steps"),
        step_options=None if step_options is None else tuple(step_options),
        default_frames=_default_frames(settings.get("default_frames"), frame_rule, f"{base}.default_frames"),
        extend_frame_step=facts["extend_frame_step"],
        carry_frames=facts["carry_frames"],
        default_upscale_steps=_default_upscale_steps(settings.get("upscale"), default_steps, f"{base}.upscale"),
        extend_fps=facts.get("extend_fps"),
        platform_carry_frames=facts.get("platform_carry_frames", {}),
    )


def _default_upscale_steps(value: Any, default_steps: int, path: str) -> int:
    """Return ``default_upscale_steps`` from a family's ``upscale`` settings, or *default_steps* when it has none."""
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
        Dict with resolved ``steps``, ``guidance``, ``scheduler``, ``upscale_sharpen`` and
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
        "upscale_sharpen": True,
        **model_capabilities(model_info.family, variant_key),
    }

    # Layer family defaults
    if "default_steps" in preset:
        effective["steps"] = preset["default_steps"]
    if "default_guidance" in preset:
        effective["guidance"] = preset["default_guidance"]
    _layer_upscale_defaults(effective, preset.get("upscale", {}))

    # Layer variant defaults
    if variant_key and "variants" in preset:
        variant = preset["variants"].get(variant_key, {})
        if "default_steps" in variant:
            effective["steps"] = variant["default_steps"]
        if "default_guidance" in variant:
            effective["guidance"] = variant["default_guidance"]
        _layer_upscale_defaults(effective, variant.get("upscale", {}))

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


def resolve_scheduler_class(scheduler: str | None, backend_name: str) -> str | None:
    """Resolve a scheduler name to its backend-specific class path, or the name itself."""
    if scheduler is None:
        return None
    return SCHEDULER_CLASSES.get(scheduler, {}).get(backend_name, scheduler)


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


def validate_scheduler(scheduler_name: str | None) -> None:
    """Check that *scheduler_name* is a known scheduler.

    Args:
        scheduler_name: Scheduler name to validate, or ``None`` (no-op).

    Raises:
        ValueError: If the scheduler name is not ``None`` and not one of ``SCHEDULER_CLASSES``.
    """
    if scheduler_name is None:
        return
    if scheduler_name not in SCHEDULER_CLASSES:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'. Valid options: {list(SCHEDULER_CLASSES)}")


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

    # Look up dimensions from flat video_sizes[ratio][size]. The model family still selects the video_model_presets settings.
    ratio_sizes = vsizes.get(ratio, {})
    size_entry = ratio_sizes.get(size, {})

    # A family whose settings carry no default_frames keeps the size table's count.
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


def _layer_upscale_defaults(effective: dict[str, Any], upscale: Any) -> None:
    """Copy a preset ``upscale`` block's ``default_upscale_steps`` and ``default_sharpen`` into *effective*."""
    if not isinstance(upscale, dict):
        return
    if "default_upscale_steps" in upscale:
        effective["upscale_steps"] = upscale["default_upscale_steps"]
    if "default_sharpen" in upscale:
        effective["upscale_sharpen"] = bool(upscale["default_sharpen"])


def resolve_upscale_steps(defaults: dict[str, Any], steps: int) -> int:
    """Return the upscale refinement steps when not set explicitly: the preset default, else ``max(1, steps // 2)``."""
    preset_steps = defaults.get("upscale_steps")
    return preset_steps if preset_steps is not None else max(1, steps // 2)


def video_load_options(family: str, config: dict[str, Any], model_name: str | None, *, platform_key: str, upscale: bool, quiet: bool = False) -> dict[str, Any]:
    """Return the keyword arguments the video backend of *family* takes in ``load_model`` for the model picked as *model_name* on *platform_key*.

    *config* supplies the settings (H3's ``ffn_chunk_tokens``). *quiet* marks a run with no terminal (the web runner): a backend that prints its own progress turns it off.

    Raises:
        ValueError: If *family* has no load options, or a setting is malformed.
    """
    if family == "ltx":
        return ltx_load_options(model_name, platform_key, upscale=upscale)
    if family == "h3":
        from zvisiongenerator.backends import h3_load_options

        return {**h3_load_options(config, model_name, platform_key, quiet=quiet), **({"upscale": True} if upscale else {})}
    raise ValueError(f"No video load options for model family '{family}'.")


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


def custom_enhancer_model(config: dict[str, Any]) -> tuple[str, str | None] | None:
    """Return the ``(repo, revision)`` of ``prompt_enhancer.user_model``, or ``None`` when it is unset or malformed."""
    user_model = (config.get("prompt_enhancer") or {}).get("user_model")
    if not isinstance(user_model, str) or not user_model.strip():
        return None
    try:
        return split_model_revision(user_model)
    except ValueError:
        return None


def configured_enhancer_models(config: dict[str, Any], *, platform_key: str) -> dict[str, tuple[str, str | None]]:
    """Return the enhancer ``(repo, revision)`` per tier: ``fast`` and ``smart`` from the metadata, then ``custom`` (``user_model``) when set."""
    models = enhancer_tier_models(platform_key=platform_key)
    custom = custom_enhancer_model(config)
    if custom is not None:
        models["custom"] = custom
    return models


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
        ValueError: If *cli_model* or ``prompt_enhancer.user_model`` is malformed.
    """
    choice = (cli_model or "").strip()
    section = config.get("prompt_enhancer") or {}
    if choice and choice not in ENHANCER_TIERS:
        return split_model_revision(choice)
    user_model = section.get("user_model")
    if not choice and isinstance(user_model, str) and user_model.strip():
        return split_model_revision(user_model)
    tiers = enhancer_tier_models(platform_key=platform_key)
    if choice == "smart":
        reason = smart_unavailable(platform_key=platform_key, budget_bytes=budget_bytes)
        if reason is None:
            return tiers["smart"]
        warnings.warn(f"The Smart enhancer is unavailable: {reason} Using Fast.", stacklevel=2)
    return tiers["fast"]
