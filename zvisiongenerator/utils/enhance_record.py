"""Record how a prompt was enhanced (options, model, tier, format) for a generated asset, and read it back."""

from __future__ import annotations

from typing import Any

from zvisiongenerator.core.job_plan import EnhancerUsed, EnhanceStatus, IterationPlan
from zvisiongenerator.utils.config import enhancer_tier_models, model_reference, split_model_revision
from zvisiongenerator.utils.paths import collapse_home, expand_home
from zvisiongenerator.utils.prompt_enhance import ENHANCE_AXES, ENHANCE_FORMATS, EnhanceSettings, settings_from_mapping, settings_to_mapping

__all__ = ["RECORD_TIERS", "build_enhance_record", "enhance_record_summary", "enhancer_tier", "parse_enhance_record", "planned_enhance_record"]

# ``custom`` is a model set on the Config page or with --enhance-model REPO.
RECORD_TIERS = ("fast", "smart", "custom")
# Every video-only option axis (camera, pace, motion, sound) plus the Rewrite choice.
_VIDEO_ONLY_KEYS = (*(axis.key for axis in ENHANCE_AXES if axis.video_only), "parts")


def enhancer_tier(config: dict[str, Any], *, platform_key: str, model: str) -> str:
    """Return which enhancer tier *model* (``REPO[@REVISION]``) is: ``fast``, ``smart``, or ``custom`` for any other model.

    A local folder matches however its home directory is written (``~`` or in full), in *model* and in the config.
    """
    try:
        repo, revision = split_model_revision(expand_home(model))
    except ValueError:
        return "custom"
    for tier, (tier_repo, tier_revision) in enhancer_tier_models(config, platform_key=platform_key).items():
        if (expand_home(tier_repo), tier_revision) == (repo, revision):
            return tier
    return "custom"


def build_enhance_record(settings: EnhanceSettings, *, mode: str, repo: str, revision: str | None, tier: str, fmt: str) -> dict[str, Any]:
    """Return the ``enhance`` block an asset records: the options used, the model, its tier and the output format.

    Image records leave out the video-only options; an empty direction is left out. A local model folder is written
    with the home directory as ``~``, so shared files do not reveal the username.
    """
    options = settings_to_mapping(settings)
    if mode != "video":
        options = {key: value for key, value in options.items() if key not in _VIDEO_ONLY_KEYS}
    if not options.get("note"):
        options.pop("note", None)
    if options.get("parts") is None:
        options.pop("parts", None)
    return {"settings": options, "model": collapse_home(model_reference(repo, revision)), "tier": tier, "format": fmt}


def planned_enhance_record(iteration: IterationPlan, enhancer: EnhancerUsed | None, *, mode: str) -> dict[str, Any] | None:
    """Return the ``enhance`` block of an iteration that preflight rewrote, or None when it was not rewritten."""
    if enhancer is None or iteration.enhance is None or iteration.enhance_status is not EnhanceStatus.ENHANCED:
        return None
    return build_enhance_record(iteration.enhance, mode=mode, repo=enhancer.repo, revision=enhancer.revision, tier=enhancer.tier, fmt=enhancer.format)


_TIER_LABELS = {"fast": "Fast", "smart": "Smart", "custom": "Custom model"}


def enhance_record_summary(record: dict[str, Any]) -> str:
    """Return a one-line summary of how a prompt was enhanced: the enhancer, then every option that is not a default.

    For example ``Fast · Cinematic, Melancholic, Orbit, Longer`` or ``Smart · JSON caption · Noir``; ``General rewrite``
    stands in for options that are all defaults.
    """
    settings, defaults = record.get("settings") or {}, settings_to_mapping(EnhanceSettings())
    picked: list[str] = []
    for axis in ENHANCE_AXES:
        value = settings.get(axis.key)
        if value is None or value == defaults[axis.key]:
            continue
        slugs = value if isinstance(value, list) else [value]
        picked.extend(axis.option(slug).label for slug in slugs if slug in axis.slugs())
    if settings.get("note"):
        picked.append(f'"{settings["note"]}"')
    parts = [_TIER_LABELS.get(record.get("tier", ""), "Custom model")]
    if record.get("format") == "json":
        parts.append("JSON caption")
    parts.append(", ".join(picked) if picked else "General rewrite")
    return " · ".join(parts)


def parse_enhance_record(raw: Any, *, mode: str, config: dict[str, Any] | None = None, platform_key: str | None = None) -> dict[str, Any] | None:
    """Validate an ``enhance`` block (from the Web UI or a saved asset) and return it normalized, or None when absent.

    Args:
        config: When given with *platform_key*, the tier is worked out from the model rather than trusted.

    Raises:
        ValueError: When the block is malformed: bad settings, an unknown tier or format, or no model.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"The enhance record must be a mapping, got {type(raw).__name__}.")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("The enhance record needs the model that wrote the rewrite.")
    fmt = raw.get("format", "text")
    if fmt not in ENHANCE_FORMATS:
        raise ValueError(f"The enhance record's format must be one of: {', '.join(ENHANCE_FORMATS)}.")
    settings = settings_from_mapping(raw.get("settings") or {}, mode=mode, motion_in_image="warn")
    repo, revision = split_model_revision(model.strip())
    if config is not None and platform_key is not None:
        tier = enhancer_tier(config, platform_key=platform_key, model=model.strip())
    else:
        tier = raw.get("tier", "custom")
        if tier not in RECORD_TIERS:
            raise ValueError(f"The enhance record's tier must be one of: {', '.join(RECORD_TIERS)}.")
    return build_enhance_record(settings, mode=mode, repo=repo, revision=revision, tier=tier, fmt=fmt)
