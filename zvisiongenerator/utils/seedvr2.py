"""SeedVR2 plain-upscale constants, weight layouts, settings, memory estimate and tier choice.

The tiers and the files they download are facts of the packaged model metadata (the ``upscaler`` entries); the settings
(``seedvr2`` in the config) choose among them. Helpers shared by the config loader, the upscale sizing, the loader
and the planner; nothing here touches the network, MLX or torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zvisiongenerator.downloads.declaration import model_targets
from zvisiongenerator.utils.model_metadata import for_platform, model_entry, model_ids

SEEDVR2_BITS = 8

_AUTO = "auto"
_UPSCALER_KIND = "upscaler"


@dataclass(frozen=True)
class SeedVR2Weights:
    """The SeedVR2 weights one platform engine reads and how it keeps them.

    Attributes:
        precision: The precision the model runs at, as labels and the weights folder name show it ("q8", "fp8").
        stores_copy: Whether the first load quantizes the Hub files and saves an 8-bit copy to the weights
            folder (deleting the download), rather than downloading the files into that folder as they are.
    """

    precision: str
    stores_copy: bool


# mflux (macOS) quantizes the 16-bit files to 8 bits while loading and keeps a stored copy.
MLX_WEIGHTS = SeedVR2Weights(precision=f"q{SEEDVR2_BITS}", stores_copy=True)

# The vendored PyTorch core (Windows and Linux) reads the Hub's FP8 files, downloaded into one folder because the
# core loads the transformer and the VAE from the same folder.
FP8_WEIGHTS = SeedVR2Weights(precision="fp8", stores_copy=False)


@dataclass(frozen=True)
class SeedVR2Tier:
    """One SeedVR2 model at 8-bit with its measured memory cost, in decimal gigabytes.

    *weights_gb* and the activation costs are accelerator memory: the whole model on Apple Silicon, the parts
    kept on the GPU on CUDA. *system_gb* is set only where the weights stream from system memory, and is the
    system memory a run needs.
    """

    model: str
    weights_gb: float
    activation_floor_gb: float
    activation_base_gb: float
    activation_gb_per_mp: float
    system_gb: float | None = None


@dataclass(frozen=True)
class SeedVR2Settings:
    """The validated ``seedvr2`` block of the config; *tiers* are in quality order."""

    model: str
    softness: float
    video_softness: float
    reference_megapixels: float
    tiers: tuple[SeedVR2Tier, ...]


@dataclass(frozen=True)
class SeedVR2Setup:
    """The tier a plain upscale uses on this machine, the weights its engine reads, and whether they are on disk."""

    settings: SeedVR2Settings
    tier: SeedVR2Tier
    budget_bytes: int | None
    downloaded: bool
    weights: SeedVR2Weights


def seedvr2_tiers(platform_key: str) -> tuple[SeedVR2Tier, ...]:
    """Return the SeedVR2 tiers the metadata declares for *platform_key*, in quality order (the order of the ``upscaler`` entries)."""
    return tuple(SeedVR2Tier(model=model_id, **{key: float(value) for key, value in for_platform(model_entry(model_id)["memory"], platform_key).items()}) for model_id in model_ids(_UPSCALER_KIND))


def seedvr2_files(model: str, platform_key: str) -> tuple[tuple[str, str], ...]:
    """Return the ``(repo, file)`` pairs the SeedVR2 *model* downloads on *platform_key*: the transformer first, then the VAE.

    The pairs follow the declared targets and their files in order, so the first is always the transformer.
    """
    return tuple((target.repo_id, name) for _role, target in model_targets(model, platform_key) for name in target.allow_patterns)


def check_seedvr2_numbers(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` when the ``seedvr2`` block of *config* is not a mapping or one of its numbers is out of range.

    ``seedvr2.model`` names a model of the metadata, so :func:`seedvr2_settings` validates it where it is used.
    """
    _numbers(config)


def seedvr2_settings(config: dict[str, Any], platform_key: str) -> SeedVR2Settings:
    """Return the validated ``seedvr2`` block of *config*, with the tiers the metadata declares for *platform_key*.

    Args:
        config: Loaded config dict.
        platform_key: The platform whose tiers apply (``darwin``, ``win32`` or ``linux``).

    Raises:
        ValueError: If a key is missing, has the wrong type or is out of range; the message names the key.
    """
    softness, video_softness, reference = _numbers(config)
    tiers = seedvr2_tiers(platform_key)
    model = config["seedvr2"].get("model", _AUTO)
    if not isinstance(model, str):
        raise ValueError(f"config 'seedvr2.model' must be a string, got {model!r}.")
    if model != _AUTO and model not in {tier.model for tier in tiers}:
        raise ValueError(f"config 'seedvr2.model' must be 'auto' or one of the SeedVR2 models ({', '.join(tier.model for tier in tiers)}), got {model!r}.")
    return SeedVR2Settings(model=model, softness=softness, video_softness=video_softness, reference_megapixels=reference, tiers=tiers)


def _numbers(config: dict[str, Any]) -> tuple[float, float, float]:
    """Return the validated ``softness``, ``video_softness`` and ``reference_megapixels`` of the ``seedvr2`` block of *config*."""
    block = config.get("seedvr2")
    if not isinstance(block, dict):
        raise ValueError("config 'seedvr2' must be a mapping.")
    softness = block.get("softness")
    if not _is_number(softness) or not 0 <= softness <= 1:
        raise ValueError(f"config 'seedvr2.softness' must be a number from 0 to 1, got {softness!r}.")
    video_softness = block.get("video_softness")
    if not _is_number(video_softness) or not 0 <= video_softness <= 1:
        raise ValueError(f"config 'seedvr2.video_softness' must be a number from 0 to 1, got {video_softness!r}.")
    reference = block.get("reference_megapixels")
    if not _is_number(reference) or reference <= 0:
        raise ValueError(f"config 'seedvr2.reference_megapixels' must be a positive number, got {reference!r}.")
    return float(softness), float(video_softness), float(reference)


def estimate_seedvr2_bytes(tier: SeedVR2Tier, megapixels: float) -> int:
    """Return the accelerator memory a SeedVR2 run needs for an output of *megapixels*, in bytes.

    The weights plus the activations, which follow a line in the output size but never drop below a floor.
    """
    activation_gb = max(tier.activation_floor_gb, tier.activation_base_gb + tier.activation_gb_per_mp * megapixels)
    return int((tier.weights_gb + activation_gb) * 1e9)


def seedvr2_over_budget(estimate_bytes: int, budget_bytes: int | None) -> bool:
    """Return whether *estimate_bytes* is more than the known memory *budget_bytes* (never, when unknown)."""
    return budget_bytes is not None and estimate_bytes > budget_bytes


def pick_seedvr2_tier(settings: SeedVR2Settings, budget_bytes: int | None, system_bytes: int | None = None) -> SeedVR2Tier:
    """Return the tier a plain upscale uses on a machine with *budget_bytes* of accelerator memory.

    The candidates are every tier (``model: auto``) or the configured model's tiers, in quality order, and the
    first that fits wins; when none fits, or the memory is unknown, the last one does. A tier fits when its
    estimate at ``reference_megapixels`` is within *budget_bytes*. A tier with ``system_gb`` streams its weights
    from system memory, so its GPU need hardly depends on the model; it fits when ``system_gb`` is within
    *system_bytes* instead.
    """
    candidates = [tier for tier in settings.tiers if settings.model in (_AUTO, tier.model)]
    for tier in candidates:
        if _tier_fits(tier, settings.reference_megapixels, budget_bytes, system_bytes):
            return tier
    return candidates[-1]


def seedvr2_tier_label(model: str, weights: SeedVR2Weights) -> str:
    """Return the display name of *model* as *weights* run it, such as "SeedVR2 7B q8" or "SeedVR2 7B fp8"."""
    return f"SeedVR2 {model.removeprefix('seedvr2-').upper()} {weights.precision}"


def seedvr2_row_name(model: str, weights: SeedVR2Weights) -> str:
    """Return the Models-page name of *model* as *weights* run it, such as "seedvr2-7b@q8"."""
    return f"{model}@{weights.precision}"


def _tier_fits(tier: SeedVR2Tier, megapixels: float, budget_bytes: int | None, system_bytes: int | None) -> bool:
    if tier.system_gb is not None:
        return system_bytes is not None and tier.system_gb * 1e9 <= system_bytes
    return budget_bytes is not None and not seedvr2_over_budget(estimate_seedvr2_bytes(tier, megapixels), budget_bytes)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
