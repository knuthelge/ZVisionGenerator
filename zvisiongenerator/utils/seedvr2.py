"""SeedVR2 plain-upscale constants, settings, memory estimate and tier choice.

Pure helpers shared by the config loader, the upscale sizing and (later) the loader and planner; nothing here
touches disk, the network or MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEEDVR2_REPO = "numz/SeedVR2_comfyUI"
SEEDVR2_BITS = 8

# Model name → (transformer file, VAE file) in SEEDVR2_REPO: the files mflux reads, in the 16-bit layout.
SEEDVR2_MODEL_FILES: dict[str, tuple[str, str]] = {
    "seedvr2-7b": ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors"),
    "seedvr2-3b": ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"),
}

# File name → size in bytes, from the repository's file tree.
SEEDVR2_FILE_BYTES: dict[str, int] = {
    "seedvr2_ema_7b_fp16.safetensors": 16_479_334_424,
    "seedvr2_ema_3b_fp16.safetensors": 6_783_018_808,
    "ema_vae_fp16.safetensors": 501_324_814,
}

_AUTO = "auto"
_TIER_COSTS = ("weights_gb", "activation_floor_gb", "activation_base_gb", "activation_gb_per_mp")


@dataclass(frozen=True)
class SeedVR2Tier:
    """One SeedVR2 model at 8-bit with its measured memory cost, in decimal gigabytes."""

    model: str
    weights_gb: float
    activation_floor_gb: float
    activation_base_gb: float
    activation_gb_per_mp: float


@dataclass(frozen=True)
class SeedVR2Settings:
    """The validated ``seedvr2`` config block; *tiers* are in quality order."""

    model: str
    softness: float
    reference_megapixels: float
    tiers: tuple[SeedVR2Tier, ...]


@dataclass(frozen=True)
class SeedVR2Setup:
    """The tier a plain upscale uses on this Mac and whether its stored copy is already on disk."""

    settings: SeedVR2Settings
    tier: SeedVR2Tier
    budget_bytes: int | None
    downloaded: bool


def seedvr2_settings(config: dict[str, Any]) -> SeedVR2Settings:
    """Return the validated ``seedvr2`` block of *config*.

    Args:
        config: Loaded config dict.

    Raises:
        ValueError: If a key is missing, has the wrong type or is out of range; the message names the key.
    """
    block = config.get("seedvr2")
    if not isinstance(block, dict):
        raise ValueError("config 'seedvr2' must be a mapping.")
    rows = block.get("tiers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("config 'seedvr2.tiers' must be a non-empty list.")
    tiers = tuple(_parse_tier(row, index) for index, row in enumerate(rows))

    model = block.get("model", _AUTO)
    if model != _AUTO and model not in {tier.model for tier in tiers}:
        raise ValueError(f"config 'seedvr2.model' must be 'auto' or one of the 'seedvr2.tiers' models ({', '.join(tier.model for tier in tiers)}), got {model!r}.")
    softness = block.get("softness")
    if not _is_number(softness) or not 0 <= softness <= 1:
        raise ValueError(f"config 'seedvr2.softness' must be a number from 0 to 1, got {softness!r}.")
    reference = block.get("reference_megapixels")
    if not _is_number(reference) or reference <= 0:
        raise ValueError(f"config 'seedvr2.reference_megapixels' must be a positive number, got {reference!r}.")
    return SeedVR2Settings(model=model, softness=float(softness), reference_megapixels=float(reference), tiers=tiers)


def estimate_seedvr2_bytes(tier: SeedVR2Tier, megapixels: float) -> int:
    """Return the memory a SeedVR2 run needs for an output of *megapixels*, in bytes.

    The weights plus the activations, which follow a line in the output size but never drop below a floor.
    """
    activation_gb = max(tier.activation_floor_gb, tier.activation_base_gb + tier.activation_gb_per_mp * megapixels)
    return int((tier.weights_gb + activation_gb) * 1e9)


def seedvr2_over_budget(estimate_bytes: int, budget_bytes: int | None) -> bool:
    """Return whether *estimate_bytes* is more than the known memory *budget_bytes* (never, when unknown)."""
    return budget_bytes is not None and estimate_bytes > budget_bytes


def pick_seedvr2_tier(settings: SeedVR2Settings, budget_bytes: int | None) -> SeedVR2Tier:
    """Return the tier a plain upscale uses on a Mac with *budget_bytes* of memory.

    The candidates are every tier (``model: auto``) or the configured model's tiers, in quality order. The first
    within budget at ``reference_megapixels`` wins; when none fits, or the budget is unknown, the last one does.
    """
    candidates = [tier for tier in settings.tiers if settings.model in (_AUTO, tier.model)]
    if budget_bytes is not None:
        for tier in candidates:
            if not seedvr2_over_budget(estimate_seedvr2_bytes(tier, settings.reference_megapixels), budget_bytes):
                return tier
    return candidates[-1]


def seedvr2_tier_label(model: str) -> str:
    """Return the display name of the 8-bit tier of *model*, such as "SeedVR2 7B q8"."""
    return f"SeedVR2 {model.removeprefix('seedvr2-').upper()} q{SEEDVR2_BITS}"


def seedvr2_download_bytes(model: str) -> int:
    """Return the bytes the first use of *model* downloads: its transformer file and the VAE file."""
    return sum(SEEDVR2_FILE_BYTES[name] for name in SEEDVR2_MODEL_FILES[model])


def seedvr2_size_label(num_bytes: int) -> str:
    """Return *num_bytes* in decimal gigabytes with one decimal, such as "17.0 GB"."""
    return f"{num_bytes / 1e9:.1f} GB"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _parse_tier(row: object, index: int) -> SeedVR2Tier:
    name = f"seedvr2.tiers[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"config '{name}' must be a mapping.")
    extra = sorted(set(row) - {"model", *_TIER_COSTS})
    if extra:
        raise ValueError(f"config '{name}' has unknown keys: {', '.join(map(str, extra))}.")
    model = row.get("model")
    if model not in SEEDVR2_MODEL_FILES:
        raise ValueError(f"config '{name}.model' must be one of {', '.join(SEEDVR2_MODEL_FILES)}, got {model!r}.")
    costs: dict[str, float] = {}
    for key in _TIER_COSTS:
        value = row.get(key)
        if not _is_number(value) or value <= 0:
            raise ValueError(f"config '{name}.{key}' must be a positive number, got {value!r}.")
        costs[key] = float(value)
    return SeedVR2Tier(model=model, **costs)
