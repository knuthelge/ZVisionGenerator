"""SeedVR2 plain-upscale constants, weight layouts, settings, memory estimate and tier choice.

Pure helpers shared by the config loader, the upscale sizing, the loader and the planner; nothing here touches
disk, the network, MLX or torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEEDVR2_REPO = "numz/SeedVR2_comfyUI"
# numz's companion repo; it holds the 7B FP8 file whose last block stays in FP16, which avoids the plain FP8
# file's streaks in flat areas.
SEEDVR2_MIXED_REPO = "AInVFX/SeedVR2_comfyUI"
SEEDVR2_BITS = 8
SEEDVR2_MODELS = ("seedvr2-7b", "seedvr2-3b")

# File name → size in bytes, from the repository's file tree.
SEEDVR2_FILE_BYTES: dict[str, int] = {
    "seedvr2_ema_7b_fp16.safetensors": 16_479_334_424,
    "seedvr2_ema_3b_fp16.safetensors": 6_783_018_808,
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": 8_466_296_338,
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": 3_391_544_696,
    "ema_vae_fp16.safetensors": 501_324_814,
}

_AUTO = "auto"
_TIER_COSTS = ("weights_gb", "activation_floor_gb", "activation_base_gb", "activation_gb_per_mp")
_SYSTEM_COST = "system_gb"


@dataclass(frozen=True)
class SeedVR2Weights:
    """The SeedVR2 weights one platform engine reads and how it keeps them.

    Attributes:
        files: Model name → its ``(repo, file)`` pairs on the Hugging Face Hub, transformer first, then the VAE.
        precision: The precision the model runs at, as labels and the weights folder name show it ("q8", "fp8").
        tiers_key: The ``seedvr2`` config list that prices this engine's tiers.
        stores_copy: Whether the first load quantizes the Hub files and saves an 8-bit copy to the weights
            folder (deleting the download), rather than downloading the files into that folder as they are.
    """

    files: dict[str, tuple[tuple[str, str], ...]]
    precision: str
    tiers_key: str
    stores_copy: bool


# mflux (macOS) quantizes the 16-bit files to 8 bits while loading and keeps a stored copy.
MLX_WEIGHTS = SeedVR2Weights(
    files={
        "seedvr2-7b": ((SEEDVR2_REPO, "seedvr2_ema_7b_fp16.safetensors"), (SEEDVR2_REPO, "ema_vae_fp16.safetensors")),
        "seedvr2-3b": ((SEEDVR2_REPO, "seedvr2_ema_3b_fp16.safetensors"), (SEEDVR2_REPO, "ema_vae_fp16.safetensors")),
    },
    precision=f"q{SEEDVR2_BITS}",
    tiers_key="tiers",
    stores_copy=True,
)

# The vendored PyTorch core (Windows and Linux) reads the Hub's FP8 files, downloaded into one folder because the
# core loads the transformer and the VAE from the same folder.
FP8_WEIGHTS = SeedVR2Weights(
    files={
        "seedvr2-7b": ((SEEDVR2_MIXED_REPO, "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"), (SEEDVR2_REPO, "ema_vae_fp16.safetensors")),
        "seedvr2-3b": ((SEEDVR2_REPO, "seedvr2_ema_3b_fp8_e4m3fn.safetensors"), (SEEDVR2_REPO, "ema_vae_fp16.safetensors")),
    },
    precision="fp8",
    tiers_key="discrete_tiers",
    stores_copy=False,
)


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


def seedvr2_settings(config: dict[str, Any], tiers_key: str = MLX_WEIGHTS.tiers_key) -> SeedVR2Settings:
    """Return the validated ``seedvr2`` block of *config*, with the tiers listed under *tiers_key*.

    Args:
        config: Loaded config dict.
        tiers_key: The tier list to read: ``tiers`` (Apple Silicon) or ``discrete_tiers`` (CUDA).

    Raises:
        ValueError: If a key is missing, has the wrong type or is out of range; the message names the key.
    """
    block = config.get("seedvr2")
    if not isinstance(block, dict):
        raise ValueError("config 'seedvr2' must be a mapping.")
    rows = block.get(tiers_key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"config 'seedvr2.{tiers_key}' must be a non-empty list.")
    tiers = tuple(_parse_tier(row, f"seedvr2.{tiers_key}[{index}]") for index, row in enumerate(rows))

    model = block.get("model", _AUTO)
    if not isinstance(model, str):
        raise ValueError(f"config 'seedvr2.model' must be a string, got {model!r}.")
    if model != _AUTO and model not in {tier.model for tier in tiers}:
        raise ValueError(f"config 'seedvr2.model' must be 'auto' or one of the 'seedvr2.{tiers_key}' models ({', '.join(tier.model for tier in tiers)}), got {model!r}.")
    softness = block.get("softness")
    if not _is_number(softness) or not 0 <= softness <= 1:
        raise ValueError(f"config 'seedvr2.softness' must be a number from 0 to 1, got {softness!r}.")
    reference = block.get("reference_megapixels")
    if not _is_number(reference) or reference <= 0:
        raise ValueError(f"config 'seedvr2.reference_megapixels' must be a positive number, got {reference!r}.")
    return SeedVR2Settings(model=model, softness=float(softness), reference_megapixels=float(reference), tiers=tiers)


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


def seedvr2_download_bytes(model: str, weights: SeedVR2Weights) -> int:
    """Return the bytes the first use of *model* downloads: its transformer file and the VAE file."""
    return sum(SEEDVR2_FILE_BYTES[name] for _repo, name in weights.files[model])


def seedvr2_size_label(num_bytes: int) -> str:
    """Return *num_bytes* in decimal gigabytes with one decimal, such as "17.0 GB"."""
    return f"{num_bytes / 1e9:.1f} GB"


def _tier_fits(tier: SeedVR2Tier, megapixels: float, budget_bytes: int | None, system_bytes: int | None) -> bool:
    if tier.system_gb is not None:
        return system_bytes is not None and tier.system_gb * 1e9 <= system_bytes
    return budget_bytes is not None and not seedvr2_over_budget(estimate_seedvr2_bytes(tier, megapixels), budget_bytes)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _parse_tier(row: object, name: str) -> SeedVR2Tier:
    if not isinstance(row, dict):
        raise ValueError(f"config '{name}' must be a mapping.")
    extra = sorted(set(row) - {"model", *_TIER_COSTS, _SYSTEM_COST})
    if extra:
        raise ValueError(f"config '{name}' has unknown keys: {', '.join(map(str, extra))}.")
    model = row.get("model")
    if not isinstance(model, str) or model not in SEEDVR2_MODELS:
        raise ValueError(f"config '{name}.model' must be one of {', '.join(SEEDVR2_MODELS)}, got {model!r}.")
    costs: dict[str, float] = {}
    for key in (*_TIER_COSTS, *([_SYSTEM_COST] if _SYSTEM_COST in row else [])):
        value = row.get(key)
        if not _is_number(value) or value <= 0:
            raise ValueError(f"config '{name}.{key}' must be a positive number, got {value!r}.")
        costs[key] = float(value)
    return SeedVR2Tier(model=model, **costs)
