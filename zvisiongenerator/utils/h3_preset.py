"""Value types of the MiniMax H3 preset and of the files it resolves to, and their construction from the model metadata and settings.

The types are plain data; the backends take them as arguments so no module has to read the config itself. The files and
numbers of the model are facts in the packaged metadata; only ``ffn_chunk_tokens`` is a setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.core.video_family import FrameRule
from zvisiongenerator.downloads.declaration import model_targets
from zvisiongenerator.downloads.target import DownloadTarget, HubFiles
from zvisiongenerator.utils.model_metadata import family_block
from zvisiongenerator.utils.paths import is_folder_name

__all__ = ["H3Files", "H3MlxFiles", "H3MlxPreset", "H3Preset", "H3Schedule", "h3_load_options", "h3_mlx_load_options", "h3_mlx_preset", "h3_preset", "h3_turbo_lora"]

_FAMILY = "h3"
_MODEL = "h3"


@dataclass(frozen=True)
class H3Schedule:
    """The sigma schedule of one step count: the number of transformer forwards and the shift of each modality."""

    forwards: int
    video_shift: float
    audio_shift: float

    @property
    def num_inference_steps(self) -> int:
        """The sigma grid size to give ``set_timesteps``: one point per forward plus the terminal zero."""
        return self.forwards + 1


@dataclass(frozen=True)
class H3Preset:
    """The CUDA (diffusers) H3 preset: the main repo's file filters, the text encoder, the turbo LoRA, the latent upscaler and the schedules."""

    repo: str
    revision: str
    repo_allow: tuple[str, ...]
    repo_ignore: tuple[str, ...]
    text_encoder: HubFiles
    text_encoder_layers: int
    lora: HubFiles
    lora_scale: float
    latent_upsampler: HubFiles
    schedules: dict[int, H3Schedule]
    ffn_chunk_tokens: int | None = None


@dataclass(frozen=True)
class H3MlxPreset:
    """The macOS (MLX) H3 preset: the transformer and component files, the quantization of the text encoder, the frame rule and the schedules.

    The frame rule and the schedules are the family's, shared with the CUDA preset, so both platforms run the same step counts, shifts and clip lengths.
    """

    transformer: HubFiles
    components: HubFiles
    text_encoder_bits: int
    text_encoder_group_size: int
    text_encoder_layers: int
    frame_rule: FrameRule
    schedules: dict[int, H3Schedule]


@dataclass(frozen=True)
class H3Files:
    """The resolved on-disk files of the CUDA H3 model."""

    repo_dir: Path
    text_encoder_dir: Path
    text_encoder_shards: tuple[Path, ...]
    lora_file: Path | None  # None for a converted folder, which holds its turbo merge
    latent_upsampler_file: Path | None = None


@dataclass(frozen=True)
class H3MlxFiles:
    """The resolved on-disk files of the macOS H3 model."""

    fl2va_dir: Path
    transformer_dir: Path
    text_encoder_source: Path
    lora_file: Path | None  # None for a converted folder, which holds its turbo merge


def h3_preset(config: dict[str, Any], platform_key: str) -> H3Preset:
    """Return the CUDA H3 preset: the files the metadata declares for *platform_key*, with ``ffn_chunk_tokens`` from *config*.

    Raises:
        ValueError: If *platform_key* declares no repository, text encoder, turbo LoRA or latent upscaler for H3 (macOS declares no text encoder or upscaler), or
            ``video_model_presets.h3.diffusers.ffn_chunk_tokens`` is not an integer of at least 1.
    """
    pairs = model_targets(_MODEL, platform_key)
    block = family_block(_FAMILY, None)
    repo = _only(pairs, "model", platform_key)
    lora, lora_scale = h3_turbo_lora(platform_key)
    return H3Preset(
        repo=repo.repo_id,
        revision=repo.revision,
        repo_allow=repo.allow_patterns,
        repo_ignore=repo.ignore_patterns or (),
        text_encoder=_hub_files(_only(pairs, "text_encoder", platform_key)),
        text_encoder_layers=block["text_encoder_layers"],
        lora=lora,
        lora_scale=lora_scale,
        latent_upsampler=_hub_files(_only(pairs, "latent_upsampler", platform_key)),
        schedules=_schedules(block),
        ffn_chunk_tokens=_ffn_chunk_tokens(config),
    )


def h3_mlx_preset() -> H3MlxPreset:
    """Return the macOS H3 preset: the INT8 transformer, the ``FL2VA`` components of the main repository, the text encoder's 4-bit form, the family's frame rule and its schedules."""
    pairs = model_targets(_MODEL, "darwin")
    block = family_block(_FAMILY, None)
    encoder = block["mlx_text_encoder"]
    return H3MlxPreset(
        transformer=_hub_files(_only(pairs, "transformer", "darwin")),
        components=_hub_files(_only(pairs, "model", "darwin")),
        text_encoder_bits=encoder["bits"],
        text_encoder_group_size=encoder["group_size"],
        text_encoder_layers=encoder["layers"],
        frame_rule=FrameRule(**block["frame_rule"]),
        schedules=_schedules(block),
    )


def h3_turbo_lora(platform_key: str) -> tuple[HubFiles, float]:
    """Return the turbo LoRA's file and strength: the ``role: lora`` companion of *platform_key* and the family's ``lora_scale``.

    Both platforms apply the same LoRA, so the macOS preset reads it from here rather than repeating it.

    Raises:
        ValueError: If *platform_key* declares no turbo LoRA for H3.
    """
    lora = _hub_files(_only(model_targets(_MODEL, platform_key), "lora", platform_key))
    return lora, float(family_block(_FAMILY, None)["lora_scale"])


def h3_mlx_load_options(model_name: str | None) -> dict[str, Any]:
    """Return the keyword arguments the macOS H3 backend's ``load_model`` takes beyond the mode and LoRAs.

    *model_name* names the text-encoder copy next to the model; a name that is not a plain folder name (a repo id or
    a path) is left out, and the backend then uses its own.
    """
    lora, lora_scale = h3_turbo_lora("darwin")
    options: dict[str, Any] = {"preset": h3_mlx_preset(), "lora": lora, "lora_scale": lora_scale}
    if model_name and is_folder_name(model_name):
        options["model_name"] = model_name
    return options


def h3_load_options(config: dict[str, Any], model_name: str | None, platform_key: str) -> dict[str, Any]:
    """Return the keyword arguments the CUDA H3 backend's ``load_model`` takes beyond the mode and LoRAs.

    Raises:
        ValueError: If the H3 files or ``ffn_chunk_tokens`` are unusable (see :func:`h3_preset`).
    """
    return {"model_name": model_name, "preset": h3_preset(config, platform_key)}


def _only(pairs: tuple[tuple[str, DownloadTarget], ...], role: str, platform_key: str) -> DownloadTarget:
    """Return the first target of *pairs* with *role*."""
    for target_role, target in pairs:
        if target_role == role:
            return target
    raise ValueError(f"MiniMax H3 declares no '{role}' download on '{platform_key}'.")


def _hub_files(target: DownloadTarget) -> HubFiles:
    """Return the pinned files of *target*."""
    return HubFiles(target.repo_id, target.revision, target.allow_patterns)


def _ffn_chunk_tokens(config: dict[str, Any]) -> int | None:
    """Return the ``video_model_presets.h3.diffusers.ffn_chunk_tokens`` setting, or ``None`` when it is unset."""
    value = ((((config.get("video_model_presets") or {}).get(_FAMILY) or {}).get("diffusers")) or {}).get("ffn_chunk_tokens")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("video_model_presets.h3.diffusers.ffn_chunk_tokens must be an integer of at least 1.")
    return value


def _schedules(block: dict[str, Any]) -> dict[int, H3Schedule]:
    """Return the sigma schedules of the family *block*, keyed by the step count a user picks."""
    return {int(steps): H3Schedule(**schedule) for steps, schedule in block["schedules"].items()}
