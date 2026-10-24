"""Value types of the MiniMax H3 preset and of the files it resolves to, and the parsing of the config entry into them.

The types are plain data; the backends take them as arguments so no module has to read the config itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.core.video_family import FrameRule
from zvisiongenerator.utils.paths import split_huggingface_revision

__all__ = ["H3Files", "H3MlxFiles", "H3MlxPreset", "H3Preset", "H3Schedule", "HubFiles", "h3_load_options", "h3_mlx_load_options", "h3_mlx_preset", "h3_preset", "h3_turbo_lora", "is_folder_name"]

_SUPPORTED_ATTENTION = ("sdpa",)
_MLX_TEXT_ENCODER_BITS = (4, 8)  # the levels a ``<name>@mlx-te<bits>`` folder name can carry
_BASE = "video_model_presets.h3"


@dataclass(frozen=True)
class HubFiles:
    """A set of files of one Hub repository, pinned to a revision."""

    repo_id: str
    revision: str
    files: tuple[str, ...]


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
    attention: str = "sdpa"
    ffn_chunk_tokens: int | None = None
    qk_norm_chunk_tokens: int | None = None
    query_chunk: int | None = None


@dataclass(frozen=True)
class H3MlxPreset:
    """The macOS (MLX) H3 preset: the transformer and component files, the quantization of the text encoder, the frame rule and the schedules.

    The frame rule (the family's) and the schedules (the ``diffusers`` block's) are shared with the CUDA preset, so a config change applies on both platforms.
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
    lora_file: Path
    latent_upsampler_file: Path | None = None


@dataclass(frozen=True)
class H3MlxFiles:
    """The resolved on-disk files of the macOS H3 model."""

    fl2va_dir: Path
    transformer_dir: Path
    text_encoder_source: Path
    lora_file: Path


def h3_preset(config: dict[str, Any]) -> H3Preset:
    """Return the CUDA H3 preset parsed from ``video_model_presets.h3`` of *config*.

    The main repository, the text encoder, the turbo LoRA and the latent upscaler must each name a revision (``owner/repo@revision``), so
    that the download plan and the runtime resolve the same files.

    Raises:
        ValueError: If the entry is missing or malformed; the message names the key path.
    """
    preset = (config.get("video_model_presets") or {}).get("h3")
    if not isinstance(preset, dict):
        raise ValueError(f"{_BASE} must be a mapping.")
    repo, revision = _pinned_repo(preset.get("repo_ref"), f"{_BASE}.repo_ref")
    diffusers = preset.get("diffusers")
    if not isinstance(diffusers, dict):
        raise ValueError(f"{_BASE}.diffusers must be a mapping.")
    path = f"{_BASE}.diffusers"
    encoder = _mapping(diffusers.get("text_encoder"), f"{path}.text_encoder")
    lora, lora_scale = _turbo_lora(diffusers, path)
    upsampler = _mapping(diffusers.get("latent_upsampler"), f"{path}.latent_upsampler")
    attention = diffusers.get("attention", "sdpa")
    if attention not in _SUPPORTED_ATTENTION:
        raise ValueError(f"{path}.attention must be one of: {', '.join(_SUPPORTED_ATTENTION)}.")
    return H3Preset(
        repo=repo,
        revision=revision,
        repo_allow=_patterns(diffusers.get("repo_allow"), f"{path}.repo_allow", required=True),
        repo_ignore=_patterns(diffusers.get("repo_ignore"), f"{path}.repo_ignore", required=False),
        text_encoder=HubFiles(*_pinned_repo(encoder.get("repo"), f"{path}.text_encoder.repo"), _patterns(encoder.get("files"), f"{path}.text_encoder.files", required=True)),
        text_encoder_layers=_positive_int(encoder.get("layers"), f"{path}.text_encoder.layers"),
        lora=lora,
        lora_scale=lora_scale,
        latent_upsampler=HubFiles(*_pinned_repo(upsampler.get("repo"), f"{path}.latent_upsampler.repo"), (_filename(upsampler.get("filename"), f"{path}.latent_upsampler.filename"),)),
        schedules=_schedules(diffusers.get("schedules"), f"{path}.schedules"),
        attention=attention,
        ffn_chunk_tokens=_optional_positive_int(diffusers.get("ffn_chunk_tokens"), f"{path}.ffn_chunk_tokens"),
    )


def h3_mlx_preset(config: dict[str, Any]) -> H3MlxPreset:
    """Return the macOS H3 preset parsed from ``video_model_presets.h3.mlx`` of *config*.

    The INT8 transformer names its own pinned repository. The component files (``FL2VA/...``) are on the main
    repository of ``repo_ref`` at its revision, so that the download plan and the runtime resolve the same files.

    The frame rule is the family's (``video_model_presets.h3.frame_rule``) and the schedules are those of
    ``video_model_presets.h3.diffusers.schedules``: both platforms run the same step counts, shifts and clip lengths.

    Raises:
        ValueError: If the entry is missing or malformed; the message names the key path.
    """
    from zvisiongenerator.utils.config import video_family  # imported here: the config module imports this one

    preset = (config.get("video_model_presets") or {}).get("h3")
    if not isinstance(preset, dict):
        raise ValueError(f"{_BASE} must be a mapping.")
    repo, revision = _pinned_repo(preset.get("repo_ref"), f"{_BASE}.repo_ref")
    family = video_family(config, "h3")
    diffusers_path = f"{_BASE}.diffusers"
    diffusers = _mapping(preset.get("diffusers"), diffusers_path)
    path = f"{_BASE}.mlx"
    mlx = _mapping(preset.get("mlx"), path)
    transformer = _mapping(mlx.get("transformer"), f"{path}.transformer")
    components = _mapping(mlx.get("components"), f"{path}.components")
    encoder = _mapping(mlx.get("text_encoder"), f"{path}.text_encoder")
    bits = _positive_int(encoder.get("bits"), f"{path}.text_encoder.bits")
    if bits not in _MLX_TEXT_ENCODER_BITS:
        raise ValueError(f"{path}.text_encoder.bits must be one of: {', '.join(str(level) for level in _MLX_TEXT_ENCODER_BITS)}.")
    return H3MlxPreset(
        transformer=HubFiles(*_pinned_repo(transformer.get("repo"), f"{path}.transformer.repo"), _patterns(transformer.get("files"), f"{path}.transformer.files", required=True)),
        components=HubFiles(repo, revision, _patterns(components.get("files"), f"{path}.components.files", required=True)),
        text_encoder_bits=bits,
        text_encoder_group_size=_positive_int(encoder.get("group_size"), f"{path}.text_encoder.group_size"),
        text_encoder_layers=_positive_int(encoder.get("layers"), f"{path}.text_encoder.layers"),
        frame_rule=family.frame_rule,
        schedules=_schedules(diffusers.get("schedules"), f"{diffusers_path}.schedules"),
    )


def h3_turbo_lora(config: dict[str, Any]) -> tuple[HubFiles, float]:
    """Return the turbo LoRA's file and strength from ``video_model_presets.h3.diffusers.lora`` of *config*.

    Both platforms apply the same LoRA, so the macOS preset reads this entry rather than repeating it.

    Raises:
        ValueError: If the entry is missing or malformed; the message names the key path.
    """
    preset = (config.get("video_model_presets") or {}).get("h3")
    if not isinstance(preset, dict):
        raise ValueError(f"{_BASE} must be a mapping.")
    path = f"{_BASE}.diffusers"
    return _turbo_lora(_mapping(preset.get("diffusers"), path), path)


def h3_mlx_load_options(config: dict[str, Any], model_name: str | None) -> dict[str, Any]:
    """Return the keyword arguments the macOS H3 backend's ``load_model`` takes beyond the mode and LoRAs.

    *model_name* names the text-encoder copy next to the model; a name that is not a plain folder name (a repo id or
    a path) is left out, and the backend then uses its own.

    Raises:
        ValueError: If the H3 entry of *config* is malformed (see :func:`h3_mlx_preset` and :func:`h3_turbo_lora`).
    """
    lora, lora_scale = h3_turbo_lora(config)
    options: dict[str, Any] = {"preset": h3_mlx_preset(config), "lora": lora, "lora_scale": lora_scale}
    if model_name and is_folder_name(model_name):
        options["model_name"] = model_name
    return options


def h3_load_options(config: dict[str, Any], model_name: str | None) -> dict[str, Any]:
    """Return the keyword arguments the CUDA H3 backend's ``load_model`` takes beyond the mode and LoRAs.

    Raises:
        ValueError: If the H3 entry of *config* is malformed (see :func:`h3_preset`).
    """
    return {"model_name": model_name, "preset": h3_preset(config)}


def is_folder_name(name: str) -> bool:
    """Return whether *name* can name a folder directly inside the models folder."""
    return name not in {".", ".."} and not any(char in name for char in "/\\:~")


def _turbo_lora(diffusers: dict[str, Any], base: str) -> tuple[HubFiles, float]:
    """Parse the ``lora`` entry of the *diffusers* block (found at *base*)."""
    lora = _mapping(diffusers.get("lora"), f"{base}.lora")
    files = HubFiles(*_pinned_repo(lora.get("repo"), f"{base}.lora.repo"), (_filename(lora.get("filename"), f"{base}.lora.filename"),))
    return files, _positive_number(lora.get("scale"), f"{base}.lora.scale")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    """Return *value* when it is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _pinned_repo(value: Any, path: str) -> tuple[str, str]:
    """Split ``owner/repo@revision`` into its parts."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a string such as owner/repo@revision.")
    repo, revision = split_huggingface_revision(value.strip())
    if not repo or revision is None:
        raise ValueError(f"{path} must name a revision (owner/repo@revision), got '{value}'.")
    return repo, revision


def _patterns(value: Any, path: str, *, required: bool) -> tuple[str, ...]:
    """Return a list of non-blank strings as a tuple; a missing list is empty unless *required*."""
    if value is None and not required:
        return ()
    if not isinstance(value, list) or (required and not value) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{path} must be a{' non-empty' if required else ''} list of non-blank strings.")
    return tuple(value)


def _filename(value: Any, path: str) -> str:
    """Return *value* when it is a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-blank string.")
    return value


def _positive_int(value: Any, path: str) -> int:
    """Return *value* when it is an integer of at least 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be an integer of at least 1.")
    return value


def _optional_positive_int(value: Any, path: str) -> int | None:
    """Return *value* when it is an integer of at least 1; a missing value is ``None``."""
    return None if value is None else _positive_int(value, path)


def _positive_number(value: Any, path: str) -> float:
    """Return *value* as a float when it is a number above 0."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{path} must be a number above 0.")
    return float(value)


def _schedules(value: Any, path: str) -> dict[int, H3Schedule]:
    """Parse the sigma schedules, keyed by the step count a user picks."""
    schedules = _mapping(value, path)
    if not schedules:
        raise ValueError(f"{path} must not be empty.")
    parsed: dict[int, H3Schedule] = {}
    for steps, entry in schedules.items():
        key = f"{path}.{steps}"
        block = _mapping(entry, key)
        parsed[_positive_int(steps, key)] = H3Schedule(
            forwards=_positive_int(block.get("forwards"), f"{key}.forwards"),
            video_shift=_positive_number(block.get("video_shift"), f"{key}.video_shift"),
            audio_shift=_positive_number(block.get("audio_shift"), f"{key}.audio_shift"),
        )
    return parsed
