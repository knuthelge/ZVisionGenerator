"""Value types of the MiniMax H3 preset and of the files it resolves to.

These are plain data: the packaged ``video_model_presets.h3`` entry is parsed into them elsewhere, and the backends take
them as arguments so no module has to read the config itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["H3Files", "H3MlxFiles", "H3MlxPreset", "H3Preset", "H3Schedule", "HubFiles"]


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
    """The CUDA (diffusers) H3 preset: the main repo's file filters, the text encoder, the turbo LoRA and the schedules."""

    repo: str
    revision: str
    repo_allow: tuple[str, ...]
    repo_ignore: tuple[str, ...]
    text_encoder: HubFiles
    text_encoder_layers: int
    lora: HubFiles
    lora_scale: float
    schedules: dict[int, H3Schedule]
    attention: str = "sdpa"
    ffn_chunk_tokens: int | None = None
    qk_norm_chunk_tokens: int | None = None
    query_chunk: int | None = None


@dataclass(frozen=True)
class H3MlxPreset:
    """The macOS (MLX) H3 preset: the transformer and component files and the quantization of the text encoder."""

    transformer: HubFiles
    components: HubFiles
    text_encoder_bits: int
    text_encoder_group_size: int
    text_encoder_layers: int


@dataclass(frozen=True)
class H3Files:
    """The resolved on-disk files of the CUDA H3 model."""

    repo_dir: Path
    text_encoder_dir: Path
    text_encoder_shards: tuple[Path, ...]
    lora_file: Path


@dataclass(frozen=True)
class H3MlxFiles:
    """The resolved on-disk files of the macOS H3 model."""

    fl2va_dir: Path
    transformer_dir: Path
    text_encoder_source: Path
    lora_file: Path
