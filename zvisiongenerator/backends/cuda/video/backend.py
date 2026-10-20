"""Windows/Linux CUDA video backend using diffusers LTX pipelines."""

from __future__ import annotations

import inspect
import json
import logging
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from zvisiongenerator.backends.cuda.driver import require_cuda
from zvisiongenerator.backends.cuda.lora_peft import is_unmatched_adapter_error, warn_unmatched_lora
from zvisiongenerator.core.video_types import ExtendSegment
from zvisiongenerator.utils.ltx_converted import read_converted_ltx
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE, CARRY_FRAMES, EXTEND_FRAME_STEP, audio_lead_samples, cut_window_audio, window_audio_cut, window_frames
from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model

if TYPE_CHECKING:
    from zvisiongenerator.backends.cuda.video.audio_carry import AudioCarry
    from zvisiongenerator.core.video_types import ExtendSource

logger = logging.getLogger(__name__)

_MINIMUM_DIFFUSERS_VERSION = (0, 41, 0)
# The pipelines' own default prompt length, in tokens.
_MAX_SEQUENCE_LENGTH = 1024
# Tokens a token-wise layer (feed-forward, RMSNorm) computes at once: a feed-forward's 4x-wide activation is then
# 128 MB and an RMSNorm's float32 copy 64 MB, however long the clip.
_TOKEN_CHUNK = 4096
# A load whose estimated peak passes this share of system memory leaves little to the rest of the machine.
_SYSTEM_MEMORY_TIGHT_RATIO = 0.8
# A distilled checkpoint runs without guidance, as on macOS. Every guidance term is set to its neutral value:
# diffusers 0.40+ default to audio guidance 7, spatio-temporal guidance 1 and modality guidance 3 even at
# guidance_scale 1.0, which over-sharpens the video, distorts the audio and adds three extra passes per step.
_DISTILLED_GUIDANCE: Final[Mapping[str, float]] = MappingProxyType(
    {"guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0, "audio_stg_scale": 0.0, "modality_scale": 1.0, "audio_modality_scale": 1.0}
)
# The pipeline class (a ``_PipelineClasses`` field) each load mode builds.
_PIPELINES_BY_MODE: Final[Mapping[str, str]] = MappingProxyType({"t2v": "text_to_video", "i2v": "image_to_video", "extend": "condition"})
# Architecture of the LTX-2.3 x2 spatial latent upsampler (matches the published weights).
_UPSAMPLER_CONFIG: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "in_channels": 128,
        "mid_channels": 1024,
        "num_blocks_per_stage": 4,
        "dims": 3,
        "spatial_upsample": True,
        "temporal_upsample": False,
        "use_rational_resampler": False,
    }
)


@dataclass(frozen=True)
class _PipelineClasses:
    """Resolved diffusers LTX pipeline classes."""

    text_to_video: type[Any]
    image_to_video: type[Any]
    condition: type[Any]
    video_condition: type[Any]
    latent_upsample_pipeline: type[Any]
    latent_upsampler_model: type[Any]


@dataclass(frozen=True)
class _RuntimeDependencies:
    """Lazy-loaded runtime dependencies for diffusers video generation."""

    torch: Any
    image_module: Any
    encode_video: Any
    pipeline_classes: _PipelineClasses
    diffusers_version: str
    distilled_sigmas: tuple[float, ...]
    stage2_sigmas: tuple[float, ...]


@dataclass(frozen=True)
class _GeneratedClip:
    """Decoded frames of one clip with its audio waveform ``(channels, samples)`` and sample rate, when the pipeline made audio."""

    frames: list[Any]
    audio: Any | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class _TextEncoderSource:
    """Where the text encoder loads from for each new prompt: the repo folder (its config) and its prepared FP8 copy."""

    repo_dir: Path
    prepared: Path
    torch_dtype: Any


@dataclass(frozen=True)
class _TransformerSource:
    """Where the transformer loads from, how it is placed, and whether it gives way to the text encoder.

    With *swap* the transformer and the text encoder do not fit in system memory together: the transformer loads
    after a prompt is encoded and is released before the next new prompt's text encoder loads.
    """

    config: dict[str, Any]
    prepared: Path
    quantize: int
    torch_dtype: Any
    low_memory: bool
    loras: tuple[tuple[str, float], ...]
    swap: bool


@dataclass
class _LoadedVideoModel:
    """Reusable loaded diffusers video pipelines for one request mode.

    The pipelines have no text encoder: it loads for a new prompt and is set only while encoding
    (see :func:`_encode_prompt`). Without *text_encoder* the pipeline encodes prompts itself. With *transformer* the
    pipeline's transformer may be absent until a prompt is encoded (see :func:`_ensure_transformer`).
    """

    text_to_video: Any | None
    image_to_video: Any | None
    upsample_pipeline: Any | None
    runtime: _RuntimeDependencies
    model_info: VideoModelInfo
    low_memory: bool
    extend: Any | None = None
    text_encoder: _TextEncoderSource | None = None
    transformer: _TransformerSource | None = None
    prompt_cache: tuple[str, dict[str, Any]] | None = None
    resident_text_encoder: Any | None = None  # kept between prompts when it fits beside the transformer


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into comparable integer parts."""

    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _resolve_pipeline_classes(diffusers_module: Any, upsampler_model: type[Any], video_condition: type[Any]) -> _PipelineClasses:
    """Resolve the LTX-2 pipeline classes from diffusers and pair them with the latent upsampler model and video condition classes."""

    names = ("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2ConditionPipeline", "LTX2LatentUpsamplePipeline")
    if not all(hasattr(diffusers_module, name) for name in names):
        raise RuntimeError(f"Installed diffusers is missing the LTX-2 video pipelines ({', '.join(names)}).")
    return _PipelineClasses(
        text_to_video=diffusers_module.LTX2Pipeline,
        image_to_video=diffusers_module.LTX2ImageToVideoPipeline,
        condition=diffusers_module.LTX2ConditionPipeline,
        video_condition=video_condition,
        latent_upsample_pipeline=diffusers_module.LTX2LatentUpsamplePipeline,
        latent_upsampler_model=upsampler_model,
    )


def _validate_diffusers_version(diffusers_version: str) -> None:
    """Ensure the installed diffusers version satisfies the video backend minimum."""

    if _parse_version(diffusers_version) < _MINIMUM_DIFFUSERS_VERSION:
        minimum = ".".join(str(part) for part in _MINIMUM_DIFFUSERS_VERSION)
        raise RuntimeError(f"diffusers>={minimum} is required for Windows/Linux LTX video support. Installed: {diffusers_version}.")


def _load_runtime_dependencies() -> _RuntimeDependencies:
    """Import heavy runtime dependencies lazily for video generation."""

    try:
        import torch
        from PIL import Image
        import av  # noqa: F401 - the encoder needs it
        import diffusers

        from zvisiongenerator.backends.cuda.video.encode import encode_video
    except ImportError as exc:
        raise RuntimeError("Video diffusers dependencies are unavailable. Install the Windows/Linux video runtime with `uv sync`.") from exc

    diffusers_version = getattr(diffusers, "__version__", "0")
    _validate_diffusers_version(diffusers_version)

    try:
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
    except ImportError as exc:
        raise RuntimeError("Installed diffusers is missing the LTX-2 distilled sigma tables, the latent upsampler model or the video condition.") from exc

    return _RuntimeDependencies(
        torch=torch,
        image_module=Image,
        encode_video=encode_video,
        pipeline_classes=_resolve_pipeline_classes(diffusers, LTX2LatentUpsamplerModel, LTX2VideoCondition),
        diffusers_version=diffusers_version,
        distilled_sigmas=tuple(DISTILLED_SIGMA_VALUES),
        stage2_sigmas=tuple(STAGE_2_DISTILLED_SIGMA_VALUES),
    )


def _platform_label() -> str:
    """Return a human-readable label for the active platform."""

    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "linux":
        return "Linux"
    return sys.platform


def _validate_cuda(runtime: _RuntimeDependencies) -> None:
    """Ensure the current runtime has an NVIDIA CUDA device available."""

    require_cuda(runtime.torch, f"The {_platform_label()} diffusers video backend")


def _configure_torch_runtime(torch: Any) -> None:
    """Apply the same lightweight CUDA tuning used by the image diffusers backend."""
    from zvisiongenerator.backends.cuda.memory import configure_allocator

    configure_allocator()
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _select_torch_dtype(torch: Any) -> Any:
    """Choose the default CUDA dtype for video generation."""

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_bf16_supported") and cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _enable_vae_tiling(pipeline: Any) -> None:
    """Make the pipeline VAE decode in tiles, in space and in time, when it supports it.

    The LTX-2 VAE tiles only in space by default, so every tile decodes all frames at once and a long clip runs out of
    GPU memory at the end. Frame-wise decoding takes a few latent frames at a time, so decoding memory no longer grows
    with the clip's length.
    """

    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        if hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = True


def _load_generation_pipeline(
    pipeline_class: type[Any],
    model_path: str,
    runtime: _RuntimeDependencies,
    *,
    torch_dtype: Any,
    low_memory: bool,
    quantize: int,
    transformer_source: tuple[str, str] | None,
    loras: list[tuple[str, float]],
    model_name: str | None,
    converted: dict[str, Any] | None = None,
) -> tuple[Any, _TextEncoderSource, _TransformerSource]:
    """Load the generation pipeline without its text encoder, placed for the memory mode.

    The transformer (at *quantize*, with *loras*) is attached now when it fits in system memory beside the text
    encoder, and otherwise only once the first prompt is encoded. A converted folder (*converted* is its manifest)
    loads its own stored weights; otherwise they are the prepared copies of the picked alias.

    Returns:
        The pipeline, and where its text encoder and transformer load from.
    """

    from zvisiongenerator.backends.cuda.memory import release_memory

    if converted is not None:
        repo_dir, prepared_transformer, prepared_text_encoder, fits_together = _converted_weights(Path(model_path).expanduser(), converted, runtime.torch.device("cuda"))
    else:
        repo_dir, prepared_transformer, prepared_text_encoder, fits_together = _stock_weights(model_path, model_name, transformer_source, quantize, runtime.torch.device("cuda"))
    connectors = _load_connectors(repo_dir, torch_dtype)
    try:
        pipeline = pipeline_class.from_pretrained(str(repo_dir), transformer=None, connectors=connectors, text_encoder=None, torch_dtype=torch_dtype)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {pipeline_class.__name__} from '{model_path}': {exc}") from exc
    _place_components(pipeline, runtime.torch, low_memory=low_memory)
    transformer = _TransformerSource(
        config=_transformer_config(repo_dir),
        prepared=prepared_transformer,
        quantize=quantize,
        torch_dtype=torch_dtype,
        low_memory=low_memory,
        loras=tuple(loras),
        swap=low_memory and not fits_together,  # without low memory the transformer lives on the GPU
    )
    if not transformer.swap:
        _attach_transformer(pipeline, transformer, runtime.torch)
    else:
        release_memory()  # the connectors' full-precision copy and the components' CPU copies
    return pipeline, _TextEncoderSource(repo_dir=repo_dir, prepared=prepared_text_encoder, torch_dtype=torch_dtype), transformer


def _attach_transformer(pipeline: Any, source: _TransformerSource, torch: Any) -> None:
    """Load the transformer from *source*, place it for the memory mode, give it to *pipeline* and apply the LoRAs."""

    from zvisiongenerator.backends.cuda.block_streaming import unload_before
    from zvisiongenerator.backends.cuda.video.weights import load_transformer
    from zvisiongenerator.backends.cuda.memory import release_memory
    from zvisiongenerator.backends.cuda.video.timestep_runs import modulate_by_runs

    transformer = load_transformer(source.config, prepared=source.prepared, bits=source.quantize, compute_dtype=source.torch_dtype)
    _place_transformer(transformer, torch, quantize=source.quantize, low_memory=source.low_memory)
    _chunk_token_wise_layers(transformer)
    modulate_by_runs(transformer)  # image-to-video's per-token timesteps without a per-token modulation copy
    if source.low_memory:
        unload_before(transformer, _codec_components(pipeline))  # the VAEs leave the GPU while it denoises
    pipeline.transformer = transformer
    # The pipeline reads its patch sizes from the transformer it was built with, which was none.
    config = getattr(transformer, "config", None)
    if config is not None:
        pipeline.transformer_spatial_patch_size = config.patch_size
        pipeline.transformer_temporal_patch_size = config.patch_size_t
    if source.loras:
        _apply_loras(pipeline, list(source.loras))
    release_memory()  # the unpinned originals the streaming copied from


def _release_transformer(pipeline: Any, source: _TransformerSource) -> None:
    """Remove the transformer (and its LoRAs, from the connectors too) from *pipeline* and free its memory."""

    from zvisiongenerator.backends.cuda.memory import release_memory

    if source.loras and hasattr(pipeline, "unload_lora_weights"):
        pipeline.unload_lora_weights()
    pipeline.transformer = None
    release_memory()


def _ensure_transformer(model: _LoadedVideoModel, pipeline: Any) -> None:
    """Attach the transformer to *pipeline* if it was left out or released for the text encoder."""

    if model.transformer is not None and getattr(pipeline, "transformer", None) is None:
        _attach_transformer(pipeline, model.transformer, model.runtime.torch)


def _repo_dir(model_path: str, *, with_transformer: bool) -> Path:
    """Return the local folder of the diffusers repo *model_path*, downloading what is missing for a repo id.

    Without *with_transformer* the repo's own transformer weights are not downloaded (only its config).
    """

    local = Path(model_path).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    ignore = None if with_transformer else ["transformer/*.safetensors", "transformer/*.safetensors.index.json"]
    try:
        return Path(snapshot_download(model_path, ignore_patterns=ignore))
    except Exception as exc:
        raise RuntimeError(f"Failed to download the LTX model '{model_path}': {exc}") from exc


def _download_hub_file(repo_id: str, filename: str, label: str) -> Path:
    """Download one file from the Hugging Face Hub (or find it in the cache) and return its local path."""

    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        raise RuntimeError(f"Failed to download the {label} '{filename}' from '{repo_id}': {exc}") from exc


def _resolve_upsampler_weights(repo_id: str, filename: str) -> str:
    """Download one latent upsampler weights file from the Hugging Face Hub and return its local path."""

    return str(_download_hub_file(repo_id, filename, "LTX latent upsampler"))


def _stock_weights(model_path: str, model_name: str | None, transformer_source: tuple[str, str] | None, quantize: int, device: Any) -> tuple[Path, Path, Path, bool]:
    """Return the repo folder, the prepared transformer and text encoder at *quantize*, and whether they fit together, for a stock model."""

    from zvisiongenerator.backends.cuda.video.memory_estimate import ltx_system_estimate

    checkpoint = _download_hub_file(*transformer_source, "LTX transformer") if transformer_source else None
    repo_dir = _repo_dir(model_path, with_transformer=checkpoint is None)
    transformer_input = checkpoint or repo_dir / "transformer"
    fits_together = _check_system_memory(ltx_system_estimate(repo_dir, transformer_input, quantize), quantize, converted=False)
    prepared_transformer, prepared_text_encoder = _prepared_weights(model_name, model_path, repo_dir, transformer_input, quantize, device)
    return repo_dir, prepared_transformer, prepared_text_encoder, fits_together


def _converted_weights(model_dir: Path, manifest: dict[str, Any], device: Any) -> tuple[Path, Path, Path, bool]:
    """Return the converted folder *model_dir*, its stored transformer and text encoder, and whether they fit together.

    Nothing is read from the source checkpoint, the configured transformer or the alias table.
    """

    from zvisiongenerator.backends.cuda.video.converted import check_converted
    from zvisiongenerator.backends.cuda.video.memory_estimate import converted_ltx_system_estimate

    fits_together = _check_system_memory(converted_ltx_system_estimate(model_dir, manifest), manifest["bits"], converted=True)
    check_converted(model_dir, manifest, device)
    return model_dir, model_dir / "transformer", model_dir / "text_encoder", fits_together


def _check_system_memory(estimate: Any, quantize: int, *, converted: bool) -> bool:
    """Refuse a load that cannot fit in system memory, and warn when it leaves little to the rest of the machine.

    Args:
        estimate: The model's :class:`LtxCudaMemoryEstimate`.
        quantize: Its level.
        converted: Whether it is a converted folder, which gets smaller by converting it again rather than by picking ``ltx-4``.

    Returns:
        Whether the transformer and the text encoder fit in system memory together; otherwise they take turns.

    Raises:
        RuntimeError: If even taking turns needs more than the machine's system memory.
    """

    import psutil

    total = int(psutil.virtual_memory().total)
    need = estimate.taking_turns
    level = f"q{quantize}"
    smaller = None
    if quantize == 8:
        smaller = "convert the checkpoint again with --quantize 4 under a new name" if converted else "use ltx-4"
    if need > total:
        advice = f" {smaller[0].upper()}{smaller[1:]} instead." if smaller else ""
        raise RuntimeError(f"LTX at {level} needs about {need / 1024**3:.0f} GB of system memory; this machine has {total / 1024**3:.0f} GB.{advice}")
    limit = total * _SYSTEM_MEMORY_TIGHT_RATIO
    if need > limit:
        advice = f" If it is stopped for lack of memory, {smaller}." if smaller else ""
        warnings.warn(f"LTX at {level} needs about {need / 1024**3:.0f} GB of this machine's {total / 1024**3:.0f} GB of system memory, leaving little for other apps.{advice}", stacklevel=4)
    return estimate.together <= limit


def _prepared_weights(model_name: str | None, model_path: str, repo_dir: Path, transformer_input: Path, quantize: int, device: Any) -> tuple[Path, Path]:
    """Return the prepared copies of the transformer and text encoder at *quantize*, writing them the first time.

    They live in the models folder as the stored copy of the picked alias, ``<models>/<name>@q<bits>`` (e.g.
    ``ltx-4@q4``), so the Models page counts them with the alias and deletes them with it. The text encoder is the
    same at every level: one already prepared beside another alias is hard-linked instead of written again.
    Writing reads the source weights once (tens of gigabytes) and can take several minutes.
    """

    from zvisiongenerator.backends.cuda.video.weights import ensure_prepared, prepared_text_encoders, text_encoder_sources, transformer_sources, write_text_encoder, write_transformer
    from zvisiongenerator.utils.paths import get_ziv_data_dir
    from zvisiongenerator.utils.stored_quant import stored_quant_name

    models_dir = get_ziv_data_dir() / "models"
    copy = models_dir / stored_quant_name(model_name or model_path.strip().replace("/", "--"), quantize)
    transformer, text_encoder = copy / "transformer", copy / "text_encoder"
    config = _transformer_config(repo_dir)
    if not (transformer.is_dir() and text_encoder.is_dir()):
        logger.info("Preparing the q%d LTX weights in %s once; this reads the full weights and can take several minutes.", quantize, copy)
    ensure_prepared(transformer, transformer_sources(transformer_input), quantize, lambda folder: write_transformer(transformer_input, config, folder, quantize, device))
    ensure_prepared(text_encoder, text_encoder_sources(repo_dir), 8, lambda folder: write_text_encoder(repo_dir, folder), reuse=prepared_text_encoders(models_dir))
    return transformer, text_encoder


def _transformer_config(repo_dir: Path) -> dict[str, Any]:
    """Return the diffusers transformer config of the repo at *repo_dir*."""

    return json.loads((repo_dir / "transformer" / "config.json").read_text(encoding="utf-8"))


def _load_connectors(repo_dir: Path, torch_dtype: Any) -> Any:
    """Load the text connectors with FP8 weight storage."""

    from zvisiongenerator.backends.cuda.video.weights import load_connectors

    return load_connectors(repo_dir / "connectors", compute_dtype=torch_dtype)


def _place_components(pipeline: Any, torch: Any, *, low_memory: bool) -> None:
    """Put the pipeline's parts other than the transformer where they run.

    In low-memory mode the connectors stream to the GPU, and the VAEs and vocoder move there only when they encode
    or decode (the transformer sends them back, see :func:`_attach_transformer`), so they take no GPU memory while
    it denoises. Otherwise everything moves to the GPU.
    """

    from zvisiongenerator.backends.cuda.block_streaming import group_offload, load_when_used

    device = torch.device("cuda")
    for component in _codec_components(pipeline):
        if low_memory:
            entry_points = [getattr(component, name) for name in ("encoder", "decoder") if getattr(component, name, None) is not None]
            load_when_used(component, entry_points or [component], device)
        else:
            component.to(device)
    if low_memory:
        group_offload(pipeline.connectors, device, "leaf_level")
    else:
        pipeline.connectors.to(device)
    _enable_vae_tiling(pipeline)


def _codec_components(pipeline: Any) -> list[Any]:
    """Return the pipeline's VAEs and vocoder: the parts that encode and decode around the transformer."""

    return [component for name in ("vae", "audio_vae", "vocoder") if (component := getattr(pipeline, name, None)) is not None]


def _place_transformer(transformer: Any, torch: Any, *, quantize: int, low_memory: bool) -> None:
    """Stream the transformer to the GPU block by block in low-memory mode, else move it there."""

    from zvisiongenerator.backends.cuda.block_streaming import group_offload, move_outside, stream_blocks

    device = torch.device("cuda")
    if not low_memory:
        transformer.to(device)
    elif quantize == 4:
        # NF4 weights carry quantization state that diffusers' group offloading leaves on the CPU.
        stream_blocks(transformer.transformer_blocks, device)
        move_outside(transformer, transformer.transformer_blocks, device)
    else:
        group_offload(transformer, device, "block_level")


def _chunk_token_wise_layers(transformer: Any) -> None:
    """Run the token-wise layers of every transformer block over at most ``_TOKEN_CHUNK`` tokens at a time.

    These are the video and audio feed-forward layers and every RMSNorm (including the attention q/k norms): each
    computes tokens independently, so chunking leaves the output unchanged.
    """

    from zvisiongenerator.backends.cuda.video.token_chunking import chunk_tokens

    modules: list[Any] = []
    for block in getattr(transformer, "transformer_blocks", ()):
        modules.extend(getattr(block, name) for name in ("ff", "audio_ff") if getattr(block, name, None) is not None)
        if hasattr(block, "modules"):
            modules.extend(module for module in block.modules() if type(module).__name__ == "RMSNorm")
    chunk_tokens(modules, _TOKEN_CHUNK)


def _read_state_dict(path: str) -> dict[str, Any]:
    """Read a safetensors file into a CPU state dict."""

    from safetensors.torch import load_file

    return load_file(path)


def _load_latent_upsampler(model_class: type[Any], state_dict: Mapping[str, Any], *, torch_dtype: Any, low_memory: bool) -> Any:
    """Build the x2 latent upsampler, load its weights strictly and place it for the memory mode.

    In low-memory mode the model stays on the CPU so it can be moved to the GPU only for the upsample pass.
    """

    model = model_class(**_UPSAMPLER_CONFIG)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"The latent upsampler weights do not match the LTX-2.3 x2 spatial upsampler layout (check video_model_presets.ltx.diffusers.latent_upsampler): {exc}") from exc
    if low_memory:
        model.to(dtype=torch_dtype)
    else:
        model.to(device="cuda", dtype=torch_dtype)
    model.eval()
    return model


def _build_upsample_pipeline(pipeline_class: type[Any], generation_pipeline: Any, upsampler: Any) -> Any:
    """Wrap the upsampler in an upsample pipeline that shares the generation pipeline's VAE.

    The result must never be moved with ``.to()`` or given CPU offload: either would re-hook the shared VAE.
    """

    return pipeline_class(vae=generation_pipeline.vae, latent_upsampler=upsampler)


def _call_signature_parameters(callable_obj: Any) -> dict[str, inspect.Parameter] | None:
    """Return signature parameters for a callable when introspection succeeds."""

    try:
        return inspect.signature(callable_obj).parameters
    except TypeError, ValueError:
        return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only those supported by the target callable signature."""

    parameters = _call_signature_parameters(callable_obj)
    if parameters is None or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _check_review_arguments(review: Callable[[Path], None] | None, stage1_steps: int | None) -> None:
    """Ensure a review is only asked for with the two-stage upscale, the only run that has a half-size render to look at.

    Raises:
        ValueError: If *review* is set without *stage1_steps*.
    """

    if review is not None and stage1_steps is None:
        raise ValueError("review needs stage1_steps: only the two-stage upscale has a half-size render to review")


def _preview_small_render(model: _LoadedVideoModel, pipeline: Any, stage1: Any, review: Callable[[Path], None]) -> None:
    """Decode the half-size result of stage 1 to a temporary video file and hand its path to *review*.

    The file lives in a temporary directory outside the output folder and is deleted when *review* returns or raises; a
    file still open elsewhere (the browser streaming it on Windows) is left for the system to clean up. The decode
    streams like the final one, and in low-memory mode the codecs go back to the CPU right after it, before the
    wait for the decision and the latent upsample (they would otherwise stay on the GPU until the transformer next runs).

    Args:
        model: The loaded model, for the frame rate and the video encoder.
        pipeline: The pipeline that has just run stage 1.
        stage1: Its output (``output_type="latent"``).
        review: Receives the path of the decoded video; it raises to end the generation.
    """

    with tempfile.TemporaryDirectory(prefix="ziv-review-", ignore_cleanup_errors=True) as directory:
        path = Path(directory) / "small-render.mp4"
        _export_video(model, _decode_clip(pipeline, stage1), str(path))
        if model.low_memory:
            for component in _codec_components(pipeline):
                component.to("cpu")
            model.runtime.torch.cuda.empty_cache()
        review(path)


def _fit_image(image: Any, width: int, height: int) -> Any:
    """Scale *image* to cover *width* × *height* with its aspect ratio kept, and crop the overflow from the center.

    The diffusers pipeline would otherwise stretch the image to the video's size; the macOS pipeline crops the same way.
    """

    from PIL import Image, ImageOps

    return ImageOps.fit(image, (width, height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))


def _make_generator(torch: Any, seed: int | None) -> Any | None:
    """Create a deterministic generator when a seed is provided."""

    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(seed)


def _stage1_sigmas(table: Sequence[float], steps: int) -> list[float]:
    """Return the distilled sigmas for *steps* denoising steps, keeping the start and the tail of the table."""

    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if steps >= len(table):
        return list(table)
    return [table[0], *table[len(table) - steps + 1 :]]


def _make_step_callback(step_callback: Callable[[dict[str, Any]], None] | None, *, phase: str, offset: int, total: int) -> Callable[..., Any] | None:
    """Adapt diffusers callback events to the existing backend callback payload."""

    if step_callback is None:
        return None

    safe_total = max(total, 1)

    def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: Any) -> Any:
        step_callback(
            {
                "phase": phase,
                "current_step": min(offset + step + 1, safe_total),
                "total_steps": safe_total,
            }
        )
        return callback_kwargs

    return _on_step_end


def _decode_clip(pipeline: Any, result: Any) -> _GeneratedClip:
    """Decode a final pass's output (``output_type="latent"``) to 8-bit frames, its audio waveform and the audio's sample rate.

    The video decodes a few frames at a time straight to system memory (``streaming_decode``), so neither GPU nor
    system memory holds a full-precision copy of the clip.
    """

    from zvisiongenerator.backends.cuda.video.streaming_decode import decode_frames

    frames = decode_frames(pipeline.vae, result.frames)
    return _GeneratedClip(frames=frames, audio=_decode_audio(pipeline, getattr(result, "audio", None)), sample_rate=_audio_sample_rate(pipeline))


def _decode_audio(pipeline: Any, audio_latents: Any | None) -> Any | None:
    """Return the first clip's waveform decoded from *audio_latents*, or ``None`` without audio latents or an audio decoder."""

    audio_vae = getattr(pipeline, "audio_vae", None)
    vocoder = getattr(pipeline, "vocoder", None)
    if audio_latents is None or audio_vae is None or vocoder is None:
        return None
    mel = audio_vae.decode(audio_latents.to(audio_vae.dtype), return_dict=False)[0]
    waveform = vocoder(mel)
    return waveform[0] if len(waveform) else None


def _audio_sample_rate(pipeline: Any) -> int | None:
    """Return the sample rate of the pipeline vocoder's output waveform, or ``None`` when the pipeline has no vocoder."""

    config = getattr(getattr(pipeline, "vocoder", None), "config", None)
    rate = getattr(config, "output_sampling_rate", None)
    return int(rate) if rate else None


def _export_video(model: _LoadedVideoModel, clip: _GeneratedClip, output_path: str) -> Path:
    """Export a generated clip to the requested output path, muxing its audio track when it has one."""

    if not clip.frames:
        raise RuntimeError("Diffusers video generation returned no frames to export.")
    fps = model.model_info.default_fps
    audio = clip.audio
    if audio is not None and clip.sample_rate is None:
        warnings.warn("The video pipeline returned audio without a vocoder sample rate; the video is saved without sound.", stacklevel=2)
        audio = None
    if audio is not None:
        audio = audio.float().cpu()
    model.runtime.encode_video(clip.frames, fps=fps, output_path=output_path, audio=audio, audio_sample_rate=clip.sample_rate if audio is not None else None)
    return Path(output_path)


def _audio_tokens(pipeline: Any, frames: int, fps: float) -> int:
    """Return the audio tokens that cover *frames* video frames at *fps*, at least one, as the pipeline counts them."""

    per_second = pipeline.audio_sampling_rate / pipeline.audio_hop_length / pipeline.audio_vae_temporal_compression_ratio
    return max(1, round(frames / fps * per_second))


def _write_extend_segment(clip: _GeneratedClip, source: ExtendSource, extend_frames: int, directory: Path, *, audio: bool) -> ExtendSegment:
    """Write the new part of a decoded extension window: its frames to ``segment.mp4`` and, with *audio*, its sound to ``segment.wav``.

    The video is encoded with the source's size, frame rate and colour tags so that it can be stream-copied onto it.
    The WAV starts ``audio_lead_samples`` before the new part, as the join expects.

    Raises:
        RuntimeError: If the window does not have ``25 + extend_frames`` frames, or sound is wanted and the window has
            none at the join's sample rate.
    """
    from zvisiongenerator.utils.video_splice import build_segment_command, write_pcm16_wav, write_segment_frames

    expected = window_frames(extend_frames)
    if len(clip.frames) != expected:
        raise RuntimeError(f"The extension window decoded to {len(clip.frames)} frames, expected {expected}.")
    video_path = directory / "segment.mp4"
    command = build_segment_command(
        source.width,
        source.height,
        source.fps_text,
        video_path,
        color_space=source.color_space,
        color_primaries=source.color_primaries,
        color_trc=source.color_trc,
    )
    write_segment_frames(clip.frames[CARRY_FRAMES:], command)
    if not audio:
        return ExtendSegment(video_path=video_path)
    if clip.audio is None or clip.sample_rate != AUDIO_SAMPLE_RATE:
        raise RuntimeError(f"The extension window has no sound at {AUDIO_SAMPLE_RATE} Hz to continue the clip's (got {clip.sample_rate}).")
    skip, keep = window_audio_cut(extend_frames, source.fps)
    lead = audio_lead_samples(source.fps)
    audio_path = directory / "segment.wav"
    write_pcm16_wav(cut_window_audio(clip.audio.float().cpu().numpy(), skip, keep, lead), audio_path)
    return ExtendSegment(video_path=video_path, audio_path=audio_path, lead_samples=lead)


def _check_lora_files(loras: list[tuple[str, float]]) -> None:
    """Raise when a LoRA file is missing or in a format the diffusers backend does not load.

    Raises:
        FileNotFoundError: If a LoRA path does not exist.
        ValueError: If a LoRA file is not ``.safetensors`` or ``.bin``.
    """

    for path, _weight in loras:
        lora_path = Path(path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {path}")
        if lora_path.is_file() and lora_path.suffix.lower() not in {".safetensors", ".bin"}:
            raise ValueError(f"Unsupported LoRA format for diffusers LTX backend: {path}")


def _apply_loras(pipeline: Any, loras: list[tuple[str, float]]) -> None:
    """Load diffusers-compatible LoRA adapters onto the active generation pipeline.

    A LoRA none of whose tensors match the pipeline is skipped with a warning.
    """

    if not hasattr(pipeline, "load_lora_weights"):
        raise RuntimeError("The installed diffusers LTX pipeline does not support LoRA adapters.")

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    load_signature = _call_signature_parameters(pipeline.load_lora_weights) or {}
    supports_adapter_name = "adapter_name" in load_signature or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in load_signature.values())

    _check_lora_files(loras)
    for index, (path, weight) in enumerate(loras):
        lora_path = Path(path)
        adapter_name = f"lora_{index}"
        load_kwargs = {"adapter_name": adapter_name} if supports_adapter_name else {}
        try:
            pipeline.load_lora_weights(str(lora_path), **load_kwargs)
        except TypeError as exc:
            raise RuntimeError(f"The installed diffusers LoRA API is incompatible with '{path}'.") from exc
        except Exception as exc:
            if is_unmatched_adapter_error(exc):
                warn_unmatched_lora(path)
                continue
            raise RuntimeError(f"Failed to load LoRA '{path}': {exc}") from exc

        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    if not adapter_names:
        return

    if hasattr(pipeline, "set_adapters"):
        set_kwargs = _filter_supported_kwargs(
            pipeline.set_adapters,
            {"adapter_names": adapter_names, "adapter_weights": adapter_weights},
        )
        if any(weight != 1.0 for weight in adapter_weights) and "adapter_weights" not in set_kwargs:
            raise RuntimeError("The installed diffusers adapter API does not support explicit LoRA scales.")
        try:
            pipeline.set_adapters(**set_kwargs)
        except TypeError:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except Exception as exc:
            if any(weight != 1.0 for weight in adapter_weights):
                raise RuntimeError("The installed diffusers adapter API does not support explicit LoRA scales.") from exc
            raise RuntimeError(f"Failed to activate LoRA adapters: {exc}") from exc
        return

    if len(adapter_names) > 1 or any(weight != 1.0 for weight in adapter_weights):
        raise RuntimeError("The installed diffusers adapter API does not support multiple LoRAs or explicit LoRA scales.")


def _cleanup(runtime: _RuntimeDependencies) -> None:
    """Release transient CUDA allocations after generation."""

    if hasattr(runtime.torch.cuda, "empty_cache"):
        runtime.torch.cuda.empty_cache()


def _build_generation_kwargs(
    *,
    prompt_inputs: Mapping[str, Any],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    sigmas: Sequence[float],
    generator: Any,
    output_type: str,
    callback: Callable[..., Any] | None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map the shared backend contract to diffusers pipeline kwargs."""

    kwargs: dict[str, Any] = {
        **prompt_inputs,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": fps,
        "sigmas": list(sigmas),
        "num_inference_steps": len(sigmas),
        "generator": generator,
        "output_type": output_type,
        **_DISTILLED_GUIDANCE,
    }
    if callback is not None:
        kwargs["callback_on_step_end"] = callback
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _upsample_latents(upsample_pipeline: Any, latents: Any, *, width: int, height: int, num_frames: int, low_memory: bool) -> Any:
    """Upsample stage-1 video latents 2x and return the upsampled latents.

    *width* and *height* are the final (stage 2) size. In low-memory mode the CPU-resident upsampler runs on the GPU only
    for this pass and returns to the CPU afterwards, even when the pass fails.
    """

    upsampler = upsample_pipeline.latent_upsampler
    if low_memory:
        upsampler.to("cuda")
    try:
        result = upsample_pipeline(
            latents=latents,
            height=height // 2,
            width=width // 2,
            num_frames=num_frames,
            output_type="latent",
        )
    finally:
        if low_memory:
            upsampler.to("cpu")
    return result.frames


@contextmanager
def _refine_scheduler(pipeline: Any) -> Iterator[None]:
    """Swap in a static-shift scheduler for the refinement pass and restore the original on exit."""

    original = pipeline.scheduler
    pipeline.scheduler = type(original).from_config(original.config, use_dynamic_shifting=False, shift_terminal=None)
    try:
        yield
    finally:
        pipeline.scheduler = original


def _generate_one_stage(
    model: _LoadedVideoModel,
    pipeline: Any,
    *,
    prompt_inputs: Mapping[str, Any],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
    steps: int,
    step_callback: Any | None,
    extra_kwargs: Mapping[str, Any],
    audio_carry: AudioCarry | None = None,
) -> _GeneratedClip:
    """Run one distilled denoising pass at full size and return its clip, with *audio_carry* pinned at the start of its sound."""

    sigmas = _stage1_sigmas(model.runtime.distilled_sigmas, steps)
    kwargs = _build_generation_kwargs(
        prompt_inputs=prompt_inputs,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        sigmas=sigmas,
        generator=_make_generator(model.runtime.torch, seed),
        output_type="latent",
        callback=_make_step_callback(step_callback, phase="video", offset=0, total=len(sigmas)),
        extra_kwargs=extra_kwargs,
    )
    if audio_carry is None:
        return _decode_clip(pipeline, pipeline(**kwargs))

    from zvisiongenerator.backends.cuda.video.audio_carry import pin_audio_prefix, restore_audio_prefix

    with pin_audio_prefix(pipeline.transformer, audio_carry):
        result = pipeline(**kwargs)
    restore_audio_prefix(result.audio, audio_carry)
    return _decode_clip(pipeline, result)


def _generate_two_stage(
    model: _LoadedVideoModel,
    pipeline: Any,
    *,
    prompt_inputs: Mapping[str, Any],
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
    stage1_steps: int,
    step_callback: Any | None,
    extra_kwargs: Mapping[str, Any],
    review: Callable[[Path], None] | None = None,
) -> _GeneratedClip:
    """Render at half size, upsample the latents 2x and refine at full size on the same pipeline.

    Both denoising passes run on *pipeline* so its LoRA adapters apply to each. Progress covers both passes as one
    run of ``len(stage-1 sigmas) + len(stage-2 sigmas)`` steps. With *review*, the half-size result is decoded to a
    temporary video and passed to it before the upsample; an exception from it ends the generation.
    """

    if width % 64 or height % 64:
        raise ValueError(f"Two-stage upscale needs width and height divisible by 64, got {width}x{height}.")

    runtime = model.runtime
    stage1_sigmas = _stage1_sigmas(runtime.distilled_sigmas, stage1_steps)
    stage2_sigmas = list(runtime.stage2_sigmas)
    total = len(stage1_sigmas) + len(stage2_sigmas)
    generator = _make_generator(runtime.torch, seed)
    common = {"prompt_inputs": prompt_inputs, "num_frames": num_frames, "fps": fps, "generator": generator}

    stage1 = pipeline(
        **_build_generation_kwargs(
            **common,
            width=width // 2,
            height=height // 2,
            sigmas=stage1_sigmas,
            output_type="latent",
            callback=_make_step_callback(step_callback, phase="video_upscale_stage_1", offset=0, total=total),
            extra_kwargs=extra_kwargs,
        )
    )
    if review is not None:
        _preview_small_render(model, pipeline, stage1, review)
    upsampled = _upsample_latents(model.upsample_pipeline, stage1.frames, width=width, height=height, num_frames=num_frames, low_memory=model.low_memory)

    refine_kwargs = {**extra_kwargs, "latents": upsampled, "audio_latents": stage1.audio, "noise_scale": stage2_sigmas[0]}
    with _refine_scheduler(pipeline):
        stage2 = pipeline(
            **_build_generation_kwargs(
                **common,
                width=width,
                height=height,
                sigmas=stage2_sigmas,
                output_type="latent",
                callback=_make_step_callback(step_callback, phase="video_upscale_stage_2", offset=len(stage1_sigmas), total=total),
                extra_kwargs=refine_kwargs,
            )
        )
    return _decode_clip(pipeline, stage2)


def _check_upscale_arguments(model: _LoadedVideoModel, stage1_steps: int | None) -> None:
    """Ensure the loaded model and the request agree on whether the run is upscaled."""

    if model.upsample_pipeline is not None and stage1_steps is None:
        raise ValueError("stage1_steps must be provided when upscale is enabled")
    if model.upsample_pipeline is None and stage1_steps is not None:
        raise RuntimeError("Upscale was requested, but no latent upsampler is loaded for this model.")


def _generate_clip(
    model: _LoadedVideoModel,
    pipeline: Any,
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    seed: int,
    steps: int,
    stage1_steps: int | None,
    step_callback: Any | None,
    extra_kwargs: Mapping[str, Any],
    audio_carry: AudioCarry | None = None,
    review: Callable[[Path], None] | None = None,
) -> _GeneratedClip:
    """Generate a clip at *fps* in one stage, or in two stages when *stage1_steps* is set; the prompt is encoded once for both.

    *audio_carry* (one stage only) pins sound at the start of the clip. *review* (two stages only) sees the half-size render first.
    """

    prompt_inputs = _encode_prompt(model, pipeline, prompt)
    _ensure_transformer(model, pipeline)
    if stage1_steps is None:
        return _generate_one_stage(
            model,
            pipeline,
            prompt_inputs=prompt_inputs,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            steps=steps,
            step_callback=step_callback,
            extra_kwargs=extra_kwargs,
            audio_carry=audio_carry,
        )
    if audio_carry is not None:
        raise ValueError("Pinned sound is not supported with the two-stage upscale.")
    return _generate_two_stage(
        model,
        pipeline,
        prompt_inputs=prompt_inputs,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        stage1_steps=stage1_steps,
        step_callback=step_callback,
        extra_kwargs=extra_kwargs,
        review=review,
    )


def _encode_prompt(model: _LoadedVideoModel, pipeline: Any, prompt: str) -> dict[str, Any]:
    """Return the pipeline keyword arguments that carry *prompt*: its embeddings, or the prompt itself without a text-encoder source.

    The text encoder streams to the GPU and loads for the first new prompt. When it and the transformer fit in system
    memory together it stays loaded for later prompts; otherwise the transformer is released first and attached again afterwards
    (:func:`_ensure_transformer`), and the text encoder is released after encoding. The last prompt's embeddings are
    kept, so repeating a prompt loads neither again.
    """

    source = model.text_encoder
    if source is None:
        return {"prompt": prompt}
    if model.prompt_cache is not None and model.prompt_cache[0] == prompt:
        return model.prompt_cache[1]

    from zvisiongenerator.backends.cuda.block_streaming import group_offload
    from zvisiongenerator.backends.cuda.video.weights import load_text_encoder
    from zvisiongenerator.backends.cuda.memory import release_memory

    torch = model.runtime.torch
    device = torch.device("cuda")
    model.prompt_cache = None  # the old embeddings are not needed while the new ones are made
    swap = model.transformer is not None and model.transformer.swap
    if swap and getattr(pipeline, "transformer", None) is not None:
        _release_transformer(pipeline, model.transformer)
    text_encoder = model.resident_text_encoder
    if text_encoder is None:
        text_encoder = load_text_encoder(source.repo_dir, prepared=source.prepared, compute_dtype=source.torch_dtype)
        group_offload(text_encoder, device)  # streamed in both modes: without low memory the transformer fills the GPU
    pipeline.text_encoder = text_encoder
    try:
        # No negative prompt: the distilled guidance (_DISTILLED_GUIDANCE) runs no classifier-free guidance.
        embeds, mask, _, _ = pipeline.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, device=device, dtype=source.torch_dtype, max_sequence_length=_MAX_SEQUENCE_LENGTH)
        model.resident_text_encoder = None if swap else text_encoder  # in system memory, streamed to the GPU
    finally:
        pipeline.text_encoder = None
        del text_encoder
        if model.resident_text_encoder is None:
            release_memory()
    inputs = {"prompt_embeds": embeds, "prompt_attention_mask": mask}
    model.prompt_cache = (prompt, inputs)
    return inputs


class DiffusersVideoBackend:
    """Shared CUDA-backed diffusers LTX video backend for Windows and Linux."""

    name = "ltx"

    def load_model(
        self,
        model_path: str,
        **kwargs: Any,
    ) -> tuple[_LoadedVideoModel, VideoModelInfo]:
        """Load the diffusers pipeline for one video request mode.

        Keyword Args:
            mode: ``t2v``, ``i2v`` or ``extend`` (continue a clip, see :meth:`extend_video`).
            low_memory: Stream the transformer and connectors to the GPU instead of keeping them there.
            loras: ``(path, weight)`` pairs.
            upscale: Load the latent upsampler for the two-stage upscale.
            latent_upsampler: ``(repo_id, filename)`` of the upsampler weights.
            quantize: 4 (NF4 transformer) or 8 (FP8 transformer); both levels use an FP8 text encoder and connectors.
            transformer_source: ``(repo_id, filename)`` of a single-file transformer checkpoint that replaces the repo's own.
            model_name: The name the model was picked by (``ltx-4``); its prepared copy is ``<models>/<name>@q<bits>``.

        A converted folder (``ziv model model --model-type ltx-2.3``) ignores ``quantize``, ``transformer_source``
        and ``model_name``: it loads the weights and level stored in it.
        """

        mode = kwargs.get("mode", "t2v")
        low_memory = bool(kwargs.get("low_memory", True))
        loras = kwargs.get("loras") or []
        upscale = bool(kwargs.get("upscale"))

        if mode not in _PIPELINES_BY_MODE:
            raise ValueError(f"Unsupported video generation mode: {mode}")

        # A converted folder carries its own level; the alias table's (``quantize``) is for stock models only.
        converted = read_converted_ltx(Path(model_path).expanduser())
        quantize = converted["bits"] if converted is not None else kwargs.get("quantize")
        if quantize not in (4, 8):
            raise ValueError(f"LTX on Windows/Linux runs 4-bit or 8-bit: use the model 'ltx-4' or 'ltx-8' (got quantize level {quantize!r}).")

        latent_upsampler = kwargs.get("latent_upsampler")
        if upscale and latent_upsampler is None:
            raise ValueError("--upscale 2 needs video_model_presets.ltx.diffusers.latent_upsampler (repo and filename) in the config.")

        runtime = _load_runtime_dependencies()
        _validate_cuda(runtime)
        _configure_torch_runtime(runtime.torch)
        torch_dtype = _select_torch_dtype(runtime.torch)

        model_info = detect_video_model(model_path)
        if model_info.family != "ltx":
            raise RuntimeError(f"Unsupported diffusers video model: '{model_path}'.")

        # Fail fast on a missing upsampler file before the large generation pipeline loads.
        upsampler_weights = _resolve_upsampler_weights(*latent_upsampler) if upscale else None

        pipeline_class = getattr(runtime.pipeline_classes, _PIPELINES_BY_MODE[mode])
        _check_lora_files(loras)
        active_pipeline, text_encoder, transformer = _load_generation_pipeline(
            pipeline_class,
            model_path,
            runtime,
            torch_dtype=torch_dtype,
            low_memory=low_memory,
            quantize=quantize,
            transformer_source=kwargs.get("transformer_source"),
            loras=loras,
            model_name=kwargs.get("model_name"),
            converted=converted,
        )

        upsample_pipeline = None
        if upsampler_weights is not None:
            upsampler = _load_latent_upsampler(
                runtime.pipeline_classes.latent_upsampler_model,
                _read_state_dict(upsampler_weights),
                torch_dtype=torch_dtype,
                low_memory=low_memory,
            )
            upsample_pipeline = _build_upsample_pipeline(runtime.pipeline_classes.latent_upsample_pipeline, active_pipeline, upsampler)

        return (
            _LoadedVideoModel(
                text_to_video=active_pipeline if mode == "t2v" else None,
                image_to_video=active_pipeline if mode == "i2v" else None,
                extend=active_pipeline if mode == "extend" else None,
                upsample_pipeline=upsample_pipeline,
                runtime=runtime,
                model_info=model_info,
                low_memory=low_memory,
                text_encoder=text_encoder,
                transformer=transformer,
            ),
            model_info,
        )

    def text_to_video(
        self,
        model: _LoadedVideoModel,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate video frames from text and export them to disk."""

        if model.text_to_video is None:
            raise RuntimeError("This loaded video model does not include a text-to-video pipeline.")

        stage1_steps = kwargs.get("stage1_steps")
        review = kwargs.get("review")
        _check_upscale_arguments(model, stage1_steps)
        _check_review_arguments(review, stage1_steps)

        try:
            with model.runtime.torch.inference_mode():
                clip = _generate_clip(
                    model,
                    model.text_to_video,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=model.model_info.default_fps,
                    seed=seed,
                    steps=steps,
                    stage1_steps=stage1_steps,
                    step_callback=step_callback,
                    extra_kwargs={},
                    review=review,
                )
                output = _export_video(model, clip, output_path)
        finally:
            _cleanup(model.runtime)
        return output

    def image_to_video(
        self,
        model: _LoadedVideoModel,
        image_path: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate video frames from an image prompt pair and export them to disk."""

        if model.image_to_video is None:
            raise RuntimeError("This loaded video model does not include an image-to-video pipeline.")

        stage1_steps = kwargs.get("stage1_steps")
        review = kwargs.get("review")
        _check_upscale_arguments(model, stage1_steps)
        _check_review_arguments(review, stage1_steps)

        with model.runtime.image_module.open(image_path) as image:
            input_image = _fit_image(image.convert("RGB"), width, height)

        try:
            with model.runtime.torch.inference_mode():
                clip = _generate_clip(
                    model,
                    model.image_to_video,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=model.model_info.default_fps,
                    seed=seed,
                    steps=steps,
                    stage1_steps=stage1_steps,
                    step_callback=step_callback,
                    extra_kwargs={"image": input_image},
                    review=review,
                )
                output = _export_video(model, clip, output_path)
        finally:
            _cleanup(model.runtime)
        return output

    def extend_video(
        self,
        model: _LoadedVideoModel,
        source: ExtendSource,
        prompt: str,
        *,
        extend_frames: int,
        seed: int,
        steps: int,
        segment_dir: str | Path,
        audio_output: bool,
        step_callback: Any | None = None,
    ) -> ExtendSegment | None:
        """Generate the continuation of a clip: new frames, and their sound, that follow its last frame.

        The clip's last 25 frames are pinned at the start of a window of ``25 + extend_frames`` frames at the clip's
        own size and frame rate (``LTX2VideoCondition`` at latent index 0, full strength), and so is the sound under
        them when the clip has any (see ``audio_carry``). The window is generated and decoded, and only its new part
        is written.

        Args:
            model: Handle returned by ``load_model`` with ``mode="extend"``.
            source: The checked clip to continue.
            prompt: Text prompt describing the scene and what happens next.
            extend_frames: New frames to generate; a multiple of 8, at least 8.
            seed: Random seed.
            steps: Distilled denoising steps.
            segment_dir: Existing directory that receives ``segment.mp4`` and, with audio, ``segment.wav``.
            audio_output: Whether to write the new part's sound. The sound is only written when the clip has any; a
                silent clip is continued without it. The clip's sound is pinned whenever it has some.
            step_callback: Receives progress event dicts.

        Returns:
            The generated segment: its video and, when ``audio_output`` is set and the clip has sound, its audio.

        Raises:
            ValueError: If the model was loaded for upscale, or ``extend_frames`` is not a positive multiple of 8.
            RuntimeError: If the model was not loaded for extend, or the clip's tail frames or audio cannot be read.
        """

        from zvisiongenerator.backends.cuda.video.audio_carry import encode_audio_carry
        from zvisiongenerator.utils.video_tail import read_source_tail, read_source_tail_audio

        if model.extend is None:
            raise RuntimeError("This loaded video model does not include an extend pipeline (load it with mode='extend').")
        if model.upsample_pipeline is not None:
            raise ValueError("Extending a video is not supported together with upscale.")
        if extend_frames < EXTEND_FRAME_STEP or extend_frames % EXTEND_FRAME_STEP:
            raise ValueError(f"extend_frames must be a positive multiple of {EXTEND_FRAME_STEP}, got {extend_frames}.")

        pipeline = model.extend
        tail = read_source_tail(source)
        tail_audio = read_source_tail_audio(source, pipeline.audio_vae.config.sample_rate) if source.has_audio else None
        carry = model.runtime.pipeline_classes.video_condition(frames=tail, index=0, strength=1.0)
        try:
            with model.runtime.torch.inference_mode():
                audio_carry = encode_audio_carry(pipeline, tail_audio, _audio_tokens(pipeline, CARRY_FRAMES, source.fps)) if tail_audio is not None else None
                clip = _generate_clip(
                    model,
                    pipeline,
                    prompt=prompt,
                    width=source.width,
                    height=source.height,
                    num_frames=window_frames(extend_frames),
                    fps=source.fps,
                    seed=seed,
                    steps=steps,
                    stage1_steps=None,
                    step_callback=step_callback,
                    extra_kwargs={"conditions": [carry]},
                    audio_carry=audio_carry,
                )
                segment = _write_extend_segment(clip, source, extend_frames, Path(segment_dir), audio=audio_output and source.has_audio)
        finally:
            _cleanup(model.runtime)
        return segment
