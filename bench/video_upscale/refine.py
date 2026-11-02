"""Upscale a finished video 2x with LTX's stage 2: encode it, upsample the latent 2x, refine it and decode it.

The stage-1 result a two-stage render hands to stage 2 is built from the encoded video instead of a half-size
render, through the pipeline's own resume path (``generate_two_stage(stage1=...)``): it encodes the prompt, builds
the states, checks the tokens' shapes against them and skips the stage-1 loop. The source's sound is encoded and held
frozen in stage 2, and the source's own audio stream is muxed onto the result.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zvisiongenerator.core.video_types import ExtendSource

_GIB = 1024**3
_AUDIO_ENCODER_RATE = 16000


@dataclass(frozen=True)
class RefineSettings:
    """How one upscale runs: refine steps, the prompt sent, the seed and the DiT tile counts (frames, height, width)."""

    steps: int
    prompt: str
    seed: int
    tiles: tuple[int, int, int] = (1, 1, 1)
    tile_overlap: int = 1
    encode_tiles: tuple[int, int] | None = None


@dataclass(frozen=True)
class LoadedModel:
    """The app's LTX handle loaded for the two-stage upscale, and what it was resolved from."""

    handle: Any
    model_name: str
    model_path: str
    config: dict[str, Any]


def load_upscale_model(model_name: str, *, low_memory: bool = True) -> LoadedModel:
    """Load *model_name* the way the app loads a video model for ``--upscale 2``."""
    from zvisiongenerator.backends.mac.video import LtxVideoBackend
    from zvisiongenerator.utils.config import load_config, video_load_options
    from zvisiongenerator.utils.paths import resolve_model_path

    config = load_config()
    model_path = resolve_model_path(model_name, aliases=config.get("model_aliases", {}), platform_key=sys.platform)
    handle, _info = LtxVideoBackend().load_model(model_path, low_memory=low_memory, **video_load_options("ltx", config, model_name, upscale=True))
    return LoadedModel(handle=handle, model_name=model_name, model_path=model_path, config=config)


def estimate_fit(model: LoadedModel, source: ExtendSource, *, low_memory: bool = True) -> dict[str, Any] | None:
    """Return the app's memory estimate for a two-stage render at the upscale's output size."""
    from zvisiongenerator.backends import accelerator_memory_budget_or_none
    from zvisiongenerator.utils.model_memory import clip_memory_fit

    return clip_memory_fit(
        model.model_path,
        width=source.width * 2,
        height=source.height * 2,
        frames=source.frame_count,
        low_memory=low_memory,
        upscale=True,
        budget=accelerator_memory_budget_or_none(),
        model_name=model.model_name,
        app_config=model.config,
    )


@contextmanager
def _measured(phases: dict[str, dict[str, float]], name: str) -> Iterator[None]:
    """Record the wall time and the MLX peak memory of the block under *name*."""
    import mlx.core as mx

    mx.reset_peak_memory()
    start = time.monotonic()
    try:
        yield
    finally:
        phases[name] = {"seconds": round(time.monotonic() - start, 1), "peak_gib": round(mx.get_peak_memory() / _GIB, 2)}


def _encoder_tiling(tiles: tuple[int, int]) -> Any:
    """Return the encoder tiling for ``(pixels, frames)`` tiles, with 64 px and 16 or 24 frames of overlap."""
    from ltx_core_mlx.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

    size, frames = tiles
    return TilingConfig(spatial_config=SpatialTilingConfig(size, 64), temporal_config=TemporalTilingConfig(frames, 16 if frames < 64 else 24))


def _encode_video_tokens(pipe: Any, source: ExtendSource, tiles: tuple[int, int] | None = None) -> Any:
    """Return the patchified VAE latent of the whole source clip, ``(1, F*H*W, 128)``; *tiles* overrides the encoder's default tiling."""
    import mlx.core as mx
    from ltx_core_mlx.utils.image import load_video_frames
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.retake import encode_video_tensor

    pixels = load_video_frames(str(source.path), source.height, source.width, source.frame_count)
    if pixels.shape[2] != source.frame_count:
        raise RuntimeError(f"Read {pixels.shape[2]} frames from {source.path}, expected {source.frame_count}.")
    if tiles is None:
        latent = pipe.image_conditioner(lambda encoder: encode_video_tensor(encoder, pixels), free_after=pipe.low_memory)
    else:
        tiling = _encoder_tiling(tiles)
        latent = pipe.image_conditioner(lambda encoder: encoder.tiled_encode(pixels, tiling).astype(mx.bfloat16), free_after=pipe.low_memory)
    tokens = pipe.video_patchifier.patchify(latent)[0]
    mx.eval(tokens)
    if pipe.low_memory:
        aggressive_cleanup()
    return tokens


def encode_check(model: LoadedModel, source: ExtendSource, tiles: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Encode *source* with each ``(pixels, frames)`` encoder tile size and compare each latent with the default tiling's.

    Returns:
        One record per tile size, the default first: time, MLX peak, and the mean and max absolute latent difference.
    """
    import mlx.core as mx
    from ltx_core_mlx.model.video_vae.tiling import TilingConfig
    from ltx_core_mlx.utils.image import load_video_frames

    pipe = model.handle.pipeline
    pixels = load_video_frames(str(source.path), source.height, source.width, source.frame_count)
    configs = [("default", TilingConfig.default())]
    for size, frames in tiles:
        configs.append((f"{size}px-{frames}f", _encoder_tiling((size, frames))))
    records, reference = [], None
    for name, config in configs:
        phases: dict[str, dict[str, float]] = {}
        with _measured(phases, name):
            latent = pipe.image_conditioner(lambda encoder, config=config: encoder.tiled_encode(pixels, config).astype(mx.bfloat16), free_after=False)
            mx.eval(latent)
        latent = latent.astype(mx.float32)
        if reference is None:
            reference = latent
        difference = mx.abs(latent - reference)
        records.append(
            {
                "tiles": name,
                **phases[name],
                "mean_abs_latent": round(mx.abs(latent).mean().item(), 4),
                "mean_abs_diff": round(difference.mean().item(), 4),
                "max_abs_diff": round(difference.max().item(), 4),
            }
        )
    pipe.image_conditioner.free()
    return records


def _encode_audio_tokens(pipe: Any, source: ExtendSource, token_count: int) -> Any:
    """Return the source's sound as frozen-able audio tokens ``(1, token_count, 128)``; silence when it has none."""
    import mlx.core as mx
    from ltx_core_mlx.utils.audio import load_audio

    from zvisiongenerator.backends.mac.video import _encode_audio_tokens as encode_tokens

    audio = load_audio(str(source.path), target_sample_rate=_AUDIO_ENCODER_RATE, max_duration=source.frame_count / source.fps) if source.has_audio else None
    if audio is None:
        return mx.zeros((1, token_count, 128), dtype=mx.bfloat16)
    return encode_tokens(pipe, audio.waveform, audio.sample_rate, token_count)


def _stage1_from_tokens(video_tokens: Any, audio_tokens: Any, latent_dims: tuple[int, int, int]) -> Any:
    """Wrap the encoded tokens as a stage-1 result; ``_stage1`` rebuilds the rest for the request."""
    from ltx_pipelines_mlx.distilled import Stage1Result

    return Stage1Result(
        video_tokens=video_tokens,
        audio_tokens=audio_tokens,
        generated_keyframes=None,
        latent_dims=latent_dims,
        resolved_images=[],
        video_embeds=None,
        audio_embeds=None,
        relay_mask=None,
        relay_audio_mask=None,
        x0_model=None,
    )


def _tile_count(settings: RefineSettings) -> Any:
    """Return the DiT tile layout of *settings*, ``None`` for one tile."""
    if settings.tiles == (1, 1, 1):
        return None
    from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig

    frames, height, width = (DimensionTilingConfig(count, settings.tile_overlap if count > 1 else 0) for count in settings.tiles)
    return TileCountConfig(frames=frames, height=height, width=width)


def mux_source_audio(video_path: Path, source_path: Path, output_path: Path) -> None:
    """Write *video_path*'s video with *source_path*'s first audio stream (when it has one), both copied, to *output_path*.

    No ``-shortest``: LTX's sound ends a little before its video and would cut frames.
    """
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-i", str(source_path), "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", str(output_path)]
    subprocess.run(command, check=True)


def upscale_video(model: LoadedModel, source: ExtendSource, settings: RefineSettings, output_path: Path, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Upscale *source* 2x into *output_path* and return the timing and memory of each phase.

    Returns:
        ``{"phases": {name: {"seconds", "peak_gib"}}, "seconds": total, "peak_gib": max}``.
    """
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.positions import compute_audio_token_count

    from zvisiongenerator.backends.mac.video import _generation_scope

    pipe = model.handle.pipeline
    pipe.stepwise = None
    pipe._tile_count = _tile_count(settings)  # noqa: SLF001
    latent_dims = compute_video_latent_shape(source.frame_count, source.height, source.width)
    audio_count = compute_audio_token_count(source.frame_count, frame_rate=source.fps)
    phases: dict[str, dict[str, float]] = {}
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    start = time.monotonic()
    with _generation_scope(model.handle):
        with _measured(phases, "encode"):
            video_tokens = _encode_video_tokens(pipe, source, settings.encode_tiles)
            audio_tokens = _encode_audio_tokens(pipe, source, audio_count)
        log(f"  encode {phases['encode']}")
        # Frozen in stage 2: the sound is the source's and is replaced by its own stream afterwards anyway.
        pipe._source_audio_tokens = audio_tokens  # noqa: SLF001
        try:
            with _measured(phases, "refine"):
                video_latent, audio_latent = pipe.generate_two_stage(
                    prompt=settings.prompt,
                    height=source.height * 2,
                    width=source.width * 2,
                    num_frames=source.frame_count,
                    frame_rate=source.fps,
                    seed=settings.seed,
                    stage2_steps=settings.steps,
                    stage1=_stage1_from_tokens(video_tokens, audio_tokens, latent_dims),
                )
        finally:
            pipe._source_audio_tokens = None  # noqa: SLF001
        log(f"  refine {phases['refine']}")
        del video_tokens
        if pipe.low_memory:
            pipe.dit = None
            pipe.prompt_encoder.free()
            pipe.image_conditioner.free()
            pipe.upsampler = None
            pipe._loaded = False  # noqa: SLF001
            aggressive_cleanup()
        with _measured(phases, "decode"):
            pipe._load_decoders()  # noqa: SLF001
            pipe._decode_and_save_video(video_latent, audio_latent, str(raw_path), frame_rate=source.fps, seed=settings.seed)  # noqa: SLF001
        log(f"  decode {phases['decode']}")
        if pipe.low_memory:
            pipe.vae_decoder = None
            pipe.audio_decoder = None
            pipe.vocoder = None
            aggressive_cleanup()
    mux_source_audio(raw_path, source.path, output_path)
    raw_path.unlink()
    return {"phases": phases, "seconds": round(time.monotonic() - start, 1), "peak_gib": max(phase["peak_gib"] for phase in phases.values())}


def iclora_upscale(model: LoadedModel, source: ExtendSource, *, lora_path: str, strength: float, prompt: str, seed: int, output_path: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Re-render *source* at 2x with the Pixel Spatial Upscaler IC-LoRA and return the timing and memory of each phase.

    One distilled pass from noise at the output size, conditioned on the source as the half-size reference (the
    LoRA's ``reference_downscale_factor`` is 2), as upstream's IC-LoRA video-to-video flow runs it.
    """
    import contextlib

    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    from zvisiongenerator.backends.mac.memory import memory_size_bytes, mlx_cache_limit
    from zvisiongenerator.backends.mac.video import _DEFAULT_LORA_MODE, _LORA_MODE_ENV, _env_default, _generation_scope, _LtxModel, _resolve_model_dir, _text_encoder_options, _use_low_ram_streaming
    from zvisiongenerator.utils.config import video_load_options

    base = model.handle.pipeline
    options = video_load_options("ltx", model.config, model.model_name, upscale=True)
    model_dir = _resolve_model_dir(model.model_path, packs=options.get("mlx_packs") or {})
    streaming = _use_low_ram_streaming(base.low_memory, memory_size_bytes())
    phases: dict[str, dict[str, float]] = {}
    raw_path = output_path.with_name(output_path.stem + ".raw.mp4")
    start = time.monotonic()
    with _env_default(_LORA_MODE_ENV, _DEFAULT_LORA_MODE), mlx_cache_limit(0) if streaming else contextlib.nullcontext():
        pipe = ICLoraPipeline(
            model_dir,
            lora_paths=[(lora_path, strength)],
            low_memory=base.low_memory,
            low_ram_streaming=streaming,
            **_text_encoder_options(model_dir, options.get("mlx_text_encoder")),
        )
    handle = _LtxModel(pipeline=pipe, frame_rate=source.fps, upscale=False, low_ram_streaming=streaming)
    with _generation_scope(handle), _measured(phases, "render"):
        pipe.generate_and_save(
            prompt=prompt,
            output_path=str(raw_path),
            video_conditioning=[(str(source.path), 1.0)],
            height=source.height * 2,
            width=source.width * 2,
            num_frames=source.frame_count,
            frame_rate=source.fps,
            seed=seed,
            single_stage=True,
        )
    log(f"  render {phases['render']}")
    del pipe, handle
    aggressive_cleanup()
    mux_source_audio(raw_path, source.path, output_path)
    raw_path.unlink()
    return {"phases": phases, "seconds": round(time.monotonic() - start, 1), "peak_gib": phases["render"]["peak_gib"]}


def bicubic_upscale(source: ExtendSource, output_path: Path) -> dict[str, Any]:
    """Write a plain 2x bicubic enlargement of *source* with its audio copied, the baseline every refine must beat."""
    start = time.monotonic()
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source.path),
        "-vf",
        f"scale={source.width * 2}:{source.height * 2}:flags=bicubic",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return {"phases": {}, "seconds": round(time.monotonic() - start, 1), "peak_gib": 0.0}
