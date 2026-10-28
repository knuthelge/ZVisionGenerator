"""macOS video backends — LTX (ltx-pipelines-mlx)."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS
from zvisiongenerator.backends.mac.memory import memory_size_bytes, mlx_cache_limit
from zvisiongenerator.core.video_types import ExtendSegment
from zvisiongenerator.utils.hf_snapshot import pack_for_reference, snapshot_with
from zvisiongenerator.utils.model_files import find_local_model_dir, ltx_mlx_bundled_text_encoder
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference
from zvisiongenerator.utils.input_images import upright_image_copies
from zvisiongenerator.utils.video_extend import CARRY_FRAMES, EXTEND_FRAME_STEP, audio_lead_samples, carry_latent_frames, cut_window_audio, window_audio_cut, window_frames
from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model
from zvisiongenerator.utils.video_splice import build_segment_command, write_pcm16_wav
from zvisiongenerator.utils.video_tail import read_source_tail

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from zvisiongenerator.core.video_types import ExtendSource
    from zvisiongenerator.utils.h3_preset import HubFiles

# Pixels per latent cell of the video VAE encoder.
_LATENT_PIXELS = 32
# Channel count of the LTX video and audio token streams.
_LATENT_CHANNELS = 128
# Macs reporting at most this much unified memory stream the transformer blocks under ``--low-memory``.
# A Mac sold as "24 GB" reports exactly 24 GiB (25769803776 bytes), so the comparison is inclusive.
# Larger Macs do not stream: streaming lowers the memory held while denoising but not the peak, which is set
# while the text encoder loads, and it slows LoRA runs.
_LOW_RAM_THRESHOLD_BYTES = 24 * 1024**3
# Environment variable the vendored pipeline reads, while it loads the transformer, to pick the compute dtype.
_COMPUTE_DTYPE_ENV = "LTX2_COMPUTE_DTYPE"
# The pipeline computes in float32 unless told otherwise. float16 is faster per step on Apple Silicon with
# near-identical output, so a generation uses it unless the user has set the variable themselves.
_DEFAULT_COMPUTE_DTYPE = "float16"
# Environment variable the vendored pipeline reads, when it is constructed, to fuse LoRAs into the weights or attach them.
_LORA_MODE_ENV = "LTX2_LORA_MODE"
# Fusing re-quantizes the q4/q8 weights, which loses most of a small LoRA's update; run-time adapters keep it at about
# the same speed. A pipeline is built with unfused LoRAs unless the user has set the variable themselves.
_DEFAULT_LORA_MODE = "unfused"
# Sample rate the audio VAE encoder reads.
_AUDIO_ENCODER_RATE = 16000


def _use_low_ram_streaming(low_memory: bool, memory_bytes: int | None, threshold_bytes: int = _LOW_RAM_THRESHOLD_BYTES) -> bool:
    """Return whether the transformer should be streamed from disk.

    Args:
        low_memory: Whether the low-memory mode is requested.
        memory_bytes: Total unified memory in bytes, or ``None`` when unknown.
        threshold_bytes: Largest memory size, inclusive, that still streams.

    Returns:
        True only when ``low_memory`` is set and ``memory_bytes`` is known and within the threshold.
    """
    return low_memory and memory_bytes is not None and memory_bytes <= threshold_bytes


@dataclass(frozen=True)
class _LtxModel:
    """Handle returned by ``load_model``: a pipeline plus the settings generation needs.

    Attributes:
        pipeline: ``BasePipeline`` (one-stage) or ``DistilledPipeline`` (two-stage upscale).
        frame_rate: Frames per second used for positions, audio length and the muxed output.
        upscale: Whether ``pipeline`` is the two-stage upscale pipeline.
        low_ram_streaming: Whether the pipeline streams the transformer blocks, which also needs
            the MLX buffer cache disabled for the duration of each generation.
        ltx25: Whether the loaded pack is an LTX-2.5 pack (it bundles its own Gemma 4 text encoder).
    """

    pipeline: Any
    frame_rate: float
    upscale: bool
    low_ram_streaming: bool
    ltx25: bool = False


def _snapshot_download(repo_id: str, revision: str, allow_patterns: list[str] | None = None) -> str:
    """Download the *revision* snapshot of *repo_id* (only *allow_patterns* when given) and return its local folder."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, allow_patterns=allow_patterns)


def _resolve_model_dir(
    model_path: str,
    *,
    packs: Mapping[str, HubFiles],
    find_local_dir: Callable[[str], Path | None] | None = None,
    download: Callable[..., str] | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Resolve a model reference to the local folder the vendored pipeline loads.

    The vendored pipeline downloads by repo id and cannot take a revision or a file list, so a pinned reference is
    resolved here. A reference to a pinned MLX pack of *packs* (its repo id in any letter case, with or without its
    pinned revision) is exactly its listed files at its revision: a cached snapshot that holds them all, otherwise a
    download of just those files. Any other pinned reference is the complete cached snapshot when there is one,
    otherwise a download of that revision (only *allow_patterns* of it when given).

    Args:
        model_path: Local path or HuggingFace repo id, optionally ``@revision``.
        packs: The pinned MLX packs, each with the files the app loads.
        find_local_dir: Cache lookup returning the complete snapshot folder or ``None``; defaults to ``find_local_model_dir``.
        download: Called as ``download(repo_id=..., revision=...[, allow_patterns=...])`` to fetch the snapshot; defaults to the Hub download.
        allow_patterns: The patterns a download of a reference that is no pack is limited to; the whole revision when ``None``.

    Returns:
        The snapshot folder for a pinned reference or a pack. An unpinned reference to no pack, or a local path, comes back unchanged.

    Raises:
        FileNotFoundError: If a pinned pack's folder lacks a listed file after the download.
    """
    if Path(model_path).expanduser().is_dir():
        return model_path
    download = _snapshot_download if download is None else download
    pack = pack_for_reference(packs, model_path)
    if pack is not None:
        folder = snapshot_with(pack, download)
        missing = [name for name in pack.files if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{pack.repo_id}@{pack.revision} has no {', '.join(missing)}; the pinned file list in the config does not match that revision.")
        return str(folder)
    reference = parse_huggingface_repo_reference(model_path)
    if reference is None or reference.revision is None:
        return model_path
    find_local_dir = find_local_model_dir if find_local_dir is None else find_local_dir
    local = find_local_dir(model_path)
    if local is not None:
        return str(local)
    patterns = {} if allow_patterns is None else {"allow_patterns": list(allow_patterns)}
    return str(download(repo_id=reference.repo_id, revision=reference.revision, **patterns))


def _text_encoder_options(model_dir: str, text_encoder: str | None) -> dict[str, str]:
    """Return the pipeline's ``gemma_model_id`` argument: the local folder of the configured Gemma 3 reference *text_encoder*, fetched with the files ``mlx_lm`` loads.

    Empty (the pipeline's own default applies) when no reference is configured, *model_dir* is not a local folder yet
    (the pipeline downloads it, and decides then what text encoder it needs) or the folder bundles its own text encoder.
    """
    folder = Path(model_dir)
    if text_encoder is None or not folder.is_dir() or ltx_mlx_bundled_text_encoder(folder) is not None:
        return {}
    return {"gemma_model_id": _resolve_model_dir(text_encoder, packs={}, allow_patterns=MLX_LM_ALLOW_PATTERNS)}


class _TwoStageProgress:
    """Adapt upstream's per-stage ``on_step`` hook to cumulative upscale progress events.

    An LTX pipeline's duck-typed ``stepwise`` slot is called once per denoising stage
    through ``bind(...)``, which must return that loop's ``on_step`` callable. This adapter
    occupies the slot; it relies on an internal hook pinned to the vendored release.
    It never touches the decoder block, so the VAE decoder is not kept resident.
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None], *, stage2_total: int) -> None:
        self._callback = callback
        self._stage2_total = stage2_total
        self._stage1_total = 0

    def bind(self, *, stage: int | None = None, **_: Any) -> Callable[[int, int, Any, float], None]:
        """Return the ``on_step`` callable for one denoising stage.

        Args:
            stage: 1 or ``None`` for the half-resolution stage, 2 for the refinement stage.
            **_: Latent geometry and decoder arguments supplied by the pipeline; unused.

        Returns:
            Callable invoked as ``on_step(step_idx, total, video_x0, sigma)`` with a 0-based index.
        """
        if stage == 2:

            def on_stage_2(step_idx: int, _total: int, _x0: Any, _sigma: float) -> None:
                self._callback(
                    {
                        "phase": "video_upscale_stage_2",
                        "current_step": self._stage1_total + step_idx + 1,
                        "total_steps": self._stage1_total + self._stage2_total,
                    }
                )

            return on_stage_2

        def on_stage_1(step_idx: int, total: int, _x0: Any, _sigma: float) -> None:
            self._stage1_total = total
            self._callback(
                {
                    "phase": "video_upscale_stage_1",
                    "current_step": step_idx + 1,
                    "total_steps": total + self._stage2_total,
                }
            )

        return on_stage_1


@dataclass(frozen=True)
class _Carry:
    """The source's tail, encoded and patchified, pinned at the start of the extension window.

    Attributes:
        video_tokens: Patchified video latent of the carried frames, ``(1, latent_frames * H * W, 128)``.
        audio_tokens: Patchified audio latent of the carried frames, ``(1, audio_token_count, 128)``, or ``None``
            when the source is silent.
        latent_frames: Latent frames the video tokens fill.
        audio_token_count: Audio tokens the carried frames span at the source's frame rate.
    """

    video_tokens: Any
    audio_tokens: Any | None
    latent_frames: int
    audio_token_count: int


@dataclass(frozen=True)
class _SegmentTarget:
    """Where the extension window's decode goes and how it is cut.

    Attributes:
        video_path: The segment's video file.
        width: Frame width of the segment (the source's displayed width).
        height: Frame height of the segment (the source's displayed height).
        audio_path: The segment's WAV file, or ``None`` when no audio is wanted.
        extend_frames: New frames that are written.
        fps_text: The source's frame rate as ffprobe writes it, passed to the encoder unchanged.
        color_space: The source's colour space tag.
        color_primaries: The source's colour primaries tag.
        color_trc: The source's transfer characteristics tag.
    """

    video_path: Path
    width: int
    height: int
    audio_path: Path | None
    extend_frames: int
    fps_text: str
    color_space: str | None = None
    color_primaries: str | None = None
    color_trc: str | None = None


def _one_stage_progress(callback: Callable[[dict[str, Any]], None] | None) -> Callable[[int, int, Any, float], None] | None:
    """Return a denoise-loop ``on_step`` that emits 1-based ``video_generate`` progress, or ``None``."""
    if callback is None:
        return None

    def on_step(step_idx: int, total: int, _x0: Any, _sigma: float) -> None:
        callback({"phase": "video_generate", "current_step": step_idx + 1, "total_steps": total})

    return on_step


def _image_anchors(image_path: str, end_image_path: str | None) -> list[tuple[str, int]]:
    """Return the ``(path, pixel frame)`` of each image the clip is pinned to: the first frame, and the last (``-1``) when set."""
    return [(image_path, 0)] if end_image_path is None else [(image_path, 0), (end_image_path, -1)]


def _conditioning_inputs(image_path: str, end_image_path: str | None) -> list:
    """Return the vendored ``ImageConditioningInput`` of each image the clip is pinned to, at full strength."""
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    return [ImageConditioningInput.at(path, frame, 1.0) for path, frame in _image_anchors(image_path, end_image_path)]


def _conditioning_latent(cond: Any) -> Any:
    """Return the latent a video conditioning holds: a first-frame one replaces latent frame 0, a keyframe one is appended."""
    return cond.keyframe_latent if hasattr(cond, "keyframe_latent") else cond.clean_latent


def _image_conditionings(pipe: Any, image_path: str, latent_shape: tuple[int, int, int], fps: float, *, num_frames: int, end_image_path: str | None = None) -> list:
    """Encode the first-frame image, and the last-frame image when given, into video conditionings, materialized before the encoder is freed."""
    import mlx.core as mx
    from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
    from ltx_pipelines_mlx.utils.args import resolve_frame_indices

    _, H, W = latent_shape
    images = pipe.image_conditioner.resolve_crf(resolve_frame_indices(_conditioning_inputs(image_path, end_image_path), num_frames))

    def _encode(encoder: Any) -> list:
        conds = combined_image_conditionings(
            images,
            enc_h=H * _LATENT_PIXELS,
            enc_w=W * _LATENT_PIXELS,
            spatial_dims=latent_shape,
            video_encoder=encoder,
            frame_rate=fps,
        )
        # MLX graph evaluation: materialize the conditionings before the encoder is freed.
        mx.eval(*(_conditioning_latent(cond) for cond in conds))
        return conds

    return pipe.image_conditioner(_encode, free_after=pipe.low_memory)


def _encode_carry(pipe: Any, source: ExtendSource, frames: Any, *, fps: float) -> _Carry:
    """Encode a clip's tail frames, and the sound under them when the clip has any, into pinnable tokens.

    The video goes through the VAE encoder; the audio is read from the file at the tail's own position and goes
    through the audio encoder, then is fitted to the tail's token count before it is patchified.

    Args:
        pipe: The loaded LTX pipeline.
        source: The clip the tail comes from; its size, frame rate and audio decide the encoding.
        frames: The tail as uint8 RGB frames of shape ``(count, height, width, 3)``.
        fps: Frame rate of the clip, used for the audio token count and the audio position.

    Returns:
        The encoded carry; ``audio_tokens`` is ``None`` for a silent source.

    Raises:
        RuntimeError: If the source has audio that cannot be read.
    """
    import mlx.core as mx
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.extend_distilled import carry_audio_tokens
    from ltx_pipelines_mlx.retake import encode_video_tensor
    from ltx_pipelines_mlx.utils.media_io import video_preprocess

    low_memory = pipe.low_memory
    count = len(frames)

    video_tensor = video_preprocess(list(frames), source.height, source.width)

    def _encode_video(encoder: Any) -> Any:
        latent = encode_video_tensor(encoder, video_tensor)
        mx.eval(latent)
        return latent

    video_latent = pipe.image_conditioner(_encode_video, free_after=low_memory)
    video_tokens = pipe.video_patchifier.patchify(video_latent)[0]
    mx.eval(video_tokens)
    if low_memory:
        aggressive_cleanup()

    audio_token_count = carry_audio_tokens(fps, count)
    audio_tokens = _encode_carry_audio(pipe, source, count, fps, audio_token_count) if source.has_audio else None
    return _Carry(video_tokens=video_tokens, audio_tokens=audio_tokens, latent_frames=carry_latent_frames(), audio_token_count=audio_token_count)


def _encode_carry_audio(pipe: Any, source: ExtendSource, count: int, fps: float, token_count: int) -> Any:
    """Return the patchified audio latent of a clip's last ``count`` frames, shape ``(1, token_count, 128)``.

    Raises:
        RuntimeError: If the audio cannot be read from the file.
    """
    import mlx.core as mx
    from ltx_core_mlx.model.audio_vae import encode_audio
    from ltx_core_mlx.utils.audio import load_audio
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_pipelines_mlx.extend_distilled import fit_audio_latent

    start_time = source.video_start + (source.frame_count - count) / fps
    audio = load_audio(source.path, target_sample_rate=_AUDIO_ENCODER_RATE, start_time=start_time, max_duration=count / fps)
    if audio is None:
        raise RuntimeError(f"Could not read the audio of {source.path} to continue it.")

    def _encode_audio(encoder: Any, processor: Any) -> Any:
        latent = encode_audio(audio.waveform, audio.sample_rate, encoder, processor)
        mx.eval(latent)
        return latent

    latent = pipe.audio_conditioner(_encode_audio, free_after=pipe.low_memory)
    # Fit on the (1, 8, T, 16) latent: patchifying first would fold the time axis into the channels.
    tokens = pipe.audio_patchifier.patchify(fit_audio_latent(latent, token_count))[0]
    mx.eval(tokens)
    if pipe.low_memory:
        aggressive_cleanup()
    return tokens


def _carry_conditioning(tokens: Any, frame_indices: list[int]) -> Any:
    """Return a conditioning that pins ``tokens`` at the given frames of a state with full strength."""
    from ltx_core_mlx.conditioning.types import VideoConditionByLatentIndex

    return VideoConditionByLatentIndex(frame_indices=frame_indices, clean_latent=tokens, strength=1.0)


def _drop_leading_frames(chunks: Iterable[Any], count: int) -> Iterator[Any]:
    """Yield decoded chunks ``(B, 3, T, H, W)`` without their first ``count`` frames on the time axis.

    Chunks may be any size, so the cut can fall inside a chunk, on a boundary or span several chunks.
    """
    remaining = count
    for chunk in chunks:
        frames = chunk.shape[2]
        if remaining >= frames:
            remaining -= frames
            continue
        yield chunk[:, :, remaining:] if remaining else chunk
        remaining = 0


def _free_dit(pipe: Any) -> None:
    """Release the transformer and mark the pipeline unloaded so the next ``load()`` reloads it.

    Mirrors what upstream's decode wrapper does under ``low_memory``: the latents are materialized and the DiT would
    otherwise sit beside the video decoder at its peak.
    """
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    pipe.dit = None
    pipe._loaded = False  # noqa: SLF001
    aggressive_cleanup()


def _decode_window_segment(pipe: Any, video_latent: Any, audio_latent: Any, target: _SegmentTarget, *, fps: float) -> None:
    """Decode an extension window and write only its new part: the video to ffmpeg, the sound to a WAV file.

    The window starts with the carried frames, which are decoded but not written. Decoding is tiled as the
    decoder's memory budget requires, and the cut works for any tiling. The WAV starts a little before the new
    part (the carried frames' sound), so that the splice can fill a source track that ends short of its video.
    """
    import mlx.core as mx
    import numpy as np
    from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling, _ffmpeg_sink, decode_cache_limit, stream_chunks_to_ffmpeg

    if pipe.low_memory and pipe.dit is not None:
        _free_dit(pipe)

    decoder = pipe.video_decoder_block.load()
    command = build_segment_command(
        target.width,
        target.height,
        target.fps_text,
        target.video_path,
        color_space=target.color_space,
        color_primaries=target.color_primaries,
        color_trc=target.color_trc,
    )
    with decode_cache_limit():
        chunks = decoder.tiled_decode(video_latent, _compute_decode_tiling(tuple(video_latent.shape), frame_rate=fps))
        with _ffmpeg_sink(command) as proc:
            stream_chunks_to_ffmpeg(_drop_leading_frames(chunks, CARRY_FRAMES), proc)

    if target.audio_path is None:
        return
    waveform = pipe.audio_decoder_block(audio_latent)
    mx.eval(waveform)
    samples = np.asarray(waveform[0].astype(mx.float32))
    skip, keep = window_audio_cut(target.extend_frames, fps)
    lead = audio_lead_samples(fps)
    write_pcm16_wav(cut_window_audio(samples, skip, keep, lead), target.audio_path)


def _generate_one_stage(
    model: _LtxModel,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    steps: int,
    output_path: str,
    image_path: str | None,
    step_callback: Callable[[dict[str, Any]], None] | None,
    *,
    frame_rate: float | None = None,
    carry: _Carry | None = None,
    segment: _SegmentTarget | None = None,
    end_image_path: str | None = None,
) -> None:
    """Run the distilled sampler at full resolution and write the clip to *output_path*.

    Mirrors stage 1 of upstream's distilled pipeline, without the half-resolution step. An LTX-2.5 pack uses its own
    sigma table; text-to-video and image-to-video then run the ancestral (SDE) sampler with the noise seed offset
    upstream uses, and an extension (``carry``) stays on the deterministic sampler. LTX-2.3 is deterministic throughout.

    Args:
        model: Handle returned by ``load_model``.
        prompt: Text prompt.
        width: Width in pixels.
        height: Height in pixels.
        num_frames: Frames to generate; for an extension, the whole window (carried frames plus new ones).
        seed: Random seed; the audio state uses ``seed + 1``.
        steps: Distilled denoising steps.
        output_path: Where the clip is written (unused when ``segment`` is given).
        image_path: First-frame image, or ``None``.
        step_callback: Receives ``video_generate`` progress events.
        frame_rate: Frame rate for positions, audio length and decode; defaults to the handle's.
        carry: Encoded tail of a clip to pin at the start of the window; replaces the image conditioning.
        segment: When given, decode the window into this target (dropping the carried frames) instead of writing
            a muxed clip to ``output_path``.
        end_image_path: Last-frame image (needs ``image_path``): appended as a keyframe on the clip's last frame.
    """
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
    from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, LTX_2_5_DISTILLED_SIGMAS, shorten_schedule
    from ltx_pipelines_mlx.utils.helpers import create_noised_state
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    pipe = model.pipeline
    fps = frame_rate or model.frame_rate
    low_memory = pipe.low_memory

    height, width = snap_output_dimensions(height, width, two_stage=False)
    video_embeds, audio_embeds = pipe.prompt_encoder(prompt, free_after=low_memory)

    F, H, W = compute_video_latent_shape(num_frames, height, width)
    audio_tokens = compute_audio_token_count(num_frames, frame_rate=fps)
    video_positions = compute_video_positions(F, H, W, frame_rate=fps)
    audio_positions = compute_audio_positions(audio_tokens)

    conditionings: list = []
    if carry is not None:
        conditionings = [_carry_conditioning(carry.video_tokens, list(range(carry.latent_frames)))]
    elif image_path is not None:
        conditionings = _image_conditionings(pipe, image_path, (F, H, W), fps, num_frames=num_frames, end_image_path=end_image_path)

    pipe.load()

    video_state = create_noised_state(
        base_shape=(1, F * H * W, _LATENT_CHANNELS),
        conditionings=conditionings,
        spatial_dims=(F, H, W),
        positions=video_positions,
        seed=seed,
        sigma=1.0,
        initial_latent=None,
        legacy_scalar_blend=True,
    )
    pinned_audio = carry is not None and carry.audio_tokens is not None
    audio_state = create_noised_state(
        base_shape=(1, audio_tokens, _LATENT_CHANNELS),
        conditionings=[_carry_conditioning(carry.audio_tokens, list(range(carry.audio_token_count)))] if pinned_audio else [],
        spatial_dims=(audio_tokens, 1, 1) if pinned_audio else (F, H, W),
        positions=audio_positions,
        seed=seed + 1,
        sigma=1.0,
        initial_latent=None,
        legacy_scalar_blend=True,
    )

    # Metal watchdog guard: materialize the noised states before the first denoise step.
    pipe._pre_denoise_flush(video_state, audio_state)  # noqa: SLF001
    if low_memory:
        aggressive_cleanup()

    sigmas = shorten_schedule(LTX_2_5_DISTILLED_SIGMAS if model.ltx25 else DISTILLED_SIGMAS, steps, keep="start")
    if model.ltx25 and carry is None:
        from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
        from ltx_pipelines_mlx.distilled import ANCESTRAL_ETA, ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_S_NOISE
        from ltx_pipelines_mlx.utils.samplers import euler_ancestral_denoising_loop

        output = euler_ancestral_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE),
            transformer=X0Model(pipe.dit),
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            on_step=_one_stage_progress(step_callback),
        )
    else:
        output = denoise_loop(
            model=X0Model(pipe.dit),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            on_step=_one_stage_progress(step_callback),
        )
    if low_memory:
        aggressive_cleanup()

    video_latent = pipe.video_patchifier.unpatchify(output.video_latent[:, : F * H * W, :], (F, H, W))
    audio_latent = pipe.audio_patchifier.unpatchify(output.audio_latent)
    if segment is None:
        pipe._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=fps, seed=seed)  # noqa: SLF001
    else:
        _decode_window_segment(pipe, video_latent, audio_latent, segment, fps=fps)

    if low_memory:
        # Free the decoders so the next job on this pipeline does not stack the DiT on top of them.
        pipe.vae_decoder = None
        pipe.audio_decoder = None
        pipe.vocoder = None
        aggressive_cleanup()


def _preview_small_render(pipe: Any, stage1: Any, review: Callable[[Path], None], *, frame_rate: float, seed: int) -> None:
    """Decode the half-size result of stage 1 to a temporary video file and hand its path to *review*.

    The file lives in a temporary directory outside the output folder and is deleted when *review* returns or raises.
    Stage 2 must see the pipeline as stage 1 left it, so the DiT slot is put back after the decode (under ``low_memory``
    the decode empties it, which would turn off stage-2 sparse attention and break tiling), and under ``low_memory``
    the decoders are freed again as ``generate_and_save`` expects to find them.

    Args:
        pipe: The ``DistilledPipeline`` that has just run stage 1.
        stage1: Its ``Stage1Result``.
        review: Receives the path of the decoded video.
        frame_rate: Frames per second of the clip.
        seed: Seed of the generation, for the decoder.
    """
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    video_half = pipe.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)
    audio_half = pipe.audio_patchifier.unpatchify(stage1.audio_tokens)
    with tempfile.TemporaryDirectory(prefix="ziv-review-") as directory:
        path = Path(directory) / "small-render.mp4"
        dit, loaded = pipe.dit, pipe._loaded  # noqa: SLF001
        try:
            pipe._load_decoders()  # noqa: SLF001
            pipe._decode_and_save_video(video_half, audio_half, str(path), frame_rate=frame_rate, seed=seed)  # noqa: SLF001
        finally:
            pipe.dit, pipe._loaded = dit, loaded  # noqa: SLF001
            if pipe.low_memory:
                pipe.vae_decoder = None
                pipe.audio_decoder = None
                pipe.vocoder = None
                aggressive_cleanup()
        review(path)


def _release_after_aborted_generation(pipe: Any) -> None:
    """Free what a finished generation frees, for a generation that stops early (Next, Quit, review decline or failure).

    Under ``low_memory`` the vendored code frees these after the last stage; a generation that ends earlier would leave
    the DiT resident and load the text encoder next to it, because ``load()`` returns early while ``_loaded`` is true.
    The exception in flight still references the DiT, so the cache is cleared at the start of the next generation
    (``_generation_scope``). Without ``low_memory`` nothing is freed.
    """
    if not pipe.low_memory:
        return
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    pipe.dit = None
    pipe.prompt_encoder.free()
    pipe.image_conditioner.free()
    pipe.upsampler = None
    pipe.vae_decoder = None
    pipe.audio_decoder = None
    pipe.vocoder = None
    pipe._loaded = False  # noqa: SLF001
    aggressive_cleanup()


@contextlib.contextmanager
def _review_after_stage1(pipe: Any, review: Callable[[Path], None], *, frame_rate: float, seed: int) -> Iterator[None]:
    """Let *review* see the half-size result of stage 1 while ``generate_and_save`` runs.

    ``generate_two_stage`` calls ``self._stage1(...)``, so an instance attribute shadows the method for the duration
    of the block. The wrapper returns the stage-1 result unchanged, so the generation goes on exactly as without it.
    The attribute is removed afterwards, also when *review* or the generation raises. A review that raises (a discard
    or a quit) ends the generation, and ``_generation_scope`` then releases the model.
    """
    stage1_method = pipe._stage1  # noqa: SLF001

    def _stage1_then_review(*args: Any, **kwargs: Any) -> Any:
        result = stage1_method(*args, **kwargs)
        _preview_small_render(pipe, result[0], review, frame_rate=frame_rate, seed=seed)
        return result

    pipe._stage1 = _stage1_then_review  # noqa: SLF001
    try:
        yield
    finally:
        del pipe._stage1  # noqa: SLF001


def _generate_two_stage(
    model: _LtxModel,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    stage1_steps: int,
    output_path: str,
    image_path: str | None,
    step_callback: Callable[[dict[str, Any]], None] | None,
    review: Callable[[Path], None] | None,
    end_image_path: str | None = None,
) -> None:
    """Generate at half resolution, spatially upscale 2x, and refine with the distilled pipeline.

    With *review*, the half-size result is decoded to a temporary video and passed to it between the two stages. With
    *end_image_path*, both stages pin the first and last frames (the pipeline resolves ``-1`` to the clip's last frame).
    """
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS

    pipe = model.pipeline
    pipe.stepwise = _TwoStageProgress(step_callback, stage2_total=len(STAGE_2_SIGMAS) - 1) if step_callback is not None else None
    with _review_after_stage1(pipe, review, frame_rate=model.frame_rate, seed=seed) if review is not None else contextlib.nullcontext():
        pipe.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=model.frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            images=_conditioning_inputs(image_path, end_image_path) if image_path is not None else None,
        )


@contextlib.contextmanager
def _env_default(name: str, value: str) -> Iterator[None]:
    """Set the environment variable *name* to *value* for the block; a value the user already set is kept."""
    if name in os.environ:
        yield
        return
    os.environ[name] = value
    try:
        yield
    finally:
        os.environ.pop(name, None)


@contextlib.contextmanager
def _generation_scope(model: _LtxModel) -> Iterator[None]:
    """Run a generation in its context: float16 compute by default, and the MLX buffer cache off while the transformer is streamed.

    Under ``low_memory`` it clears the MLX cache on entry and releases the model when an exception leaves the block.
    """
    pipe = model.pipeline
    if pipe.low_memory:
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        aggressive_cleanup()
    with contextlib.ExitStack() as stack:
        # The pipeline reads the compute dtype each time it loads the transformer, which happens inside a generation.
        stack.enter_context(_env_default(_COMPUTE_DTYPE_ENV, _DEFAULT_COMPUTE_DTYPE))
        if model.low_ram_streaming:
            stack.enter_context(mlx_cache_limit(0))
        try:
            yield
        except BaseException:
            _release_after_aborted_generation(pipe)
            raise


def _generate(
    model: _LtxModel,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    steps: int,
    output_path: str,
    image_path: str | None,
    stage1_steps: int | None,
    step_callback: Callable[[dict[str, Any]], None] | None,
    review: Callable[[Path], None] | None = None,
    end_image_path: str | None = None,
) -> Path:
    """Dispatch to the one-stage or two-stage generator that matches the loaded handle.

    Raises:
        ValueError: If ``stage1_steps`` is set but the handle was not loaded for upscale, or the reverse,
            if ``review`` is set without ``stage1_steps``, or ``end_image_path`` without ``image_path``.
    """
    if end_image_path is not None and image_path is None:
        raise ValueError("end_image_path needs image_path: the clip runs from the first-frame image to the last-frame image")
    if review is not None and stage1_steps is None:
        raise ValueError("review needs stage1_steps: only the two-stage upscale has a half-size render to review")
    if (stage1_steps is not None) != model.upscale:
        raise ValueError(f"stage1_steps must be set exactly when the model is loaded for upscale (upscale={model.upscale}, stage1_steps={stage1_steps})")
    # The vendored pipeline opens the images itself, so a sideways or transparent one is handed over as an upright copy.
    with _generation_scope(model), upright_image_copies(image_path, end_image_path) as (image_path, end_image_path):
        if stage1_steps is not None:
            _generate_two_stage(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, image_path, step_callback, review, end_image_path=end_image_path)
        else:
            _generate_one_stage(model, prompt, width, height, num_frames, seed, steps, output_path, image_path, step_callback, end_image_path=end_image_path)
    return Path(output_path)


class LtxVideoBackend:
    """Video backend using ltx-pipelines-mlx (dgrauet/ltx-2-mlx)."""

    name: str = "ltx"

    def load_model(
        self,
        model_path: str,
        **kwargs: Any,
    ) -> tuple[Any, VideoModelInfo]:
        """Load an LTX video pipeline.

        Args:
            model_path: Local path or HuggingFace repo ID.
            **kwargs: low_memory (bool), loras (list[tuple[str, float]]), upscale (bool), mlx_packs (the pinned
                packs by reference, each a ``HubFiles``) and mlx_text_encoder (the Gemma 3 reference).
                ``mode`` and ``end_image`` are accepted and ignored; the same pipeline serves t2v and i2v, with or without a last-frame image.

        Returns:
            Tuple of (backend-private model handle, VideoModelInfo).

        Raises:
            RuntimeError: If ltx-pipelines-mlx is not installed.
            ValueError: If the pack is LTX-2.5 but its text encoder is not at the folder root.
            FileNotFoundError: If a pinned pack lacks a file its configured list names.
        """
        low_memory: bool = kwargs.get("low_memory", True)
        loras: list[tuple[str, float]] | None = kwargs.get("loras")
        upscale: bool = bool(kwargs.get("upscale"))
        streaming = _use_low_ram_streaming(low_memory, memory_size_bytes())
        model_dir = _resolve_model_dir(model_path, packs=kwargs.get("mlx_packs") or {})
        text_encoder = _text_encoder_options(model_dir, kwargs.get("mlx_text_encoder"))

        try:
            # The constructor reads the LoRA mode, and sets the MLX cache limit to 0 process-wide when streaming; scope both.
            with _env_default(_LORA_MODE_ENV, _DEFAULT_LORA_MODE), mlx_cache_limit(0) if streaming else contextlib.nullcontext():
                if upscale:
                    from ltx_pipelines_mlx import DistilledPipeline

                    pipeline = DistilledPipeline(model_dir=model_dir, low_memory=low_memory, low_ram_streaming=streaming, **text_encoder)
                else:
                    from ltx_pipelines_mlx import BasePipeline

                    pipeline = BasePipeline(model_dir=model_dir, low_memory=low_memory, low_ram_streaming=streaming, **text_encoder)
        except ImportError as exc:
            raise RuntimeError("Video dependency ltx-pipelines-mlx not found. Reinstall with: uv sync") from exc

        from ltx_pipelines_mlx.utils.generation import is_ltx25_pack

        ltx25 = bool(is_ltx25_pack(pipeline.model_dir))
        if ltx25 and ltx_mlx_bundled_text_encoder(Path(pipeline.model_dir)) is None:
            raise ValueError(
                f"{model_path} is an LTX-2.5 pack but has no text_encoder.safetensors and text_encoder_config.json at its root; re-download it so the bundled Gemma 4 text encoder is present."
            )

        if loras:
            # ltx-pipelines-mlx attaches (or, with LTX2_LORA_MODE=fused, fuses) _pending_loras while it loads the transformer
            pipeline._pending_loras = loras  # noqa: SLF001

        model_info = detect_video_model(model_path)
        return _LtxModel(pipeline=pipeline, frame_rate=float(model_info.default_fps), upscale=upscale, low_ram_streaming=streaming, ltx25=ltx25), model_info

    def text_to_video(
        self,
        model: Any,
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
        """Generate video from text prompt.

        Args:
            model: Handle returned by load_model.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps.
            output_path: Path to save video.
            step_callback: Receives progress event dicts.
            **kwargs: stage1_steps for distilled two-stage upscale mode; review, a ``Callable[[Path], None]``
                that receives a temporary video of the half-size render before the upscale (needs stage1_steps).

        Returns:
            Path to generated video.

        Raises:
            ValueError: If stage1_steps does not match the model's upscale mode, or review is set without it.
        """
        return _generate(model, prompt, width, height, num_frames, seed, steps, output_path, None, kwargs.get("stage1_steps"), step_callback, kwargs.get("review"))

    def image_to_video(
        self,
        model: Any,
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
        """Generate video from image and text prompt.

        Args:
            model: Handle returned by load_model.
            image_path: Path to input image.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps.
            output_path: Path to save video.
            step_callback: Receives progress event dicts.
            **kwargs: stage1_steps for distilled two-stage upscale mode; review, a ``Callable[[Path], None]``
                that receives a temporary video of the half-size render before the upscale (needs stage1_steps);
                end_image, the path of a last-frame image the clip ends on.

        Returns:
            Path to generated video.

        Raises:
            ValueError: If stage1_steps does not match the model's upscale mode, or review is set without it.
        """
        return _generate(
            model, prompt, width, height, num_frames, seed, steps, output_path, image_path, kwargs.get("stage1_steps"), step_callback, kwargs.get("review"), end_image_path=kwargs.get("end_image")
        )

    def extend_video(
        self,
        model: Any,
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

        The clip's last 25 frames (and the sound under them) are encoded and pinned at the start of a window of
        ``25 + extend_frames`` frames at the clip's own size and frame rate. The window is generated and decoded,
        and only its new part is written.

        Args:
            model: Handle returned by ``load_model``.
            source: The checked clip to continue.
            prompt: Text prompt describing the scene and what happens next.
            extend_frames: New frames to generate; a multiple of 8, at least 8.
            seed: Random seed.
            steps: Distilled denoising steps.
            segment_dir: Existing directory that receives ``segment.mp4`` and, with audio, ``segment.wav``.
            audio_output: Whether to write the new part's sound. The sound is only written when the clip has
                any; a silent clip is continued without it. The clip's sound is pinned whenever it has some.
            step_callback: Receives progress event dicts.

        Returns:
            The generated segment: its video and, when ``audio_output`` is set and the clip has sound, its audio.

        Raises:
            ValueError: If the model was loaded for upscale, uses the diffusion video decoder, or ``extend_frames``
                is not a positive multiple of 8.
            RuntimeError: If the clip's tail frames or audio cannot be read.
        """
        if model.upscale:
            raise ValueError("Extending a video is not supported together with upscale.")
        if model.pipeline.video_decoder_block.video_decoder == "diffusion":
            raise ValueError("Extending a video is not supported with the diffusion video decoder.")
        if extend_frames < EXTEND_FRAME_STEP or extend_frames % EXTEND_FRAME_STEP:
            raise ValueError(f"extend_frames must be a positive multiple of {EXTEND_FRAME_STEP}, got {extend_frames}.")

        directory = Path(segment_dir)
        wants_audio = audio_output and source.has_audio
        target = _SegmentTarget(
            video_path=directory / "segment.mp4",
            width=source.width,
            height=source.height,
            audio_path=directory / "segment.wav" if wants_audio else None,
            extend_frames=extend_frames,
            fps_text=source.fps_text,
            color_space=source.color_space,
            color_primaries=source.color_primaries,
            color_trc=source.color_trc,
        )
        tail = read_source_tail(source)
        with _generation_scope(model):
            carry = _encode_carry(model.pipeline, source, tail, fps=source.fps)
            _generate_one_stage(
                model,
                prompt,
                source.width,
                source.height,
                window_frames(extend_frames),
                seed,
                steps,
                str(target.video_path),
                None,
                step_callback,
                frame_rate=source.fps,
                carry=carry,
                segment=target,
            )
        lead = audio_lead_samples(source.fps) if target.audio_path else 0
        return ExtendSegment(video_path=target.video_path, audio_path=target.audio_path, lead_samples=lead)
