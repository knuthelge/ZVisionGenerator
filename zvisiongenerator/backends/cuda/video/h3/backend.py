"""MiniMax H3 video backend for Windows and Linux (CUDA).

The model handle holds file paths and settings, never weights. Every clip loads each model just before its stage and
frees it right after, so the stages never overlap in memory:

1. the prompt (and any keyframes) become one tokenized presentation;
2. the text encoder, with its vision tower when there are keyframes, encodes it;
3. the video VAE encoder turns the keyframes or the extension tail into latents, then the audio VAE encodes the tail audio;
4. the stored 4-bit transformer copy loads, with the AdaLN tables of the clip's forwards and the user LoRAs, and denoises;
5. the video VAE decodes the video, then the audio VAE decodes the sound.

Upscale renders stage 1 at half size, lets a review look at it, and enlarges the frames with SeedVR2 to the full size.
Extend pins the last 22 frames of a clip, and their sound, at the start of a window and keeps only what follows them.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.driver import require_cuda
from zvisiongenerator.backends.cuda.memory import release_memory
from zvisiongenerator.backends.cuda.video.encode import encode_video
from zvisiongenerator.backends.cuda.video.h3 import audio as h3_audio
from zvisiongenerator.backends.cuda.video.h3 import conditioning, decode, sampler, schedule, stored_copy, text_encoder
from zvisiongenerator.backends.cuda.video.h3.frames import Geometry, geometry
from zvisiongenerator.backends.cuda.video.h3.sources import resolve_files, transformer_files
from zvisiongenerator.backends.cuda.video.runtime import configure_torch_runtime, fit_image
from zvisiongenerator.backends.cuda.video.token_chunking import chunk_tokens
from zvisiongenerator.core.video_types import ExtendSegment
from zvisiongenerator.utils.h3_memory import H3_FPS, H3_FRAMES_PER_CHUNK, H3_ROW_PIXELS, h3_cuda_system_bytes, h3_lora_bytes, h3_seedvr2_tier
from zvisiongenerator.utils.h3_preset import H3Files, H3Preset
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE as JOIN_SAMPLE_RATE
from zvisiongenerator.utils.video_extend import EXTEND_FPS_TOLERANCE, cut_window_audio
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_splice import build_segment_command, write_pcm16_wav, write_segment_frames
from zvisiongenerator.utils.video_tail import read_source_tail, read_source_tail_audio

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

    from zvisiongenerator.core.video_types import ExtendSource

__all__ = ["H3VideoBackend"]

_MODEL_INFO = VideoModelInfo(family="h3", backend="h3", supports_i2v=True, default_fps=H3_FPS, frame_alignment=H3_FRAMES_PER_CHUNK, resolution_alignment=H3_ROW_PIXELS)
_MODES = ("t2v", "i2v", "extend")
# What ``load_model`` takes. ``low_memory`` (H3 always streams its blocks) and ``model_name`` (the stored copy has one
# fixed name) are accepted for the callers that pass them to every backend, and ignored.
_LOAD_KEYS = frozenset({"mode", "low_memory", "loras", "upscale", "model_name", "preset"})
_UPSCALE_BATCH = 5  # frames SeedVR2 enlarges together; the smallest coherent batch
_SYSTEM_MEMORY_TIGHT_RATIO = 0.8  # a load needing more than this share of system memory leaves little to the rest of the machine
_RESOLUTION_FOR_UPSCALE = 2 * H3_ROW_PIXELS  # stage 1 runs at half size, which must stay a multiple of 32
_PHASE = "video"
_PHASE_STAGE_1 = "video_upscale_stage_1"
_PHASE_STAGE_2 = "video_upscale_stage_2"


@dataclass(frozen=True)
class _H3Model:
    """The handle ``load_model`` returns: the resolved files and the settings every clip is made with."""

    files: H3Files
    preset: H3Preset
    copy_dir: Path
    mode: str
    loras: tuple[tuple[Path, float], ...]
    upscale: bool
    device: Any


@dataclass(frozen=True)
class _ClipSpec:
    """One clip to render: its canvas, its settled schedule and the media it is conditioned on.

    Attributes:
        keyframes: The keyframe images as given (any size); each render fits them to its own canvas.
        anchors: ``"first"`` or ``"last"`` per keyframe.
        tail: The extension tail, ``(22, height, width, 3)`` uint8 at the canvas size.
        tail_audio: The sound under the tail, ``(2, samples)`` float32 at the audio VAE's rate.
        on_step: Receives ``{"phase", "current_step", "total_steps"}`` after every forward.
    """

    prompt: str
    width: int
    height: int
    num_frames: int
    seed: int
    forwards: int
    keyframes: tuple[Image.Image, ...] = ()
    anchors: tuple[str, ...] = ()
    tail: np.ndarray | None = None
    tail_audio: np.ndarray | None = None
    on_step: Callable[[dict[str, Any]], None] | None = None

    @property
    def row_set(self) -> str:
        """Which conditioning rows the clip has, as the AdaLN fold keys name them.

        An extension whose source has no sound holds no audio rows, so its forwards use the keys of a keyframe-only clip.
        """
        if self.tail is not None and self.tail_audio is not None:
            return "extend"
        return "keyframe" if self.keyframes or self.tail is not None else "plain"


@dataclass(frozen=True)
class _Rendered:
    """A decoded clip: 8-bit frames and a stereo waveform ``(2, samples)`` at the audio VAE's sample rate."""

    frames: list[np.ndarray]
    waveform: torch.Tensor


# ---- Small pure helpers ---------------------------------------------------------------------------------------------


def _models_dir() -> Path:
    """Return the folder that holds the stored copy."""
    return get_ziv_data_dir() / "models"


def _forwards(preset: H3Preset, steps: int) -> int:
    """Return *steps* when the preset has a schedule for it.

    Raises:
        ValueError: If there is no schedule for *steps*, naming the supported counts.
    """
    if steps not in preset.schedules:
        raise ValueError(f"MiniMax H3 runs {' or '.join(str(count) for count in sorted(preset.schedules))} steps, got {steps}.")
    return steps


def _progress(callback: Callable[[dict[str, Any]], None] | None, phase: str, *, offset: int = 0, total: int | None = None) -> Callable[[dict[str, Any]], None] | None:
    """Return *callback* adapted to report under *phase*, its steps shifted by *offset* and counted out of *total* when given."""
    if callback is None:
        return None

    def _on_step(event: dict[str, Any]) -> None:
        callback({"phase": phase, "current_step": offset + event["current_step"], "total_steps": event["total_steps"] if total is None else total})

    return _on_step


@contextmanager
def _out_of_memory_as_error(width: int, height: int, num_frames: int) -> Iterator[None]:
    """Turn a CUDA out-of-memory error inside the block into a ``RuntimeError`` that tells the user what to change."""
    import torch

    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f"MiniMax H3 ran out of GPU memory at {width}x{height}, {num_frames} frames. Try a shorter clip or a smaller size.") from exc


def _check_upscale_arguments(model: _H3Model, stage1_steps: int | None, review: Callable[[Path], None] | None, width: int, height: int) -> None:
    """Ensure the loaded model and the request agree on whether the run is upscaled, and that stage 1 gets a valid half-size canvas.

    Raises:
        ValueError: If *review* comes without *stage1_steps*, *stage1_steps* is missing for an upscale model, or the
            canvas is not a multiple of 64.
        RuntimeError: If *stage1_steps* is set but the model was not loaded with ``upscale``.
    """
    if review is not None and stage1_steps is None:
        raise ValueError("review needs stage1_steps: only the upscale has a half-size render to review")
    if model.upscale and stage1_steps is None:
        raise ValueError("stage1_steps must be provided when upscale is enabled")
    if not model.upscale and stage1_steps is not None:
        raise RuntimeError("Upscale was requested, but the model was not loaded with upscale=True.")
    if stage1_steps is not None and (width % _RESOLUTION_FOR_UPSCALE or height % _RESOLUTION_FOR_UPSCALE):
        raise ValueError(f"The upscale needs width and height divisible by {_RESOLUTION_FOR_UPSCALE}, got {width}x{height}.")


def _require_mode(model: _H3Model, mode: str) -> None:
    """Raise unless *model* was loaded for *mode*."""
    if model.mode != mode:
        raise RuntimeError(f"This loaded video model was loaded for mode {model.mode!r}, not {mode!r} (load it with mode={mode!r}).")


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Return ``(channels, samples)`` audio resampled from *source_rate* to *target_rate* (polyphase)."""
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor, axis=1).astype("float32")


def _new_audio(waveform: torch.Tensor, extend_frames: int) -> np.ndarray:
    """Return the sound of the new frames of an extension window as ``(2, extend_frames * 2000)`` float32 at the join's rate.

    The window's sound is resampled to the rate the join works at; the samples that cover the carried tail are dropped,
    the samples of the new frames kept, and a window whose sound ends early is faded out and padded with silence.
    """
    samples = _resample(waveform.float().cpu().numpy(), h3_audio.AUDIO_SAMPLE_RATE, JOIN_SAMPLE_RATE)
    return cut_window_audio(samples, round(conditioning.TAIL_FRAMES * JOIN_SAMPLE_RATE / H3_FPS), round(extend_frames * JOIN_SAMPLE_RATE / H3_FPS))


def _write_video(clip: _Rendered, output_path: Path) -> None:
    """Encode *clip* with its sound to *output_path* at the model's frame rate."""
    encode_video(clip.frames, fps=H3_FPS, output_path=str(output_path), audio=clip.waveform, audio_sample_rate=h3_audio.AUDIO_SAMPLE_RATE)


# ---- Stages: each loads its models, returns CPU results and lets go of the models on return -------------------------------


def _encode_text(model: _H3Model, prompt: str, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return the text conditioning ``(1, rows, hidden)`` of *prompt* with *images* in front of it, and the modality tag of each row."""
    repo = model.files.repo_dir
    presentation = text_encoder.tokenize_presentation(prompt, repo / "tokenizer", repo / "processor", images)
    encoder = text_encoder.load_text_encoder(model.files.text_encoder_dir, model.files.text_encoder_shards, model.device, layers=model.preset.text_encoder_layers, vision=bool(images))
    embeds = text_encoder.encode(encoder, presentation, model.device).cpu()
    return embeds, presentation.token_tags


def _encode_visuals(model: _H3Model, spec: _ClipSpec, images: Sequence[Image.Image]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Return the keyframe latents and the tail latents (``None`` without a tail), encoded by the video VAE's encoder."""
    encoder = conditioning.load_vae_encoder(model.files.repo_dir / "vae", model.device)
    keyframes = tuple(conditioning.encode_keyframe(encoder, image, spec.width, spec.height) for image in images)
    tail = conditioning.encode_tail(encoder, spec.tail) if spec.tail is not None else None
    return keyframes, tail


def _encode_tail_audio(model: _H3Model, tail_audio: np.ndarray) -> torch.Tensor:
    """Return the tail's sound as held audio rows ``(74, 32)``, encoded by the audio VAE."""
    import torch

    audio_vae = h3_audio.load_audio_vae(model.files.repo_dir / "audio_vae", model.device)
    return conditioning.encode_tail_audio(audio_vae, torch.from_numpy(tail_audio))


def _fold_keys(schedulers: tuple[Any, Any], row_set: str) -> list[Any]:
    """Return the AdaLN fold keys of every forward of *schedulers* for a clip with the conditioning rows of *row_set*."""
    from zvisiongenerator.backends.cuda.video.h3 import adaln  # imports torch at its top level

    video, audio = schedulers
    return adaln.unique_keys(adaln.step_keys(video, audio, row_set))


def _all_fold_keys(preset: H3Preset, repo_dir: Path) -> list[Any]:
    """Return the AdaLN fold keys the stored copy is built for: those of every step count of *preset*."""
    from zvisiongenerator.backends.cuda.video.h3 import adaln

    return adaln.fold_keys(preset.schedules, repo_dir)


def _chunk_feed_forwards(transformer: Any, preset: H3Preset) -> None:
    """Run the feed-forward of every block over at most ``preset.ffn_chunk_tokens`` rows at a time, when the preset sets it.

    The feed-forward widens each row before narrowing it, so its temporaries are the largest tensors of a long clip's
    forward; each row is independent, so chunking gives the same output with a fraction of that memory.
    """
    if preset.ffn_chunk_tokens is not None:
        chunk_tokens([block.ff for block in transformer.transformer_blocks], preset.ffn_chunk_tokens)


def _denoise(
    model: _H3Model,
    spec: _ClipSpec,
    geo: Geometry,
    embeds: torch.Tensor,
    token_tags: Sequence[int],
    schedulers: tuple[Any, Any],
    held: sampler.Conditioning,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the transformer for the clip's forwards and LoRAs, denoise, and return the video and audio latents on the CPU."""
    transformer = stored_copy.load_copy(model.copy_dir, _fold_keys(schedulers, spec.row_set), model.device, loras=model.loras)
    _chunk_feed_forwards(transformer, model.preset)
    video, audio = sampler.denoise(transformer, embeds, geo, schedulers, spec.seed, model.device, conditioning=held, step_callback=spec.on_step, phase=_PHASE, text_token_tags=token_tags)
    return video.cpu(), audio.cpu()


def _decode_frames(model: _H3Model, latents: torch.Tensor) -> list[np.ndarray]:
    """Return the 8-bit frames of the video *latents*."""
    vae = decode.load_video_vae(model.files.repo_dir / "vae", model.device)
    return decode.decode_video(vae, latents.to(model.device))


def _decode_waveform(model: _H3Model, latents: torch.Tensor) -> torch.Tensor:
    """Return the stereo waveform ``(2, samples)`` at 32 kHz of the audio *latents*, on the CPU."""
    audio_vae = h3_audio.load_audio_vae(model.files.repo_dir / "audio_vae", model.device)
    return h3_audio.decode_audio(audio_vae, latents.to(model.device))


def _staged(stage: Callable[..., Any], *args: Any) -> Any:
    """Run one stage, then return the memory its models held (also when it fails)."""
    try:
        return stage(*args)
    finally:
        release_memory()


def _render(model: _H3Model, spec: _ClipSpec) -> _Rendered:
    """Run every stage of one clip in order, releasing each before the next, and return the decoded frames and sound."""
    geo = geometry(spec.width, spec.height, spec.num_frames)
    images = [fit_image(image, spec.width, spec.height) for image in spec.keyframes]
    schedulers = schedule.build_schedulers(model.preset.schedules[spec.forwards], model.files.repo_dir)
    embeds, token_tags = _staged(_encode_text, model, spec.prompt, images)
    keyframes, tail_latents = _staged(_encode_visuals, model, spec, images) if images or spec.tail is not None else ((), None)
    tail_audio = _staged(_encode_tail_audio, model, spec.tail_audio) if spec.tail_audio is not None else None
    held = sampler.Conditioning(keyframes=keyframes, anchors=spec.anchors, tail_latents=tail_latents, tail_audio=tail_audio)
    video_latents, audio_latents = _staged(_denoise, model, spec, geo, embeds, token_tags, schedulers, held)
    frames = _staged(_decode_frames, model, video_latents)
    waveform = _staged(_decode_waveform, model, audio_latents)
    return _Rendered(frames=frames, waveform=waveform)


# ---- Upscale -----------------------------------------------------------------------------------------------------------


def _upscale_frames(frames: list[np.ndarray], width: int, height: int, seed: int, progress: Callable[[int, int], None] | None) -> list[np.ndarray]:
    """Enlarge *frames* to *width* x *height* with the SeedVR2 tier the config and this machine's memory pick."""
    from zvisiongenerator.backends import accelerator_memory_budget_or_none, get_seedvr2_upscaler
    from zvisiongenerator.seedvr2_loader import build_seedvr2_setup, load_seedvr2
    from zvisiongenerator.utils.config import load_config

    upscaler = get_seedvr2_upscaler()
    if upscaler is None or not hasattr(upscaler, "upscale_video"):
        raise RuntimeError("SeedVR2 video upscaling is not available on this platform.")
    budget = accelerator_memory_budget_or_none()
    models_dir = _models_dir()
    setup = build_seedvr2_setup(load_config(), weights=upscaler.weights, budget_bytes=budget.gpu_bytes if budget else None, system_bytes=budget.system_bytes if budget else None, models_dir=models_dir)
    seedvr2 = load_seedvr2(upscaler, setup.tier.model, models_dir=models_dir, on_phase=None)
    return upscaler.upscale_video(seedvr2, frames, width=width, height=height, seed=seed, softness=setup.settings.softness, batch_size=_UPSCALE_BATCH, progress=progress)


def _review_stage1(clip: _Rendered, review: Callable[[Path], None]) -> None:
    """Write the half-size render to a temporary video file and hand its path to *review*; the file is deleted when *review* returns or raises."""
    with tempfile.TemporaryDirectory(prefix="ziv-review-", ignore_cleanup_errors=True) as directory:
        path = Path(directory) / "small-render.mp4"
        _write_video(clip, path)
        review(path)


def _render_upscaled(
    model: _H3Model,
    spec: _ClipSpec,
    output_path: Path,
    *,
    step_callback: Callable[[dict[str, Any]], None] | None,
    review: Callable[[Path], None] | None,
) -> None:
    """Render *spec* at half size, review it, enlarge its frames with SeedVR2 and write them with the stage-1 sound.

    Progress is one run: the stage-1 forwards, then one step per SeedVR2 batch. An exception from *review* ends the
    generation.
    """
    batches = math.ceil(spec.num_frames / _UPSCALE_BATCH)
    total = spec.forwards + batches
    small = _render(model, replace(spec, width=spec.width // 2, height=spec.height // 2, on_step=_progress(step_callback, _PHASE_STAGE_1, total=total)))
    if review is not None:
        _review_stage1(small, review)
    release_memory()
    stage2 = _progress(step_callback, _PHASE_STAGE_2, offset=spec.forwards, total=total)

    def _on_batch(done: int, _batches: int) -> None:
        if stage2 is not None:
            stage2({"current_step": done, "total_steps": batches})

    frames = _staged(_upscale_frames, small.frames, spec.width, spec.height, spec.seed, _on_batch)
    _write_video(_Rendered(frames=frames, waveform=small.waveform), output_path)


# ---- Extend --------------------------------------------------------------------------------------------------------------


def _check_extend_arguments(model: _H3Model, source: ExtendSource, extend_frames: int) -> None:
    """Raise unless the model can extend, *extend_frames* is a positive multiple of 17, and *source* is a 24 fps clip with a size that is a multiple of 32.

    Raises:
        RuntimeError: If the model was not loaded for extend.
        ValueError: For an upscale model, a bad frame count, a source that is not 24 fps, or a size that is not a multiple of 32.
    """
    _require_mode(model, "extend")
    if model.upscale:
        raise ValueError("Extending a video is not supported together with upscale.")
    if extend_frames < H3_FRAMES_PER_CHUNK or extend_frames % H3_FRAMES_PER_CHUNK:
        raise ValueError(f"extend_frames must be a positive multiple of {H3_FRAMES_PER_CHUNK}, got {extend_frames}.")
    if abs(source.fps - H3_FPS) > EXTEND_FPS_TOLERANCE:
        raise ValueError(f"MiniMax H3 extends {H3_FPS} fps clips only; the source runs at {source.fps:g} fps.")
    if source.width % H3_ROW_PIXELS or source.height % H3_ROW_PIXELS:
        raise ValueError(f"MiniMax H3 extends clips whose size is a multiple of {H3_ROW_PIXELS}; the source is {source.width}x{source.height}.")


def _write_segment(clip: _Rendered, source: ExtendSource, extend_frames: int, directory: Path, *, audio: bool) -> ExtendSegment:
    """Write the new part of a decoded extension window: its frames to ``segment.mp4`` and, with *audio*, its sound to ``segment.wav``.

    The 22 frames of the carried tail, and the sound that covers them, are dropped. The video is encoded with the
    source's size, frame rate and colour tags so that it can be stream-copied onto it. The WAV starts exactly at the new
    part and runs at the join's sample rate.

    Raises:
        RuntimeError: If the window did not decode to ``22 + extend_frames`` frames.
    """
    expected = conditioning.TAIL_FRAMES + extend_frames
    if len(clip.frames) != expected:
        raise RuntimeError(f"The extension window decoded to {len(clip.frames)} frames, expected {expected}.")
    video_path = directory / "segment.mp4"
    command = build_segment_command(source.width, source.height, source.fps_text, video_path, color_space=source.color_space, color_primaries=source.color_primaries, color_trc=source.color_trc)
    write_segment_frames(clip.frames[conditioning.TAIL_FRAMES :], command)
    if not audio:
        return ExtendSegment(video_path=video_path)
    audio_path = directory / "segment.wav"
    write_pcm16_wav(_new_audio(clip.waveform, extend_frames), audio_path)
    return ExtendSegment(video_path=video_path, audio_path=audio_path, lead_samples=0)


def _extend(
    model: _H3Model,
    source: ExtendSource,
    prompt: str,
    *,
    extend_frames: int,
    seed: int,
    forwards: int,
    directory: Path,
    audio_output: bool,
    step_callback: Callable[[dict[str, Any]], None] | None,
) -> ExtendSegment:
    """Read the source's tail, render the window that starts with it, and write the new part."""
    tail = read_source_tail(source, frames=conditioning.TAIL_FRAMES)
    tail_audio = read_source_tail_audio(source, h3_audio.AUDIO_SAMPLE_RATE, frames=conditioning.TAIL_FRAMES) if source.has_audio else None
    window = conditioning.TAIL_FRAMES + extend_frames
    spec = _ClipSpec(
        prompt=prompt,
        width=source.width,
        height=source.height,
        num_frames=window,
        seed=seed,
        forwards=forwards,
        tail=tail,
        tail_audio=tail_audio,
        on_step=_progress(step_callback, _PHASE),
    )
    with _out_of_memory_as_error(source.width, source.height, window):
        clip = _render(model, spec)
    return _write_segment(clip, source, extend_frames, directory, audio=audio_output and source.has_audio)


# ---- The backend ----------------------------------------------------------------------------------------------------------


def _open_rgb(path: str | Path) -> Image.Image:
    """Return the image at *path* converted to RGB."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _render_clip(
    model: _H3Model,
    spec: _ClipSpec,
    output_path: Path,
    *,
    stage1_steps: int | None,
    step_callback: Callable[[dict[str, Any]], None] | None,
    review: Callable[[Path], None] | None,
) -> None:
    """Render *spec* in one stage, or at half size plus the SeedVR2 upscale when *stage1_steps* is set, and write the video to *output_path*."""
    with _out_of_memory_as_error(spec.width, spec.height, spec.num_frames):
        if stage1_steps is None:
            _write_video(_render(model, replace(spec, on_step=_progress(step_callback, _PHASE))), output_path)
        else:
            _render_upscaled(model, spec, output_path, step_callback=step_callback, review=review)


def _check_system_memory(loras: Sequence[tuple[Path, float]], upscale: bool) -> None:
    """Refuse a load whose system memory (base, user LoRAs and the upscale's SeedVR2 tier) exceeds the machine's, and warn above 80% of it.

    The decoded frames add to this per clip, so the figure is a floor.

    Raises:
        RuntimeError: If the estimate is more than the machine's system memory.
    """
    import psutil

    from zvisiongenerator.backends import accelerator_memory_budget_or_none
    from zvisiongenerator.utils.config import load_config

    total = int(psutil.virtual_memory().total)
    tier = h3_seedvr2_tier(load_config(), accelerator_memory_budget_or_none()) if upscale else None
    need = h3_cuda_system_bytes(loras_bytes=h3_lora_bytes(path for path, _weight in loras), seedvr2_tier=tier)
    if need > total:
        raise RuntimeError(
            f"MiniMax H3 needs about {need / 1024**3:.0f} GB of system memory; this machine has {total / 1024**3:.0f} GB." + (" Generate without Upscale to need less." if upscale else "")
        )
    if need > total * _SYSTEM_MEMORY_TIGHT_RATIO:
        warnings.warn(f"MiniMax H3 needs about {need / 1024**3:.0f} GB of this machine's {total / 1024**3:.0f} GB of system memory, leaving little for other apps.", stacklevel=2)


class H3VideoBackend:
    """Video backend for MiniMax H3 on Windows and Linux: a stored 4-bit transformer copy streamed block by block."""

    name = "h3"

    def load_model(self, model_path: str, **kwargs: Any) -> tuple[_H3Model, VideoModelInfo]:
        """Resolve the model's files and make sure the stored 4-bit copy exists; the models themselves load per clip.

        The first load of a repository writes the copy once, which reads the full transformer and can take several minutes.

        Args:
            model_path: A local H3 folder or the Hub reference of the MiniMax H3 repository.
            **kwargs:
                mode: ``"t2v"`` (default), ``"i2v"`` or ``"extend"``; the clip methods check it.
                preset: The ``H3Preset`` (required).
                loras: ``(path, weight)`` of each user LoRA.
                upscale: Whether the clips are made with the SeedVR2 upscale.
                low_memory, model_name: Accepted and ignored.

        Returns:
            Tuple of (backend-private model handle, VideoModelInfo).

        Raises:
            ValueError: For a keyword other than the above (quantize, transformer_source, sampler, image_crf and
                repo_ignore are among those refused), an unknown mode, or a missing preset.
            FileNotFoundError: If a LoRA file is missing.
            RuntimeError: If CUDA is unavailable, or the load needs more system memory than the machine has.
        """
        unknown = sorted(set(kwargs) - _LOAD_KEYS)
        if unknown:
            raise ValueError(f"MiniMax H3 does not take the option(s): {', '.join(unknown)}.")
        mode = kwargs.get("mode", "t2v")
        if mode not in _MODES:
            raise ValueError(f"Unsupported video generation mode: {mode}")
        preset = kwargs.get("preset")
        if not isinstance(preset, H3Preset):
            raise ValueError("MiniMax H3 needs its preset to load (preset=).")
        loras = tuple((Path(path), float(weight)) for path, weight in kwargs.get("loras") or ())
        for path, _weight in loras:
            if not path.exists():
                raise FileNotFoundError(f"LoRA weights not found: {path}")

        import torch

        require_cuda(torch, "MiniMax H3 on Windows and Linux")
        configure_torch_runtime(torch)
        _check_system_memory(loras, bool(kwargs.get("upscale")))
        device = torch.device("cuda")
        files = resolve_files(model_path, preset)
        shards, config = transformer_files(files.repo_dir)
        copy_dir = stored_copy.ensure_copy(_models_dir(), shards, config, files.lora_file, preset.lora_scale, _all_fold_keys(preset, files.repo_dir), device)
        return _H3Model(files=files, preset=preset, copy_dir=copy_dir, mode=mode, loras=loras, upscale=bool(kwargs.get("upscale")), device=device), _MODEL_INFO

    def text_to_video(
        self,
        model: _H3Model,
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
        """Generate a video, with sound, from a text prompt and write it to *output_path*.

        Args:
            model: Handle returned by ``load_model`` with ``mode="t2v"``.
            prompt: Text prompt.
            width: Video width (a multiple of 32; of 64 with the upscale).
            height: Video height.
            num_frames: Frames at 24 fps, of the form ``17 * n + 5``.
            seed: Random seed.
            steps: Transformer forwards; a step count of the preset's schedules (4 or 8).
            output_path: Path to save the video.
            step_callback: Receives ``{"phase", "current_step", "total_steps"}`` after every forward.
            **kwargs: ``stage1_steps`` (the forwards of the half-size render, with the upscale) and ``review`` (see
                ``VideoBackend.text_to_video``).

        Returns:
            Path to the generated video.

        Raises:
            ValueError: For unsupported steps, a canvas or frame count off the grid, or inconsistent upscale arguments.
            RuntimeError: If the model was not loaded for this mode, or the GPU runs out of memory.
        """
        _require_mode(model, "t2v")
        stage1_steps, review = kwargs.get("stage1_steps"), kwargs.get("review")
        _check_upscale_arguments(model, stage1_steps, review, width, height)
        forwards = _forwards(model.preset, steps if stage1_steps is None else stage1_steps)
        spec = _ClipSpec(prompt=prompt, width=width, height=height, num_frames=num_frames, seed=seed, forwards=forwards)
        try:
            _render_clip(model, spec, Path(output_path), stage1_steps=stage1_steps, step_callback=step_callback, review=review)
        finally:
            release_memory()
        return Path(output_path)

    def image_to_video(
        self,
        model: _H3Model,
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
        """Generate a video, with sound, from a first-frame image (and optionally a last-frame image) and a prompt.

        Each image is scaled to cover the canvas and centre-cropped.

        Args:
            model: Handle returned by ``load_model`` with ``mode="i2v"``.
            image_path: The first frame.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Frames at 24 fps, of the form ``17 * n + 5``.
            seed: Random seed.
            steps: Transformer forwards (4 or 8).
            output_path: Path to save the video.
            step_callback: Receives ``{"phase", "current_step", "total_steps"}`` after every forward.
            **kwargs: ``end_image`` (the path of the last frame), ``stage1_steps`` and ``review`` (see ``text_to_video``).

        Returns:
            Path to the generated video.

        Raises:
            ValueError: As ``text_to_video``.
            RuntimeError: If the model was not loaded for this mode, or the GPU runs out of memory.
        """
        _require_mode(model, "i2v")
        stage1_steps, review = kwargs.get("stage1_steps"), kwargs.get("review")
        _check_upscale_arguments(model, stage1_steps, review, width, height)
        forwards = _forwards(model.preset, steps if stage1_steps is None else stage1_steps)
        end_image = kwargs.get("end_image")
        images = (_open_rgb(image_path), *((_open_rgb(end_image),) if end_image else ()))
        spec = _ClipSpec(prompt=prompt, width=width, height=height, num_frames=num_frames, seed=seed, forwards=forwards, keyframes=images, anchors=("first", "last") if end_image else ("first",))
        try:
            _render_clip(model, spec, Path(output_path), stage1_steps=stage1_steps, step_callback=step_callback, review=review)
        finally:
            release_memory()
        return Path(output_path)

    def extend_video(
        self,
        model: _H3Model,
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

        The clip's last 22 frames, and the sound under them, are pinned at the start of a window of ``22 + extend_frames``
        frames at the clip's own size. The window is generated and decoded, and only its new part is written.

        Args:
            model: Handle returned by ``load_model`` with ``mode="extend"``.
            source: The checked clip to continue (24 fps, a size that is a multiple of 32).
            prompt: Text prompt for what happens next.
            extend_frames: New frames; a positive multiple of 17.
            seed: Random seed.
            steps: Transformer forwards (4 or 8).
            segment_dir: Existing directory that receives ``segment.mp4`` and, with audio, ``segment.wav``.
            audio_output: Whether to write the new part's sound (only when the clip has any). The clip's sound is pinned
                whenever it has some.
            step_callback: Receives ``{"phase", "current_step", "total_steps"}`` after every forward.

        Returns:
            The generated segment: its video and, with audio, a 48 kHz WAV that starts at the new part.

        Raises:
            ValueError: If the model was loaded for upscale, ``extend_frames`` is not a positive multiple of 17, the
                source is not 24 fps or not a multiple of 32 in size, or *steps* is unsupported.
            RuntimeError: If the model was not loaded for extend, the clip's tail cannot be read, or the GPU runs out of memory.
        """
        _check_extend_arguments(model, source, extend_frames)
        forwards = _forwards(model.preset, steps)
        try:
            return _extend(
                model,
                source,
                prompt,
                extend_frames=extend_frames,
                seed=seed,
                forwards=forwards,
                directory=Path(segment_dir),
                audio_output=audio_output,
                step_callback=step_callback,
            )
        finally:
            release_memory()
