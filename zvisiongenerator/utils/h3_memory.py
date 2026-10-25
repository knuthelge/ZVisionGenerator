"""Row and frame arithmetic of the MiniMax H3 transformer sequence.

H3 sees one packed sequence of text rows, audio rows and video rows. The counts follow from the clip size alone, so the
geometry of a canvas, the memory estimate and the sampler all share these pure functions. They mirror diffusers'
``video_latent_num_frames`` and ``audio_latent_num_frames`` (checked by a contract test against the pinned release).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_memory import GIB, CudaMemoryEstimate, MemoryBudget, classify_cuda_memory_fit, classify_memory_fit, discrete_clip_fit

__all__ = [
    "H3_AUDIO_CHANNELS",
    "H3_AUDIO_LATENTS_PER_SECOND",
    "H3_FPS",
    "H3_FRAMES_PER_CHUNK",
    "H3_LATENTS_PER_CHUNK",
    "H3_PATCH",
    "H3_REFINE_FORWARDS",
    "H3_REFINE_OVERLAP_LATENT_FRAMES",
    "H3_REFINE_SCHEDULE",
    "H3_REFINE_WINDOW_LATENT_FRAMES",
    "H3_ROW_PIXELS",
    "H3_SPATIAL_RATIO",
    "H3_TAIL_FRAMES",
    "H3_TEXT_TOKENS",
    "estimate_h3_cuda_clip_memory",
    "estimate_h3_mlx_clip_memory",
    "estimate_h3_mlx_seconds",
    "h3_audio_latents",
    "h3_clip_fit",
    "h3_cuda_system_bytes",
    "h3_expected_seconds",
    "h3_extend_window_frames",
    "h3_frames_of_latent_frames",
    "h3_latent_frames",
    "h3_lora_bytes",
    "h3_mlx_clip_fit",
    "h3_refine_windows",
    "h3_sequence_rows",
    "h3_upscale_keeps_transformer",
    "h3_video_rows",
]

H3_FRAMES_PER_CHUNK = 17  # a clip is 17 * n + 5 frames
H3_LATENTS_PER_CHUNK = 5  # each further 17 frames add 5 latent frames; the leading 5 frames give 2
H3_FPS = 24
H3_AUDIO_LATENTS_PER_SECOND = 40
H3_SPATIAL_RATIO = 16  # pixels per latent cell
H3_PATCH = 2  # latent cells per row, along each spatial axis
H3_ROW_PIXELS = H3_SPATIAL_RATIO * H3_PATCH  # 32 pixels per row, along each spatial axis
H3_AUDIO_CHANNELS = 2  # the mono audio VAE codes each of the two channels on its own
H3_TEXT_TOKENS = 100  # approximation of the prompt length in rows
H3_TAIL_FRAMES = 22  # an extension carries 17 * 1 + 5 frames of its source, the smallest valid clip
# The upscale refines the enlarged latents with the last forwards of the 8-step schedule, in overlapping windows of latent frames.
# They are code constants, not config: there is no user control, and they are tied to the stored copy's fold tables.
H3_REFINE_SCHEDULE = 8  # the schedule the refine runs the tail of, whatever the stage-1 step count
H3_REFINE_FORWARDS = 2  # forwards of that schedule's tail
H3_REFINE_WINDOW_LATENT_FRAMES = 22  # 73 frames
H3_REFINE_OVERLAP_LATENT_FRAMES = 7

# macOS (MLX) estimates: text-to-video from the fork's M4 Pro measurements, the image modes' conditioning phase from a Mac M5 with 32 GB.
_MLX_IMAGE_MODES = ("i2v", "extend")
_MLX_REFERENCE_ROWS = 12_946  # 768x448 at 124 frames: text, audio and video rows
_MLX_REFERENCE_BYTES = 15 * GIB  # peak of a text-to-video run at the reference rows
_MLX_BYTES_PER_ROW = 160 * 1024  # growth of the denoise peak per sequence row
# The conditioning phase of the image modes (keyframe encode, vision tower, prompt encode) peaks at 14.2 GiB, measured on a Mac M5
# with 32 GB at 576x320 and 768x448, 73 frames; the 4-bit text encoder sets it.
_MLX_LOAD_PHASE_BYTES = int(14.2 * GIB)
_MLX_FIXED_SECONDS = 240  # load, text encode, decode and mux
_MLX_REFERENCE_SECONDS = 23 * 60  # the fork's time at the reference clip and 8 forwards
_MLX_REFERENCE_FORWARDS = 8
_MLX_LINEAR_SHARE = 0.675  # share of a forward at the reference rows that grows linearly with rows (the rest is attention)
_MLX_FORWARD_SECONDS = (_MLX_REFERENCE_SECONDS - _MLX_FIXED_SECONDS) / _MLX_REFERENCE_FORWARDS
_MLX_LINEAR_SECONDS_PER_ROW = _MLX_LINEAR_SHARE * _MLX_FORWARD_SECONDS / _MLX_REFERENCE_ROWS
_MLX_QUADRATIC_SECONDS_PER_ROW2 = (1 - _MLX_LINEAR_SHARE) * _MLX_FORWARD_SECONDS / _MLX_REFERENCE_ROWS**2

# Windows and Linux (CUDA) estimates, calibrated on an RTX 3080 (10 GiB) with 30 GiB of system memory. The GPU figures
# are the peaks of real runs with the feed-forward chunked (the preset's ``ffn_chunk_tokens``) at 15,499 to 42,560 rows;
# every estimate sits at or above the measured peak.
_CUDA_MODES = ("t2v", "i2v", "extend")
_CUDA_REFERENCE_ROWS = 15_499  # 864x480 at 124 frames
_CUDA_TEXT_ENCODER_BYTES = int(2.10 * GIB)  # the 4-bit text encoder while the prompt is encoded
_CUDA_VISION_BYTES = int(1.11 * GIB)  # ... plus its vision tower when keyframes are encoded
_CUDA_DIT_REFERENCE_BYTES = int(4.9 * GIB)  # the streamed transformer's process peak at the reference rows: 4.5 GiB reserved plus the CUDA context
_CUDA_DIT_BYTES_PER_ROW = 151 * 1024  # growth of the reserved peak per further row, with the feed-forward chunked
_CUDA_DIT_LORA_BYTES_PER_BYTE = 2.7  # GPU bytes per byte of user LoRA file: the run-time merge holds the factors and a float copy
_CUDA_GPU_HEADROOM_BYTES = GIB  # the desktop's share of the card (0.65 to 0.93 GiB); the allocator's slack is in the peaks above
_CUDA_DECODE_BASE_BYTES = int((5.157 + 0.70) * GIB)  # the video VAE decode: weights and working set, plus allocator slack
_CUDA_DECODE_BYTES_PER_PIXEL = 9.48e-7 * GIB
_CUDA_SYSTEM_BASE_BYTES = int(17.7 * GIB)  # the CUDA runtime, the stored copy's streaming buffers and the encoders' CPU side
_CUDA_VISION_SYSTEM_BYTES = int(1.5 * GIB)  # the vision tower's CPU side while keyframes are encoded
_CUDA_FIXED_SECONDS = 116  # load, text encode, prepare the transformer and mux
_CUDA_DECODE_BASE_SECONDS = 8.8
_CUDA_DECODE_SECONDS_PER_PIXEL_FRAME = 3.01e-7
_CUDA_EXPORT_SECONDS = 0.8  # per 124 frames
_CUDA_REFERENCE_FRAMES = 124
_CUDA_FORWARD_SECONDS = 16.95  # one transformer forward at the reference rows
_CUDA_LINEAR_SHARE = 0.675  # share of that forward that grows linearly with rows (the rest is attention)
_CUDA_C1 = _CUDA_LINEAR_SHARE * _CUDA_FORWARD_SECONDS / _CUDA_REFERENCE_ROWS  # seconds per row
_CUDA_C2 = (1 - _CUDA_LINEAR_SHARE) * _CUDA_FORWARD_SECONDS / _CUDA_REFERENCE_ROWS**2  # seconds per row squared
_CUDA_IMAGE_ENCODE_SECONDS = 30  # the video VAE encoder (and the vision tower) of a keyframe or an extension tail
_CUDA_LORA_FORWARD_FACTOR = 1.08  # a user LoRA is merged at run time
# The latent upscale (RTX 3080): the network runs unchunked, so its GPU peak is linear in the full-size latent volume
# ``cells = (width / 16) * (height / 16) * latent_frames``. The line passes above the four peaks measured with the network alone:
# 1.71 GiB at 130,240 cells (1280x704, 124 frames), 2.28 at 207,200 (1600x896, 124), 4.14 at 403,200 (1600x896, 243) and 7.44 at
# 795,200 (1600x896, 483).
_CUDA_UPSAMPLER_BASE_BYTES = int(0.75 * GIB)
_CUDA_UPSAMPLER_BYTES_PER_CELL = 8.43e-6 * GIB
_CUDA_LATENT_UPSCALE_SECONDS = 10  # the network: 6.7 to 10.9 s measured for 37 to 72 latent frames
_CUDA_REFINE_LOAD_SECONDS = 62  # the second load of the stored transformer copy for the refine: 35 to 62 s measured
# An upscale without review keeps the transformer loaded from stage 1 to the refine (one load, not two), so it stays on the GPU
# while the waveform is decoded, the latents are enlarged and the keyframes are encoded. What stays is the part outside the
# streamed blocks plus the quantization state and fold tables of the blocks: 2016 MiB measured on an RTX 3080 (a stage-1 size of
# 640x352), rounded up. The upscaler stage measured about 3.0 GiB beside it at 1280x704.
_CUDA_RESIDENT_TRANSFORMER_BYTES = int(2.1 * GIB)
_CUDA_AUDIO_DECODE_BYTES = GIB  # the audio VAE's 0.58 GiB of weights and a few seconds of audio
_CUDA_ENCODE_BASE_BYTES = int(1.5 * GIB)  # the keyframe encode: the video VAE's encoder (0.67 GiB in float32) and slack
_CUDA_ENCODE_BYTES_PER_PIXEL = 2 * _CUDA_DECODE_BYTES_PER_PIXEL  # one frame in float32 against the decoder's chunks in bfloat16


def h3_latent_frames(num_frames: int) -> int:
    """Return the latent frame count of a *num_frames* clip.

    Raises:
        ValueError: If *num_frames* is not of the form ``17 * n + 5``.
    """
    if num_frames < H3_LATENTS_PER_CHUNK or num_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form {H3_FRAMES_PER_CHUNK} * n + {H3_LATENTS_PER_CHUNK}, got {num_frames}.")
    return (num_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2


def h3_frames_of_latent_frames(latent_frames: int) -> int:
    """Return the frame count of a clip with *latent_frames* latent frames: the inverse of :func:`h3_latent_frames`.

    Raises:
        ValueError: If *latent_frames* is not of the form ``5 * n + 2``.
    """
    if latent_frames < 2 or (latent_frames - 2) % H3_LATENTS_PER_CHUNK:
        raise ValueError(f"latent_frames must be of the form {H3_LATENTS_PER_CHUNK} * n + 2, got {latent_frames}.")
    return (latent_frames - 2) // H3_LATENTS_PER_CHUNK * H3_FRAMES_PER_CHUNK + H3_LATENTS_PER_CHUNK


def h3_refine_windows(latent_frames: int) -> tuple[tuple[int, int], ...]:
    """Return the ``(start, end)`` latent-frame windows the upscale refines, which overlap by 7 frames and are never shorter than 22.

    A clip of 22 latent frames (73 frames) or fewer is one window. A longer one gets windows of 22 latent frames
    starting every 15, and a last window pulled back to end at the last latent frame when the steps stop short of it,
    so every window is 22 long and the last may overlap its neighbour by more than 7. The plan depends on the clip
    length alone, so a seed gives the same clip on every card.
    """
    if latent_frames <= H3_REFINE_WINDOW_LATENT_FRAMES:
        return ((0, latent_frames),)
    step = H3_REFINE_WINDOW_LATENT_FRAMES - H3_REFINE_OVERLAP_LATENT_FRAMES
    starts = list(range(0, latent_frames - H3_REFINE_WINDOW_LATENT_FRAMES + 1, step))
    if starts[-1] + H3_REFINE_WINDOW_LATENT_FRAMES < latent_frames:
        starts.append(latent_frames - H3_REFINE_WINDOW_LATENT_FRAMES)
    return tuple((start, start + H3_REFINE_WINDOW_LATENT_FRAMES) for start in starts)


def h3_audio_latents(num_frames: int) -> int:
    """Return the number of audio latents that cover a *num_frames* clip."""
    return round(num_frames / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND)


def _cell_rows(width: int, height: int) -> int:
    """Return the rows of one latent frame of a ``width x height`` canvas.

    Raises:
        ValueError: If the canvas is not a positive multiple of 32 on both sides.
    """
    if width < 1 or height < 1 or width % H3_ROW_PIXELS or height % H3_ROW_PIXELS:
        raise ValueError(f"Canvas {width}x{height} must be a positive multiple of {H3_ROW_PIXELS}.")
    return (width // H3_ROW_PIXELS) * (height // H3_ROW_PIXELS)


def h3_video_rows(width: int, height: int, num_frames: int) -> int:
    """Return the video rows of a ``width x height x num_frames`` clip.

    Raises:
        ValueError: If the canvas is not a multiple of 32 or *num_frames* is not ``17 * n + 5``.
    """
    return h3_latent_frames(num_frames) * _cell_rows(width, height)


def h3_sequence_rows(width: int, height: int, num_frames: int, *, keyframes: int = 0, tail_frames: int = 0) -> int:
    """Return the length of the packed sequence: text, audio and video rows, plus the rows of conditioning keyframes.

    Each keyframe adds one latent frame of rows. An extend tail adds none: its frames and audio are the first frames
    of the *num_frames* window, so *tail_frames* is only checked, not counted.

    Raises:
        ValueError: If the canvas or frame count is invalid, *keyframes* is negative, or *tail_frames* is neither 0 nor
            a valid clip length within *num_frames*.
    """
    if keyframes < 0:
        raise ValueError(f"keyframes must not be negative, got {keyframes}.")
    if tail_frames:
        h3_latent_frames(tail_frames)
        if tail_frames > num_frames:
            raise ValueError(f"tail_frames ({tail_frames}) must not exceed num_frames ({num_frames}).")
    audio_rows = h3_audio_latents(num_frames) * H3_AUDIO_CHANNELS
    keyframe_rows = keyframes * _cell_rows(width, height)
    return H3_TEXT_TOKENS + audio_rows + h3_video_rows(width, height, num_frames) + keyframe_rows


def _mlx_rows(width: int, height: int, frames: int, mode: str) -> int:
    """Return the sequence rows of a macOS clip; the image modes carry one keyframe (an extend continues from the last frame).

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    if mode != "t2v" and mode not in _MLX_IMAGE_MODES:
        raise ValueError(f"mode must be one of t2v, {', '.join(_MLX_IMAGE_MODES)}, got {mode!r}.")
    return h3_sequence_rows(width, height, frames, keyframes=1 if mode in _MLX_IMAGE_MODES else 0)


def estimate_h3_mlx_clip_memory(width: int, height: int, frames: int, *, mode: str = "t2v") -> int:
    """Estimate the peak unified memory, in bytes, of an H3 clip on macOS.

    The denoise phase peaks at 15.0 GiB for the reference clip (12,946 rows) and grows by 160 KiB per further row.
    Text-to-video streams the text encoder, so that is its peak. The image modes run in stages, each holding one
    component: the conditioning phase (keyframe encode, vision tower and prompt encode) peaks at 14.2 GiB, and the
    denoise phase also holds the keyframe rows. The peak is the larger of the two.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    rows = _mlx_rows(width, height, frames, mode)
    denoise = _MLX_REFERENCE_BYTES + _MLX_BYTES_PER_ROW * (rows - _MLX_REFERENCE_ROWS)
    return int(max(denoise, _MLX_LOAD_PHASE_BYTES) if mode in _MLX_IMAGE_MODES else denoise)


def estimate_h3_mlx_seconds(width: int, height: int, frames: int, *, forwards: int, mode: str = "t2v") -> int:
    """Estimate the generation time, in seconds, of an H3 clip on macOS (an estimate from an M4 Pro, not a measurement of this machine).

    A fixed 240 s plus, per forward, a term linear and a term quadratic in the sequence rows, calibrated to the
    fork's 23 minutes for 768x448 at 124 frames and 8 forwards. It excludes the one-time weight preparation.

    Raises:
        ValueError: If *forwards* is below 1, *mode* is unknown, or the canvas or frame count is invalid.
    """
    if forwards < 1:
        raise ValueError(f"forwards must be at least 1, got {forwards}.")
    rows = _mlx_rows(width, height, frames, mode)
    return round(_MLX_FIXED_SECONDS + forwards * (_MLX_LINEAR_SECONDS_PER_ROW * rows + _MLX_QUADRATIC_SECONDS_PER_ROW2 * rows**2))


def h3_mlx_clip_fit(width: int, height: int, frames: int, budget: MemoryBudget | None, *, forwards: int, mode: str = "t2v") -> dict[str, Any] | None:
    """Check an H3 clip against a Mac's unified-memory budget; callers warn on ``tight`` or ``too_large`` and never block.

    Returns:
        ``{"status", "required_gb", "budget_gb", "expected_seconds"}`` (the time is an estimate), or ``None`` without a
        budget or on a discrete GPU, whose memory is estimated separately.

    Raises:
        ValueError: If *mode* is unknown, *forwards* is below 1, or the canvas or frame count is invalid.
    """
    if budget is None or not budget.gpu_bytes or budget.system_bytes is not None:
        return None
    required = estimate_h3_mlx_clip_memory(width, height, frames, mode=mode)
    return {
        "status": classify_memory_fit(required, budget.gpu_bytes),
        "required_gb": round(required / GIB, 1),
        "budget_gb": round(budget.gpu_bytes / GIB, 1),
        "expected_seconds": estimate_h3_mlx_seconds(width, height, frames, forwards=forwards, mode=mode),
    }


def h3_extend_window_frames(extend_frames: int) -> int:
    """Return the frames H3 generates for an extension: its carried tail plus the *extend_frames* new ones."""
    return H3_TAIL_FRAMES + extend_frames


def h3_lora_bytes(paths: Iterable[str | Path]) -> int:
    """Return the size in bytes of the LoRA files at *paths* (a folder counts its files); a path that is not there counts nothing."""
    total = 0
    for path in map(Path, paths):
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(child.stat().st_size for child in path.iterdir() if child.is_file())
    return total


def _cuda_rows(width: int, height: int, frames: int, mode: str) -> int:
    """Return the sequence rows of a CUDA clip; only an image-to-video clip carries a keyframe (an extension's tail adds no rows).

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    if mode not in _CUDA_MODES:
        raise ValueError(f"mode must be one of {', '.join(_CUDA_MODES)}, got {mode!r}.")
    return h3_sequence_rows(width, height, frames, keyframes=1 if mode == "i2v" else 0)


def _stage_one_size(width: int, height: int, *, upscale: bool) -> tuple[int, int]:
    """Return the canvas the transformer and the decode work at: half the target size when it is upscaled."""
    return (width // 2, height // 2) if upscale else (width, height)


def h3_cuda_system_bytes(*, loras_bytes: int = 0, frame_bytes: int = 0, vision: bool = False) -> int:
    """Return the peak system memory of an H3 clip on CUDA: the base, the keyframe vision tower, the user LoRAs and the decoded frames.

    Args:
        loras_bytes: Total size of the user LoRA files.
        frame_bytes: The decoded frames of the clip.
        vision: Whether keyframes are encoded, which keeps the text encoder's vision tower resident.
    """
    return _CUDA_SYSTEM_BASE_BYTES + (_CUDA_VISION_SYSTEM_BYTES if vision else 0) + loras_bytes + frame_bytes


def _cuda_upsampler_bytes(width: int, height: int, frames: int) -> float:
    """Return the GPU peak of the latent upscaler on a ``width x height x frames`` output: linear in the full-size latent volume."""
    cells = (width // H3_SPATIAL_RATIO) * (height // H3_SPATIAL_RATIO) * h3_latent_frames(frames)
    return _CUDA_UPSAMPLER_BASE_BYTES + _CUDA_UPSAMPLER_BYTES_PER_CELL * cells


def _refine_windows_of(frames: int) -> tuple[int, int]:
    """Return ``(window frames, windows)`` of the refine of a *frames*-frame clip: the pixel frames of its largest window and how many windows there are.

    Raises:
        ValueError: If *frames* is not ``17 * n + 5``.
    """
    windows = h3_refine_windows(h3_latent_frames(frames))
    return h3_frames_of_latent_frames(max(end - start for start, end in windows)), len(windows)


def estimate_h3_cuda_clip_memory(
    width: int,
    height: int,
    frames: int,
    *,
    mode: str = "t2v",
    loras_bytes: int = 0,
    upscale: bool = False,
    keep_transformer: bool = False,
) -> CudaMemoryEstimate:
    """Estimate the peak GPU and system memory of an H3 clip on a CUDA machine.

    The stages never overlap, so the GPU peak is the largest of the text encoder (with its vision tower for a
    keyframe), the streamed transformer at the sequence's rows (user LoRAs add 2.7 times their size), the video
    decode at the output size, and, when upscaling, the latent upscaler and the transformer again for the refine,
    plus 1 GiB the desktop takes. An upscaled clip renders stage 1 at half size; the refine works in windows of at most
    73 frames at the full size, so its rows are those of the largest window. The latent upscaler grows linearly with
    the full-size latent volume. The full-size keyframe encode of an image mode stays below the decode, so it is not a
    stage of its own. System memory is the base, the vision tower, the LoRAs and the decoded output frames.

    With *keep_transformer* the upscale keeps the transformer loaded from stage 1 to the refine, so the audio decode, the
    latent upscaler and the keyframe encode (an image-to-video clip) run beside its resident part, and each of their peaks
    adds it.

    Args:
        width: Output width in pixels (the enlarged size when *upscale*).
        height: Output height in pixels.
        frames: Output frame count (an extension's window, with its carried tail).
        mode: ``"t2v"``, ``"i2v"`` or ``"extend"``.
        loras_bytes: Total size of the user LoRA files.
        upscale: Whether the clip is enlarged 2x with the latent upscaler.
        keep_transformer: Whether the upscale keeps the transformer loaded between its two passes (no review; see
            :func:`h3_upscale_keeps_transformer`). Ignored without *upscale*.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    work_width, work_height = _stage_one_size(width, height, upscale=upscale)
    rows = _cuda_rows(work_width, work_height, frames, mode)
    lora = _CUDA_DIT_LORA_BYTES_PER_BYTE * loras_bytes
    text_encoder = _CUDA_TEXT_ENCODER_BYTES + (_CUDA_VISION_BYTES if mode == "i2v" else 0)
    transformer = _CUDA_DIT_REFERENCE_BYTES + _CUDA_DIT_BYTES_PER_ROW * (rows - _CUDA_REFERENCE_ROWS) + lora
    decode = _CUDA_DECODE_BASE_BYTES + _CUDA_DECODE_BYTES_PER_PIXEL * width * height
    stages = [text_encoder, transformer, decode]
    if upscale:
        stages.append(_cuda_upsampler_bytes(width, height, frames))
        window_frames, _windows = _refine_windows_of(frames)
        refine_rows = _cuda_rows(width, height, window_frames, mode)
        stages.append(_CUDA_DIT_REFERENCE_BYTES + _CUDA_DIT_BYTES_PER_ROW * (refine_rows - _CUDA_REFERENCE_ROWS) + lora)
        if keep_transformer:
            resident = _CUDA_RESIDENT_TRANSFORMER_BYTES  # user LoRAs sit in the streamed blocks (CPU) and the token refiner, which the figure covers
            stages.append(_CUDA_AUDIO_DECODE_BYTES + resident)
            stages.append(_cuda_upsampler_bytes(width, height, frames) + resident)
            if mode == "i2v":
                stages.append(_CUDA_ENCODE_BASE_BYTES + _CUDA_ENCODE_BYTES_PER_PIXEL * width * height + resident)
    return CudaMemoryEstimate(
        gpu_bytes=int(max(stages) + _CUDA_GPU_HEADROOM_BYTES),
        system_bytes=h3_cuda_system_bytes(loras_bytes=loras_bytes, frame_bytes=frames * width * height * 3, vision=mode == "i2v"),
    )


def _cuda_forward_seconds(rows: int) -> float:
    """Return the seconds of one transformer forward over *rows* rows: a linear and a quadratic (attention) term."""
    return _CUDA_C1 * rows + _CUDA_C2 * rows**2


def h3_expected_seconds(
    width: int,
    height: int,
    frames: int,
    *,
    forwards: int,
    mode: str = "t2v",
    loras: bool = False,
    upscale: bool = False,
    platform: str = "cuda",
    keep_transformer: bool = False,
) -> int:
    """Estimate the generation time, in seconds, of an H3 clip; it excludes the one-time weight preparation.

    On CUDA (an RTX 3080 calibration): a fixed 116 s, the decode and export at the output size, one linear-plus-quadratic
    term per forward (8% more with a user LoRA) and 30 s to encode a keyframe or an extension tail. When upscaling, the
    *forwards* run at half size, and the latent upscale, the transformer's second load and the refine's forwards (two per
    window, at the rows of the largest window) are added, with a second keyframe encode at the full size for an image mode.
    An upscale that keeps the transformer loaded between its passes (*keep_transformer*) has no second load.
    On ``"mlx"`` it is :func:`estimate_h3_mlx_seconds`.

    Args:
        width: Output width in pixels (the enlarged size when *upscale*).
        height: Output height in pixels.
        frames: Output frame count (an extension's window).
        forwards: Transformer forwards of the run (of stage 1 when upscaling).
        mode: ``"t2v"``, ``"i2v"`` or ``"extend"``.
        loras: Whether a user LoRA is applied.
        upscale: Whether the clip is enlarged 2x with the latent upscaler (CUDA only).
        platform: ``"cuda"`` or ``"mlx"``.
        keep_transformer: Whether the upscale loads the transformer once for both passes (see :func:`h3_upscale_keeps_transformer`).

    Raises:
        ValueError: If *forwards* is below 1, *platform* or *mode* is unknown, or the canvas or frame count is invalid.
    """
    if forwards < 1:
        raise ValueError(f"forwards must be at least 1, got {forwards}.")
    if platform == "mlx":
        return estimate_h3_mlx_seconds(width, height, frames, forwards=forwards, mode=mode)
    if platform != "cuda":
        raise ValueError(f"platform must be cuda or mlx, got {platform!r}.")
    work_width, work_height = _stage_one_size(width, height, upscale=upscale)
    rows = _cuda_rows(work_width, work_height, frames, mode)
    lora_factor = _CUDA_LORA_FORWARD_FACTOR if loras else 1
    seconds = _CUDA_FIXED_SECONDS + _CUDA_DECODE_BASE_SECONDS + _CUDA_DECODE_SECONDS_PER_PIXEL_FRAME * width * height * frames + _CUDA_EXPORT_SECONDS * frames / _CUDA_REFERENCE_FRAMES
    seconds += forwards * _cuda_forward_seconds(rows) * lora_factor
    if mode != "t2v":
        seconds += _CUDA_IMAGE_ENCODE_SECONDS * (2 if upscale else 1)  # an upscale encodes the images again at the full size
    if upscale:
        window_frames, windows = _refine_windows_of(frames)
        refine_forwards = H3_REFINE_FORWARDS * windows
        seconds += (
            _CUDA_LATENT_UPSCALE_SECONDS
            + (0 if keep_transformer else _CUDA_REFINE_LOAD_SECONDS)
            + refine_forwards * _cuda_forward_seconds(_cuda_rows(width, height, window_frames, mode)) * lora_factor
        )
    return round(seconds)


def h3_upscale_keeps_transformer(width: int, height: int, frames: int, gpu_bytes: int, *, mode: str = "t2v", loras_bytes: int = 0) -> bool:
    """Return whether an upscale without review can keep the transformer loaded between its two passes on a card of *gpu_bytes*.

    The single load skips the second load of the stored copy, but the transformer's resident part then sits beside the audio
    decode, the latent upscaler and the keyframe encode. It is used where the estimate of that flow stays within the share of the
    card a clip may use (the ``fits`` line of :func:`~zvisiongenerator.utils.model_memory.classify_cuda_memory_fit`); a clip
    that would not fit loads twice, as before. The backend and the estimates both ask this, so they agree.

    Args:
        width: Output width in pixels (the enlarged size).
        height: Output height in pixels.
        frames: Output frame count.
        gpu_bytes: The card's memory.
        mode: ``"t2v"`` or ``"i2v"``.
        loras_bytes: Total size of the user LoRA files.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    estimate = estimate_h3_cuda_clip_memory(width, height, frames, mode=mode, loras_bytes=loras_bytes, upscale=True, keep_transformer=True)
    # Only the GPU decides here. System memory does not differ between the flows: the kept transformer loads where the first
    # load did (after the encoders have let go), and the stages it stays through hold less than the denoise, which the base
    # system figure already covers.
    return classify_cuda_memory_fit(CudaMemoryEstimate(estimate.gpu_bytes, 0), MemoryBudget(gpu_bytes, system_bytes=0)) == "fits"


def h3_clip_fit(
    width: int,
    height: int,
    frames: int,
    budget: MemoryBudget | None,
    *,
    forwards: int,
    mode: str = "t2v",
    loras_bytes: int = 0,
    upscale: bool = False,
    review: bool = False,
) -> dict[str, Any] | None:
    """Check an H3 clip against a discrete GPU's budget; callers warn on ``tight`` or ``too_large`` and the backend refuses only what cannot fit system memory.

    An upscale without *review* keeps the transformer loaded between its passes where the card holds that
    (:func:`h3_upscale_keeps_transformer`), which skips the second load; with review it loads twice, as the half-size decode needs the memory.

    Returns:
        ``{"status", "required_gb", "budget_gb", "kind": "discrete", "resource", "family": "h3", "expected_seconds"}`` (the time is an
        estimate), or ``None`` without a budget or on unified memory, which :func:`h3_mlx_clip_fit` estimates.

    Raises:
        ValueError: If *mode* is unknown, *forwards* is below 1, or the canvas or frame count is invalid.
    """
    if budget is None or not budget.gpu_bytes or budget.system_bytes is None:
        return None
    keep = upscale and not review and h3_upscale_keeps_transformer(width, height, frames, budget.gpu_bytes, mode=mode, loras_bytes=loras_bytes)
    estimate = estimate_h3_cuda_clip_memory(width, height, frames, mode=mode, loras_bytes=loras_bytes, upscale=upscale, keep_transformer=keep)
    seconds = h3_expected_seconds(width, height, frames, forwards=forwards, mode=mode, loras=loras_bytes > 0, upscale=upscale, keep_transformer=keep)
    return {**discrete_clip_fit(estimate, budget), "family": "h3", "expected_seconds": seconds}
