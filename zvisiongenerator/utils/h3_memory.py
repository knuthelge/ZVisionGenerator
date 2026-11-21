"""Row and frame arithmetic of the MiniMax H3 transformer sequence, and the memory and time estimates built on it.

H3 sees one packed sequence of text rows, audio rows and video rows. The counts follow from the clip size alone, so the
geometry of a canvas, the sampler and the estimates all share these pure functions. They mirror diffusers'
``video_latent_num_frames`` and ``audio_latent_num_frames`` (checked by a contract test against the pinned release).
The estimates give an H3 clip's unified memory on MLX, its GPU and system memory on CUDA, and its expected generation time.
"""

from __future__ import annotations

from collections.abc import Sequence

from zvisiongenerator.utils.h3_ref2va import h3_reference_size
from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS, FITS, grade
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.model_memory import GIB, CudaMemoryEstimate

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
    "H3_VISION_MERGE",
    "H3_VISION_PATCH",
    "estimate_h3_cuda_clip_memory",
    "estimate_h3_mlx_clip_memory",
    "estimate_h3_mlx_seconds",
    "h3_audio_latents",
    "h3_cuda_system_bytes",
    "h3_expected_seconds",
    "h3_frames_of_latent_frames",
    "h3_latent_frames",
    "h3_nominal_references",
    "h3_reference_block_rows",
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
# Qwen3-VL's image processor (the H3 processor's ``preprocessor_config.json``) cuts an image into 16-pixel patches and merges 2x2 of
# them into one token, after resizing both sides to a multiple of 16 * 2.
H3_VISION_PATCH = 16
H3_VISION_MERGE = 2
_REFERENCE_FRAMING_TOKENS = 8  # a reference's text rows besides its image tokens: the 6-token "<Picture i>: " label and the vision start and end
# An Extend with references shows the encoder its tail as one video block: the tail's 22 frames at 24 fps are read at 2 fps (two frames),
# which merge into one block of one token per 32x32 pixels of the canvas, framed by the 6-token "<Video 1>: " label, a 6-token
# "<0.2 seconds>" stamp and the vision start and end.
_TAIL_FRAMING_TOKENS = 14
H3_TAIL_FRAMES = 22  # an extension carries 17 * 1 + 5 frames of its source, the smallest valid clip
# The upscale refines the enlarged latents with the last forwards of the 8-step schedule, in overlapping windows of latent frames.
# They are code constants, not config: there is no user control, and they are tied to the stored copy's fold tables.
H3_REFINE_SCHEDULE = 8  # the schedule the refine runs the tail of, whatever the stage-1 step count
H3_REFINE_FORWARDS = 2  # forwards of that schedule's tail
H3_REFINE_WINDOW_LATENT_FRAMES = 22  # 73 frames
H3_REFINE_OVERLAP_LATENT_FRAMES = 7

# macOS (MLX) estimates: the peaks are the process's peak footprint, and the times are the fork's own measurements.
_MLX_IMAGE_MODES = ("i2v", "extend")
# The denoise phase is a line over the sequence rows through two text-to-video clips (768x448 at 22 frames, 2,526 rows; 512x320 at 124 frames, 6,434 rows). measured: h3-s, h3-5s; longer sequences are extrapolated, not measured.
_MLX_PEAK_ROWS = 6_434  # 512x320 at 124 frames: text, audio and video rows
_MLX_PEAK_BYTES = int(12.03 * GIB)  # peak of a text-to-video run at those rows
_MLX_BYTES_PER_ROW = int(186.5 * 1024)  # growth of the denoise peak per sequence row
# The conditioning phase (reference encode, vision tower, prompt encode) is set by the 4-bit text encoder. measured with 2 references: h3-5s-refs. The image modes (i2v, extend) run the same phase: not measured there.
_MLX_LOAD_PHASE_BYTES = int(15.34 * GIB)
_MLX_REFERENCE_ROWS = 12_946  # 768x448 at 124 frames: text, audio and video rows, the clip the time estimate is anchored to
_MLX_FIXED_SECONDS = 240  # load, text encode, decode and mux
_MLX_REFERENCE_SECONDS = 23 * 60  # the fork's time at the reference clip and 8 forwards
_MLX_REFERENCE_FORWARDS = 8
_MLX_LINEAR_SHARE = 0.675  # share of a forward at the reference rows that grows linearly with rows (the rest is attention)
_MLX_FORWARD_SECONDS = (_MLX_REFERENCE_SECONDS - _MLX_FIXED_SECONDS) / _MLX_REFERENCE_FORWARDS
_MLX_LINEAR_SECONDS_PER_ROW = _MLX_LINEAR_SHARE * _MLX_FORWARD_SECONDS / _MLX_REFERENCE_ROWS
_MLX_QUADRATIC_SECONDS_PER_ROW2 = (1 - _MLX_LINEAR_SHARE) * _MLX_FORWARD_SECONDS / _MLX_REFERENCE_ROWS**2

# Windows and Linux (CUDA) estimates. The GPU figures are allocated peaks (the memory the tensors held), before the CUDA offset the verdict adds; the transformer's growth with the rows is measured with the feed-forward chunked
# (the preset's ``ffn_chunk_tokens``), see below.
_CUDA_MODES = ("t2v", "i2v", "extend")
_CUDA_REFERENCE_ROWS = 15_499  # 864x480 at 124 frames
_CUDA_TEXT_ENCODER_BYTES = int(2.10 * GIB)  # the 4-bit text encoder while the prompt is encoded
_CUDA_VISION_BYTES = int(1.11 * GIB)  # ... plus its vision tower when keyframes are encoded
# The text encoder's turn with references: its figure above plus this per reference, from the allocated peak of the 2-reference clip less the text encoder and the vision tower, per reference. measured at 2 references: h3-5s-refs;
# not measured at 1 (the same per-reference term) or with a keyframe.
_CUDA_REFERENCE_ENCODE_BYTES = int(1.62 * GIB)
# The streamed transformer's allocated-equivalent peak (the figure less the CUDA offset the verdict adds) is a least-squares line over the sequence rows, through the three clips that the transformer, not the decode, sets the peak of,
# with the feed-forward chunked: h3-10s-864 (30,070 rows), h3-5s-1280 (34,554) and h3-14s-864 (42,560). Each figure is the middle of the case's hold-probe bracket, less the offset; h3-10s-864 is not probed, so its stand-in is its allocated peak plus
# the offset (not measured). The three do not lie on one line, and the line reads below the video decode at the reference rows, so no measured clip there sets the peak at it. measured between 30,070 and 42,560 rows; smaller clips are extrapolated and their peak is the decode's or the text encoder's.
_CUDA_DIT_REFERENCE_BYTES = int(3.65 * GIB)
_CUDA_DIT_BYTES_PER_ROW = 150 * 1024  # growth of the figure per further row, with the feed-forward chunked
# A user LoRA adds to the transformer phase this many GPU bytes per byte of its file, the difference of the hold-probe brackets of h3-5s-1280-lora and h3-5s-1280. measured with one LoRA, at one size; the bracket's width allows 0 to 0.7.
# The LoRA does not lift a small clip's peak above the decode: h3-5s-864-lora peaks where h3-5s-864 does.
_CUDA_DIT_LORA_BYTES_PER_BYTE = 0.27
# The video VAE decode, weights and working set. The base is the middle of the allocated peaks of the two plain clips (a peak no other stage of the estimate reaches) less the size term below. measured as a base: h3-s, h3-5s; the per-pixel term is
# not measured as an allocated peak (the two clips' sizes differ by less than it predicts), and h3-5s-864 checks it.
_CUDA_DECODE_BASE_BYTES = int(4.53 * GIB)
_CUDA_DECODE_BYTES_PER_PIXEL = 9.48e-7 * GIB
# The app process's system memory (anonymous plus shared pages, the highest the two summed to in one sample) of a plain clip, less the decoded frames. The text encoder's anonymous memory and the pinned transformer blocks peak at different
# moments, so the sum of their separate peaks overstates it. measured at one size: h3-5s.
_CUDA_SYSTEM_BASE_BYTES = int(15.54 * GIB)
# A user LoRA's system memory: 2.9 times the file, because the run-time merge reads the factors and holds a float copy. measured with one LoRA: h3-5s-864-lora, h3-5s-1280-lora.
_CUDA_SYSTEM_LORA_BYTES_PER_BYTE = 2.9
# The vision tower's CPU side while references are encoded. measured with references: h3-5s-refs; not measured with a keyframe (the same term).
_CUDA_VISION_SYSTEM_BYTES = int(0.71 * GIB)
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
# The latent upscale: the network runs unchunked, so its GPU peak is linear in the full-size latent volume ``cells = (width / 16) * (height / 16) * latent_frames``. The line passes above the four peaks measured with the network alone
# (130,240 to 795,200 cells; h3-5s-1600-upscale is one of the sizes).
_CUDA_UPSAMPLER_BASE_BYTES = int(0.75 * GIB)
_CUDA_UPSAMPLER_BYTES_PER_CELL = 8.43e-6 * GIB
_CUDA_LATENT_UPSCALE_SECONDS = 10  # the network: 6.7 to 10.9 s measured for 37 to 72 latent frames
_CUDA_REFINE_LOAD_SECONDS = 62  # the second load of the stored transformer copy for the refine: 35 to 62 s measured
# An upscale without review keeps the transformer loaded from stage 1 to the refine (one load, not two), so it stays on the GPU while the waveform is decoded, the latents are enlarged and the keyframes are encoded. What stays is the part outside
# the streamed blocks plus the quantization state and fold tables of the blocks, rounded up. measured: h3-5s-1600-upscale.
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


def h3_reference_block_rows(width: int, height: int) -> int:
    """Return the sequence rows one reference of ``width x height`` pixels adds, as prepared (see :func:`~zvisiongenerator.utils.h3_ref2va.h3_reference_size`).

    Its latent is one frame at 1/16 of the size, packed like the clip's own video rows: one row per 32x32 pixels. Its
    picture also enters the text block, as one token per merged 2x2 of the processor's 16-pixel patches (a prepared
    reference already has sides of a multiple of the processor's 32, so it keeps its size), plus its label and the vision
    start and end tokens.

    Raises:
        ValueError: If a side is not a positive multiple of 32.
    """
    latent_rows = _cell_rows(width, height)
    factor = H3_VISION_PATCH * H3_VISION_MERGE
    return latent_rows + (width // factor) * (height // factor) + _REFERENCE_FRAMING_TOKENS


def h3_nominal_references(count: int, width: int, height: int) -> tuple[tuple[int, int], ...]:
    """Return the prepared sizes of *count* references that share the aspect ratio of a ``width x height`` canvas.

    A caller that knows only how many references a job holds, not their pictures, estimates with these.

    Raises:
        ValueError: If *count* is negative, or the canvas is not positive and there is a reference.
    """
    if count < 0:
        raise ValueError(f"references must not be negative, got {count}.")
    if count and (width < 1 or height < 1):
        raise ValueError(f"Canvas {width}x{height} must be positive.")
    return (h3_reference_size(width, height),) * count if count else ()


def h3_sequence_rows(width: int, height: int, num_frames: int, *, keyframes: int = 0, tail_frames: int = 0, references: Sequence[tuple[int, int]] = ()) -> int:
    """Return the length of the packed sequence: text, audio and video rows, plus the rows of conditioning keyframes and references.

    Each keyframe adds one latent frame of rows. An extend tail adds none to the video: its frames and audio are the first frames
    of the *num_frames* window, so *tail_frames* is only checked, not counted, except that a tail beside *references* is also shown
    to the text encoder, as one vision block of the canvas's size. Each of the *references*, given as its prepared
    ``(width, height)``, adds its latent rows and its tokens of the text block (:func:`h3_reference_block_rows`).

    Raises:
        ValueError: If the canvas or frame count is invalid, *keyframes* is negative, a reference side is not a multiple of 32,
            or *tail_frames* is neither 0 nor a valid clip length within *num_frames*.
    """
    if keyframes < 0:
        raise ValueError(f"keyframes must not be negative, got {keyframes}.")
    if tail_frames:
        h3_latent_frames(tail_frames)
        if tail_frames > num_frames:
            raise ValueError(f"tail_frames ({tail_frames}) must not exceed num_frames ({num_frames}).")
    audio_rows = h3_audio_latents(num_frames) * H3_AUDIO_CHANNELS
    keyframe_rows = keyframes * _cell_rows(width, height)
    reference_rows = sum(h3_reference_block_rows(*size) for size in references)
    tail_rows = _cell_rows(width, height) + _TAIL_FRAMING_TOKENS if tail_frames and references else 0
    return H3_TEXT_TOKENS + audio_rows + h3_video_rows(width, height, num_frames) + keyframe_rows + reference_rows + tail_rows


def _mlx_rows(width: int, height: int, frames: int, mode: str, references: Sequence[tuple[int, int]] = ()) -> int:
    """Return the sequence rows of a macOS clip; the image modes carry one keyframe (an extend continues from the last frame), and each reference its block.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    if mode != "t2v" and mode not in _MLX_IMAGE_MODES:
        raise ValueError(f"mode must be one of t2v, {', '.join(_MLX_IMAGE_MODES)}, got {mode!r}.")
    return h3_sequence_rows(width, height, frames, keyframes=1 if mode in _MLX_IMAGE_MODES else 0, references=references)


def estimate_h3_mlx_clip_memory(width: int, height: int, frames: int, *, mode: str = "t2v", references: Sequence[tuple[int, int]] = ()) -> int:
    """Estimate the peak unified memory, in bytes, of an H3 clip on macOS.

    The denoise phase peaks at 12.03 GiB for a clip of 6,434 rows (512x320, 124 frames) and grows by 186.5 KiB per row.
    Text-to-video streams the text encoder, so that is its peak. The image modes, and any clip with *references* (the
    prepared ``(width, height)`` of each; their rows count, see :func:`h3_sequence_rows`), run in stages, each holding one
    component: the conditioning phase (keyframe encode, vision tower and prompt encode) peaks at 15.34 GiB, and the
    denoise phase also holds the keyframe and reference rows. The peak is the larger of the two.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    rows = _mlx_rows(width, height, frames, mode, references)
    denoise = _MLX_PEAK_BYTES + _MLX_BYTES_PER_ROW * (rows - _MLX_PEAK_ROWS)
    return int(max(denoise, _MLX_LOAD_PHASE_BYTES) if mode in _MLX_IMAGE_MODES or references else denoise)


def estimate_h3_mlx_seconds(width: int, height: int, frames: int, *, forwards: int, mode: str = "t2v", references: Sequence[tuple[int, int]] = ()) -> int:
    """Estimate the generation time, in seconds, of an H3 clip on macOS (calibrated to the fork's own timing, not measured by the bench).

    A fixed 240 s plus, per forward, a term linear and a term quadratic in the sequence rows, calibrated to the
    fork's 23 minutes for 768x448 at 124 frames and 8 forwards, at the rows of the clip including its *references*. It excludes the one-time weight preparation.

    Raises:
        ValueError: If *forwards* is below 1, *mode* is unknown, or the canvas or frame count is invalid.
    """
    if forwards < 1:
        raise ValueError(f"forwards must be at least 1, got {forwards}.")
    rows = _mlx_rows(width, height, frames, mode, references)
    return round(_MLX_FIXED_SECONDS + forwards * (_MLX_LINEAR_SECONDS_PER_ROW * rows + _MLX_QUADRATIC_SECONDS_PER_ROW2 * rows**2))


def _cuda_rows(width: int, height: int, frames: int, mode: str, references: Sequence[tuple[int, int]] = ()) -> int:
    """Return the sequence rows of a CUDA clip; only an image-to-video clip carries a keyframe (an extension's tail adds no rows, but with references it enters the text block), and each reference its block.

    Raises:
        ValueError: If *mode* is unknown, or the canvas or frame count is invalid.
    """
    if mode not in _CUDA_MODES:
        raise ValueError(f"mode must be one of {', '.join(_CUDA_MODES)}, got {mode!r}.")
    return h3_sequence_rows(width, height, frames, keyframes=1 if mode == "i2v" else 0, tail_frames=H3_TAIL_FRAMES if mode == "extend" else 0, references=references)


def _cuda_vision(mode: str, references: Sequence[tuple[int, int]]) -> bool:
    """Return whether the text encoder runs with its vision tower: an image-to-video clip has a keyframe, and any clip with references shows them."""
    return mode == "i2v" or bool(references)


def _stage_one_size(width: int, height: int, *, upscale: bool) -> tuple[int, int]:
    """Return the canvas the transformer and the decode work at: half the target size when it is upscaled."""
    return (width // 2, height // 2) if upscale else (width, height)


def h3_cuda_system_bytes(*, loras_bytes: int = 0, frame_bytes: int = 0, vision: bool = False) -> int:
    """Return the peak system memory of an H3 clip on CUDA: the base, the vision tower (keyframes or references), the user LoRAs and the decoded frames.

    Args:
        loras_bytes: Total size of the user LoRA files.
        frame_bytes: The decoded frames of the clip.
        vision: Whether keyframes or references are encoded, which keeps the text encoder's vision tower resident.
    """
    return int(_CUDA_SYSTEM_BASE_BYTES + (_CUDA_VISION_SYSTEM_BYTES if vision else 0) + _CUDA_SYSTEM_LORA_BYTES_PER_BYTE * loras_bytes + frame_bytes)


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
    references: Sequence[tuple[int, int]] = (),
) -> CudaMemoryEstimate:
    """Estimate the peak GPU and system memory of an H3 clip on a CUDA machine, before the CUDA offset the verdict adds.

    The stages never overlap, so the GPU peak is the largest of the text encoder (with its vision tower for a
    keyframe), the streamed transformer at the sequence's rows (user LoRAs add 0.27 times their size), the video
    decode at the output size, and, when upscaling, the latent upscaler and the transformer again for the refine. An upscaled clip renders stage 1 at half size; the refine works in windows of at most
    73 frames at the full size, so its rows are those of the largest window. References keep their own size and are held
    in every pass, so their rows (:func:`h3_reference_block_rows`) are in each, and they load the vision tower in any mode. The latent upscaler grows linearly with
    the full-size latent volume. The full-size keyframe encode of an image mode stays below the decode, so it is not a
    stage of its own. System memory is the base, the vision tower, the LoRAs (2.9 times their size) and the decoded output frames.

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
        references: The prepared ``(width, height)`` of each reference image.

    Raises:
        ValueError: If *mode* is unknown, or the canvas, frame count or a reference size is invalid.
    """
    work_width, work_height = _stage_one_size(width, height, upscale=upscale)
    rows = _cuda_rows(work_width, work_height, frames, mode, references)
    lora = _CUDA_DIT_LORA_BYTES_PER_BYTE * loras_bytes
    vision = _cuda_vision(mode, references)
    text_encoder = _CUDA_TEXT_ENCODER_BYTES + (_CUDA_VISION_BYTES if vision else 0) + _CUDA_REFERENCE_ENCODE_BYTES * len(references)
    transformer = _CUDA_DIT_REFERENCE_BYTES + _CUDA_DIT_BYTES_PER_ROW * (rows - _CUDA_REFERENCE_ROWS) + lora
    decode = _CUDA_DECODE_BASE_BYTES + _CUDA_DECODE_BYTES_PER_PIXEL * width * height
    stages = [text_encoder, transformer, decode]
    if upscale:
        stages.append(_cuda_upsampler_bytes(width, height, frames))
        window_frames, _windows = _refine_windows_of(frames)
        refine_rows = _cuda_rows(width, height, window_frames, mode, references)
        stages.append(_CUDA_DIT_REFERENCE_BYTES + _CUDA_DIT_BYTES_PER_ROW * (refine_rows - _CUDA_REFERENCE_ROWS) + lora)
        if keep_transformer:
            resident = _CUDA_RESIDENT_TRANSFORMER_BYTES  # user LoRAs sit in the streamed blocks (CPU) and the token refiner, which the figure covers
            stages.append(_CUDA_AUDIO_DECODE_BYTES + resident)
            stages.append(_cuda_upsampler_bytes(width, height, frames) + resident)
            if mode == "i2v":
                stages.append(_CUDA_ENCODE_BASE_BYTES + _CUDA_ENCODE_BYTES_PER_PIXEL * width * height + resident)
    return CudaMemoryEstimate(
        gpu_bytes=int(max(stages)),
        system_bytes=h3_cuda_system_bytes(loras_bytes=loras_bytes, frame_bytes=frames * width * height * 3, vision=vision),
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
    references: Sequence[tuple[int, int]] = (),
) -> int:
    """Estimate the generation time, in seconds, of an H3 clip; it excludes the one-time weight preparation.

    On CUDA: a fixed 116 s, the decode and export at the output size, one linear-plus-quadratic
    term per forward (8% more with a user LoRA) and 30 s to encode a keyframe or an extension tail. When upscaling, the
    *forwards* run at half size, and the latent upscale, the transformer's second load and the refine's forwards (two per
    window, at the rows of the largest window) are added, with a second keyframe encode at the full size for an image mode.
    An upscale that keeps the transformer loaded between its passes (*keep_transformer*) has no second load. The rows of the
    *references* are in every forward, and a text-to-video clip with references also pays the encode once (they are encoded in stage 1 only).
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
        references: The prepared ``(width, height)`` of each reference image.

    Raises:
        ValueError: If *forwards* is below 1, *platform* or *mode* is unknown, or the canvas, frame count or a reference size is invalid.
    """
    if forwards < 1:
        raise ValueError(f"forwards must be at least 1, got {forwards}.")
    if platform == "mlx":
        return estimate_h3_mlx_seconds(width, height, frames, forwards=forwards, mode=mode, references=references)
    if platform != "cuda":
        raise ValueError(f"platform must be cuda or mlx, got {platform!r}.")
    work_width, work_height = _stage_one_size(width, height, upscale=upscale)
    rows = _cuda_rows(work_width, work_height, frames, mode, references)
    lora_factor = _CUDA_LORA_FORWARD_FACTOR if loras else 1
    seconds = _CUDA_FIXED_SECONDS + _CUDA_DECODE_BASE_SECONDS + _CUDA_DECODE_SECONDS_PER_PIXEL_FRAME * width * height * frames + _CUDA_EXPORT_SECONDS * frames / _CUDA_REFERENCE_FRAMES
    seconds += forwards * _cuda_forward_seconds(rows) * lora_factor
    if mode != "t2v":
        seconds += _CUDA_IMAGE_ENCODE_SECONDS * (2 if upscale else 1)  # an upscale encodes the images again at the full size
    elif references:
        seconds += _CUDA_IMAGE_ENCODE_SECONDS  # the references are encoded once, in stage 1
    if upscale:
        window_frames, windows = _refine_windows_of(frames)
        refine_forwards = H3_REFINE_FORWARDS * windows
        seconds += (
            _CUDA_LATENT_UPSCALE_SECONDS
            + (0 if keep_transformer else _CUDA_REFINE_LOAD_SECONDS)
            + refine_forwards * _cuda_forward_seconds(_cuda_rows(width, height, window_frames, mode, references)) * lora_factor
        )
    return round(seconds)


def h3_upscale_keeps_transformer(width: int, height: int, frames: int, gpu_bytes: int, *, mode: str = "t2v", loras_bytes: int = 0, references: Sequence[tuple[int, int]] = ()) -> bool:
    """Return whether an upscale without review can keep the transformer loaded between its two passes on a card of *gpu_bytes*.

    The single load skips the second load of the stored copy, but the transformer's resident part then sits beside the audio
    decode, the latent upscaler and the keyframe encode. It is used where the estimate of that flow, with the CUDA offset, stays on
    the ``fits`` line of :func:`~zvisiongenerator.utils.memory_grade.grade`; a clip that would not fit loads
    twice. The backend and the estimates both ask this, so they agree.

    Args:
        width: Output width in pixels (the enlarged size).
        height: Output height in pixels.
        frames: Output frame count.
        gpu_bytes: The card's memory.
        mode: ``"t2v"`` or ``"i2v"``.
        loras_bytes: Total size of the user LoRA files.
        references: The prepared ``(width, height)`` of each reference image.

    Raises:
        ValueError: If *mode* is unknown, or the canvas, frame count or a reference size is invalid.
    """
    estimate = estimate_h3_cuda_clip_memory(width, height, frames, mode=mode, loras_bytes=loras_bytes, upscale=True, keep_transformer=True, references=references)
    # Only the GPU decides here. System memory does not differ between the flows: the kept transformer loads where the first
    # load did (after the encoders have let go), and the stages it stays through hold less than the denoise, which the base
    # system figure already covers.
    return grade(estimate.gpu_bytes + CUDA_OFFSET_BYTES, gpu_bytes, CUDA_GPU_RATIOS) == FITS
