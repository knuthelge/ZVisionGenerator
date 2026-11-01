"""Staged MiniMax H3 generation on macOS: from keyframe images (image-to-video, first and last frame, Extend) and from an audio track.

The vendored fork keeps the text encoder, its vision tower, both VAEs and the transformer resident for an image run, which
peaks near 29 GiB. Unified memory makes staging cheap: each phase here loads one component, uses it, drops it and returns
host (numpy float32) copies, so the peak is about the largest single component and a re-read from SSD is the only cost.
``release_memory()`` runs after every phase.

The fork stays untouched; its pieces are composed here. The fork methods that read only a few ``self`` attributes get a
``SimpleNamespace`` as ``self``, the vision tower is mlx-vlm's, and the image processor is transformers' PIL one (the fast one needs torchvision,
which macOS does not install). The denoise loop is the one reimplemented part: it mirrors ``MiniMaxH3Pipeline.__call__``
with the transformer streamed from disk.

A user's audio track is held, not generated: its window is encoded by the audio VAE into clean audio rows that replace the
audio noise, and every audio row of the clip is a conditioning row. The model then makes the picture only, and the clip's
own sound is silence that a later stage replaces with the track. A track without images is text to video on this path.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.mac.memory import release_memory
from zvisiongenerator.utils.h3_memory import H3_FPS

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from zvisiongenerator.utils.h3_preset import H3Schedule

__all__ = ["TRACK_SAMPLE_RATE", "generate_conditioned", "pack_track_latents"]

# The audio VAE's sample rate, which a track window is read at (the release's ``audio_vae`` metadata; a contract test pins the fork's default).
TRACK_SAMPLE_RATE = 32000

_VISION_PREFIX = "model.visual."
_INDEX_NAME = "model.safetensors.index.json"
_GIB = 2**30


@dataclass(frozen=True)
class _Canvas:
    """The geometry of one clip: frame and latent sizes with the configs they derive from."""

    height: int
    width: int
    frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int
    video_config: Any
    audio_config: Any
    dit_config: Any


def generate_conditioned(
    *,
    fl2va_dir: Path,
    transformer_dir: Path,
    text_encoder_dir: Path,
    turbo_lora_path: Path | None,
    turbo_lora_alpha: float | None,
    turbo_lora_scale: float,
    schedule: H3Schedule,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    images: Sequence[Image.Image],
    anchors: Sequence[str],
    step_callback: Callable[[int, int], None] | None,
    verbose: bool,
    track_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a clip conditioned on keyframe images and, optionally, an audio track, one component in memory at a time.

    Args:
        fl2va_dir: The release's ``FL2VA`` folder (VAEs, tokenizer, processor).
        transformer_dir: The MLX INT8 transformer folder.
        text_encoder_dir: The 4-bit text-encoder copy, which also holds the vision tower.
        turbo_lora_path: The turbo or combined adapter, or ``None`` for a converted folder that holds its own merge.
        turbo_lora_alpha: The adapter's alpha, or ``None`` to take the file's.
        turbo_lora_scale: The adapter's strength.
        schedule: The sigma schedule (step count and the two shifts).
        prompt: Text prompt.
        width: Video width, a multiple of 32.
        height: Video height, a multiple of 32.
        frames: Frames at 24 fps; rounded up to the model's ``17 * n + 5`` grid.
        seed: Request seed; the keyframe encode uses the fork's fixed seed before it.
        images: The keyframes, already sized to ``width x height``; none with a *track_audio* (text to video).
        anchors: ``"first"`` or ``"last"`` per image.
        step_callback: Called as ``step_callback(done, total)`` after every denoise step; it may raise to cancel.
        verbose: Print each phase's MLX peak and the denoise steps.
        track_audio: The sound of the whole clip to hold, stereo float32 ``(2, samples)`` at :data:`TRACK_SAMPLE_RATE`; it
            is encoded to clean audio rows and the model generates no sound.

    Returns:
        ``(video, audio, sample_rate)``: RGB uint8 frames ``(frames, height, width, 3)`` and float32 stereo audio ``(2, samples)``.
        With a *track_audio* the audio is silence as long as the video (the caller replaces it with the track).

    Raises:
        ValueError: If there are neither images nor a track, the images and anchors differ in count, or the size is not a multiple of 32.
    """
    import numpy as np
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.load import read_audio_vae_config, read_video_vae_config
    from minimax_h3_mlx.packing import align_num_frames, audio_latent_num_frames, video_latent_num_frames

    if not images and track_audio is None:
        raise ValueError("Staged MiniMax H3 generation needs at least one keyframe image or an audio track.")
    if len(images) != len(anchors):
        raise ValueError(f"Each keyframe needs an anchor: {len(images)} image(s), {len(anchors)} anchor(s).")
    if height % 32 or width % 32:
        raise ValueError(f"`height` and `width` must be multiples of 32, got {height}x{width}.")
    video_config = read_video_vae_config(fl2va_dir / "video_vae")
    audio_config = read_audio_vae_config(fl2va_dir / "audio_vae")
    num_frames = align_num_frames(frames)
    ratio = video_config.spatial_compression_ratio
    canvas = _Canvas(
        height=height,
        width=width,
        frames=num_frames,
        latent_frames=video_latent_num_frames(num_frames),
        latent_height=height // ratio,
        latent_width=width // ratio,
        audio_latents=audio_latent_num_frames(num_frames),
        video_config=video_config,
        audio_config=audio_config,
        dit_config=DiTConfig.from_json(transformer_dir / "config.json"),
    )
    images = list(images)

    if images:
        condition_rows = _phase("keyframes", verbose, _keyframe_rows, fl2va_dir, canvas, images)
        vision_rows, deepstack_rows = _phase("vision", verbose, _vision_rows, text_encoder_dir, fl2va_dir / "processor", images)
        prompt_rows, text_tags = _phase("prompt", verbose, _prompt_rows, text_encoder_dir, fl2va_dir, prompt, images, vision_rows, deepstack_rows)
    else:
        condition_rows = np.zeros((0, video_config.latent_channels * math.prod(canvas.dit_config.patch_size)), dtype=np.float32)
        prompt_rows, text_tags = _phase("prompt", verbose, _text_rows, text_encoder_dir, fl2va_dir, prompt)
    track_rows = _phase("track audio", verbose, _track_rows, fl2va_dir, canvas, track_audio) if track_audio is not None else None
    video_rows, audio_rows = _phase(
        "denoise",
        verbose,
        _denoise,
        transformer_dir,
        canvas=canvas,
        turbo=(turbo_lora_path, turbo_lora_alpha, turbo_lora_scale),
        schedule=schedule,
        prompt_rows=prompt_rows,
        text_tags=text_tags,
        condition_rows=condition_rows,
        anchors=tuple(anchors),
        track_rows=track_rows,
        seed=seed,
        step_callback=step_callback,
        verbose=verbose,
    )
    video = _phase("video decode", verbose, _decode_video, fl2va_dir, canvas, video_rows)
    rate = int(audio_config.sampling_rate)
    if track_rows is not None:
        # The track's own sound is what the clip is given afterwards; the held rows are not decoded.
        return video, np.zeros((2, round(len(video) / H3_FPS * rate)), dtype=np.float32), rate
    audio = _phase("audio decode", verbose, _decode_audio, fl2va_dir, canvas, audio_rows)
    return video, audio, rate


def _phase(name: str, verbose: bool, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run one phase, report its MLX peak, and release what it left behind.

    The phase's result holds host arrays only, so nothing it loaded outlives the call.
    """
    import mlx.core as mx

    mx.reset_peak_memory()
    result = function(*args, **kwargs)
    if verbose:
        print(f"  {name}: MLX peak {mx.get_peak_memory() / _GIB:.1f} GiB", flush=True)
    release_memory()
    return result


def _keyframe_rows(fl2va_dir: Path, canvas: _Canvas, images: list[Image.Image]) -> np.ndarray:
    """Return the packed conditioning rows of the keyframes (float32), encoded by the video VAE."""
    import numpy as np
    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    host = SimpleNamespace(video_vae=load_video_vae(fl2va_dir / "video_vae"), _video_config=canvas.video_config, _dit_config=canvas.dit_config)
    return np.array(MiniMaxH3Pipeline._encode_keyframes(host, images, canvas.height, canvas.width))


def _image_processor(processor_dir: Path) -> Any:
    """Return the PIL Qwen2-VL image processor (the default fast one needs torchvision, which macOS does not have)."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(str(processor_dir), backend="pil")


def _vision_rows(text_encoder_dir: Path, processor_dir: Path, images: list[Image.Image]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the vision tower's merged rows and deepstack features for *images* (float32)."""
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    processed = _image_processor(processor_dir)(images=images, return_tensors="np")
    config = json.loads((text_encoder_dir / "config.json").read_text(encoding="utf-8"))
    vision = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    weight_map = json.loads((text_encoder_dir / _INDEX_NAME).read_text(encoding="utf-8"))["weight_map"]
    tensors = {}
    for shard in sorted({shard for key, shard in weight_map.items() if key.startswith(_VISION_PREFIX)}):
        tensors.update({key.removeprefix(_VISION_PREFIX): tensor for key, tensor in mx.load(str(text_encoder_dir / shard)).items() if key.startswith(_VISION_PREFIX)})
    # The copy keeps the checkpoint's PyTorch conv layout; sanitize moves the patch embedding to MLX's.
    vision.load_weights(list(vision.sanitize(tensors).items()), strict=True)
    pixels = mx.array(np.asarray(processed["pixel_values"])).astype(mx.bfloat16)
    hidden, deepstack = vision(pixels, mx.array(np.asarray(processed["image_grid_thw"]).astype(np.int32)))
    return np.array(hidden.astype(mx.float32)), [np.array(feature.astype(mx.float32)) for feature in deepstack]


def _prompt_rows(text_encoder_dir: Path, fl2va_dir: Path, prompt: str, images: list[Image.Image], vision_rows: np.ndarray, deepstack_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the text encoder's hidden rows for the request with the vision rows in its image positions, and the row tags.

    This is the fork's ``encode`` with its image-token merge corrected: the vision rows go into the image-token positions
    instead of being broadcast against the whole sequence.

    Raises:
        ValueError: If the request holds a different number of image tokens than the vision tower produced rows.
    """
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
    from mlx_vlm.models.qwen3_vl.language import LanguageModel

    encoder = MiniMaxH3TextEncoder(text_encoder_dir, load_vision=False, stream_layers=False, tokenizer_dir=fl2va_dir / "tokenizer", processor_dir=fl2va_dir / "processor")
    encoder._processor = SimpleNamespace(image_processor=_image_processor(fl2va_dir / "processor"))
    input_ids, token_tags, vision_inputs = encoder.build_request(prompt, images)
    positions = np.flatnonzero(np.array(input_ids)[0] == encoder.image_token_id)
    if len(positions) != len(vision_rows):
        raise ValueError(f"The request has {len(positions)} image positions but the vision tower produced {len(vision_rows)} rows.")
    inputs_embeds = encoder.language.embed_tokens(input_ids)
    inputs_embeds[0, mx.array(positions, dtype=mx.uint32)] = mx.array(vision_rows).astype(inputs_embeds.dtype)
    position_ids, _ = LanguageModel.get_rope_index(encoder, input_ids, image_grid_thw=mx.array(vision_inputs[1].astype(np.int32)), video_grid_thw=None, attention_mask=None)
    hidden = encoder._hidden_states(
        input_ids,
        position_ids,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=input_ids == encoder.image_token_id,
        deepstack_visual_embeds=[mx.array(feature).astype(mx.bfloat16) for feature in deepstack_rows],
    )
    return np.array(hidden.astype(mx.float32)), token_tags


def _text_rows(text_encoder_dir: Path, fl2va_dir: Path, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the text encoder's hidden rows for a prompt with no images, and the row tags (float32)."""
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    encoder = MiniMaxH3TextEncoder(text_encoder_dir, load_vision=False, stream_layers=False, tokenizer_dir=fl2va_dir / "tokenizer", processor_dir=fl2va_dir / "processor")
    hidden, token_tags = encoder.encode(prompt)
    return np.array(hidden.astype(mx.float32)), token_tags


def pack_track_latents(mean: np.ndarray, latents_mean: Sequence[float], latents_std: Sequence[float], latents: int) -> np.ndarray:
    """Return the audio VAE's posterior mean as *latents* normalised audio rows per stereo channel, channel-major.

    The mean is normalised ``(mean - latents_mean) / latents_std``, as the pipeline's decode undoes it, then each channel is
    trimmed, or padded by repeating its last latent, to *latents* (as the Windows/Linux backend does).

    Args:
        mean: The posterior mean ``(2, latent_channels, frames)``, one batch item per stereo channel.
        latents_mean: The VAE's per-channel mean, ``latent_channels`` values.
        latents_std: The VAE's per-channel standard deviation, ``latent_channels`` values.
        latents: The latents to hold per channel.

    Returns:
        Float32 rows ``(2 * latents, latent_channels)``: channel 0's latents, then channel 1's.

    Raises:
        ValueError: If *mean* is not ``(2, latent_channels, frames)`` with at least one frame, the statistics do not match
            ``latent_channels``, or *latents* is not positive.
    """
    import numpy as np

    if mean.ndim != 3 or mean.shape[0] != 2 or mean.shape[2] < 1:
        raise ValueError(f"Expected a posterior mean shaped (2, latent_channels, frames); got {tuple(mean.shape)}.")
    if len(latents_mean) != mean.shape[1] or len(latents_std) != mean.shape[1]:
        raise ValueError(f"The latent statistics must have {mean.shape[1]} values; got {len(latents_mean)} and {len(latents_std)}.")
    if latents < 1:
        raise ValueError(f"The held audio must be at least one latent per channel, got {latents}.")
    centre = np.asarray(latents_mean, dtype=np.float32).reshape(1, -1, 1)
    scale = np.asarray(latents_std, dtype=np.float32).reshape(1, -1, 1)
    normalised = ((mean.astype(np.float32) - centre) / scale).transpose(0, 2, 1)  # (2, frames, channels)
    if normalised.shape[1] < latents:
        normalised = np.concatenate([normalised, np.repeat(normalised[:, -1:], latents - normalised.shape[1], axis=1)], axis=1)
    return np.ascontiguousarray(normalised[:, :latents].reshape(2 * latents, -1))


def _track_rows(fl2va_dir: Path, canvas: _Canvas, waveform: np.ndarray) -> np.ndarray:
    """Return the clip's held audio rows ``(2 * audio_latents, latent_channels)``: *waveform* encoded by the audio VAE (float32).

    Raises:
        ValueError: If *waveform* is not stereo ``(2, samples)``.
    """
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx.load import load_audio_vae

    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError(f"Expected a stereo waveform shaped (2, samples); got {tuple(waveform.shape)}.")
    audio_vae = load_audio_vae(fl2va_dir / "audio_vae")
    mean, _logs = audio_vae.encode(mx.array(waveform.astype(np.float32))[:, None])
    return pack_track_latents(np.array(mean.astype(mx.float32)), canvas.audio_config.latents_mean, canvas.audio_config.latents_std, canvas.audio_latents)


def _denoise(
    transformer_dir: Path,
    *,
    canvas: _Canvas,
    turbo: tuple[Path | None, float | None, float],
    schedule: H3Schedule,
    prompt_rows: np.ndarray,
    text_tags: np.ndarray,
    condition_rows: np.ndarray,
    anchors: tuple[str, ...],
    track_rows: np.ndarray | None,
    seed: int,
    step_callback: Callable[[int, int], None] | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the joint video and audio denoise loop with the transformer streamed from disk; return the generated rows (float32).

    Mirrors ``MiniMaxH3Pipeline.__call__``: noise is drawn conditioning, video, audio from the request seed; conditioning
    rows are noised to the keyframe level, held there, and never written back. With no condition rows (text to video) nothing
    is drawn for them. With *track_rows* every audio row is a clean conditioning row at timestep 1.0, no audio noise is
    drawn, the audio is never stepped, and the returned audio is empty.
    """
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx.adaln import ModulationCache
    from minimax_h3_mlx.packing import AUDIO_CHANNELS, KEYFRAME_NOISE_AUG, build_packed_sequence, patchify_video_latents
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
    from minimax_h3_mlx.streaming import load_streaming_dit

    patch_size = canvas.dit_config.patch_size
    layout = build_packed_sequence(text_tags, canvas.latent_frames, canvas.latent_height, canvas.latent_width, canvas.audio_latents, patch_size, anchors)
    if track_rows is not None:
        layout = replace(layout, num_condition_audio_rows=AUDIO_CHANNELS * canvas.audio_latents)

    mx.random.seed(seed)
    condition = None
    if len(condition_rows):
        condition = MiniMaxH3Scheduler(shift=schedule.video_shift).scale_noise(mx.array(condition_rows), KEYFRAME_NOISE_AUG, mx.random.normal(condition_rows.shape).astype(mx.float32))
    latents = mx.random.normal((1, canvas.video_config.latent_channels, canvas.latent_frames, canvas.latent_height, canvas.latent_width)).astype(mx.float32)
    video_rows = patchify_video_latents(latents, patch_size)
    if condition is not None:
        video_rows = mx.concatenate([condition, video_rows])
    if track_rows is not None:
        audio_rows = mx.array(track_rows).astype(mx.float32)
    else:
        audio_rows = mx.random.normal((canvas.audio_latents * AUDIO_CHANNELS, canvas.audio_config.latent_channels)).astype(mx.float32)

    video_sched, audio_sched = MiniMaxH3Scheduler(shift=schedule.video_shift), MiniMaxH3Scheduler(shift=schedule.audio_shift)
    video_sched.set_timesteps(schedule.num_inference_steps)
    audio_sched.set_timesteps(schedule.num_inference_steps)
    timestep_table, plan = MiniMaxH3Pipeline._row_timestep_plan(None, layout, video_sched.timesteps, audio_sched.timesteps)

    lora_path, lora_alpha, lora_scale = turbo
    dit, provider = load_streaming_dit(transformer_dir, turbo_lora_path=lora_path, turbo_lora_alpha=lora_alpha, turbo_lora_scale=lora_scale, verbose=verbose)
    modulation = ModulationCache.build_streaming(dit, provider, timestep_table, dtype=mx.bfloat16)

    n_cond_v, n_cond_a = layout.num_condition_video_rows, layout.num_condition_audio_rows
    embeds = mx.array(prompt_rows).astype(mx.bfloat16)
    total = len(video_sched.timesteps)
    for i, t in enumerate(video_sched.timesteps.tolist()):
        started = time.perf_counter()
        video_pred, audio_pred = dit(
            video_rows[None].astype(mx.bfloat16),
            audio_rows[None].astype(mx.bfloat16),
            embeds,
            timestep_table,
            plan[i],
            layout.token_tags,
            layout.position_ids,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
            modulation_cache=modulation,
            block_provider=provider,
        )
        # Only the generated rows are stepped; the conditioning rows are concatenated back unchanged.
        stepped_video = video_sched.step(video_pred[0, n_cond_v:].astype(mx.float32), float(t), video_rows[n_cond_v:])
        video_rows = mx.concatenate([video_rows[:n_cond_v], stepped_video]) if n_cond_v else stepped_video
        if n_cond_a < audio_rows.shape[0]:
            stepped_audio = audio_sched.step(audio_pred[0, n_cond_a:].astype(mx.float32), float(audio_sched.timesteps[i].item()), audio_rows[n_cond_a:])
            audio_rows = mx.concatenate([audio_rows[:n_cond_a], stepped_audio]) if n_cond_a else stepped_audio
        mx.eval(video_rows, audio_rows)
        if verbose:
            print(f"  step {i + 1}/{total}  {time.perf_counter() - started:.1f}s", flush=True)
        if step_callback is not None:
            step_callback(i + 1, total)
    generated_audio = np.array(audio_rows[n_cond_a:]) if n_cond_a < audio_rows.shape[0] else np.zeros((0, audio_rows.shape[1]), dtype=np.float32)
    return np.array(video_rows[n_cond_v:]), generated_audio


def _decode_video(fl2va_dir: Path, canvas: _Canvas, rows: np.ndarray) -> np.ndarray:
    """Return the RGB uint8 frames ``(frames, height, width, 3)`` of the generated video rows, decoded by the video VAE."""
    import mlx.core as mx
    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    host = SimpleNamespace(video_vae=load_video_vae(fl2va_dir / "video_vae"), _video_config=canvas.video_config, _dit_config=canvas.dit_config)
    return MiniMaxH3Pipeline._decode_video(host, mx.array(rows), canvas.latent_frames, canvas.latent_height, canvas.latent_width)


def _decode_audio(fl2va_dir: Path, canvas: _Canvas, rows: np.ndarray) -> np.ndarray:
    """Return the float32 stereo waveform ``(2, samples)`` of the generated audio rows, decoded by the audio VAE."""
    import mlx.core as mx
    from minimax_h3_mlx.load import load_audio_vae
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    host = SimpleNamespace(audio_vae=load_audio_vae(fl2va_dir / "audio_vae"), _audio_config=canvas.audio_config)
    return MiniMaxH3Pipeline._decode_audio(host, mx.array(rows), canvas.audio_latents)
