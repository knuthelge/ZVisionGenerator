"""macOS video backends — LTX (ltx-pipelines-mlx)."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.backends.memory_mac import memory_size_bytes, mlx_cache_limit
from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model

# Pixels per latent cell of the video VAE encoder.
_LATENT_PIXELS = 32
# Channel count of the LTX video and audio token streams.
_LATENT_CHANNELS = 128
# Macs reporting at most this much unified memory stream the transformer blocks under ``--low-memory``.
# A Mac sold as "24 GB" reports exactly 24 GiB (25769803776 bytes), so the comparison is inclusive.
# Larger Macs do not stream: streaming lowers the memory held while denoising but not the peak, which is set
# while the text encoder loads, and it slows LoRA runs.
_LOW_RAM_THRESHOLD_BYTES = 24 * 1024**3


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
    """

    pipeline: Any
    frame_rate: float
    upscale: bool
    low_ram_streaming: bool


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


def _one_stage_progress(callback: Callable[[dict[str, Any]], None] | None) -> Callable[[int, int, Any, float], None] | None:
    """Return a denoise-loop ``on_step`` that emits 1-based ``video_generate`` progress, or ``None``."""
    if callback is None:
        return None

    def on_step(step_idx: int, total: int, _x0: Any, _sigma: float) -> None:
        callback({"phase": "video_generate", "current_step": step_idx + 1, "total_steps": total})

    return on_step


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
) -> None:
    """Run the distilled sampler at full resolution and write the clip to *output_path*.

    Mirrors stage 1 of upstream's distilled pipeline, without the half-resolution step.
    """
    import mlx.core as mx
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
    from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, shorten_schedule
    from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput
    from ltx_pipelines_mlx.utils.helpers import create_noised_state
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    pipe = model.pipeline
    fps = model.frame_rate
    low_memory = pipe.low_memory

    height, width = snap_output_dimensions(height, width, two_stage=False)
    video_embeds, audio_embeds = pipe.prompt_encoder(prompt, free_after=low_memory)

    F, H, W = compute_video_latent_shape(num_frames, height, width)
    audio_tokens = compute_audio_token_count(num_frames, frame_rate=fps)
    video_positions = compute_video_positions(F, H, W, frame_rate=fps)
    audio_positions = compute_audio_positions(audio_tokens)

    conditionings: list = []
    if image_path is not None:
        images = pipe.image_conditioner.resolve_crf([ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)])

        def _encode(encoder: Any) -> list:
            conds = combined_image_conditionings(
                images,
                enc_h=H * _LATENT_PIXELS,
                enc_w=W * _LATENT_PIXELS,
                spatial_dims=(F, H, W),
                video_encoder=encoder,
                frame_rate=fps,
            )
            # MLX graph evaluation: materialize the conditionings before the encoder is freed.
            mx.eval(*(cond.clean_latent for cond in conds))
            return conds

        conditionings = pipe.image_conditioner(_encode, free_after=low_memory)

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
    audio_state = create_noised_state(
        base_shape=(1, audio_tokens, _LATENT_CHANNELS),
        conditionings=[],
        spatial_dims=(F, H, W),
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

    output = denoise_loop(
        model=X0Model(pipe.dit),
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=video_embeds,
        audio_text_embeds=audio_embeds,
        sigmas=shorten_schedule(DISTILLED_SIGMAS, steps, keep="start"),
        on_step=_one_stage_progress(step_callback),
    )
    if low_memory:
        aggressive_cleanup()

    video_latent = pipe.video_patchifier.unpatchify(output.video_latent[:, : F * H * W, :], (F, H, W))
    audio_latent = pipe.audio_patchifier.unpatchify(output.audio_latent)
    pipe._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=fps, seed=seed)  # noqa: SLF001

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
    """Free what the rest of ``generate_and_save`` would have freed, for a generation that stops after stage 1.

    Under ``low_memory`` the vendored code releases the DiT, the prompt encoder, the image conditioner and the upsampler
    after stage 2 (``DistilledPipeline._stage2`` and ``generate_and_save``). A generation that ends between the stages
    never gets there, and the next ``_stage1`` would load the text encoder next to the resident DiT, because ``load()``
    returns early while ``_loaded`` is true. The next ``load()`` reloads all of it (the DiT when ``dit`` is ``None``,
    the VAE encoder through the image conditioner, the upsampler when ``upsampler`` is ``None``).
    The DiT is not freed at this moment: the exception in flight still references it through its traceback, so the
    cache is only cleared at the start of the next reviewed generation (``_generate_two_stage``).
    Without ``low_memory`` nothing is freed, as in a generation that runs through.
    """
    if not pipe.low_memory:
        return
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    pipe.dit = None
    pipe.prompt_encoder.free()
    pipe.image_conditioner.free()
    pipe.upsampler = None
    pipe._loaded = False  # noqa: SLF001
    aggressive_cleanup()


@contextlib.contextmanager
def _review_after_stage1(pipe: Any, review: Callable[[Path], None], *, frame_rate: float, seed: int) -> Iterator[None]:
    """Let *review* see the half-size result of stage 1 while ``generate_and_save`` runs.

    ``generate_two_stage`` calls ``self._stage1(...)``, so an instance attribute shadows the method for the duration
    of the block. The wrapper returns the stage-1 result unchanged, so the generation goes on exactly as without it.
    The attribute is removed afterwards, also when *review* or the generation raises. When the review raises (a
    discard or a quit), the model is released as a finished generation would have released it.
    """
    stage1_method = pipe._stage1  # noqa: SLF001

    def _stage1_then_review(*args: Any, **kwargs: Any) -> Any:
        result = stage1_method(*args, **kwargs)
        try:
            _preview_small_render(pipe, result[0], review, frame_rate=frame_rate, seed=seed)
        except BaseException:
            _release_after_aborted_generation(pipe)
            raise
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
) -> None:
    """Generate at half resolution, spatially upscale 2x, and refine with the distilled pipeline.

    With *review*, the half-size result is decoded to a temporary video and passed to it between the two stages.
    """
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS

    pipe = model.pipeline
    if review is not None and pipe.low_memory:
        # A discarded or quit review left the DiT referenced by its traceback, so the release after it could not free it.
        # With that traceback gone, this frees what the previous video left in the MLX cache before stage 1 loads the text encoder.
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        aggressive_cleanup()
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
            image=image_path,
        )


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
) -> Path:
    """Dispatch to the one-stage or two-stage generator that matches the loaded handle.

    Raises:
        ValueError: If ``stage1_steps`` is set but the handle was not loaded for upscale, or the reverse,
            or if ``review`` is set without ``stage1_steps``.
    """
    if review is not None and stage1_steps is None:
        raise ValueError("review needs stage1_steps: only the two-stage upscale has a half-size render to review")
    if (stage1_steps is not None) != model.upscale:
        raise ValueError(f"stage1_steps must be set exactly when the model is loaded for upscale (upscale={model.upscale}, stage1_steps={stage1_steps})")
    with mlx_cache_limit(0) if model.low_ram_streaming else contextlib.nullcontext():
        if stage1_steps is not None:
            _generate_two_stage(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, image_path, step_callback, review)
        else:
            _generate_one_stage(model, prompt, width, height, num_frames, seed, steps, output_path, image_path, step_callback)
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
            **kwargs: low_memory (bool), loras (list[tuple[str, float]]), upscale (bool).
                ``mode`` is accepted and ignored; the same pipeline serves t2v and i2v.

        Returns:
            Tuple of (backend-private model handle, VideoModelInfo).

        Raises:
            RuntimeError: If ltx-pipelines-mlx is not installed.
        """
        low_memory: bool = kwargs.get("low_memory", True)
        loras: list[tuple[str, float]] | None = kwargs.get("loras")
        upscale: bool = bool(kwargs.get("upscale"))
        streaming = _use_low_ram_streaming(low_memory, memory_size_bytes())

        try:
            # The constructor sets the MLX cache limit to 0 process-wide when streaming; scope it.
            with mlx_cache_limit(0) if streaming else contextlib.nullcontext():
                if upscale:
                    from ltx_pipelines_mlx import DistilledPipeline

                    pipeline = DistilledPipeline(model_dir=model_path, low_memory=low_memory, low_ram_streaming=streaming)
                else:
                    from ltx_pipelines_mlx import BasePipeline

                    pipeline = BasePipeline(model_dir=model_path, low_memory=low_memory, low_ram_streaming=streaming)
        except ImportError as exc:
            raise RuntimeError("Video dependency ltx-pipelines-mlx not found. Reinstall with: uv sync") from exc

        if loras:
            # ltx-pipelines-mlx fuses _pending_loras while it loads the transformer
            pipeline._pending_loras = loras  # noqa: SLF001

        model_info = detect_video_model(model_path)
        return _LtxModel(pipeline=pipeline, frame_rate=float(model_info.default_fps), upscale=upscale, low_ram_streaming=streaming), model_info

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
                that receives a temporary video of the half-size render before the upscale (needs stage1_steps).

        Returns:
            Path to generated video.

        Raises:
            ValueError: If stage1_steps does not match the model's upscale mode, or review is set without it.
        """
        return _generate(model, prompt, width, height, num_frames, seed, steps, output_path, image_path, kwargs.get("stage1_steps"), step_callback, kwargs.get("review"))
