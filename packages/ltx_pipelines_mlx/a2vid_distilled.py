"""Audio-to-Video on the distilled two-stage path — no dev model, no CFG.

Mirrors Lightricks' ``LTX-2.5_A2V_Two_Stage_Distilled`` ComfyUI workflow
(ComfyUI-LTXVideo ``example_workflows/2.5/``):
  Stage 1: distilled model at half resolution, the input audio's latent frozen.
  Stage 2: spatial upsample, distilled refine at full resolution, the input audio still frozen.
The output carries the original waveform, trimmed to the clip.

Same transformer, sigmas and sampler as ``generate --distilled`` (ancestral on LTX-2.5
packs), so an audio-conditioned clip costs a distilled render instead of the dev
model's 30 CFG steps (:class:`~ltx_pipelines_mlx.a2vid_two_stage.A2VidPipelineTwoStage`).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.a2vid_two_stage import decode_with_source_audio, encode_source_audio
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.utils.blocks import snap_num_frames
from ltx_pipelines_mlx.utils.helpers import create_noised_state


class A2VidDistilledPipeline(DistilledPipeline):
    """Audio-to-video with the distilled transformer (``a2v --distilled``).

    The input audio enters only as a frozen conditioning stream (zero denoise
    mask, per-modality sigma 0) in both stages, so the video is generated to
    match it, exactly as upstream's ``freeze_audio=True``. Everything else is
    :class:`~ltx_pipelines_mlx.distilled.DistilledPipeline`: weights, steps,
    sampler, I2V anchors, ``--low-ram`` streaming.
    """

    #: Encoded input audio for the render in progress. ``None`` outside
    #: :meth:`generate_and_save`, where the stage hooks fall back to the
    #: parent's generated audio.
    _source_audio_tokens: mx.array | None = None

    def _stage1_audio_state(
        self,
        audio_shape: tuple[int, int, int],
        audio_positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
    ) -> LatentState:
        """Stage 1's audio state: the input track, frozen."""
        if self._source_audio_tokens is None:
            return super()._stage1_audio_state(audio_shape, audio_positions, spatial_dims, seed)
        return self._frozen_audio_state(audio_shape, audio_positions, spatial_dims, seed)

    def _stage2_audio_state(
        self,
        audio_tokens: mx.array,
        audio_positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
        sigma: float,
    ) -> LatentState:
        """Stage 2's audio state: the input track again, frozen (not re-noised at ``sigma``)."""
        if self._source_audio_tokens is None:
            return super()._stage2_audio_state(audio_tokens, audio_positions, spatial_dims, seed, sigma)
        return self._frozen_audio_state(tuple(audio_tokens.shape), audio_positions, spatial_dims, seed)

    def _frozen_audio_state(
        self,
        shape: tuple[int, ...],
        positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
    ) -> LatentState:
        """Build the frozen audio state from the encoded input track.

        Raises:
            ValueError: when the encoded track does not cover the clip's audio tokens.
        """
        tokens = self._source_audio_tokens
        assert tokens is not None
        if tuple(tokens.shape) != tuple(shape):
            raise ValueError(
                f"Encoded audio has shape {tuple(tokens.shape)}, but the clip's audio state needs {tuple(shape)}."
            )
        return create_noised_state(
            base_shape=tokens.shape,
            conditionings=[],
            spatial_dims=spatial_dims,  # unused for audio
            positions=positions,
            seed=seed,
            initial_latent=tokens,
            frozen=True,
        )

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        audio_path: str | Path | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        image: str | None = None,
        images=None,
        audio_start_time: float = 0.0,
    ) -> str:
        """Generate a video for the input audio and save it with that audio.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            audio_path: Path to input audio file (required). With music or
                loud ambience, pass the isolated vocals.
            height: Video height.
            width: Video width.
            num_frames: Number of frames (floored to the 8k+1 grid).
            frame_rate: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: the full distilled table, 8).
            stage2_steps: Stage 2 steps (default: the full stage-2 table, 3).
            image: Optional reference image for I2V conditioning (first frame).
            images: Optional multi-anchor I2V conditioning inputs.
            audio_start_time: Start time in seconds for audio.

        Returns:
            Path to the output video file.

        Raises:
            ValueError: when ``audio_path`` is missing, has no audio, or is
                shorter than the clip -- before any model is loaded.
        """
        if audio_path is None:
            raise ValueError("audio_path is required for A2VidDistilledPipeline")
        num_frames = snap_num_frames(num_frames)

        # Encode the input audio before anything else: a silent or short file
        # fails before the text encoder or the transformer is loaded.
        self._source_audio_tokens = encode_source_audio(
            self,
            audio_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
            start_time=audio_start_time,
            max_duration=num_frames / frame_rate,
        )
        try:
            video_latent, _ = self.generate_two_stage(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                image=image,
                images=images,
            )
        finally:
            self._source_audio_tokens = None

        # Free transformer + encoders to make room for the decoder
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()
        decode_with_source_audio(
            self,
            video_latent,
            output_path,
            audio_path=audio_path,
            start_time=audio_start_time,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )

        if self.low_memory:
            self.vae_decoder = None
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        return output_path


__all__ = ["A2VidDistilledPipeline"]
