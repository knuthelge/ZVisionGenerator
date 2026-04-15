"""HQ Audio-to-Video pipeline — res_2s second-order sampler for Stage 1.

Same architecture as AudioToVideoPipeline but uses the res_2s sampler
instead of Euler for Stage 1, producing higher quality output.

Reference defaults (LTX_2_3_HQ_PARAMS): 15 steps, no STG, rescale=0.45.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_pipelines_mlx.a2vid_two_stage import AudioToVideoPipeline
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE
from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop

# Reference HQ defaults (LTX_2_3_HQ_PARAMS)
HQ_DEFAULT_STAGE1_STEPS = 15
HQ_DEFAULT_STG_SCALE = 0.0
HQ_DEFAULT_RESCALE_SCALE = 0.45


class AudioToVideoHQPipeline(AudioToVideoPipeline):
    """HQ Audio-to-Video with res_2s second-order sampler for Stage 1.

    Inherits from AudioToVideoPipeline and overrides Stage 1 denoising
    to use res_2s instead of Euler. Stage 2 is identical.

    Reference HQ defaults: 15 steps, no STG, rescale_scale=0.45.
    """

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        audio_path: str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int = HQ_DEFAULT_STAGE1_STEPS,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = HQ_DEFAULT_STG_SCALE,
        image: str | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> str:
        """Generate video with HQ defaults (15 steps, no STG, rescale=0.45)."""
        return super().generate_and_save(
            prompt=prompt,
            output_path=output_path,
            audio_path=audio_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            image=image,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
        )

    def _denoise_stage1(
        self,
        x0_model: X0Model,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        neg_video_embeds: mx.array,
        neg_audio_embeds: mx.array,
        sigmas: list[float],
        cfg_scale: float = 3.0,
        stg_scale: float = HQ_DEFAULT_STG_SCALE,
    ) -> object:
        """Run Stage 1 denoising with res_2s + CFG (HQ defaults)."""
        # Video: HQ guidance (no STG, lower rescale)
        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=HQ_DEFAULT_RESCALE_SCALE,
            modality_scale=3.0,
            stg_blocks=[],
        )
        # Audio: no guidance (frozen in Stage 1)
        audio_gp = MultiModalGuiderParams()

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        return res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
        )
