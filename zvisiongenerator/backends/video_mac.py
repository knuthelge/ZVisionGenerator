"""macOS video backends — LTX (ltx-pipelines-mlx)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model


def _load_upsampler(model_dir: Path, name: str = "spatial_upscaler_x2_v1_1") -> Any:
    """Load the spatial upsampler from model weights.

    Args:
        model_dir: Directory containing upsampler weights.
        name: Weight file basename (without extension).

    Returns:
        Loaded LatentUpsampler instance.

    Raises:
        FileNotFoundError: If upsampler weights are not found.
    """
    import json

    from ltx_core_mlx.model.upsampler import LatentUpsampler
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import load_split_safetensors

    weights_path = model_dir / f"{name}.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Upsampler weights not found: {weights_path}. Download the model with upscaler weights.")

    config_path = model_dir / f"{name}_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text()).get("config", {})
        upsampler = LatentUpsampler.from_config(config)
    else:
        upsampler = LatentUpsampler()
    weights = load_split_safetensors(weights_path, prefix=f"{name}.")
    upsampler.load_weights(list(weights.items()))
    aggressive_cleanup()
    return upsampler


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
            **kwargs: mode ("t2v"|"i2v"), low_memory (bool), loras (list[tuple[str, float]]), upscale (bool).

        Returns:
            Tuple of (pipeline, VideoModelInfo).

        Raises:
            RuntimeError: If ltx-pipelines-mlx is not installed.
        """
        mode: str = kwargs.get("mode", "t2v")
        low_memory: bool = kwargs.get("low_memory", True)
        loras: list[tuple[str, float]] | None = kwargs.get("loras")
        upscale: bool = bool(kwargs.get("upscale"))

        try:
            if mode == "i2v" and not upscale:
                from ltx_pipelines_mlx import ImageToVideoPipeline

                pipeline = ImageToVideoPipeline(model_dir=model_path, low_memory=low_memory)
            else:
                from ltx_pipelines_mlx import TextToVideoPipeline

                pipeline = TextToVideoPipeline(model_dir=model_path, low_memory=low_memory)
        except ImportError as exc:
            raise RuntimeError("Video dependency ltx-pipelines-mlx not found. Reinstall with: uv sync") from exc

        if loras:
            # ltx-pipelines-mlx reads _pending_loras during generation
            pipeline._pending_loras = loras  # noqa: SLF001

        model_info = detect_video_model(model_path)
        return pipeline, model_info

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
            model: LTX pipeline object from load_model.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps.
            output_path: Path to save video.
            **kwargs: stage1_steps for distilled two-stage upscale mode.

        Returns:
            Path to generated video.
        """
        stage1_steps = kwargs.get("stage1_steps")
        if stage1_steps is not None:
            return self._generate_upscaled(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, step_callback=step_callback)
        model.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=steps,
            progress_callback=step_callback,
        )
        return Path(output_path)

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
            model: LTX pipeline object from load_model.
            image_path: Path to input image.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps.
            output_path: Path to save video.
            **kwargs: stage1_steps for distilled two-stage upscale mode.

        Returns:
            Path to generated video.
        """
        stage1_steps = kwargs.get("stage1_steps")
        if stage1_steps is not None:
            return self._generate_upscaled(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, image_path=image_path, step_callback=step_callback)
        model.generate_and_save(
            prompt=prompt,
            image=image_path,
            output_path=output_path,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=steps,
            progress_callback=step_callback,
        )
        return Path(output_path)

    def _generate_upscaled(
        self,
        pipeline: Any,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        stage1_steps: int,
        output_path: str,
        *,
        image_path: str | None = None,
        step_callback: Any | None = None,
    ) -> Path:
        """Generate video with distilled-only two-stage upscaling.

        Runs Stage 1 at half resolution with DISTILLED_SIGMAS, spatially upscales
        2x via LatentUpsampler, then refines at full resolution with STAGE_2_SIGMAS.
        Uses the same distilled transformer for both stages (no dev model, no CFG).

        Args:
            pipeline: TextToVideoPipeline instance.
            prompt: Text prompt.
            width: Full target width.
            height: Full target height.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Number of denoising steps for Stage 1.
            output_path: Path to save the output video.
            image_path: Optional path to input image for I2V conditioning.

        Returns:
            Path to the generated video file.

        Raises:
            ValueError: If width or height is less than 64.
        """
        if width < 64 or height < 64:
            raise ValueError(f"Upscale requires width and height >= 64, got {width}x{height}")
        import mlx.core as mx
        from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state, noise_latent_state
        from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        if image_path is not None:
            from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex, apply_conditioning
            from ltx_core_mlx.utils.image import prepare_image_for_encoding

        model_dir = Path(pipeline.model_dir)

        # --- 1. Encode text ---
        pipeline._load_text_encoder()  # noqa: SLF001
        video_embeds, audio_embeds = pipeline._encode_text(prompt)  # noqa: SLF001
        mx.eval(video_embeds, audio_embeds)
        if pipeline.low_memory:
            pipeline.text_encoder = None
            pipeline.feature_extractor = None
            aggressive_cleanup()

        # --- 2. Load transformer (if not already loaded) ---
        if pipeline.dit is None:
            pipeline.dit = LTXModel()
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = model_dir / "transformer-distilled.safetensors"
            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
            pending_loras = getattr(pipeline, "_pending_loras", None)
            if pending_loras:
                transformer_weights = pipeline._fuse_pending_loras(transformer_weights, pending_loras)  # noqa: SLF001
            apply_quantization(pipeline.dit, transformer_weights)
            pipeline.dit.load_weights(list(transformer_weights.items()))
            aggressive_cleanup()

        # --- 3. Load VAE encoder + upsampler ---
        pipeline._load_vae_encoder()  # noqa: SLF001
        upsampler = _load_upsampler(model_dir)

        # --- 4. Compute half-resolution (floor-aligned to 32px) ---
        half_w = (width // 2 // 32) * 32
        half_h = (height // 2 // 32) * 32
        if half_w * 2 != width or half_h * 2 != height:
            warnings.warn(f"Upscale half-resolution {half_w}x{half_h} does not double exactly to {width}x{height}; output will be {half_w * 2}x{half_h * 2}", stacklevel=2)

        # --- 5. Stage 1: half-res denoising (distilled, no CFG) ---
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # I2V conditioning: encode image at half-res, apply to frame 0
        if image_path is not None:
            img_tensor = prepare_image_for_encoding(image_path, half_h, half_w)
            ref_latent = pipeline.vae_encoder.encode(img_tensor[:, :, None, :, :])
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            cond = [VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)]
            video_state = apply_conditioning(video_state, cond, (F, H_half, W_half))

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1]
        x0_model = X0Model(pipeline.dit)

        stage1_total_steps = len(sigmas_1) - 1
        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            step_callback=(
                None
                if step_callback is None
                else lambda event: step_callback(
                    {
                        "phase": "video_upscale_stage_1",
                        "current_step": event["current_step"],
                        "total_steps": stage1_total_steps + len(STAGE_2_SIGMAS) - 1,
                    }
                )
            ),
        )

        # --- 6. Upscale: denormalize → spatial 2x → renormalize ---
        video_half = pipeline.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # BCFHW -> BFHWC
        video_denorm = pipeline.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)  # BFHWC -> BCFHW
        video_upscaled = upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)  # BCFHW -> BFHWC
        video_upscaled = pipeline.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)  # BFHWC -> BCFHW
        mx.eval(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # --- 7. Stage 2: full-res refinement (distilled, no CFG) ---
        video_tokens, _ = pipeline.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens,
            denoise_mask=mx.ones((1, video_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # I2V conditioning: re-encode image at full-res, apply to frame 0
        if image_path is not None:
            img_tensor_full = prepare_image_for_encoding(image_path, half_h * 2, half_w * 2)
            ref_latent_full = pipeline.vae_encoder.encode(img_tensor_full[:, :, None, :, :])
            ref_tokens_full = ref_latent_full.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            cond_full = [VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens_full)]
            video_state_2 = apply_conditioning(video_state_2, cond_full, (F, H_full, W_full))

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        stage2_total_steps = len(sigmas_2) - 1
        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
            step_callback=(
                None
                if step_callback is None
                else lambda event: step_callback(
                    {
                        "phase": "video_upscale_stage_2",
                        "current_step": stage1_total_steps + event["current_step"],
                        "total_steps": stage1_total_steps + stage2_total_steps,
                    }
                )
            ),
        )

        video_latent = pipeline.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = pipeline.audio_patchifier.unpatchify(output_2.audio_latent)

        # --- 8. Free heavy components, decode, save ---
        if pipeline.low_memory:
            pipeline.dit = None
            pipeline.vae_encoder = None
            pipeline._loaded = False  # noqa: SLF001
            del upsampler
            aggressive_cleanup()

        pipeline._load_decoders()  # noqa: SLF001
        pipeline._decode_and_save_video(video_latent, audio_latent, output_path)  # noqa: SLF001
        return Path(output_path)
