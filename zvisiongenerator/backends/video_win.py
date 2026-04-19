"""Windows/CUDA video backend using ltx-pipelines."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch

from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model

# CUDA optimization hints for Windows/WDDM
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


_TEXT_ENCODER_ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*"]


def _resolve_model_paths(model_dir: str, *, require_upsampler: bool = True) -> tuple[str, str | None, str | None]:
    """Auto-discover checkpoint, gemma directory, and spatial upsampler from a model directory.

    Args:
        model_dir: Path to the model directory.
        require_upsampler: If True, raise when upsampler is missing. If False, return None.

    Returns:
        Tuple of (checkpoint_path, gemma_root, spatial_upsampler_path_or_None).

    Raises:
        FileNotFoundError: If any required component is missing.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 1. Distilled checkpoint — a .safetensors file (prefer one with "distilled" in the name)
    safetensors = list(root.glob("*.safetensors"))
    distilled = [f for f in safetensors if "distilled" in f.name.lower()]
    # Exclude spatial upscaler files from checkpoint candidates
    checkpoint_candidates = distilled or [f for f in safetensors if "spatial_upscal" not in f.name.lower()]
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No .safetensors checkpoint found in {model_dir}. Expected a distilled checkpoint file (e.g. ltx-video-2b-v0.9.7-distilled.safetensors).")
    checkpoint_path = str(checkpoint_candidates[0])

    # 2. Text encoder — prefer gemma*/, but fall back to text_encoder/.
    # Flat repositories such as LTX-2.3-fp8 ship only the video checkpoint,
    # so absence here is handled later via model metadata.
    gemma_dirs = list(root.glob("gemma*/"))
    text_encoder_dir = root / "text_encoder"
    if gemma_dirs:
        gemma_root = str(gemma_dirs[0])
    elif text_encoder_dir.is_dir():
        gemma_root = str(text_encoder_dir)
    else:
        gemma_root = None

    # 3. Spatial upsampler — a file matching *spatial_upscal*.safetensors
    upsampler_files = list(root.glob("*spatial_upscal*.safetensors"))
    if not upsampler_files:
        if require_upsampler:
            raise FileNotFoundError(f"No spatial upsampler found in {model_dir}. Expected a file matching *spatial_upscal*.safetensors.")
        return checkpoint_path, gemma_root, None
    spatial_upsampler_path = str(upsampler_files[0])

    return checkpoint_path, gemma_root, spatial_upsampler_path


def _maybe_download(model_path: str, *, allow_patterns: list[str] | None = None) -> str:
    """If model_path looks like a HuggingFace repo ID, download it. Otherwise return as-is.

    Args:
        model_path: Local path or HuggingFace repo ID (org/name).
        allow_patterns: Optional Hugging Face snapshot filters.

    Returns:
        Local directory path.
    """
    if "/" in model_path and not Path(model_path).exists():
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, allow_patterns=allow_patterns)
    return model_path


class LtxCudaVideoBackend:
    """CUDA/ltx-pipelines video backend — implements VideoBackend Protocol."""

    name: str = "ltx"

    def __init__(self) -> None:
        self._model_info: VideoModelInfo | None = None

    def load_model(
        self,
        model_path: str,
        **kwargs: Any,
    ) -> tuple[Any, VideoModelInfo]:
        """Load the video pipeline from a model directory.

        Args:
            model_path: Local directory or HuggingFace repo ID.
            **kwargs: loras (list[tuple[str, float]]), upscale (bool).

        Returns:
            Tuple of (pipeline, VideoModelInfo).

        Raises:
            RuntimeError: If CUDA is not available.
            FileNotFoundError: If model components are missing.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. The Windows video backend requires an NVIDIA GPU with CUDA support.")

        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import SDOps

        model_info = detect_video_model(model_path)
        local_dir = _maybe_download(model_path)
        upscale: bool = bool(kwargs.get("upscale"))

        # LoRA conversion
        raw_loras: list[tuple[str, float]] = kwargs.get("loras") or []
        loras = [LoraPathStrengthAndSDOps(path=p, strength=s, sd_ops=SDOps(name="identity")) for p, s in raw_loras]

        checkpoint, gemma_root, upsampler = _resolve_model_paths(local_dir, require_upsampler=upscale)
        if gemma_root is None:
            if model_info.default_text_encoder is None:
                raise FileNotFoundError(f"No gemma*/ or text_encoder/ subdirectory found in {local_dir}, and no default text encoder is configured for {model_path}.")
            gemma_root = _maybe_download(
                model_info.default_text_encoder,
                allow_patterns=_TEXT_ENCODER_ALLOW_PATTERNS,
            )

        if upscale:
            from ltx_pipelines.distilled import DistilledPipeline

            pipeline = DistilledPipeline(
                distilled_checkpoint_path=checkpoint,
                gemma_root=gemma_root,
                spatial_upsampler_path=upsampler,
                loras=loras,
                device=torch.device("cuda"),
            )
        else:
            from ltx_pipelines.distilled_single_stage import DistilledSingleStagePipeline

            pipeline = DistilledSingleStagePipeline(
                distilled_checkpoint_path=checkpoint,
                gemma_root=gemma_root,
                loras=loras,
                device=torch.device("cuda"),
            )

        self._model_info = model_info
        return pipeline, model_info

    @torch.inference_mode()
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
        **kwargs: Any,
    ) -> Path | None:
        """Generate video from text prompt.

        Args:
            model: Pipeline from load_model.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps (slices DISTILLED_SIGMAS).
            output_path: Path to save video.
            **kwargs: stage1_steps (int) for two-stage upscale mode.

        Returns:
            Path to generated video, or None on failure.
        """
        stage1_steps = kwargs.get("stage1_steps")
        if stage1_steps is not None:
            return self._generate_upscaled(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, images=[])
        return self._generate(model, prompt, width, height, num_frames, seed, steps, output_path, images=[])

    @torch.inference_mode()
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
        **kwargs: Any,
    ) -> Path | None:
        """Generate video from image and text prompt.

        Args:
            model: Pipeline from load_model.
            image_path: Path to conditioning image.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps (slices DISTILLED_SIGMAS).
            output_path: Path to save video.
            **kwargs: stage1_steps (int) for two-stage upscale mode.

        Returns:
            Path to generated video, or None on failure.
        """
        from ltx_pipelines.utils.args import ImageConditioningInput

        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
        stage1_steps = kwargs.get("stage1_steps")
        if stage1_steps is not None:
            return self._generate_upscaled(model, prompt, width, height, num_frames, seed, stage1_steps, output_path, images=images)
        return self._generate(model, prompt, width, height, num_frames, seed, steps, output_path, images=images)

    def _generate(
        self,
        pipeline: Any,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        *,
        images: list[Any],
    ) -> Path | None:
        """Run the single-stage pipeline and encode the output video.

        Args:
            pipeline: DistilledSingleStagePipeline instance.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            steps: Inference steps (slices DISTILLED_SIGMAS).
            output_path: Path to save video.
            images: Image conditioning inputs (empty list for T2V).

        Returns:
            Path to generated video, or None on failure.
        """
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")

        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS

        fps = self._model_info.default_fps
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        sigmas = DISTILLED_SIGMAS[: steps + 1] if steps is not None else DISTILLED_SIGMAS

        try:
            video_iter, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=images,
                tiling_config=tiling_config,
                sigmas=sigmas,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn("CUDA out of memory during video generation. Try reducing resolution or frame count, or use ltx-8 for a pre-quantized model.", stacklevel=2)
            return None
        except Exception as exc:
            warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
            return None

        return self._encode_output(video_iter, audio, fps, output_path, video_chunks_number)

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
        images: list[Any],
    ) -> Path | None:
        """Run the two-stage distilled pipeline with spatial upscaling.

        Args:
            pipeline: DistilledPipeline instance.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for Stage 1.
            output_path: Path to save video.
            images: Image conditioning inputs (empty list for T2V).

        Returns:
            Path to generated video, or None on failure.
        """
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")

        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS

        fps = self._model_info.default_fps
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        try:
            video_iter, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=images,
                tiling_config=tiling_config,
                stage_1_sigmas=DISTILLED_SIGMAS[: stage1_steps + 1],
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn("CUDA out of memory during video generation. Try reducing resolution or frame count, or use ltx-8 for a pre-quantized model.", stacklevel=2)
            return None
        except Exception as exc:
            warnings.warn(f"Video generation failed: {exc}", stacklevel=2)
            return None

        return self._encode_output(video_iter, audio, fps, output_path, video_chunks_number)

    def _encode_output(
        self,
        video_iter: Any,
        audio: Any,
        fps: int,
        output_path: str,
        video_chunks_number: int,
    ) -> Path | None:
        """Encode pipeline output to a video file.

        Args:
            video_iter: Video iterator from pipeline.
            audio: Audio object (may be None or malformed).
            fps: Frames per second.
            output_path: Path to save video.
            video_chunks_number: Number of video chunks for encoding.

        Returns:
            Path to generated video, or None on failure.
        """
        from ltx_pipelines.utils.media_io import encode_video

        # Audio may fail to decode — produce video without audio in that case
        safe_audio = audio
        try:
            # Validate audio is usable (audio object exists but may be malformed)
            if audio is not None and not hasattr(audio, "sampling_rate"):
                safe_audio = None
        except Exception:
            safe_audio = None
            warnings.warn("Audio decoding failed; producing video without audio.", stacklevel=2)

        try:
            encode_video(
                video=video_iter,
                fps=fps,
                audio=safe_audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn("CUDA out of memory during video encoding. Try reducing resolution or frame count, or use ltx-8 for a pre-quantized model.", stacklevel=2)
            return None
        except Exception as exc:
            warnings.warn(f"Video encoding failed: {exc}", stacklevel=2)
            return None

        return Path(output_path)
