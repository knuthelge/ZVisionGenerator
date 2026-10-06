"""diffusers/CUDA image backend for Windows and Linux."""

from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

from zvisiongenerator.backends.image_win_preview import LivePreview, create_live_preview
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model

# CUDA allocation and kernel tuning hints for the diffusers image backend.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Families whose NF4 text encoder and transformer do not fit a 10-12 GB card together (Krea 2: ~3 + ~7 GB).
# They take turns on the GPU through model CPU offload instead of both staying resident.
_Q4_CPU_OFFLOAD_FAMILIES = frozenset({"krea2"})


def _krea2_guidance_scale(guidance: float | None) -> float:
    """Convert a standard CFG scale (1.0 = off) to Krea 2's ``guidance_scale`` (0.0 = off).

    Krea 2's pipeline computes ``cond + g * (cond - uncond)``, which is standard CFG at ``1 + g``.
    """
    return max((1.0 if guidance is None else guidance) - 1.0, 0.0)


def _make_bnb_configs(quantize: int, compute_dtype: torch.dtype):
    """Create matched BitsAndBytesConfig for transformers and diffusers components."""
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from transformers import BitsAndBytesConfig as TransformersBnBConfig

    if quantize == 4:
        te_config = TransformersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tx_config = DiffusersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        te_config = TransformersBnBConfig(load_in_8bit=True)
        tx_config = DiffusersBnBConfig(load_in_8bit=True)

    return te_config, tx_config


def _load_quantized(model_path: str, quantize: int, torch_dtype: torch.dtype, family: str):
    """Load pipeline with bitsandbytes-quantized transformer and text encoder."""
    from diffusers import AutoModel as DiffusersAutoModel
    from transformers import AutoModel as HFAutoModel

    te_config, tx_config = _make_bnb_configs(quantize, torch_dtype)

    offload_q4 = quantize == 4 and family in _Q4_CPU_OFFLOAD_FAMILIES

    # Load text encoder (Qwen3Model) with quantization — placed on CUDA by bnb
    text_encoder = HFAutoModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        quantization_config=te_config,
        torch_dtype=torch_dtype,
    )
    if offload_q4:
        # Free the GPU for the transformer load; model CPU offload moves the encoder back when it runs.
        text_encoder.to("cpu")

    # Load transformer (ZImageTransformer2DModel) with quantization — placed on CUDA by bnb
    transformer = DiffusersAutoModel.from_pretrained(
        model_path,
        subfolder="transformer",
        quantization_config=tx_config,
        torch_dtype=torch_dtype,
    )

    # Build pipeline with pre-quantized components
    # Pipeline skips loading these from disk since we pass the instances
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )

    if quantize == 4 and not offload_q4:
        # NF4: ~5GB total — everything fits on GPU, no offloading needed
        pipeline.vae.to(device="cuda", dtype=torch_dtype)
    else:
        # INT8 (and NF4 for _Q4_CPU_OFFLOAD_FAMILIES): use model CPU offloading via accelerate hooks.
        # Group offloading is ineffective for INT8 because bitsandbytes
        # stores quantized weights in state.CB/state.SCB (not parameters/buffers).
        pipeline.enable_model_cpu_offload()
        pipeline.vae.to(device="cuda", dtype=torch_dtype)

    return pipeline


def _make_step_callback(skip_signal, *, total_steps: int, step_callback=None, live_preview: LivePreview | None = None):
    """Create a callback_on_step_end that reports progress and interrupts on skip.

    With ``live_preview``, milestone steps also carry a cheap preview of the predicted final image under the
    ``preview`` key.
    """

    def _on_step_end(pipe, step, timestep, callback_kwargs):
        del timestep
        if step_callback is not None:
            # img2img runs only the tail of the schedule; the pipeline knows how many steps that is.
            run_steps = getattr(pipe, "num_timesteps", None)
            steps_total = run_steps if isinstance(run_steps, int) and run_steps > 0 else max(total_steps, 1)
            payload = {
                "current_step": min(step + 1, steps_total),
                "total_steps": steps_total,
            }
            preview = live_preview.observe(pipe, step, callback_kwargs.get("latents")) if live_preview is not None else None
            if preview is not None:
                payload["preview"] = preview
            step_callback(payload)
        if skip_signal is not None and skip_signal.check():
            pipe._interrupt = True
        return callback_kwargs

    return _on_step_end


def _make_skip_callback(skip_signal):
    """Create a callback_on_step_end that interrupts on skip."""
    return _make_step_callback(skip_signal, total_steps=1)


class DiffusersBackend:
    """diffusers/CUDA backend for Windows and Linux.

    Stateful: holds a lazy-initialized img2img pipeline in self._img2img_pipe.
    """

    name = "diffusers"

    def __init__(self):
        self._img2img_pipe = None
        self._model_info: ImageModelInfo | None = None

    def stored_quant_format(self) -> str | None:
        """Return ``None``: bitsandbytes quantization happens at load and is not stored."""
        return None

    def save_quantized(self, model: Any, path: str) -> None:
        """Raise: this backend does not store quantized weights."""
        raise NotImplementedError("The diffusers backend does not store quantized weights.")

    def load_model(
        self,
        model_path: str,
        quantize: int | None = None,
        precision: str = "bfloat16",
        lora_paths: list[str] | None = None,
        lora_weights: list[float] | None = None,
    ) -> tuple[Any, "ImageModelInfo"]:
        self._img2img_pipe = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. The diffusers/CUDA image backend requires an NVIDIA GPU with CUDA support on Windows and Linux.")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        model_info = detect_image_model(model_path)
        self._model_info = model_info

        if model_info.family == "ideogram4":
            raise RuntimeError("Ideogram 4 is not supported on this platform (macOS/MLX only).")

        if quantize in (4, 8):
            pipeline = _load_quantized(model_path, quantize, torch_dtype, model_info.family)
        else:
            if quantize is not None:
                print(f"Unsupported quantize value {quantize!r}; falling back to full precision. Use 4 (NF4) or 8 (INT8).")

            from diffusers.hooks import apply_group_offloading

            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )

            # Stream transformer blocks to GPU with async prefetch (like safetensors)
            apply_group_offloading(
                pipeline.transformer,
                onload_device=torch.device("cuda"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
                record_stream=True,
                non_blocking=True,
                low_cpu_mem_usage=True,
            )

            # Stream text encoder layers similarly
            apply_group_offloading(
                pipeline.text_encoder,
                onload_device=torch.device("cuda"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
                record_stream=True,
                non_blocking=True,
                low_cpu_mem_usage=True,
            )

            # VAE is tiny (160MB) — just place on GPU
            pipeline.vae.to(device="cuda", dtype=torch_dtype)

        if lora_paths:
            for i, path in enumerate(lora_paths):
                pipeline.load_lora_weights(path, adapter_name=f"lora_{i}")
            adapter_names = [f"lora_{i}" for i in range(len(lora_paths))]
            pipeline.set_adapters(adapter_names, adapter_weights=lora_weights)

        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        return pipeline, model_info

    def _step_callback_kwargs(self, skip_signal, step_callback, steps: int, height: int, width: int) -> dict[str, Any]:
        """Build the pipeline's step-end callback kwargs, with live previews when progress is reported."""
        live_preview = None
        if step_callback is not None and self._model_info is not None:
            live_preview = create_live_preview(self._model_info.family, steps, height, width)
        kwargs: dict[str, Any] = {
            "callback_on_step_end": _make_step_callback(skip_signal, total_steps=steps, step_callback=step_callback, live_preview=live_preview),
        }
        if live_preview is not None:
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        return kwargs

    @torch.inference_mode()
    def image_to_image(
        self,
        model: Any,
        image: Image.Image,
        prompt: str,
        strength: float,
        steps: int,
        seed: int,
        guidance: float,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        skip_signal: Any | None = None,
        step_callback: Any | None = None,
    ) -> Image.Image | None:
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        if self._model_info.family == "krea2":
            raise ValueError("img2img is not supported for Krea 2 on this platform: diffusers has no Krea 2 image-to-image pipeline.")
        if self._img2img_pipe is None:
            from diffusers import AutoPipelineForImage2Image

            self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(model)

        original_scheduler = self._img2img_pipe.scheduler
        try:
            if scheduler == "beta":
                from diffusers import FlowMatchEulerDiscreteScheduler

                self._img2img_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._img2img_pipe.scheduler.config, use_beta_sigmas=True)

            _is_flux = self._model_info is not None and self._model_info.family in ("flux1", "flux2", "flux2_klein")

            # Free VRAM from the generation pass before refinement
            torch.cuda.empty_cache()

            generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
            pipe_kwargs = dict(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance if guidance is not None else (1.0 if _is_flux else 0.0),
                generator=generator,
            )
            if not _is_flux and negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt
            if skip_signal is not None or step_callback is not None:
                pipe_kwargs.update(self._step_callback_kwargs(skip_signal, step_callback, steps, image.height, image.width))

            result = self._img2img_pipe(**pipe_kwargs)

            if skip_signal is not None and self._img2img_pipe._interrupt:
                self._img2img_pipe._interrupt = False
                torch.cuda.empty_cache()
                return None

            torch.cuda.empty_cache()
            return result.images[0]
        finally:
            self._img2img_pipe.scheduler = original_scheduler

    @torch.inference_mode()
    def text_to_image(
        self,
        model: Any,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        skip_signal: Any | None = None,
        step_callback: Any | None = None,
        steps_explicit: bool = False,
        guidance_explicit: bool = False,
        first_sigma: float | None = None,
    ) -> Image.Image | None:
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        original_scheduler = model.scheduler
        try:
            if scheduler == "beta":
                from diffusers import FlowMatchEulerDiscreteScheduler

                model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config, use_beta_sigmas=True)
            _is_flux = self._model_info is not None and self._model_info.family in ("flux1", "flux2", "flux2_klein")

            generator = torch.Generator(device="cpu").manual_seed(seed)
            kwargs = dict(
                prompt=prompt,
                width=width,
                height=height,
                generator=generator,
                num_inference_steps=steps,
            )
            if self._model_info.family == "krea2":
                kwargs["guidance_scale"] = _krea2_guidance_scale(guidance)
            elif guidance is not None:
                kwargs["guidance_scale"] = guidance
            elif _is_flux:
                kwargs["guidance_scale"] = 1.0
            if not _is_flux and negative_prompt is not None:
                kwargs["negative_prompt"] = negative_prompt
            if skip_signal is not None or step_callback is not None:
                kwargs.update(self._step_callback_kwargs(skip_signal, step_callback, steps, height, width))

            result = model(**kwargs)

            if skip_signal is not None and model._interrupt:
                model._interrupt = False
                torch.cuda.empty_cache()
                return None

            torch.cuda.empty_cache()
            return result.images[0]
        finally:
            model.scheduler = original_scheduler
