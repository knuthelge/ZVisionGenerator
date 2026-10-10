"""diffusers/CUDA image backend for Windows and Linux."""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

from zvisiongenerator.backends.cuda_driver import cuda_driver_hint
from zvisiongenerator.backends.image_win_preview import LivePreview, create_live_preview
from zvisiongenerator.backends.lora_peft import is_unmatched_adapter_error, restore_cpu_offload, warn_unmatched_lora
from zvisiongenerator.backends.memory_cuda import configure_allocator
from zvisiongenerator.backends.image_win_quant import load_fp8_components, load_nf4_components, write_fp8_copy
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.stored_quant import STORED_QUANT_BITS, stored_quant_bits

# CUDA allocation and kernel tuning hints for the diffusers image backend.
configure_allocator()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# Families whose NF4 text encoder and transformer do not fit a 10-12 GB card together (Krea 2: ~3 + ~7 GB).
# They take turns on the GPU through model CPU offload instead of both staying resident.
_Q4_CPU_OFFLOAD_FAMILIES = frozenset({"krea2"})


def _krea2_guidance_scale(guidance: float) -> float:
    """Convert a standard CFG scale (1.0 = off) to Krea 2's ``guidance_scale`` (0.0 = off).

    Krea 2's pipeline computes ``cond + g * (cond - uncond)``, which is standard CFG at ``1 + g``.
    """
    return max(guidance - 1.0, 0.0)


# Krea2Pipeline's fixed timestep shift for Krea 2 Turbo. Copied from diffusers 0.41's Krea2Pipeline.__call__,
# which does not expose it: recheck it when upgrading diffusers.
_KREA2_TURBO_MU = 1.15


def _img2img_sigmas(steps: int, strength: float) -> list[float]:
    """Return the tail of Krea 2's unshifted sigma grid that an img2img run at *strength* denoises over.

    The grid is the one Krea2Pipeline builds itself (``linspace(1, 1/steps, steps)``). Like diffusers' img2img
    pipelines, ``strength`` skips the first ``int(steps - steps * strength)`` entries and always keeps at least one;
    the workflow inflates ``steps`` by ``1 / strength`` and relies on this rounding to run the steps the user asked for.
    """
    grid = [1.0 - i / steps for i in range(steps)]
    run = min(max(steps - int(max(steps - steps * strength, 0.0)), 1), steps)
    return grid[steps - run :]


def _krea2_size(pipe: Any, image: Image.Image) -> tuple[int, int]:
    """Return *image*'s ``(width, height)`` rounded down to the pixel multiple Krea 2's latent patches need."""
    multiple = pipe.vae_scale_factor * pipe.patch_size
    return max(image.width // multiple, 1) * multiple, max(image.height // multiple, 1) * multiple


def _krea2_encode_image(pipe: Any, image: Image.Image, width: int, height: int) -> torch.Tensor:
    """Encode *image* to the packed, normalized latents Krea2Pipeline denoises."""
    vae = pipe.vae
    pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(device=pipe._execution_device, dtype=vae.dtype)
    # The Qwen-Image VAE encodes video-shaped (batch, channels, frames, height, width) input.
    encoded = vae.encode(pixels.unsqueeze(2)).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(encoded)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(encoded)
    normalized = ((encoded - mean) / std)[:, :, 0]
    batch, channels, latent_height, latent_width = normalized.shape
    return pipe._pack_latents(normalized, batch, channels, latent_height, latent_width)


def _krea2_start_sigma(pipe: Any, sigmas: list[float]) -> float:
    """Return the noise level Krea2Pipeline starts *sigmas* at once its scheduler applies Turbo's timestep shift."""
    scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    scheduler.set_timesteps(sigmas=sigmas, mu=_KREA2_TURBO_MU)
    return float(scheduler.sigmas[0])


def _krea2_img2img_latents(pipe: Any, image: Image.Image, sigmas: list[float], width: int, height: int, generator: torch.Generator | None) -> torch.Tensor:
    """Return *image*'s latents noised to the first of *sigmas*, ready to pass to Krea2Pipeline as ``latents``."""
    clean = _krea2_encode_image(pipe, image, width, height)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device=clean.device, dtype=clean.dtype)
    sigma = _krea2_start_sigma(pipe, sigmas)
    return (1.0 - sigma) * clean + sigma * noise


def _load_nf4(model_path: str, torch_dtype: torch.dtype, family: str, *, prequantized: bool):
    """Load the pipeline with its transformer and text encoder in NF4 on the GPU (q4)."""
    offload = family in _Q4_CPU_OFFLOAD_FAMILIES
    components = load_nf4_components(model_path, torch_dtype, prequantized=prequantized, text_encoder_to_cpu=offload)
    pipeline = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch_dtype, **components)
    if offload:
        # Model CPU offload moves the text encoder and transformer onto the GPU in turn.
        pipeline.enable_model_cpu_offload()
    pipeline.vae.to(device="cuda", dtype=torch_dtype)
    return pipeline


def _stream_to_gpu(pipeline: Any, torch_dtype: torch.dtype) -> None:
    """Keep the transformer and text encoder in system memory and stream their layers to the GPU as they run."""
    from diffusers.hooks import apply_group_offloading

    options = {"onload_device": torch.device("cuda"), "num_blocks_per_group": 1, "use_stream": True, "record_stream": True, "non_blocking": True, "low_cpu_mem_usage": True}
    apply_group_offloading(pipeline.transformer, offload_type="block_level", **options)
    apply_group_offloading(pipeline.text_encoder, offload_type=_text_encoder_offload_type(pipeline.text_encoder), **options)
    pipeline.vae.to(device="cuda", dtype=torch_dtype)


def _text_encoder_offload_type(text_encoder: Any) -> str:
    """Return how to stream *text_encoder*: block by block when its layers form a top-level list, else leaf by leaf.

    Block-level offloading only splits a top-level layer list. Qwen3-VL (Krea 2) nests its layers under
    ``language_model`` and calls its embedding directly, so at block level its whole language model would move to
    the GPU at once.
    """
    has_block_list = any(type(child).__name__ in ("ModuleList", "Sequential") for child in text_encoder.children())
    return "block_level" if has_block_list else "leaf_level"


def _load_loras(pipeline: Any, lora_paths: list[str], lora_weights: list[float] | None, *, cpu_offload: bool = False) -> None:
    """Load LoRA files onto *pipeline*: through diffusers, with LoKr layers added as adapter terms.

    A file without LoKr tensors goes to diffusers by path, so its metadata (such as ``lora_alpha``) applies. A file
    with LoKr tensors is split: the rest goes to diffusers, the LoKr layers to :func:`apply_lokr` once every
    standard LoRA has loaded, so diffusers always finds the layers it expects. LoKr layers that do not map onto the
    model are skipped with a warning, as mflux does. A file none of whose tensors match the model is skipped with a warning. When *cpu_offload*
    is set, model CPU offload is turned back on if loading a LoRA left it off.
    """
    from safetensors.torch import load_file

    from zvisiongenerator.backends.image_win_lokr import apply_lokr, split_lora_tensors

    weights = lora_weights if lora_weights is not None else [1.0] * len(lora_paths)
    lokr_files: list[tuple[str, Any, float]] = []
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for i, (path, weight) in enumerate(zip(lora_paths, weights, strict=True)):
        parts = split_lora_tensors(load_file(path)) if _has_lokr(path) else None
        if parts is not None:
            lokr_files.append((path, parts, weight))
        if (parts is None or parts.rest) and _load_lora_file(pipeline, path, parts, f"lora_{i}"):
            adapter_names.append(f"lora_{i}")
            adapter_weights.append(weight)
    if adapter_names:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    restore_cpu_offload(pipeline, enabled=cpu_offload)
    for path, parts, weight in lokr_files:
        skipped = apply_lokr(pipeline, parts.lokr, weight, pin_memory=True)
        if skipped:
            warnings.warn(f"LoRA {Path(path).name}: skipped {len(skipped)} LoKr layers this model does not have (e.g. {skipped[0]}).", stacklevel=2)


def _has_lokr(path: str) -> bool:
    """Return whether the LoRA file at *path* holds LoKr tensors, reading only its tensor names."""
    if not path.endswith(".safetensors"):
        return False
    from safetensors import safe_open

    from zvisiongenerator.backends.image_win_lokr import has_lokr_tensors

    with safe_open(path, framework="pt") as handle:
        return has_lokr_tensors(handle.keys())


def _load_lora_file(pipeline: Any, path: str, parts: Any, adapter_name: str) -> bool:
    """Load the LoRA at *path* through diffusers, or only its non-LoKr tensors when it was split into *parts*.

    When diffusers rejects tensors it cannot convert (such as LyCORIS ``diff`` deltas next to a LoRA), the file's
    standard LoRA tensors load on their own and the rest is skipped with a warning, as mflux does. A file is read
    and split for that only when it was not split already.

    Returns:
        Whether an adapter loaded.

    Raises:
        ValueError: When diffusers rejects the file and it has no other tensors to leave out.
    """
    from safetensors.torch import load_file

    from zvisiongenerator.backends.image_win_lokr import split_lora_tensors

    try:
        pipeline.load_lora_weights(path if parts is None else parts.rest, adapter_name=adapter_name)
        return _has_adapter(pipeline, adapter_name, path)
    except ValueError as exc:
        if is_unmatched_adapter_error(exc):
            warn_unmatched_lora(path)
            return False
        if parts is None and path.endswith(".safetensors"):
            parts = split_lora_tensors(load_file(path))
        if parts is None or not parts.unsupported:
            raise
    warnings.warn(f"LoRA {Path(path).name}: skipped {len(parts.unsupported)} tensors of an unsupported type (e.g. {parts.unsupported[0]}).", stacklevel=3)
    if not parts.lora:
        return False
    try:
        pipeline.load_lora_weights(parts.lora, adapter_name=adapter_name)
    except ValueError as exc:
        if not is_unmatched_adapter_error(exc):
            raise
        warn_unmatched_lora(path)
        return False
    return _has_adapter(pipeline, adapter_name, path)


def _has_adapter(pipeline: Any, adapter_name: str, path: str) -> bool:
    """Return whether diffusers registered *adapter_name*, warning when it loaded nothing from *path*.

    diffusers skips tensors no component of this model uses without raising, so a file can load no adapter.
    """
    if any(adapter_name in names for names in pipeline.get_list_adapters().values()):
        return True
    warn_unmatched_lora(path)
    return False


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

    Holds only the loaded model's detected info. It keeps no pipeline between calls: the backend lives for the
    whole process, so a cached pipeline would keep a finished job's model in memory.
    """

    name = "diffusers"

    def __init__(self):
        self._model_info: ImageModelInfo | None = None

    def stored_quant_format(self, bits: int) -> str | None:
        """Return the format tag of stored quants at *bits*: the versions of the libraries that write and read them.

        FP8 (q8) copies depend on diffusers and transformers; NF4 (q4) copies also on bitsandbytes. Other levels
        are not quantized on this backend, so they are not stored (``None``).
        """
        if bits not in STORED_QUANT_BITS:
            return None
        packages = ("diffusers", "transformers", "bitsandbytes") if bits == 4 else ("diffusers", "transformers")
        return "+".join(f"{package}-{importlib.metadata.version(package)}" for package in packages)

    def quantizes_from_files(self, bits: int) -> bool:
        """Return whether *bits* is stored from the source files: FP8 (q8) is, NF4 (q4) is saved from a loaded model."""
        return bits == 8

    def save_quantized(self, model: Any, path: str) -> None:
        """Write a pipeline loaded at q4 (NF4, no LoRAs) to *path* in diffusers' pre-quantized format."""
        model.save_pretrained(path)

    def write_quantized_files(self, source: str, path: str, bits: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Write the FP8 (q8) copy of the model at *source* into *path*, one component at a time.

        Raises:
            ValueError: For a level that is saved from a loaded model instead (q4).
        """
        if not self.quantizes_from_files(bits):
            raise ValueError(f"q{bits} copies are saved from a loaded model, not written from files.")
        write_fp8_copy(Path(source), Path(path), torch.bfloat16, cancelled)

    def load_model(
        self,
        model_path: str,
        quantize: int | None = None,
        precision: str = "bfloat16",
        lora_paths: list[str] | None = None,
        lora_weights: list[float] | None = None,
    ) -> tuple[Any, "ImageModelInfo"]:
        if not torch.cuda.is_available():
            message = "CUDA is not available. The diffusers/CUDA image backend requires an NVIDIA GPU with CUDA support on Windows and Linux."
            hint = cuda_driver_hint(False, getattr(torch.version, "cuda", None))
            raise RuntimeError(f"{message} {hint}" if hint else message)
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

        stored_bits = stored_quant_bits(Path(model_path))
        bits = stored_bits if stored_bits is not None else quantize
        if bits not in (None, 4, 8):
            warnings.warn(f"Unsupported quantize value {bits!r}; loading at full precision. Use 4 (NF4) or 8 (FP8).", stacklevel=2)
            bits = None

        if bits == 4:
            pipeline = _load_nf4(model_path, torch_dtype, model_info.family, prequantized=stored_bits is not None)
        else:
            components = {}
            if bits == 8:
                components = load_fp8_components(model_path, torch_dtype)
            pipeline = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch_dtype, **components)
            _stream_to_gpu(pipeline, torch_dtype)

        if lora_paths:
            _load_loras(pipeline, lora_paths, lora_weights, cpu_offload=bits == 4 and model_info.family in _Q4_CPU_OFFLOAD_FAMILIES)

        # Decode in slices and tiles to bound VAE memory. Called on the VAE itself: some pipelines (Krea 2) have no
        # enable_vae_tiling wrapper and would otherwise decode the whole frame at once.
        for method in ("enable_slicing", "enable_tiling"):
            try:
                getattr(pipeline.vae, method, lambda: None)()
            except NotImplementedError:
                pass  # diffusers' base VAE class has the method but this VAE cannot tile or slice

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
            return self._krea2_image_to_image(model, image, prompt, strength, steps, seed, guidance, skip_signal, step_callback)
        from diffusers import AutoPipelineForImage2Image

        # Built per call from the loaded components (cheap): a scheduler swap stays on this pipeline only, and
        # no reference to the model outlives the job. Without a dtype, from_pipe casts the shared components to
        # float32; the VAE carries the dtype the model loaded in.
        img2img = AutoPipelineForImage2Image.from_pipe(model, torch_dtype=model.vae.dtype)
        if scheduler == "beta":
            from diffusers import FlowMatchEulerDiscreteScheduler

            img2img.scheduler = FlowMatchEulerDiscreteScheduler.from_config(img2img.scheduler.config, use_beta_sigmas=True)

        _is_flux = self._model_info.family in ("flux1", "flux2", "flux2_klein")

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

        result = img2img(**pipe_kwargs)
        torch.cuda.empty_cache()
        if skip_signal is not None and img2img._interrupt:
            return None
        return result.images[0]

    def _krea2_image_to_image(
        self,
        pipe: Any,
        image: Image.Image,
        prompt: str,
        strength: float,
        steps: int,
        seed: int | None,
        guidance: float | None,
        skip_signal: Any | None,
        step_callback: Any | None,
    ) -> Image.Image | None:
        """Run Krea 2 img2img through its text-to-image pipeline, which diffusers ships without an img2img variant.

        The image is VAE-encoded, noised to the run's first sigma and denoised over the tail of the schedule.
        """
        torch.cuda.empty_cache()
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        width, height = _krea2_size(pipe, image)
        sigmas = _img2img_sigmas(steps, strength)
        pipe_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=len(sigmas),
            sigmas=sigmas,
            latents=_krea2_img2img_latents(pipe, image, sigmas, width, height, generator),
            guidance_scale=_krea2_guidance_scale(guidance),
            generator=generator,
        )
        if skip_signal is not None or step_callback is not None:
            pipe_kwargs.update(self._step_callback_kwargs(skip_signal, step_callback, len(sigmas), height, width))

        result = pipe(**pipe_kwargs)

        torch.cuda.empty_cache()
        if skip_signal is not None and pipe._interrupt:
            pipe._interrupt = False
            return None
        return result.images[0]

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
