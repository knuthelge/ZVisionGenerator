"""Windows/Linux CUDA video backend using diffusers LTX pipelines."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from zvisiongenerator.backends.cuda_driver import cuda_driver_hint
from zvisiongenerator.backends.lora_peft import is_unmatched_adapter_error, restore_cpu_offload, warn_unmatched_lora
from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model

_MINIMUM_DIFFUSERS_VERSION = (0, 38, 0)
# Distilled checkpoints run without classifier-free guidance. Only guidance_scale is passed, so the other guidance
# terms (audio, spatio-temporal, modality) follow the pipeline defaults of the installed diffusers version.
_DISTILLED_GUIDANCE: Final[Mapping[str, float]] = MappingProxyType({"guidance_scale": 1.0})


@dataclass(frozen=True)
class _PipelineClasses:
    """Resolved diffusers LTX pipeline classes."""

    text_to_video: type[Any]
    image_to_video: type[Any]
    latent_upscaler: type[Any]


@dataclass(frozen=True)
class _RuntimeDependencies:
    """Lazy-loaded runtime dependencies for diffusers video generation."""

    torch: Any
    image_module: Any
    export_to_video: Any
    pipeline_classes: _PipelineClasses
    diffusers_version: str
    distilled_sigmas: tuple[float, ...]
    stage2_sigmas: tuple[float, ...]


@dataclass
class _LoadedVideoModel:
    """Reusable loaded diffusers video pipelines for one request mode."""

    text_to_video: Any | None
    image_to_video: Any | None
    latent_upscaler: Any | None
    runtime: _RuntimeDependencies
    model_info: VideoModelInfo
    low_memory: bool


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into comparable integer parts."""

    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _resolve_pipeline_classes(diffusers_module: Any) -> _PipelineClasses:
    """Resolve the LTX-2 pipeline classes from diffusers."""

    names = ("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2LatentUpsamplePipeline")
    if not all(hasattr(diffusers_module, name) for name in names):
        raise RuntimeError("Installed diffusers is missing the LTX-2 video pipelines (LTX2Pipeline, LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline).")
    return _PipelineClasses(
        text_to_video=diffusers_module.LTX2Pipeline,
        image_to_video=diffusers_module.LTX2ImageToVideoPipeline,
        latent_upscaler=diffusers_module.LTX2LatentUpsamplePipeline,
    )


def _validate_diffusers_version(diffusers_version: str) -> None:
    """Ensure the installed diffusers version satisfies the video backend minimum."""

    if _parse_version(diffusers_version) < _MINIMUM_DIFFUSERS_VERSION:
        minimum = ".".join(str(part) for part in _MINIMUM_DIFFUSERS_VERSION)
        raise RuntimeError(f"diffusers>={minimum} is required for Windows/Linux LTX video support. Installed: {diffusers_version}.")


def _load_runtime_dependencies() -> _RuntimeDependencies:
    """Import heavy runtime dependencies lazily for video generation."""

    try:
        import torch
        from PIL import Image
        import diffusers
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise RuntimeError("Video diffusers dependencies are unavailable. Install the Windows/Linux video runtime with `uv sync`.") from exc

    diffusers_version = getattr(diffusers, "__version__", "0")
    _validate_diffusers_version(diffusers_version)

    try:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
    except ImportError as exc:
        raise RuntimeError("Installed diffusers is missing the LTX-2 distilled sigma tables.") from exc

    return _RuntimeDependencies(
        torch=torch,
        image_module=Image,
        export_to_video=export_to_video,
        pipeline_classes=_resolve_pipeline_classes(diffusers),
        diffusers_version=diffusers_version,
        distilled_sigmas=tuple(DISTILLED_SIGMA_VALUES),
        stage2_sigmas=tuple(STAGE_2_DISTILLED_SIGMA_VALUES),
    )


def _platform_label() -> str:
    """Return a human-readable label for the active platform."""

    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "linux":
        return "Linux"
    return sys.platform


def _validate_cuda(runtime: _RuntimeDependencies) -> None:
    """Ensure the current runtime has an NVIDIA CUDA device available."""

    torch = runtime.torch
    if not torch.cuda.is_available() or getattr(getattr(torch, "version", None), "cuda", None) is None:
        message = f"CUDA is not available. The {_platform_label()} diffusers video backend requires an NVIDIA GPU with CUDA support."
        hint = cuda_driver_hint(False, getattr(getattr(torch, "version", None), "cuda", None))
        raise RuntimeError(f"{message} {hint}" if hint else message)


def _configure_torch_runtime(torch: Any) -> None:
    """Apply the same lightweight CUDA tuning used by the image diffusers backend."""
    from zvisiongenerator.backends.memory_cuda import configure_allocator

    configure_allocator()
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _select_torch_dtype(torch: Any) -> Any:
    """Choose the default CUDA dtype for video generation."""

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_bf16_supported") and cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _enable_vae_tiling(pipeline: Any) -> None:
    """Enable tiled decoding on the pipeline VAE when it supports it."""

    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def _configure_pipeline(pipeline: Any, runtime: _RuntimeDependencies, *, torch_dtype: Any, low_memory: bool) -> Any:
    """Apply offload and VAE tuning on a loaded diffusers pipeline."""

    if low_memory and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    elif hasattr(pipeline, "to"):
        pipeline.to(device="cuda", dtype=torch_dtype)

    _enable_vae_tiling(pipeline)
    return pipeline


def _load_pipeline(
    pipeline_class: type[Any],
    model_path: str,
    runtime: _RuntimeDependencies,
    *,
    torch_dtype: Any,
    low_memory: bool,
) -> Any:
    """Instantiate and configure one diffusers pipeline."""

    try:
        pipeline = pipeline_class.from_pretrained(model_path, torch_dtype=torch_dtype)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {pipeline_class.__name__} from '{model_path}': {exc}") from exc
    return _configure_pipeline(pipeline, runtime, torch_dtype=torch_dtype, low_memory=low_memory)


def _call_signature_parameters(callable_obj: Any) -> dict[str, inspect.Parameter] | None:
    """Return signature parameters for a callable when introspection succeeds."""

    try:
        return inspect.signature(callable_obj).parameters
    except TypeError, ValueError:
        return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only those supported by the target callable signature."""

    parameters = _call_signature_parameters(callable_obj)
    if parameters is None or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _refuse_review(kwargs: dict[str, Any]) -> None:
    """Raise when a review before upscaling is requested: the diffusers upscale has no half-size render to pause at.

    Raises:
        ValueError: If *kwargs* carries a ``review`` callable.
    """

    if kwargs.get("review") is not None:
        raise ValueError("Review before upscaling is not supported by the diffusers video backend.")


def _make_generator(torch: Any, seed: int | None) -> Any | None:
    """Create a deterministic generator when a seed is provided."""

    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(seed)


def _stage1_sigmas(table: Sequence[float], steps: int) -> list[float]:
    """Return the distilled sigmas for *steps* denoising steps, keeping the start and the tail of the table."""

    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if steps >= len(table):
        return list(table)
    return [table[0], *table[len(table) - steps + 1 :]]


def _make_step_callback(step_callback: Callable[[dict[str, Any]], None] | None, *, phase: str, offset: int, total: int) -> Callable[..., Any] | None:
    """Adapt diffusers callback events to the existing backend callback payload."""

    if step_callback is None:
        return None

    safe_total = max(total, 1)

    def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: Any) -> Any:
        step_callback(
            {
                "phase": phase,
                "current_step": min(offset + step + 1, safe_total),
                "total_steps": safe_total,
            }
        )
        return callback_kwargs

    return _on_step_end


def _extract_frames(result: Any) -> list[Any]:
    """Normalize diffusers video outputs to a single list of frames."""

    if result is None:
        return []
    if hasattr(result, "frames"):
        frames = result.frames
    elif hasattr(result, "images"):
        frames = result.images
    else:
        frames = result

    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        return frames[0]
    if isinstance(frames, list):
        return frames
    return []


def _export_video(model: _LoadedVideoModel, frames: list[Any], output_path: str) -> Path:
    """Export generated frames to the requested output path."""

    if not frames:
        raise RuntimeError("Diffusers video generation returned no frames to export.")
    model.runtime.export_to_video(frames, output_video_path=output_path, fps=model.model_info.default_fps)
    return Path(output_path)


def _apply_loras(pipeline: Any, loras: list[tuple[str, float]], *, cpu_offload: bool = False) -> None:
    """Load diffusers-compatible LoRA adapters onto the active generation pipeline.

    A LoRA none of whose tensors match the pipeline is skipped with a warning. When *cpu_offload* is set, model CPU
    offload is turned back on if loading a LoRA left it off.
    """

    if not hasattr(pipeline, "load_lora_weights"):
        raise RuntimeError("The installed diffusers LTX pipeline does not support LoRA adapters.")

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    load_signature = _call_signature_parameters(pipeline.load_lora_weights) or {}
    supports_adapter_name = "adapter_name" in load_signature or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in load_signature.values())

    for index, (path, weight) in enumerate(loras):
        lora_path = Path(path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {path}")
        if lora_path.is_file() and lora_path.suffix.lower() not in {".safetensors", ".bin"}:
            raise ValueError(f"Unsupported LoRA format for diffusers LTX backend: {path}")

        adapter_name = f"lora_{index}"
        load_kwargs = {"adapter_name": adapter_name} if supports_adapter_name else {}
        try:
            pipeline.load_lora_weights(str(lora_path), **load_kwargs)
        except TypeError as exc:
            raise RuntimeError(f"The installed diffusers LoRA API is incompatible with '{path}'.") from exc
        except Exception as exc:
            if is_unmatched_adapter_error(exc):
                warn_unmatched_lora(path)
                continue
            raise RuntimeError(f"Failed to load LoRA '{path}': {exc}") from exc

        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    restore_cpu_offload(pipeline, enabled=cpu_offload)
    if not adapter_names:
        return

    if hasattr(pipeline, "set_adapters"):
        set_kwargs = _filter_supported_kwargs(
            pipeline.set_adapters,
            {"adapter_names": adapter_names, "adapter_weights": adapter_weights},
        )
        if any(weight != 1.0 for weight in adapter_weights) and "adapter_weights" not in set_kwargs:
            raise RuntimeError("The installed diffusers adapter API does not support explicit LoRA scales.")
        try:
            pipeline.set_adapters(**set_kwargs)
        except TypeError:
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        except Exception as exc:
            if any(weight != 1.0 for weight in adapter_weights):
                raise RuntimeError("The installed diffusers adapter API does not support explicit LoRA scales.") from exc
            raise RuntimeError(f"Failed to activate LoRA adapters: {exc}") from exc
        return

    if len(adapter_names) > 1 or any(weight != 1.0 for weight in adapter_weights):
        raise RuntimeError("The installed diffusers adapter API does not support multiple LoRAs or explicit LoRA scales.")


def _cleanup(runtime: _RuntimeDependencies) -> None:
    """Release transient CUDA allocations after generation."""

    if hasattr(runtime.torch.cuda, "empty_cache"):
        runtime.torch.cuda.empty_cache()


def _build_generation_kwargs(
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    sigmas: Sequence[float],
    generator: Any,
    output_type: str,
    callback: Callable[..., Any] | None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map the shared backend contract to diffusers pipeline kwargs."""

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": fps,
        "sigmas": list(sigmas),
        "num_inference_steps": len(sigmas),
        "generator": generator,
        "output_type": output_type,
        **_DISTILLED_GUIDANCE,
    }
    if callback is not None:
        kwargs["callback_on_step_end"] = callback
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _generate_one_stage(
    model: _LoadedVideoModel,
    pipeline: Any,
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    steps: int,
    step_callback: Any | None,
    extra_kwargs: Mapping[str, Any],
    phase: str = "video",
) -> list[Any]:
    """Run one distilled denoising pass at full size and return its frames."""

    sigmas = _stage1_sigmas(model.runtime.distilled_sigmas, steps)
    kwargs = _build_generation_kwargs(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=model.model_info.default_fps,
        sigmas=sigmas,
        generator=_make_generator(model.runtime.torch, seed),
        output_type="pil",
        callback=_make_step_callback(step_callback, phase=phase, offset=0, total=len(sigmas)),
        extra_kwargs=extra_kwargs,
    )
    return _extract_frames(pipeline(**kwargs))


class DiffusersVideoBackend:
    """Shared CUDA-backed diffusers LTX video backend for Windows and Linux."""

    name = "ltx"

    def load_model(
        self,
        model_path: str,
        **kwargs: Any,
    ) -> tuple[_LoadedVideoModel, VideoModelInfo]:
        """Load the active diffusers pipeline(s) for one video request mode."""

        mode = kwargs.get("mode", "t2v")
        low_memory = bool(kwargs.get("low_memory", True))
        loras = kwargs.get("loras") or []
        upscale = bool(kwargs.get("upscale"))

        if mode not in {"t2v", "i2v"}:
            raise ValueError(f"Unsupported video generation mode: {mode}")

        runtime = _load_runtime_dependencies()
        _validate_cuda(runtime)
        _configure_torch_runtime(runtime.torch)
        torch_dtype = _select_torch_dtype(runtime.torch)

        model_info = detect_video_model(model_path)
        if model_info.family != "ltx":
            raise RuntimeError(f"Unsupported diffusers video model: '{model_path}'.")

        text_pipeline = _load_pipeline(runtime.pipeline_classes.text_to_video, model_path, runtime, torch_dtype=torch_dtype, low_memory=low_memory) if mode == "t2v" else None
        image_pipeline = _load_pipeline(runtime.pipeline_classes.image_to_video, model_path, runtime, torch_dtype=torch_dtype, low_memory=low_memory) if mode == "i2v" else None

        if text_pipeline is None and image_pipeline is None:
            raise RuntimeError(f"No diffusers generation pipeline was loaded for mode '{mode}'.")

        active_pipeline = text_pipeline or image_pipeline
        if loras:
            _apply_loras(active_pipeline, loras, cpu_offload=low_memory)

        latent_upscaler = None
        if upscale:
            latent_upscaler = _load_pipeline(runtime.pipeline_classes.latent_upscaler, model_path, runtime, torch_dtype=torch_dtype, low_memory=low_memory)

        return (
            _LoadedVideoModel(
                text_to_video=text_pipeline,
                image_to_video=image_pipeline,
                latent_upscaler=latent_upscaler,
                runtime=runtime,
                model_info=model_info,
                low_memory=low_memory,
            ),
            model_info,
        )

    def text_to_video(
        self,
        model: _LoadedVideoModel,
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
        """Generate video frames from text and export them to disk."""

        if model.text_to_video is None:
            raise RuntimeError("This loaded video model does not include a text-to-video pipeline.")

        _refuse_review(kwargs)
        stage1_steps = kwargs.get("stage1_steps")
        if model.latent_upscaler is not None and stage1_steps is None:
            raise ValueError("stage1_steps must be provided when upscale is enabled")
        if model.latent_upscaler is None and stage1_steps is not None:
            raise RuntimeError("Upscale was requested, but the diffusers latent upscaler is unavailable for this model.")

        base_steps = stage1_steps if stage1_steps is not None else steps

        with model.runtime.torch.inference_mode():
            frames = _generate_one_stage(
                model,
                model.text_to_video,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                seed=seed,
                steps=base_steps,
                step_callback=step_callback,
                extra_kwargs={},
                phase="video_upscale_stage_1" if stage1_steps is not None else "video",
            )
            if model.latent_upscaler is not None:
                frames = self._upscale_video(
                    model,
                    frames=frames,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    seed=seed,
                    base_steps=base_steps,
                    step_callback=step_callback,
                )
            output = _export_video(model, frames, output_path)
        _cleanup(model.runtime)
        return output

    def image_to_video(
        self,
        model: _LoadedVideoModel,
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
        """Generate video frames from an image prompt pair and export them to disk."""

        if model.image_to_video is None:
            raise RuntimeError("This loaded video model does not include an image-to-video pipeline.")

        _refuse_review(kwargs)
        stage1_steps = kwargs.get("stage1_steps")
        if model.latent_upscaler is not None and stage1_steps is None:
            raise ValueError("stage1_steps must be provided when upscale is enabled")
        if model.latent_upscaler is None and stage1_steps is not None:
            raise RuntimeError("Upscale was requested, but the diffusers latent upscaler is unavailable for this model.")

        base_steps = stage1_steps if stage1_steps is not None else steps
        with model.runtime.image_module.open(image_path) as image:
            input_image = image.convert("RGB")

        with model.runtime.torch.inference_mode():
            frames = _generate_one_stage(
                model,
                model.image_to_video,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                seed=seed,
                steps=base_steps,
                step_callback=step_callback,
                extra_kwargs={"image": input_image},
                phase="video_upscale_stage_1" if stage1_steps is not None else "video",
            )
            if model.latent_upscaler is not None:
                frames = self._upscale_video(
                    model,
                    frames=frames,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    seed=seed,
                    base_steps=base_steps,
                    step_callback=step_callback,
                )
            output = _export_video(model, frames, output_path)
        _cleanup(model.runtime)
        return output

    def _upscale_video(
        self,
        model: _LoadedVideoModel,
        *,
        frames: list[Any],
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        base_steps: int,
        step_callback: Any | None,
    ) -> list[Any]:
        """Run the optional latent upscaler and emit a terminal progress event."""

        if model.latent_upscaler is None:
            raise RuntimeError("Upscale was requested, but no latent upscaler pipeline is loaded.")

        upscale_kwargs = {
            "video": frames,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "generator": _make_generator(model.runtime.torch, seed),
            "output_type": "pil",
        }
        result = model.latent_upscaler(**_filter_supported_kwargs(model.latent_upscaler, upscale_kwargs))
        if step_callback is not None:
            step_callback(
                {
                    "phase": "video_upscale_stage_2",
                    "current_step": max(base_steps, 1) + 1,
                    "total_steps": max(base_steps, 1) + 1,
                }
            )
        return _extract_frames(result)
