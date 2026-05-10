"""macOS image backend using mflux and MLX."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from PIL import Image
from mflux.models.flux2 import Flux2Klein
from mflux.models.z_image import ZImageTurbo
from mflux.models.common.config.model_config import ModelConfig
from mflux.utils.exceptions import StopImageGenerationException
import mlx.core as mx
from mlx.utils import tree_map

from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model


class _SkipChecker:
    """InLoopCallback that aborts generation when skip is requested."""

    def __init__(self, skip_signal):
        self._skip_signal = skip_signal

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self._skip_signal.check():
            raise StopImageGenerationException("Skipped by user")


class _ProgressChecker:
    """InLoopCallback that reports denoising progress for each step."""

    def __init__(self, total_steps: int, step_callback):
        self._total_steps = max(total_steps, 1)
        self._step_callback = step_callback
        self._current_step = 0

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        del t, seed, prompt, latents, config, time_steps
        self._current_step = min(self._current_step + 1, self._total_steps)
        self._step_callback(
            {
                "current_step": self._current_step,
                "total_steps": self._total_steps,
            }
        )


def _upcast_model_weights(model, components):
    """Cast model component weights to float32."""

    def to_float32(p):
        return p.astype(mx.float32) if isinstance(p, mx.array) else p

    for name in components:
        component = getattr(model, name, None)
        if component is not None:
            component.update(tree_map(to_float32, component.parameters()))


class MfluxBackend:
    """mflux/MLX backend for macOS — implements ImageBackend Protocol."""

    name = "mflux"

    def __init__(self):
        self._model_info: ImageModelInfo | None = None

    def load_model(
        self,
        model_path: str,
        quantize: int | None = None,
        precision: str = "bfloat16",
        lora_paths: list[str] | None = None,
        lora_weights: list[float] | None = None,
    ) -> tuple[Any, "ImageModelInfo"]:
        """Load a model from the given path with optional quantization, precision, and LoRA.

        Args:
            model_path: Path to the model directory.
            quantize: Quantization level (None, 4, or 8).
            precision: "bfloat16" (default, fast) or "float32" (slower, better detail).
            lora_paths: List of paths to LoRA .safetensors files, or None to disable.
            lora_weights: List of scale factors for each LoRA (default None).
        """
        model_info = detect_image_model(model_path)

        ModelConfig.precision = mx.float32 if precision == "float32" else mx.bfloat16

        lora_kwargs = {}
        if lora_paths:
            lora_kwargs["lora_paths"] = lora_paths
            lora_kwargs["lora_scales"] = lora_weights

        if model_info.family == "flux2_klein":
            if model_info.size is None:
                raise ValueError("Could not determine Klein model size. Specify the model explicitly or check the model files.")
            if model_info.is_distilled:
                config = ModelConfig.flux2_klein_4b() if model_info.size == "4b" else ModelConfig.flux2_klein_9b()
            else:
                config = ModelConfig.flux2_klein_base_4b() if model_info.size == "4b" else ModelConfig.flux2_klein_base_9b()

            model = Flux2Klein(
                quantize=quantize,
                model_path=model_path,
                model_config=config,
                **lora_kwargs,
            )
        elif model_info.family == "zimage":
            model = ZImageTurbo(
                quantize=quantize,
                model_path=model_path,
                model_config=ModelConfig.z_image(),
                **lora_kwargs,
            )
        else:
            raise ValueError(f"Model family '{model_info.family}' is not supported by the mflux backend. Supported families: zimage, flux2_klein")

        if precision == "float32":
            _upcast_model_weights(model, ["transformer", "text_encoder", "vae"])
        else:
            # Even in bfloat16 mode, upcast VAE for better color gradients
            _upcast_model_weights(model, ["vae"])

        self._model_info = model_info
        return model, model_info

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
        # mflux requires a file path, not a PIL Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            temp_path = f.name

        checker = None
        progress_checker = None
        if skip_signal is not None:
            checker = _SkipChecker(skip_signal)
            model.callbacks.register(checker)
        if step_callback is not None:
            progress_checker = _ProgressChecker(steps, step_callback)
            model.callbacks.register(progress_checker)

        try:
            image_strength = 1.0 - strength  # invert: mflux convention

            if seed is not None:
                mx.random.seed(seed)

            _is_flux = self._model_info is not None and self._model_info.family in ("flux1", "flux2", "flux2_klein")

            gen_kwargs = dict(
                prompt=prompt,
                width=image.width,
                height=image.height,
                seed=seed,
                num_inference_steps=steps,
                image_path=temp_path,
                image_strength=image_strength,
                guidance=guidance if guidance is not None else (1.0 if _is_flux else 0.0),
            )
            if scheduler is not None:
                gen_kwargs["scheduler"] = scheduler
            if not _is_flux and negative_prompt is not None:
                gen_kwargs["negative_prompt"] = negative_prompt

            result = model.generate_image(**gen_kwargs)
            return result.image
        except StopImageGenerationException:
            return None
        finally:
            if checker is not None:
                try:
                    model.callbacks.in_loop.remove(checker)
                except ValueError:
                    pass
            if progress_checker is not None:
                try:
                    model.callbacks.in_loop.remove(progress_checker)
                except ValueError:
                    pass
            try:
                os.unlink(temp_path)
            except OSError:
                pass

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
    ) -> Image.Image | None:
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        # Seed the global MLX RNG so that ancestral scheduler noise injection
        # is deterministic per seed (mflux only seeds the initial latent noise
        # with an explicit key, not the global state).
        mx.random.seed(seed)

        checker = None
        progress_checker = None
        if skip_signal is not None:
            checker = _SkipChecker(skip_signal)
            model.callbacks.register(checker)
        if step_callback is not None:
            progress_checker = _ProgressChecker(steps, step_callback)
            model.callbacks.register(progress_checker)

        try:
            _is_flux = self._model_info is not None and self._model_info.family in ("flux1", "flux2", "flux2_klein")

            gen_kwargs = dict(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                num_inference_steps=steps,
                guidance=guidance if guidance is not None else (1.0 if _is_flux else 0.0),
            )
            if scheduler is not None:
                gen_kwargs["scheduler"] = scheduler
            if not _is_flux and negative_prompt is not None:
                gen_kwargs["negative_prompt"] = negative_prompt

            result = model.generate_image(**gen_kwargs)
            return result.image
        except StopImageGenerationException:
            return None
        finally:
            if checker is not None:
                try:
                    model.callbacks.in_loop.remove(checker)
                except ValueError:
                    pass
            if progress_checker is not None:
                try:
                    model.callbacks.in_loop.remove(progress_checker)
                except ValueError:
                    pass
