"""macOS image backend using mflux and MLX."""

from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import os
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterator
from typing import Any

from PIL import Image
from mflux.models.flux2 import Flux2Klein
from mflux.models.z_image import ZImageTurbo
from mflux.models.ideogram4 import Ideogram4
from mflux.models.ideogram4.model.ideogram4_scheduler.scheduler import Ideogram4Scheduler
from mflux.models.krea2 import Krea2
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import FlowMatchEulerDiscreteScheduler, LinearScheduler
from mflux.utils.exceptions import StopImageGenerationException
import mlx.core as mx
from mlx.utils import tree_map

from zvisiongenerator.backends.image_mac_preview import estimate_clean_latents, render_latent_preview
from zvisiongenerator.core.progress_events import preview_milestone_steps
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model

# First-step sigma override applied to every Ideogram 4 generation to reduce
# false positives from the safety filter on harmless prompts (the grey
# "Image blocked by safety filter" frame).
# Overrides only the first denoising step's timestep: t_values[-1] = 1.0 - value.
# Best-effort mitigation (not guaranteed); retune or set to None to disable.
IDEOGRAM4_INITIAL_SIGMA: float | None = 1.004

# Per-run override slot, distinct from the always-on default constant above.
# The sentinel distinguishes "no override" (use the default) from "override to
# None" (disable the sigma adjustment for one run).
# The override is thread-local so concurrent web generations (the web runner uses a
# ThreadPoolExecutor) are isolated: each thread only sees its own override, and the
# always-on default (no override set on a thread) uses the module constant.
_INITIAL_SIGMA_UNSET = object()
_initial_sigma_override = threading.local()


# Last Ideogram 4 ``(t_values, s_values)`` built on this thread, so live previews can read
# each step's noise levels (they are locals of mflux's denoising loop).
_ideogram4_timesteps = threading.local()


def _effective_initial_sigma() -> float | None:
    """Return the sigma the shim should apply: the per-thread override when set, else the default."""
    value = getattr(_initial_sigma_override, "value", _INITIAL_SIGMA_UNSET)
    return IDEOGRAM4_INITIAL_SIGMA if value is _INITIAL_SIGMA_UNSET else value


@contextlib.contextmanager
def _use_initial_sigma(value: float | None) -> Iterator[None]:
    """Temporarily override the Ideogram 4 first-step sigma for the current thread, restoring the prior state on exit.

    The override is stored thread-locally so concurrent generations do not race on a
    shared slot.

    Args:
        value: Sigma to apply for the duration of the block. ``None`` disables the
            adjustment for that run without touching the always-on default constant.
    """
    prev = getattr(_initial_sigma_override, "value", _INITIAL_SIGMA_UNSET)
    _initial_sigma_override.value = value
    try:
        yield
    finally:
        _initial_sigma_override.value = prev


def _install_ideogram4_initial_sigma() -> None:
    """Wrap ``Ideogram4Scheduler.make_timesteps`` to override the first denoising step's timestep.

    Ideogram 4 is the only family that uses ``make_timesteps``, so wrapping it here
    does not affect other families (zimage/flux). The wrapper reads the effective sigma
    dynamically at call time via ``_effective_initial_sigma()`` (the per-run override when
    set, else the ``IDEOGRAM4_INITIAL_SIGMA`` default): when it is ``None`` the schedule is
    returned unchanged; otherwise it copies ``t_values`` and sets ``t_values[-1] = 1.0 - sigma``,
    leaving ``s_values`` and every other ``t_values`` entry untouched.

    Either way the final ``(t_values, s_values)`` is recorded in ``_ideogram4_timesteps`` for the
    current thread: live previews read each step's noise levels from it, so the recording must
    stay even when no sigma override applies.

    Idempotent: re-importing this module does not double-wrap the staticmethod.
    """
    original = Ideogram4Scheduler.make_timesteps
    if getattr(original, "_ziv_initial_sigma_wrapped", False):
        return

    def make_timesteps(**kwargs):
        t_values, s_values = original(**kwargs)
        sigma = _effective_initial_sigma()
        if sigma is not None:
            t_values = t_values.copy()
            t_values[-1] = 1.0 - sigma
        _ideogram4_timesteps.value = (t_values, s_values)
        return t_values, s_values

    make_timesteps._ziv_initial_sigma_wrapped = True
    Ideogram4Scheduler.make_timesteps = staticmethod(make_timesteps)


# Ideogram4-only: activate the first-step sigma override for the whole process.
_install_ideogram4_initial_sigma()

# Krea 2 Turbo samples with plain Euler, as Krea's own inference code does. mflux defaults to er_sde, which adds
# fresh noise every step and leaves grain when refining an existing image (img2img, upscale).
_KREA2_SAMPLER = "euler"


def _krea2_turbo_config() -> ModelConfig:
    """Return mflux's Krea 2 config with Turbo's fixed timestep shift (mu = 1.15), the shift Krea trained it at.

    mflux derives the shift from the image size instead; at upscale sizes (about 4 MP, mu above 2) an img2img
    refinement then starts from far more noise than its denoise asks for.
    """
    config = copy.copy(ModelConfig.krea2())
    config.sigma_base_shift = config.sigma_max_shift  # equal endpoints: the shift no longer depends on image size
    return config


def _unregister_callback(model: Any, callback: Any) -> None:
    """Remove a callback from every mflux registry list so it never fires on later runs of a cached model."""
    registry = model.callbacks
    for callbacks in (registry.before_loop, registry.in_loop, registry.after_loop, registry.interrupt):
        try:
            callbacks.remove(callback)
        except ValueError:
            pass


class _SkipChecker:
    """InLoopCallback that aborts generation when skip is requested."""

    def __init__(self, skip_signal):
        self._skip_signal = skip_signal

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **_):
        if self._skip_signal.check():
            raise StopImageGenerationException("Skipped by user")


def _is_euler_scheduler(scheduler: Any) -> bool:
    """Return whether a scheduler's steps are plain flow-matching Euler updates over ``scheduler.sigmas``.

    Live previews can only recover the predicted clean image from those; others preview the raw latents.
    """
    if isinstance(scheduler, (LinearScheduler, FlowMatchEulerDiscreteScheduler)):
        return True
    # Never import the beta scheduler here (it pulls in SciPy mid-loop): if its module is not loaded yet,
    # this scheduler cannot be one.
    beta_module = sys.modules.get("zvisiongenerator.schedulers.beta_scheduler")
    return beta_module is not None and isinstance(scheduler, beta_module.BetaScheduler)


def _step_noise_levels(family: str, t: int, config) -> tuple[float, float] | None:
    """Return the noise levels before and after denoising step ``t``, or ``None`` when unknown."""
    if family == "ideogram4":
        timesteps = getattr(_ideogram4_timesteps, "value", None)
        if timesteps is None:
            return None
        # Ideogram 4 walks its schedule backwards and uses t = 1 - noise level.
        t_values, s_values = timesteps
        if len(t_values) != config.num_inference_steps:
            return None  # recorded schedule belongs to another run
        index = len(t_values) - 1 - t
        return 1.0 - float(t_values[index]), 1.0 - float(s_values[index])
    if not _is_euler_scheduler(config.scheduler):
        return None
    sigmas = config.scheduler.sigmas
    return float(sigmas[t]), float(sigmas[t + 1])


class _ProgressChecker:
    """Before/in-loop callback that reports denoising progress for each step.

    When ``model`` and ``family`` are given, milestone steps also carry a cheap
    preview of the predicted final image under the ``preview`` key.
    """

    def __init__(self, total_steps: int, step_callback, *, model: Any = None, family: str | None = None):
        self._total_steps = max(total_steps, 1)
        self._step_callback = step_callback
        self._current_step = 0
        self._model = model
        self._family = family
        self._previews_enabled = family is not None
        self._previous_latents = None
        self._preview_steps: frozenset[int] = frozenset()

    # ``**_``: newer mflux passes extra keywords (0.20 added ``control_images``); a strict signature fails every generation.
    def call_before_loop(self, seed, prompt, latents, config, **_):
        del seed, prompt
        # img2img, refine and upscale runs start part-way through the schedule; count only the steps that run.
        try:
            self._total_steps = max(config.num_inference_steps - config.init_time_step, 1)
        except AttributeError, TypeError:
            pass
        if not self._previews_enabled:
            return
        self._previous_latents = latents
        try:
            # img2img, refine and upscale runs start part-way through the schedule; place milestones within the steps actually run.
            start = config.init_time_step
            self._preview_steps = frozenset(start + step for step in preview_milestone_steps(config.num_inference_steps - start))
        except Exception as exc:  # noqa: BLE001 - previews are best-effort
            warnings.warn(f"Live previews disabled for {self._family}: {exc}", stacklevel=2)
            self._previews_enabled = False

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None, **_):
        del seed, prompt, time_steps
        # mflux calls this before it evaluates the step. Evaluate it first so every report is a finished step, and
        # send a preview as a second report so its decode never holds this one back (steps would arrive in pairs).
        mx.eval(latents)
        self._current_step = min(self._current_step + 1, self._total_steps)
        payload = {
            "current_step": self._current_step,
            "total_steps": self._total_steps,
        }
        self._step_callback(payload)
        preview = self._render_preview(t, latents, config, denoised)
        # Keep these latents only when the next step renders a preview from them; a reported prediction needs none.
        self._previous_latents = latents if self._previews_enabled and denoised is None and t + 2 in self._preview_steps else None
        if preview is not None:
            self._step_callback({**payload, "preview": preview})

    def _render_preview(self, t, latents, config, denoised=None) -> Image.Image | None:
        """Render a preview at milestone steps; never let a preview failure stop generation.

        Samplers that report their clean-image prediction (``denoised``, e.g. Krea 2's) preview it directly.
        """
        if not self._previews_enabled or t + 1 not in self._preview_steps:
            return None
        try:
            if denoised is not None:
                latents = denoised
            elif self._previous_latents is not None and (levels := _step_noise_levels(self._family, t, config)) is not None:
                latents = estimate_clean_latents(self._previous_latents, latents, *levels)
            return render_latent_preview(self._model, self._family, latents, config.height, config.width)
        except Exception as exc:  # noqa: BLE001 - previews are best-effort
            warnings.warn(f"Live preview failed for {self._family}: {exc}", stacklevel=2)
            self._previews_enabled = False
            return None


def _wrap_ideogram4_prompt(prompt: str) -> str | dict[str, Any]:
    """Wrap a plain-text prompt into Ideogram 4's minimal structured JSON caption.

    Prompts that already look like JSON captions (start with "{") are returned
    unchanged so callers retain full control.

    Args:
        prompt: The user-supplied prompt text.

    Returns:
        The original prompt when it is already a JSON caption, otherwise a
        minimal structured caption dict that mflux serializes without emitting
        its plain-text caption warning.
    """
    if prompt.strip().startswith("{"):
        return prompt
    return {
        "high_level_description": prompt,
        "compositional_deconstruction": {"background": prompt, "elements": []},
    }


def _upcast_model_weights(model, components):
    """Cast model component weights to float32."""

    def to_float32(p):
        if isinstance(p, mx.array) and p.dtype in (mx.bfloat16, mx.float16):
            return p.astype(mx.float32)
        return p

    for name in components:
        component = getattr(model, name, None)
        if component is not None:
            component.update(tree_map(to_float32, component.parameters()))


# Freed buffers MLX may keep for reuse. Its default (the whole memory limit) let the cache grow to 16-29 GB
# alongside the model and pushed macOS into swap; 4 GB showed no step slowdown.
_BUFFER_CACHE_LIMIT_BYTES = 4 * 1024**3


def _materialize_weights(model: Any) -> None:
    """Evaluate every parameter once so the weights stay resident.

    mflux loads weights lazily; unevaluated, every generation rebuilds them from the safetensors
    files (and quantizes them again), costing tens of seconds and a near-bf16 memory peak per image.
    """
    mx.eval(model.parameters())


def _apply_buffer_cache_policy() -> None:
    """Cap MLX's free-buffer cache and drop what loading left behind."""
    mx.set_cache_limit(_BUFFER_CACHE_LIMIT_BYTES)
    mx.clear_cache()


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
        elif model_info.family == "ideogram4":
            # Ideogram 4 transformer is FP8-locked; quantize is not forwarded.
            model = Ideogram4(
                model_path=model_path,
                model_config=ModelConfig.ideogram4_fp8(),
                **lora_kwargs,
            )
        elif model_info.family == "krea2":
            model = Krea2(
                quantize=quantize,
                model_path=model_path,
                model_config=_krea2_turbo_config(),
                **lora_kwargs,
            )
        else:
            raise ValueError(f"Model family '{model_info.family}' is not supported by the mflux backend. Supported families: zimage, flux2_klein, ideogram4, krea2")

        if precision == "float32":
            _upcast_model_weights(model, ["transformer", "text_encoder", "vae"])
        else:
            # Even in bfloat16 mode, upcast VAE for better color gradients
            _upcast_model_weights(model, ["vae"])

        _materialize_weights(model)
        _apply_buffer_cache_policy()

        self._model_info = model_info
        return model, model_info

    def stored_quant_format(self, bits: int) -> str | None:
        """Return the format tag of quantized weights this backend saves: the mflux version, at every level."""
        return f"mflux-{importlib.metadata.version('mflux')}"

    def quantizes_from_files(self, bits: int) -> bool:
        """Return ``False``: mflux saves every level from a model loaded at that level."""
        return False

    def write_quantized_files(self, source: str, path: str, bits: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Raise: mflux saves stored quants from a loaded model (see :meth:`save_quantized`)."""
        raise NotImplementedError("mflux saves stored quants from a loaded model.")

    def save_quantized(self, model: Any, path: str) -> None:
        """Write a loaded, quantized model's weights to *path* in mflux's own format.

        Args:
            model: A model returned by :meth:`load_model` with a quantize level and no LoRAs
                (mflux bakes LoRAs into saved weights).
            path: Destination directory.
        """
        model.save_model(path)

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
        if self._model_info.family == "ideogram4":
            raise ValueError("img2img is not supported for Ideogram 4.")
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
            progress_checker = _ProgressChecker(steps, step_callback, model=model, family=self._model_info.family)
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
            if self._model_info.family == "krea2":
                gen_kwargs["scheduler"] = _KREA2_SAMPLER
            if not _is_flux and negative_prompt is not None:
                gen_kwargs["negative_prompt"] = negative_prompt

            result = model.generate_image(**gen_kwargs)
            return result.image
        except StopImageGenerationException:
            return None
        finally:
            for callback in (checker, progress_checker):
                if callback is not None:
                    _unregister_callback(model, callback)
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            mx.clear_cache()

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
            progress_checker = _ProgressChecker(steps, step_callback, model=model, family=self._model_info.family)
            model.callbacks.register(progress_checker)

        try:
            if self._model_info.family == "ideogram4":
                # Ideogram 4 has no scheduler/negative_prompt params; omit steps/guidance to keep mflux's shaped preset schedule.
                gen_kwargs = dict(prompt=_wrap_ideogram4_prompt(prompt), width=width, height=height, seed=seed, preset="V4_DEFAULT_20")
                if steps_explicit:
                    gen_kwargs["num_inference_steps"] = steps
                if guidance_explicit:
                    gen_kwargs["guidance"] = guidance
            else:
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
                if self._model_info.family == "krea2":
                    gen_kwargs["scheduler"] = _KREA2_SAMPLER
                if not _is_flux and negative_prompt is not None:
                    gen_kwargs["negative_prompt"] = negative_prompt

            # Per-run Ideogram 4 first-step sigma override (affects the scheduler shim, not generate_image itself).
            sigma_ctx = _use_initial_sigma(first_sigma) if first_sigma is not None else contextlib.nullcontext()
            with sigma_ctx:
                result = model.generate_image(**gen_kwargs)
            return result.image
        except StopImageGenerationException:
            return None
        finally:
            # The recorded Ideogram 4 schedule belongs to this run only; never let a later run read it.
            _ideogram4_timesteps.value = None
            for callback in (checker, progress_checker):
                if callback is not None:
                    _unregister_callback(model, callback)
            mx.clear_cache()
