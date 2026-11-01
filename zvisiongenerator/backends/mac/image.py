"""macOS image backend using mflux and MLX."""

from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import sys
import threading
import warnings
from pathlib import Path
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from PIL import Image
from mflux.models.flux2 import Flux2Klein
from mflux.models.z_image import ZImageTurbo
from mflux.models.ideogram4 import Ideogram4
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.model.ideogram4_scheduler.scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.weights import Ideogram4LoRAMapping, Ideogram4WeightDefinition
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.krea2 import Krea2
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import SCHEDULER_REGISTRY, FlowMatchEulerDiscreteScheduler, LinearScheduler, try_import_external_scheduler
from mflux.utils.exceptions import StopImageGenerationException
import mlx.core as mx
from mlx import nn

from zvisiongenerator.backends.mac import moss as _moss
from zvisiongenerator.backends.mac._temp_images import remove_file, save_temp_png
from zvisiongenerator.backends.mac.image_preview import estimate_clean_latents, render_latent_preview
from zvisiongenerator.backends.mac.klein_references import generate_with_references
from zvisiongenerator.backends.mac.memory import load_resident, upcast_weights
from zvisiongenerator.backends.mac.prompt_encoding import prepare_prompts as _prepare_prompts
from zvisiongenerator.core import JobPrompt
from zvisiongenerator.core.ideogram4 import IDEOGRAM4_INITIAL_SIGMA, INSTANT_GUIDANCE, sampler_settings, wrap_ideogram4_prompt as _wrap_ideogram4_prompt
from zvisiongenerator.core.moss import lora_stack as moss_lora_stack, run_size as moss_run_size, sampler_settings as moss_sampler_settings
from zvisiongenerator.core.progress_events import preview_steps
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.lora_files import readable_lora_files
from zvisiongenerator.utils.moss_build import is_moss_dir, require_built, shared_quant_source, turbo_lora_for

# Per-run override slot, distinct from the always-on default constant (IDEOGRAM4_INITIAL_SIGMA).
# The sentinel distinguishes "no override" (use the default) from "override to
# None" (disable the sigma adjustment for one run).
# The override is thread-local, so a run only ever sees the override set on its own thread; the web runner runs one
# job at a time on its single worker thread.
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

_UNCONDITIONAL = "unconditional_transformer"


class _NoUnconditionalTransformer(nn.Module):
    """Take the place of mflux's unconditional Ideogram 4 transformer, which Instant does not have.

    mflux skips the unconditional pass at guidance 1.0, Instant's built-in guidance, so this is never called; it holds
    no weights, so nothing is loaded, quantized or saved for it.
    """

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Ideogram 4 Instant has no unconditional transformer; it runs at guidance 1.0 only.")


@contextlib.contextmanager
def _instant_ideogram4() -> Iterator[None]:
    """Build, quantize or save mflux's Ideogram 4 without its unconditional transformer, for the duration of the block.

    mflux always builds both transformers of the base model. Inside the block its weight definition omits the
    unconditional one (so it is not read from disk or written by ``save_model``), the initializer puts
    :class:`_NoUnconditionalTransformer` in its place before weights are applied, so no skeleton of 9.3B parameters
    gets quantized either, and LoRAs are applied to the conditional transformer only.
    """
    original_components = Ideogram4WeightDefinition.get_components
    original_init_models = Ideogram4Initializer._init_models
    original_apply_lora = Ideogram4Initializer._apply_lora

    def get_components() -> list[Any]:
        return [component for component in original_components() if component.name != _UNCONDITIONAL]

    def init_models(model: Any, model_path: Any) -> None:
        original_init_models(model, model_path)
        setattr(model, _UNCONDITIONAL, _NoUnconditionalTransformer())

    def apply_lora(model: Any, lora_paths: list[str] | None, lora_scales: list[float] | None, bake_lora: bool) -> None:
        # mflux also applies every LoRA to the unconditional transformer, which Instant does not have.
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Ideogram4LoRAMapping.get_mapping(),
            transformer=model.conditional_transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    Ideogram4WeightDefinition.get_components = staticmethod(get_components)
    Ideogram4Initializer._init_models = staticmethod(init_models)
    Ideogram4Initializer._apply_lora = staticmethod(apply_lora)
    try:
        yield
    finally:
        Ideogram4WeightDefinition.get_components = staticmethod(original_components)
        Ideogram4Initializer._init_models = staticmethod(original_init_models)
        Ideogram4Initializer._apply_lora = staticmethod(original_apply_lora)


# Krea 2 Turbo samples with plain Euler, as Krea's own inference code does. mflux defaults to er_sde, which adds
# fresh noise every step and leaves grain when refining an existing image (img2img, upscale).
_KREA2_SAMPLER = "euler"


_BETA_SCHEDULER = "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
_KLEIN_BETA_SCHEDULER = "zvisiongenerator.schedulers.beta_scheduler.KleinBetaScheduler"


def _family_scheduler(family: str, scheduler: str | None) -> str | None:
    """Return the mflux scheduler path for *family*: FLUX.2 Klein's beta keeps Klein's own step-dependent shift."""
    return _KLEIN_BETA_SCHEDULER if family == "flux2_klein" and scheduler == _BETA_SCHEDULER else scheduler


@contextlib.contextmanager
def _krea2_sigma_schedule(scheduler_path: str | None) -> Iterator[None]:
    """Give Krea 2's Euler sampler the sigmas of the scheduler at *scheduler_path* for one run; ``None`` keeps mflux's own.

    mflux steps Krea 2 with its own Euler stepper and takes the sigmas from the scheduler registered under the sampler's
    name, so that registry entry is swapped for the run and restored after.
    """
    if scheduler_path is None:
        yield
        return
    original = SCHEDULER_REGISTRY[_KREA2_SAMPLER]
    SCHEDULER_REGISTRY[_KREA2_SAMPLER] = try_import_external_scheduler(scheduler_path)
    try:
        yield
    finally:
        SCHEDULER_REGISTRY[_KREA2_SAMPLER] = original


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
    # Never import a scheduler here that the run did not: an unloaded module's scheduler cannot be this one.
    moss_module = sys.modules.get("mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler")
    if moss_module is not None and isinstance(scheduler, moss_module.MOSSImageScheduler):
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

    When ``model`` and ``family`` are given, every step but the last also carries a cheap
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
            # img2img, refine and upscale runs start part-way through the schedule; place previews within the steps actually run.
            start = config.init_time_step
            self._preview_steps = frozenset(start + step for step in preview_steps(config.num_inference_steps - start))
        except Exception as exc:  # noqa: BLE001 - previews are best-effort
            warnings.warn(f"Live previews disabled for {self._family}: {exc}", stacklevel=2)
            self._previews_enabled = False

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None, **_):
        del seed, prompt, time_steps
        # mflux calls this before it evaluates the step. Evaluate it first so every report is a finished step; the
        # preview's projection then costs well under a millisecond and rides on the same report.
        mx.eval(latents)
        self._current_step = min(self._current_step + 1, self._total_steps)
        payload = {
            "current_step": self._current_step,
            "total_steps": self._total_steps,
        }
        preview = self._render_preview(t, latents, config, denoised)
        # Keep these latents only when the next step renders a preview from them; a reported prediction needs none.
        self._previous_latents = latents if self._previews_enabled and denoised is None and t + 2 in self._preview_steps else None
        if preview is not None:
            payload["preview"] = preview
        self._step_callback(payload)

    def _render_preview(self, t, latents, config, denoised=None) -> Image.Image | None:
        """Render a preview on a preview step; never let a preview failure stop generation.

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


def _weight_attributes(model: Any) -> list[str]:
    """Return the names of *model*'s attributes that hold an MLX module.

    mflux families name their weights differently (Ideogram 4 keeps its denoisers in ``conditional_transformer`` and
    ``unconditional_transformer``), so the release looks for modules instead of a fixed list of names. An mflux model is
    itself an ``nn.Module``, which stores its child modules as dict entries rather than in ``__dict__``.
    """
    entries = dict(model) if isinstance(model, dict) else {}
    entries.update(getattr(model, "__dict__", {}))
    return [name for name, value in entries.items() if isinstance(value, nn.Module)]


@contextlib.contextmanager
def _reporting(model: Any, steps: int, family: str, skip_signal: Any | None, step_callback: Any | None) -> Iterator[None]:
    """Report one generation's steps from *model*'s mflux callbacks: honour a skip and send progress and previews.

    A skip ends the block quietly, so the caller's code after it runs; the callbacks are removed and MLX's buffer cache
    cleared however the block ends.
    """
    callbacks = []
    if skip_signal is not None:
        callbacks.append(_SkipChecker(skip_signal))
    if step_callback is not None:
        callbacks.append(_ProgressChecker(steps, step_callback, model=model, family=family))
    for callback in callbacks:
        model.callbacks.register(callback)
    try:
        yield
    except StopImageGenerationException:
        pass
    finally:
        for callback in callbacks:
            _unregister_callback(model, callback)
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
        if model_info.family == "moss":
            model = _load_moss(model_path, model_info, quantize, precision, lora_paths, lora_weights)
            self._model_info = model_info
            return model, model_info

        lora_kwargs = {}
        if lora_paths:
            lora_kwargs["lora_scales"] = lora_weights

        def build() -> Any:
            """Construct the family's model and upcast the requested weights."""
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
                # Ideogram 4 Instant: the FP8 transformer, quantized to q4/q8 when asked; no unconditional transformer.
                with _instant_ideogram4():
                    model = Ideogram4(
                        quantize=quantize,
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
                raise ValueError(f"Model family '{model_info.family}' is not supported by the mflux backend. Supported families: zimage, flux2_klein, ideogram4, krea2, moss")

            if precision == "float32":
                upcast_weights(model, ["transformer", "text_encoder", "vae"])
            else:
                # Even in bfloat16 mode, upcast VAE for better color gradients
                upcast_weights(model, ["vae"])
            return model

        # mflux reads LoRAs only from files, so a file with names it cannot match is handed over as a renamed copy. mflux
        # bakes the LoRAs into the weights while it builds the model, and load_resident evaluates every parameter, so
        # the copies are not read again once it returns (or raises) and are removed then.
        with readable_lora_files(lora_paths or [], model_info.family, "mflux") as readable_paths:
            if lora_paths:
                lora_kwargs["lora_paths"] = readable_paths
            model = load_resident(build)

        self._model_info = model_info
        return model, model_info

    def stored_quant_format(self, bits: int) -> str | None:
        """Return the format tag of quantized weights this backend saves: the mflux version, at every level."""
        return f"mflux-{importlib.metadata.version('mflux')}"

    def quantizes_from_files(self, bits: int, source: str | None = None) -> bool:
        """Return whether the stored quant at *bits* is linked from files: only a MOSS-Image copy the other alias shares.

        mflux saves every other level from a model loaded at that level (see :meth:`save_quantized`).
        """
        return source is not None and is_moss_dir(Path(source)) and shared_quant_source(Path(source), bits, self.stored_quant_format(bits)) is not None

    def write_quantized_files(self, source: str, path: str, bits: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Hard-link the other MOSS-Image alias's stored copy at *bits* into *path*: Turbo and SFT hold the same weights.

        Raises:
            RuntimeError: If *source* has no shared copy at *bits* (see :meth:`quantizes_from_files`).
        """
        shared = shared_quant_source(Path(source), bits, self.stored_quant_format(bits))
        if shared is None:
            raise RuntimeError(f"No stored copy at q{bits} to share for '{source}'; mflux saves it from a loaded model.")
        _moss.link_saved_copy(shared, Path(path))

    def save_quantized(self, model: Any, path: str) -> None:
        """Write a loaded, quantized model's weights to *path* in mflux's own format.

        Args:
            model: A model returned by :meth:`load_model` with a quantize level and no LoRAs
                (mflux bakes LoRAs into saved weights).
            path: Destination directory.
        """
        if self._model_info is not None and self._model_info.family == "ideogram4":
            with _instant_ideogram4():
                model.save_model(path)
            return
        if self._model_info is not None and self._model_info.family == "moss":
            _moss.save(model, path)
            return
        model.save_model(path)

    def runs_after_save(self, model: Any) -> bool:
        """Return ``False`` for MOSS-Image, whose staged save leaves nothing loaded; other families stay as loaded."""
        return self._model_info is None or self._model_info.family != "moss"

    def prepare_prompts(self, model: Any, prompts: Sequence[JobPrompt], *, width: int, height: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Encode the job's prompts and free *model*'s text encoder, so its memory is available for generation.

        Families without an entry in :mod:`~zvisiongenerator.backends.mac.prompt_encoding` keep their encoder loaded.
        Returns early, leaving *model* as loaded, once *cancelled* returns true.

        Args:
            model: A model returned by :meth:`load_model`.
            prompts: Every prompt and negative prompt the job's generations use.
            width: The width the job generates at.
            height: The height the job generates at.
            cancelled: Polled before each encode.
        """
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        if self._model_info.family == "moss":
            # The text encoder is freed once the prompts are encoded, and only then does the denoiser load.
            if prompts and _moss.encode_prompts(model, _moss.job_texts(prompts, turbo=self._model_info.is_distilled), cancelled) and not (cancelled is not None and cancelled()):
                _moss.ensure_denoiser(model)
            return
        _prepare_prompts(model, self._model_info.family, prompts, width=width, height=height, cancelled=cancelled)

    def release_model(self, model: Any) -> None:
        """Drop every MLX module *model* holds and clear its prompt cache, so the MLX arrays are unreferenced.

        Calling it again does nothing.
        """
        for name in _weight_attributes(model):
            setattr(model, name, None)
        prompt_cache = getattr(model, "prompt_cache", None)
        if prompt_cache is not None:
            prompt_cache.clear()

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
        *,
        reference_images: list[Image.Image] | None = None,
    ) -> Image.Image | None:
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        if self._model_info.family == "ideogram4":
            raise ValueError("img2img is not supported for Ideogram 4.")
        if self._model_info.family == "moss":
            width, height = moss_run_size(image.width, image.height)
            return self._moss_generate(model, prompt, width, height, seed, steps, guidance, negative_prompt, skip_signal, step_callback, image=image, strength=strength)
        # mflux requires a file path, not a PIL Image
        temp_path = save_temp_png(image)
        try:
            with _reporting(model, steps, self._model_info.family, skip_signal, step_callback):
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
                # As on Windows/Linux, Klein's refine passes (image to image, upscale) keep Klein's own schedule.
                if scheduler is not None and self._model_info.family != "flux2_klein":
                    gen_kwargs["scheduler"] = scheduler
                if self._model_info.family == "krea2":
                    gen_kwargs["scheduler"] = _KREA2_SAMPLER
                if not _is_flux and negative_prompt is not None:
                    gen_kwargs["negative_prompt"] = negative_prompt

                if reference_images:
                    # The start image is the img2img source; the references ride along as edit conditioning.
                    start_kwargs = {key: value for key, value in gen_kwargs.items() if key not in ("image_path", "image_strength")}
                    result = generate_with_references(model, reference_images, start_path=temp_path, image_strength=image_strength, **start_kwargs)
                else:
                    result = model.generate_image(**gen_kwargs)
                return result.image
            return None
        finally:
            remove_file(temp_path)

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
        *,
        reference_images: list[Image.Image] | None = None,
    ) -> Image.Image | None:
        if self._model_info is None:
            raise RuntimeError("load_model() must be called before generation")
        if self._model_info.family == "moss":
            return self._moss_generate(model, prompt, width, height, seed, steps, guidance, negative_prompt, skip_signal, step_callback)
        # Seed the global MLX RNG so that ancestral scheduler noise injection
        # is deterministic per seed (mflux only seeds the initial latent noise
        # with an explicit key, not the global state).
        mx.random.seed(seed)
        try:
            with _reporting(model, steps, self._model_info.family, skip_signal, step_callback):
                if self._model_info.family == "ideogram4":
                    # Ideogram 4 has no scheduler/negative_prompt params. Instant runs mflux's default preset's noise schedule
                    # (mean 0.0, standard deviation 1.75) over the given steps, at its built-in guidance, which skips the unconditional pass.
                    sampler = sampler_settings(steps, guidance)
                    gen_kwargs = dict(prompt=_wrap_ideogram4_prompt(prompt), width=width, height=height, seed=seed, num_inference_steps=sampler.steps, guidance=INSTANT_GUIDANCE)
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
                        gen_kwargs["scheduler"] = _family_scheduler(self._model_info.family, scheduler)
                    if self._model_info.family == "krea2":
                        gen_kwargs["scheduler"] = _KREA2_SAMPLER
                    if not _is_flux and negative_prompt is not None:
                        gen_kwargs["negative_prompt"] = negative_prompt

                # Per-run Ideogram 4 first-step sigma override (affects the scheduler shim, not generate_image itself).
                sigma_ctx = _use_initial_sigma(first_sigma) if first_sigma is not None else contextlib.nullcontext()
                krea2_ctx = _krea2_sigma_schedule(scheduler) if self._model_info.family == "krea2" else contextlib.nullcontext()
                with sigma_ctx, krea2_ctx:
                    if reference_images:
                        result = generate_with_references(model, reference_images, **gen_kwargs)
                    else:
                        result = model.generate_image(**gen_kwargs)
                return result.image
            return None
        finally:
            # The recorded Ideogram 4 schedule belongs to this run only; never let a later run read it.
            _ideogram4_timesteps.value = None

    def _moss_generate(
        self,
        model: Any,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float | None,
        negative_prompt: str | None,
        skip_signal: Any | None,
        step_callback: Any | None,
        *,
        image: Image.Image | None = None,
        strength: float = 1.0,
    ) -> Image.Image | None:
        """Run MOSS-Image for *steps* (from *image* at *strength* when given), reporting progress and honouring a skip."""
        sampler = moss_sampler_settings(turbo=self._model_info.is_distilled, steps=steps, guidance=guidance if guidance is not None else 1.0, negative_prompt=negative_prompt)
        with _reporting(model, steps, "moss", skip_signal, step_callback):
            return _moss.generate(model, prompt, width=width, height=height, seed=seed, sampler=sampler, image=image, strength=strength)
        return None


def _load_moss(model_path: str, model_info: ImageModelInfo, quantize: int | None, precision: str, lora_paths: list[str] | None, lora_weights: list[float] | None) -> Any:
    """Load MOSS-Image's text encoder stage, with the Turbo LoRA (for Turbo) and the user LoRAs set for its denoiser.

    mflux keeps every LoRA as a separate layer. The VAE (and, in float32 mode, every component) is upcast as each stage
    loads.

    Raises:
        RuntimeError: If *model_path* is not a built ``moss`` / ``moss-sft`` folder or a stored copy of one.
    """
    require_built(model_path)
    turbo_lora = turbo_lora_for(model_path, turbo=model_info.is_distilled)
    stack = moss_lora_stack(str(turbo_lora) if turbo_lora is not None else None, lora_paths, lora_weights)
    return _moss.load(
        model_path,
        quantize,
        lora_paths=[lora.path for lora in stack],
        lora_scales=[lora.strength for lora in stack],
        upcast=("transformer", "text_encoder", "vae") if precision == "float32" else ("vae",),
    )
