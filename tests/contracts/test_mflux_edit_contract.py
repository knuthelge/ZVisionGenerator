"""Pin the mflux 0.24 internals that ``backends/mac/image_edit.py`` relies on.

Behaviour on stub models and signature pins: no weights are loaded. A failure here after an mflux bump means upstream
changed a hook, a signature or an ordering the mask-edit blend depends on. Runs on macOS; it is
skipped where mlx or mflux is missing.
"""

from __future__ import annotations

import importlib.metadata
import inspect
from unittest.mock import patch

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.callbacks.callback_registry import CallbackRegistry  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.vae.vae_util import VAEUtil  # noqa: E402
from mflux.models.flux2 import Flux2Klein  # noqa: E402
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402
from mflux.models.krea2.latent_creator import Krea2LatentCreator  # noqa: E402
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler  # noqa: E402
from mflux.models.z_image.latent_creator import ZImageLatentCreator  # noqa: E402
from mflux.utils.image_util import ImageUtil  # noqa: E402

from tests._mflux_shells import make_shell, stub_prompt_encoders  # noqa: E402
from zvisiongenerator.backends.mac import image_edit  # noqa: E402

_BETA = "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
_KLEIN_BETA = "zvisiongenerator.schedulers.beta_scheduler.KleinBetaScheduler"
_SIZE = 64
_STEPS = 4


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


class _Spy:
    """Callback that wraps the run's ``config.scheduler.step`` and records, in order, what the loop does."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.step_kwargs: list[dict] = []
        self.config = None
        self.before_loop_kwargs: dict = {}

    def call_before_loop(self, **kwargs):
        self.events.append("before_loop")
        self.before_loop_kwargs = kwargs
        self.config = kwargs["config"]
        scheduler = self.config.scheduler
        original = scheduler.step

        def step(*args, **step_kwargs):
            self.events.append("step")
            self.step_kwargs.append({"args": args, **step_kwargs})
            return original(*args, **step_kwargs)

        scheduler.step = step

    def call_in_loop(self, t, **_):
        self.events.append("in_loop")


def _run_loop(family: str, scheduler: str | None = None) -> _Spy:
    shell = make_shell(family, _SIZE, _SIZE)
    spy = _Spy()
    shell.model.callbacks.register(spy)
    kwargs = dict(seed=1, prompt="p", num_inference_steps=_STEPS, height=_SIZE, width=_SIZE)
    if scheduler is not None:
        kwargs["scheduler"] = scheduler
    with stub_prompt_encoders():
        shell.model.generate_image(**kwargs)
    return spy


class TestSchedulerStepHook:
    """Z-Image and Klein: ``config.scheduler.step`` is called with keywords between ``ctx.before_loop`` and ``ctx.in_loop``."""

    @pytest.mark.parametrize(("family", "extra"), [("zimage", set()), ("flux2_klein", {"sigmas"})])
    def test_step_is_called_by_keyword_once_per_step_between_before_loop_and_in_loop(self, family, extra):
        spy = _run_loop(family)

        assert spy.events == ["before_loop"] + ["step", "in_loop"] * _STEPS
        for t, call in enumerate(spy.step_kwargs):
            assert call["args"] == ()
            assert set(call) - {"args"} == {"noise", "timestep", "latents"} | extra
            assert call["timestep"] == t and isinstance(call["timestep"], int)

    def test_klein_passes_the_schedulers_own_sigmas(self):
        spy = _run_loop("flux2_klein")

        assert all(call["sigmas"] is spy.config.scheduler.sigmas for call in spy.step_kwargs)

    @pytest.mark.parametrize("family", ["zimage", "flux2_klein"])
    def test_the_wrapped_step_is_per_run(self, family):
        first, second = _run_loop(family), _run_loop(family)

        assert first.config is not second.config
        assert first.config.scheduler is not second.config.scheduler
        assert "step" in vars(first.config.scheduler)  # the wrap sits on the run's own instance...
        assert inspect.isfunction(vars(type(first.config.scheduler))["step"])  # ...never on the class
        fresh = Config(model_config=first.config.model_config, num_inference_steps=_STEPS, height=_SIZE, width=_SIZE).scheduler
        assert "step" not in vars(fresh)

    def test_before_loop_receives_the_keywords_the_installer_accepts(self):
        spy = _run_loop("flux2_klein")

        assert {"seed", "prompt", "latents", "config"} <= set(spy.before_loop_kwargs)
        parameters = inspect.signature(image_edit._BlendInstaller.call_before_loop).parameters
        assert [name for name in parameters if name != "self"][:4] == ["seed", "prompt", "latents", "config"]
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    @pytest.mark.parametrize("scheduler", [None, _BETA, _KLEIN_BETA])
    def test_the_installer_wraps_whatever_scheduler_the_run_made(self, scheduler):
        config = Config(model_config=ModelConfig.flux2_klein_4b(), num_inference_steps=_STEPS, height=_SIZE, width=_SIZE, scheduler=scheduler or "flow_match_euler_discrete")
        registry = CallbackRegistry()
        registry.register(image_edit._BlendInstaller(_prepared()))

        registry.start(seed=1, prompt="p", config=config).before_loop(mx.zeros((1, 16, 128)))

        assert "step" in vars(config.scheduler)


def _prepared():
    """Return arrays of Klein's packed layout for a 64 px crop."""
    zeros = mx.zeros((1, 16, 128))
    return image_edit.Prepared(clean=zeros, noise=zeros, mask=mx.ones((1, 16, 1)))


class TestKrea2StepperHook:
    """Krea 2: the stepper is a local built before ``ctx.before_loop``, so ``make_stepper`` is the hook."""

    def test_make_stepper_is_a_static_method_taking_name_sigmas_seed(self):
        assert isinstance(vars(Krea2Sampler)["make_stepper"], staticmethod)
        assert _params(Krea2Sampler.make_stepper) == ["name", "sigmas", "seed"]

    def test_the_loop_builds_the_stepper_before_before_loop_and_steps_it_positionally(self):
        events: list[object] = []
        real = vars(Krea2Sampler)["make_stepper"].__func__

        def make_stepper(*args, **kwargs):
            events.append(("make_stepper", args, kwargs))
            stepper = real(*args, **kwargs)
            stepped = stepper.step

            def step(*step_args, **step_kwargs):
                events.append(("step", step_args, step_kwargs))
                return stepped(*step_args, **step_kwargs)

            stepper.step = step
            return stepper

        shell = make_shell("krea2", _SIZE, _SIZE)
        shell.model.callbacks.register(type("Mark", (), {"call_before_loop": lambda self, **_: events.append("before_loop")})())
        with patch.object(Krea2Sampler, "make_stepper", staticmethod(make_stepper)), stub_prompt_encoders():
            shell.model.generate_image(seed=3, prompt="p", num_inference_steps=_STEPS, height=_SIZE, width=_SIZE, scheduler="euler")

        assert [event if isinstance(event, str) else event[0] for event in events] == ["make_stepper", "before_loop"] + ["step"] * _STEPS
        _, args, kwargs = events[0]
        name, sigmas, seed = args
        assert (name, seed, kwargs) == ("euler", 3, {})
        assert sigmas.shape == (_STEPS + 1,)
        for t, (_, step_args, step_kwargs) in enumerate(events[2:]):
            assert step_kwargs == {} and len(step_args) == 4  # (t, latents, v, denoised)
            assert step_args[0] == t and step_args[1].shape == step_args[2].shape == step_args[3].shape


class TestSchedules:
    @pytest.mark.parametrize(
        ("model_config", "scheduler"),
        [
            (ModelConfig.z_image_turbo(), "linear"),
            (ModelConfig.z_image_turbo(), _BETA),
            (ModelConfig.flux2_klein_4b(), "flow_match_euler_discrete"),
            (ModelConfig.flux2_klein_4b(), _KLEIN_BETA),
            (ModelConfig.krea2(), "euler"),
        ],
    )
    def test_sigmas_have_one_entry_per_step_and_end_in_zero(self, model_config, scheduler):
        config = Config(model_config=model_config, num_inference_steps=_STEPS, height=_SIZE, width=_SIZE, scheduler=scheduler)

        sigmas = [float(sigma) for sigma in config.scheduler.sigmas]

        assert len(sigmas) == _STEPS + 1
        assert sigmas[-1] == 0.0 and sigmas[0] > sigmas[-2] > 0.0

    def test_time_steps_are_plain_ints_from_zero(self):
        config = Config(model_config=ModelConfig.z_image_turbo(), num_inference_steps=_STEPS, height=_SIZE, width=_SIZE)

        assert list(config.time_steps) == list(range(_STEPS))


class TestLatentCreators:
    def test_noise_creators_take_seed_height_width(self):
        assert _params(ZImageLatentCreator.create_noise) == ["seed", "height", "width"]
        assert _params(Krea2LatentCreator.create_noise) == ["seed", "height", "width"]
        assert _params(Flux2LatentCreator.prepare_packed_latents)[:4] == ["seed", "height", "width", "batch_size"]

    def test_noise_is_drawn_from_the_seed_alone_in_the_loops_layouts(self):
        assert ZImageLatentCreator.create_noise(5, 64, 96).shape == (16, 1, 8, 12)
        assert Krea2LatentCreator.create_noise(5, 64, 96).shape == (1, 16, 8, 12)
        packed, _ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(seed=5, height=64, width=96, batch_size=1)
        assert (packed.shape, latent_height, latent_width) == ((1, 4 * 6, 128), 4, 6)
        for draw in (lambda: ZImageLatentCreator.create_noise(5, 64, 96), lambda: Krea2LatentCreator.create_noise(5, 64, 96)):
            assert mx.array_equal(draw(), draw())

    def test_pack_latents_signatures(self):
        assert _params(ZImageLatentCreator.pack_latents) == ["latents", "height", "width"]
        assert _params(Krea2LatentCreator.pack_latents) == ["latents", "height", "width"]
        assert _params(Flux2LatentCreator.pack_latents) == ["latents"]
        assert _params(Flux2LatentCreator.patchify_latents) == ["latents"]

    def test_klein_helpers_prepare_img2img_latents_calls(self):
        assert _params(Flux2Klein._match_latent_spatial_size) == ["encoded", "target_height", "target_width"]
        assert _params(_Flux2KleinEditHelpers.ensure_4d_latents) == ["latents"]
        assert _params(_Flux2KleinEditHelpers.crop_to_even_spatial) == ["latents"]
        assert _params(_Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents) == ["encoded", "vae"]

    def test_encode_takes_a_pixel_array(self):
        assert _params(VAEUtil.encode) == ["vae", "image", "tiling_config"]
        assert _params(ImageUtil.to_array) == ["image", "is_mask"]


def test_mflux_is_the_pinned_release_line():
    """The hooks above are private to 0.24; ``pyproject.toml`` bounds mflux below 0.25 for the same reason."""
    version = tuple(int(part) for part in importlib.metadata.version("mflux").split(".")[:2])

    assert version == (0, 24)
