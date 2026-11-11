"""Real-array check of the mflux latent blend: mflux's own loops, schedulers and Krea 2 stepper run on MLX arrays.

The models are real ``ZImage``, ``Flux2Klein`` and ``Krea2`` objects with a stub transformer that predicts a constant
velocity, a stub VAE with known latents and no prompt encoders, so no weights load. ``MfluxBackend.edit_region`` drives
the real ``generate_image``; a recorder callback sees the latents after every step. Runs on macOS; it is skipped
where mlx or mflux is missing.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mlx")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator  # noqa: E402
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler  # noqa: E402

from tests._mflux_shells import BN_EPS, BN_MEAN, BN_VAR, make_shell, stub_prompt_encoders  # noqa: E402
from zvisiongenerator.backends.mac import image_edit  # noqa: E402
from zvisiongenerator.backends.mac.image import MfluxBackend  # noqa: E402
from zvisiongenerator.utils.image_model_detect import ImageModelInfo  # noqa: E402

HEIGHT, WIDTH = 64, 96  # not square, so a transposed mask or token order cannot pass
STEPS = 4
SEED = 11
VELOCITY = 0.5
BETA = "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
FAMILIES = ("zimage", "flux2_klein", "krea2")
SCHEDULERS = [pytest.param(None, id="default"), pytest.param(BETA, id="beta")]


def _paint() -> np.ndarray:
    """Return a mask with whole cells on both grids, plus cells that are half painted on the 8 px grid and on the 16 px grid."""
    paint = np.zeros((HEIGHT, WIDTH), dtype=bool)
    paint[16:48, 32:64] = True
    paint[0:8, 0:16] = True  # a whole 8 px cell pair, half of a 16 px cell
    paint[56:64, 80:84] = True  # half of an 8 px cell, a quarter of a 16 px cell
    return paint


class _Recorder:
    """Callback that keeps the run's config, its starting latents and the latents after every step."""

    def __init__(self) -> None:
        self.config = None
        self.start = None
        self.start_dtype = None
        self.steps: list[np.ndarray] = []
        self.dtypes: list[mx.Dtype] = []

    def call_before_loop(self, seed, prompt, latents, config, **_):
        self.config = config
        self.start = np.array(latents.astype(mx.float32))
        self.start_dtype = latents.dtype

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **_):
        self.steps.append(np.array(latents.astype(mx.float32)))
        self.dtypes.append(latents.dtype)


def _backend(family: str) -> MfluxBackend:
    backend = MfluxBackend()
    backend._model_info = ImageModelInfo(family=family, is_distilled=True, size=None)
    return backend


def _run(family: str, scheduler: str | None = None, velocity: float = VELOCITY, skip_signal=None):
    """Run ``edit_region`` on a stub shell; return the shell, the backend's result and the recorder."""
    shell = make_shell(family, HEIGHT, WIDTH, velocity)
    recorder = _Recorder()
    shell.model.callbacks.register(recorder)
    with stub_prompt_encoders():
        result = _backend(family).edit_region(shell.model, Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0, scheduler=scheduler, skip_signal=skip_signal)
    return shell, result, recorder


def _cell_fractions(paint: np.ndarray, cell: int) -> np.ndarray:
    """Return the painted fraction of each ``cell`` px cell, counted one cell at a time."""
    rows, cols = paint.shape[0] // cell, paint.shape[1] // cell
    out = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            out[row, col] = paint[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell].sum() / cell**2
    return out


def _expected_mask(family: str) -> np.ndarray:
    paint = _paint()
    if family == "zimage":
        return _cell_fractions(paint, 8)[None, None]
    grid = _cell_fractions(paint, 16)
    if family == "flux2_klein":
        return np.array(Flux2LatentCreator.pack_latents(mx.array(grid)[None, None]))  # mflux's own token order
    return np.kron(grid, np.ones((2, 2), dtype=np.float32))[None, None]


def _expected_clean(family: str, encoded: mx.array) -> np.ndarray:
    if family == "zimage":
        return np.array(encoded[0])  # (16, 1, h, w)
    if family == "krea2":
        return np.array(encoded[:, :, 0])  # (1, 16, h, w)
    patched = Flux2LatentCreator.patchify_latents(encoded)
    normalised = (patched - BN_MEAN) / np.sqrt(BN_VAR + BN_EPS)
    return np.array(Flux2LatentCreator.pack_latents(normalised))


@pytest.fixture(autouse=True)
def float32_latents(monkeypatch):
    """Run in float32 so the formula is checked tightly; the dtype test sets bfloat16 itself."""
    monkeypatch.setattr(ModelConfig, "precision", mx.float32)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
@pytest.mark.parametrize("family", FAMILIES)
def test_every_step_blends_the_source_at_the_next_sigma(family, scheduler):
    shell, result, recorder = _run(family, scheduler)

    sigmas = np.array(recorder.config.scheduler.sigmas.astype(mx.float32))
    assert len(sigmas) == STEPS + 1 and sigmas[-1] == 0.0
    assert len(recorder.steps) == STEPS
    mask = _expected_mask(family)
    clean = _expected_clean(family, shell.encoded)
    noise = recorder.start
    previous = noise
    for t, latents in enumerate(recorder.steps):
        stepped = previous + VELOCITY * (sigmas[t + 1] - sigmas[t])
        noised = (1 - sigmas[t + 1]) * clean + sigmas[t + 1] * noise
        np.testing.assert_allclose(latents, (1 - mask) * noised + mask * stepped, atol=1e-5)
        previous = latents
    assert result.size == (WIDTH, HEIGHT)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_unpainted_area_ends_as_exactly_the_clean_latents(family):
    shell, _, recorder = _run(family)

    mask = _expected_mask(family)
    clean = _expected_clean(family, shell.encoded)
    unpainted = np.broadcast_to(mask == 0, clean.shape)
    assert unpainted.any()
    np.testing.assert_allclose(recorder.steps[-1][unpainted], clean[unpainted], atol=1e-6)
    painted = np.broadcast_to(mask == 1, clean.shape)
    assert painted.any()
    assert not np.allclose(recorder.steps[-1][painted], clean[painted], atol=1e-3)  # the paint is free to change


@pytest.mark.parametrize("family", FAMILIES)
def test_the_noise_prepare_draws_is_the_noise_the_loop_starts_from(family):
    shell, _, recorder = _run(family)

    prepared = image_edit.prepare(family, shell.model, Image.new("RGB", (WIDTH, HEIGHT)), _paint(), SEED)

    np.testing.assert_array_equal(np.array(prepared.noise), recorder.start)
    np.testing.assert_allclose(np.array(prepared.clean), _expected_clean(family, shell.encoded), atol=1e-6)
    np.testing.assert_allclose(np.array(prepared.mask), _expected_mask(family))


@pytest.mark.parametrize("family", FAMILIES)
def test_the_blend_keeps_the_loops_dtype_and_the_source(family, monkeypatch):
    monkeypatch.setattr(ModelConfig, "precision", mx.bfloat16)

    shell, _, recorder = _run(family)

    # Krea 2 draws its noise in float32 whatever the precision; the others follow it.
    assert recorder.start_dtype == (mx.float32 if family == "krea2" else mx.bfloat16)
    assert set(recorder.dtypes) == {recorder.start_dtype}
    mask = _expected_mask(family)
    clean = _expected_clean(family, shell.encoded)
    unpainted = np.broadcast_to(mask == 0, clean.shape)
    np.testing.assert_allclose(recorder.steps[-1][unpainted], clean[unpainted], atol=5e-2)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_skip_returns_none_and_leaves_no_hook_behind(family):
    stepper = vars(Krea2Sampler)["make_stepper"]
    skip = type("Skip", (), {"check": staticmethod(lambda: True)})()

    shell, result, _ = _run(family, skip_signal=skip)

    assert result is None
    assert not any(isinstance(callback, image_edit._BlendInstaller) for callback in shell.model.callbacks.before_loop)
    assert vars(Krea2Sampler)["make_stepper"] is stepper


@pytest.mark.parametrize("family", FAMILIES)
def test_a_plain_run_after_an_edit_is_not_blended(family):
    shell, _, _ = _run(family)
    recorder = _Recorder()
    shell.model.callbacks.register(recorder)

    with stub_prompt_encoders():
        _backend(family).text_to_image(shell.model, "p", WIDTH, HEIGHT, SEED, STEPS, 1.0)

    clean = _expected_clean(family, shell.encoded)
    unpainted = np.broadcast_to(_expected_mask(family) == 0, clean.shape)
    sigma = float(recorder.config.scheduler.sigmas[1])
    noised = (1 - sigma) * clean + sigma * recorder.start
    assert not np.allclose(recorder.steps[0][unpainted], noised[unpainted], atol=1e-3)


class TestEditRegionGuards:
    def test_a_model_must_be_loaded_first(self):
        with pytest.raises(RuntimeError, match="load_model"):
            MfluxBackend().edit_region(object(), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0)

    @pytest.mark.parametrize("family", ["flux1", "flux2", "ideogram4", "moss"])
    def test_a_family_without_mask_editing_is_refused(self, family):
        with pytest.raises(ValueError, match="not supported"):
            _backend(family).edit_region(object(), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0)
