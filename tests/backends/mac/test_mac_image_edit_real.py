"""Real-array check of the mflux latent blend and the clean-crop reference: mflux's own edit loop and schedulers run on MLX arrays.

The model is a real ``Flux2Klein`` object with a stub transformer that predicts a constant velocity, a stub VAE with
known latents and no prompt encoders, so no weights load. ``MfluxBackend.edit_region`` drives the real
``Flux2KleinEdit.generate_image`` through the references view; a recorder callback sees the latents after every step. Runs on macOS; it is
skipped where mlx or mflux is missing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mlx")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator  # noqa: E402

from tests._images import painted_box  # noqa: E402
from tests._mflux_shells import BN_EPS, BN_MEAN, BN_VAR, make_shell, stub_prompt_encoders  # noqa: E402
from zvisiongenerator.backends.mac import image_edit, klein_references  # noqa: E402
from zvisiongenerator.backends.mac.image import MfluxBackend  # noqa: E402
from zvisiongenerator.utils.image_model_detect import ImageModelInfo  # noqa: E402

HEIGHT, WIDTH = 64, 96  # not square, so a transposed mask or token order cannot pass
STEPS = 4
SEED = 11
VELOCITY = 0.5
BETA = "zvisiongenerator.schedulers.beta_scheduler.BetaScheduler"
FAMILY = "flux2_klein"
SCHEDULERS = [pytest.param(None, id="default"), pytest.param(BETA, id="beta")]


def _paint() -> np.ndarray:
    """Return a mask with whole 16 px cells, plus cells that are half and an eighth painted."""
    return painted_box(HEIGHT, WIDTH, (32, 16, 64, 48), (0, 0, 16, 8), (80, 56, 84, 64))


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


def _backend() -> MfluxBackend:
    backend = MfluxBackend()
    backend._model_info = ImageModelInfo(family=FAMILY, is_distilled=True, size=None)
    return backend


def _run(scheduler: str | None = None, velocity: float = VELOCITY, skip_signal=None):
    """Run ``edit_region`` on a stub shell; return the shell, the backend's result and the recorder."""
    shell = make_shell(FAMILY, HEIGHT, WIDTH, velocity)
    recorder = _Recorder()
    shell.model.callbacks.register(recorder)
    with stub_prompt_encoders():
        result = _backend().edit_region(
            shell.model, Image.new("RGB", (WIDTH, HEIGHT)), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0, scheduler=scheduler, skip_signal=skip_signal
        )
    return shell, result, recorder


def _cell_fractions(paint: np.ndarray, cell: int) -> np.ndarray:
    """Return the painted fraction of each ``cell`` px cell, counted one cell at a time."""
    rows, cols = paint.shape[0] // cell, paint.shape[1] // cell
    out = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            out[row, col] = paint[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell].sum() / cell**2
    return out


def _expected_mask() -> np.ndarray:
    return np.array(Flux2LatentCreator.pack_latents(mx.array(_cell_fractions(_paint(), 16))[None, None]))  # mflux's own token order


def _expected_clean(encoded: mx.array) -> np.ndarray:
    patched = Flux2LatentCreator.patchify_latents(encoded)
    normalised = (patched - BN_MEAN) / np.sqrt(BN_VAR + BN_EPS)
    return np.array(Flux2LatentCreator.pack_latents(normalised))


@pytest.fixture(autouse=True)
def float32_latents(monkeypatch):
    """Run in float32 so the formula is checked tightly; the dtype test sets bfloat16 itself."""
    monkeypatch.setattr(ModelConfig, "precision", mx.float32)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_every_step_blends_the_source_at_the_next_sigma(scheduler):
    shell, result, recorder = _run(scheduler)

    sigmas = np.array(recorder.config.scheduler.sigmas.astype(mx.float32))
    assert len(sigmas) == STEPS + 1 and sigmas[-1] == 0.0
    assert len(recorder.steps) == STEPS
    mask = _expected_mask()
    clean = _expected_clean(shell.encoded)
    noise = recorder.start
    previous = noise
    for t, latents in enumerate(recorder.steps):
        stepped = previous + VELOCITY * (sigmas[t + 1] - sigmas[t])
        noised = (1 - sigmas[t + 1]) * clean + sigmas[t + 1] * noise
        np.testing.assert_allclose(latents, (1 - mask) * noised + mask * stepped, atol=1e-5)
        previous = latents
    assert result.size == (WIDTH, HEIGHT)


def test_the_noise_prepare_draws_is_the_noise_the_loop_starts_from():
    shell, _, recorder = _run()

    prepared = image_edit.prepare(shell.model, Image.new("RGB", (WIDTH, HEIGHT)), _paint(), SEED)

    np.testing.assert_array_equal(np.array(prepared.noise), recorder.start)
    np.testing.assert_allclose(np.array(prepared.clean), _expected_clean(shell.encoded), atol=1e-6)
    np.testing.assert_allclose(np.array(prepared.mask), _expected_mask())


def test_the_blend_keeps_the_loops_dtype_and_the_source(monkeypatch):
    monkeypatch.setattr(ModelConfig, "precision", mx.bfloat16)

    shell, _, recorder = _run()

    assert recorder.start_dtype == mx.bfloat16
    assert set(recorder.dtypes) == {recorder.start_dtype}
    clean = _expected_clean(shell.encoded)
    unpainted = np.broadcast_to(_expected_mask() == 0, clean.shape)
    np.testing.assert_allclose(recorder.steps[-1][unpainted], clean[unpainted], atol=5e-2)


def test_the_given_reference_rides_along_and_its_file_is_removed(monkeypatch):
    shell = make_shell(FAMILY, HEIGHT, WIDTH, VELOCITY)
    tokens: list[int] = []
    stub = shell.model.transformer
    shell.model.transformer = lambda **kwargs: (tokens.append(kwargs["hidden_states"].shape[1]), stub(**kwargs))[1]
    written: list[str] = []
    save = klein_references.save_temp_png
    monkeypatch.setattr(klein_references, "save_temp_png", lambda image: written.append(save(image)) or written[-1])

    reference = Image.new("RGB", (48, 32))  # not the crop: an Extend's reference is the source alone
    encoded_sizes: list[tuple[int, ...]] = []
    encode = shell.model.vae.encode
    shell.model.vae.encode = lambda image: (encoded_sizes.append(tuple(image.shape[-2:])), encode(image))[1]
    with stub_prompt_encoders():
        _backend().edit_region(shell.model, Image.new("RGB", (WIDTH, HEIGHT)), reference, _paint(), "p", SEED, STEPS, 1.0)

    crop_tokens = (HEIGHT // 16) * (WIDTH // 16)
    assert encoded_sizes == [(HEIGHT, WIDTH), (32, 48)]  # the crop for the blend, then the reference as given
    assert tokens and set(tokens) == {2 * crop_tokens}  # the stub encodes any picture to the crop's latents; mflux compiles the step, so it may trace once
    assert len(written) == 1 and not os.path.exists(written[0])


def test_the_join_fix_decodes_the_last_steps_latents_before_the_pin():
    shell = make_shell(FAMILY, HEIGHT, WIDTH, VELOCITY)
    decoded: list[np.ndarray] = []
    decode = shell.model.vae.decode_packed_latents
    shell.model.vae.decode_packed_latents = lambda packed, tiling_config=None: (decoded.append(np.array(packed.astype(mx.float32))), decode(packed, tiling_config=tiling_config))[1]
    recorder = _Recorder()
    shell.model.callbacks.register(recorder)

    with stub_prompt_encoders():
        result = _backend().edit_region(shell.model, Image.new("RGB", (WIDTH, HEIGHT)), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0, fix_join=True)

    clean = _expected_clean(shell.encoded)
    unpainted = np.broadcast_to(_expected_mask() == 0, clean.shape)
    assert len(decoded) == 2  # the loop's own decode, then the one for the join
    packed_loop = decoded[0].transpose(0, 2, 3, 1).reshape(1, -1, 128)
    packed_ours = decoded[1].transpose(0, 2, 3, 1).reshape(1, -1, 128)
    np.testing.assert_allclose(packed_loop[unpainted], clean[unpainted], atol=1e-5)  # the loop's final latents are pinned to the source...
    assert not np.allclose(packed_ours[unpainted], clean[unpainted], atol=1e-3)  # ...ours are the model's own, before the pin
    painted = np.broadcast_to(_expected_mask() == 1, clean.shape)
    np.testing.assert_allclose(packed_ours[painted], packed_loop[painted], atol=1e-5)
    assert result.size == (WIDTH, HEIGHT)


def test_without_the_join_fix_the_last_latents_are_decoded_once():
    shell = make_shell(FAMILY, HEIGHT, WIDTH, VELOCITY)
    count: list[int] = []
    decode = shell.model.vae.decode_packed_latents
    shell.model.vae.decode_packed_latents = lambda packed, tiling_config=None: (count.append(1), decode(packed, tiling_config=tiling_config))[1]

    with stub_prompt_encoders():
        _backend().edit_region(shell.model, Image.new("RGB", (WIDTH, HEIGHT)), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0)

    assert count == [1]


def test_a_skip_returns_none_and_leaves_no_hook_behind():
    skip = type("Skip", (), {"check": staticmethod(lambda: True)})()

    shell, result, _ = _run(skip_signal=skip)

    assert result is None
    assert not any(isinstance(callback, image_edit._BlendInstaller) for callback in shell.model.callbacks.before_loop)


def test_a_plain_run_after_an_edit_is_not_blended():
    shell, _, _ = _run()
    recorder = _Recorder()
    shell.model.callbacks.register(recorder)

    with stub_prompt_encoders():
        _backend().text_to_image(shell.model, "p", WIDTH, HEIGHT, SEED, STEPS, 1.0)

    clean = _expected_clean(shell.encoded)
    unpainted = np.broadcast_to(_expected_mask() == 0, clean.shape)
    sigma = float(recorder.config.scheduler.sigmas[1])
    noised = (1 - sigma) * clean + sigma * recorder.start
    assert not np.allclose(recorder.steps[0][unpainted], noised[unpainted], atol=1e-3)


class TestEditRegionGuards:
    def test_a_model_must_be_loaded_first(self):
        with pytest.raises(RuntimeError, match="load_model"):
            MfluxBackend().edit_region(object(), Image.new("RGB", (WIDTH, HEIGHT)), Image.new("RGB", (WIDTH, HEIGHT)), _paint(), "p", SEED, STEPS, 1.0)
