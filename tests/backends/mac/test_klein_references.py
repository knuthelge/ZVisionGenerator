"""Tests for Klein references on the mflux backend: the view over a loaded model and the scoped start-image hook.

The model is a hand-built ``Flux2Klein`` shell with fake components, and mflux's encoder and reference conditioning are
patched, so the real ``Flux2KleinEdit`` loop runs without any weights.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

pytest.importorskip("mflux")
mx = pytest.importorskip("mlx.core")

from mflux.callbacks.callback_registry import CallbackRegistry  # noqa: E402
from mflux.models.common.compute_precision import ComputePrecision  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.vae.vae_util import VAEUtil  # noqa: E402
from mflux.models.flux2 import Flux2Klein  # noqa: E402
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402
from mlx import nn  # noqa: E402

import zvisiongenerator.backends.mac.klein_references as klein_references  # noqa: E402
from zvisiongenerator.backends.mac._temp_images import save_temp_png  # noqa: E402

_GEN = dict(prompt="p", width=64, height=64, seed=1, num_inference_steps=8, guidance=1.0)


def _fake_klein() -> Flux2Klein:
    """Return a ``Flux2Klein`` carrying every attribute the initializer sets, with fake components."""
    model = Flux2Klein.__new__(Flux2Klein)
    nn.Module.__init__(model)
    model.prompt_cache = {}
    model.model_config = ModelConfig.flux2_klein_4b()
    model.callbacks = CallbackRegistry()
    model.tiling_config = None
    model.tokenizers = {"qwen3": object()}
    model.vae = MagicMock(**{"decode_packed_latents.side_effect": lambda packed, tiling_config=None: mx.zeros((1, 3, 64, 64))})
    model.transformer = lambda **kwargs: mx.zeros(kwargs["hidden_states"].shape)
    model.text_encoder = None
    model.bits = None
    model.small_text_encoder = False
    model.small_decoder = False
    model.lora_paths = None
    model.lora_scales = None
    model.compute_precision = ComputePrecision(None)
    return model


@pytest.fixture
def mflux_stubs():
    """Patch mflux's prompt encoding and reference conditioning; record the reference paths and the img2img configs."""
    seen = {"reference_paths": [], "img2img_configs": []}

    def encode_text(prompt, *, tokenizer, text_encoder):
        return mx.zeros((1, 8, 16)), mx.zeros((1, 8, 4), dtype=mx.int32)

    def reference_conditioning(*, vae, tiling_config, image_paths=None, batch_size=1):
        seen["reference_paths"] = list(image_paths)
        assert all(os.path.isfile(path) for path in image_paths)
        tokens = 12 * len(image_paths)
        return mx.zeros((1, tokens, 128)), mx.zeros((1, tokens, 4), dtype=mx.int32)

    def img2img_latents(self, *, seed, config):
        seen["img2img_configs"].append(config)
        return Flux2LatentCreator.prepare_packed_latents(seed=seed, height=config.height, width=config.width, batch_size=1)

    with (
        patch.object(_Flux2KleinEditHelpers, "encode_text", staticmethod(encode_text)),
        patch.object(_Flux2KleinEditHelpers, "prepare_reference_image_conditioning", staticmethod(reference_conditioning)),
        patch.object(Flux2Klein, "_prepare_img2img_latents", img2img_latents),
    ):
        yield seen


def test_references_only_runs_the_full_schedule_without_a_start_image(mflux_stubs):
    refs = [Image.new("RGB", (64, 48)), Image.new("RGB", (32, 80))]
    steps = []
    model = _fake_klein()
    model.callbacks.register(MagicMock(call_in_loop=lambda t, **kw: steps.append(t)))

    result = klein_references.generate_with_references(model, refs, **_GEN)

    assert result.image.size == (64, 64)
    assert len(mflux_stubs["reference_paths"]) == 2
    assert mflux_stubs["img2img_configs"] == []
    assert steps == list(range(8))  # the shared callback registry saw every step of the untruncated schedule
    assert not any(os.path.exists(path) for path in mflux_stubs["reference_paths"])


def test_no_strength_means_no_start_image(mflux_stubs):
    klein_references.generate_with_references(_fake_klein(), [Image.new("RGB", (64, 64))], start_path="start.png", image_strength=0.0, **_GEN)

    assert mflux_stubs["img2img_configs"] == []


def test_a_start_image_builds_the_first_latents_and_truncates_the_schedule(mflux_stubs):
    model = _fake_klein()
    steps = []
    model.callbacks.register(MagicMock(call_in_loop=lambda t, **kw: steps.append(t)))
    original = _Flux2KleinEditHelpers.prepare_generation_latents
    start_path = save_temp_png(Image.new("RGB", (64, 64)))

    try:
        klein_references.generate_with_references(model, [Image.new("RGB", (64, 48))], start_path=start_path, image_strength=0.5, **_GEN)
    finally:
        os.unlink(start_path)

    (config,) = mflux_stubs["img2img_configs"]
    assert str(config.image_path) == start_path  # the start image, not the first reference
    assert config.init_time_step == 4
    assert steps == [4, 5, 6, 7]
    assert _Flux2KleinEditHelpers.prepare_generation_latents is original


def test_the_hook_is_removed_when_generation_fails(mflux_stubs):
    model = _fake_klein()
    original = _Flux2KleinEditHelpers.prepare_generation_latents

    def fail(**kwargs):
        raise RuntimeError("boom")

    model.transformer = fail
    with pytest.raises(RuntimeError, match="boom"):
        klein_references.generate_with_references(model, [Image.new("RGB", (64, 64))], start_path="start.png", image_strength=0.5, **_GEN)

    assert _Flux2KleinEditHelpers.prepare_generation_latents is original
    assert not any(os.path.exists(path) for path in mflux_stubs["reference_paths"])


def _stub_vae() -> SimpleNamespace:
    """Return a VAE stand-in whose encoder yields 32-channel latents at an eighth of the image size."""

    def encode(image):
        return mx.ones((1, 32, image.shape[-2] // 8, image.shape[-1] // 8))

    bn = SimpleNamespace(running_mean=mx.zeros(128), running_var=mx.ones(128), eps=1e-5)
    return SimpleNamespace(encode=encode, bn=bn)


@pytest.fixture
def reference_paths(tmp_path):
    """Return a factory for PNG files on disk, one per requested size."""

    def make(count: int) -> list[str]:
        paths = []
        for index in range(count):
            path = tmp_path / f"ref{index}.png"
            Image.new("RGB", (64, 48)).save(path)
            paths.append(str(path))
        return paths

    return make


@pytest.fixture
def evaluations():
    """Count the arrays ``evaluate`` is asked to compute, while still computing them."""
    real = klein_references.evaluate
    calls = []

    def counting(array):
        calls.append(array)
        return real(array)

    with patch.object(klein_references, "evaluate", counting):
        yield calls


def _stub_klein(vae: SimpleNamespace) -> Flux2Klein:
    """Return a fake ``Flux2Klein`` whose VAE is *vae* and whose reference conditioning is mflux's own."""
    model = _fake_klein()
    model.vae = vae
    vae.decode_packed_latents = lambda packed, tiling_config=None: mx.zeros((1, 3, 64, 64))
    return model


def _stub_prompt_encoding():
    return patch.object(_Flux2KleinEditHelpers, "encode_text", staticmethod(lambda *a, **k: (mx.zeros((1, 8, 16)), mx.zeros((1, 8, 4), dtype=mx.int32))))


class TestEvaluatingVAE:
    def test_each_reference_is_evaluated_before_the_next_is_encoded(self, reference_paths):
        events = []
        vae = _stub_vae()
        encode = vae.encode
        vae.encode = lambda image: (events.append("encode"), encode(image))[1]
        real = klein_references.evaluate

        def recording(array):
            events.append("evaluate")
            return real(array)

        with patch.object(klein_references, "evaluate", recording):
            _Flux2KleinEditHelpers.prepare_reference_image_conditioning(vae=klein_references._EvaluatingVAE(vae), tiling_config=None, image_paths=reference_paths(3))

        assert events == ["encode", "evaluate"] * 3

    def test_the_conditioning_is_identical_with_and_without_the_wrapper(self, reference_paths):
        paths = reference_paths(3)
        vae = _stub_vae()

        plain_latents, plain_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(vae=vae, tiling_config=None, image_paths=paths)
        latents, ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(vae=klein_references._EvaluatingVAE(vae), tiling_config=None, image_paths=paths)

        assert mx.array_equal(latents, plain_latents)
        assert mx.array_equal(ids, plain_ids)
        assert sorted({int(t) for t in ids[0, :, 0].tolist()}) == [10, 20, 30]

    def test_other_attributes_are_the_wrapped_vaes_own(self):
        vae = _stub_vae()
        wrapper = klein_references._EvaluatingVAE(vae)

        assert wrapper.bn is vae.bn
        assert getattr(wrapper, "latent_channels", 16) == 16
        with pytest.raises(AttributeError):
            wrapper.missing

    def test_the_tiled_path_encodes_through_the_wrapper(self):
        vae = _stub_vae()
        vae.latent_channels = 32  # read through the wrapper
        tiling = SimpleNamespace(vae_encode_tiled=True, vae_encode_tile_size=64, vae_encode_tile_overlap=16)
        evaluated = []
        real = klein_references.evaluate

        def recording(array):
            evaluated.append(array)
            return real(array)

        with patch.object(klein_references, "evaluate", recording):
            encoded = VAEUtil.encode(vae=klein_references._EvaluatingVAE(vae), image=mx.zeros((1, 3, 128, 128)), tiling_config=tiling)

        assert encoded.shape[1] == 32
        assert evaluated


class TestReferenceView:
    def test_the_view_shares_the_models_components_and_wraps_its_vae(self):
        model = _stub_klein(_stub_vae())

        view = klein_references.reference_view(model)

        for name in klein_references.SHARED_ATTRIBUTES:
            if name != "vae":
                assert getattr(view, name) is getattr(model, name)
        assert view.vae.bn is model.vae.bn
        assert view.vae.decode_packed_latents is model.vae.decode_packed_latents

    def test_the_loaded_model_keeps_its_own_vae(self):
        vae = _stub_vae()
        model = _stub_klein(vae)

        klein_references.reference_view(model)

        assert model.vae is vae

    def test_a_run_evaluates_one_encode_per_reference(self, evaluations):
        model = _stub_klein(_stub_vae())
        original = VAEUtil.encode

        with _stub_prompt_encoding():
            klein_references.generate_with_references(model, [Image.new("RGB", (32, 32))] * 3, **_GEN)

        assert len(evaluations) == 3
        assert VAEUtil.encode is original  # nothing outside the view is patched

    def test_a_start_image_run_evaluates_the_references_and_the_start_latents(self, evaluations):
        model = _stub_klein(_stub_vae())
        start_encodes = []

        def img2img_latents(self, *, seed, config):
            start_encodes.append(VAEUtil.encode(vae=model.vae, image=mx.zeros((1, 3, 16, 16))))
            return Flux2LatentCreator.prepare_packed_latents(seed=seed, height=config.height, width=config.width, batch_size=1)

        with _stub_prompt_encoding(), patch.object(Flux2Klein, "_prepare_img2img_latents", img2img_latents):
            klein_references.generate_with_references(model, [Image.new("RGB", (32, 32))] * 2, start_path="start.png", image_strength=0.5, **_GEN)

        assert len(evaluations) == 3  # the two references, then the start image's latents
        assert len(start_encodes) == 1


def test_the_start_image_hook_evaluates_its_latents_before_returning(mflux_stubs):
    model = _fake_klein()
    config = MagicMock()
    evaluated = []

    with patch.object(klein_references, "evaluate", lambda value: evaluated.append(value) or value):
        with klein_references._start_image_latents(model, config):
            latents, latent_ids, _, _ = _Flux2KleinEditHelpers.prepare_generation_latents(seed=1, height=64, width=64)

    assert [array for batch in evaluated for array in batch] == [latents, latent_ids]


class TestBackendDispatch:
    @pytest.fixture
    def backend(self):
        import zvisiongenerator.backends.mac.image as image_mac

        instance = image_mac.MfluxBackend()
        instance._model_info = image_mac.ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        return image_mac, instance

    def test_text_to_image_sends_references_through_the_view(self, backend):
        image_mac, instance = backend
        refs = [Image.new("RGB", (32, 32))]
        model = MagicMock()
        generated = MagicMock(image=Image.new("RGB", (64, 64)))

        with patch.object(image_mac, "generate_with_references", return_value=generated) as generate:
            result = instance.text_to_image(model, "p", 64, 64, 1, 4, 1.0, reference_images=refs)

        assert result is generated.image
        assert generate.call_args.args == (model, refs)
        assert "image_strength" not in generate.call_args.kwargs
        model.generate_image.assert_not_called()

    def test_image_to_image_passes_the_start_image_and_inverted_strength(self, backend):
        image_mac, instance = backend
        refs = [Image.new("RGB", (32, 32))]
        generated = MagicMock(image=Image.new("RGB", (64, 64)))

        with patch.object(image_mac, "generate_with_references", return_value=generated) as generate:
            instance.image_to_image(MagicMock(), Image.new("RGB", (64, 64)), "p", 0.4, 8, 1, 1.0, reference_images=refs)

        kwargs = generate.call_args.kwargs
        assert kwargs["image_strength"] == pytest.approx(0.6)
        assert kwargs["start_path"].endswith(".png")
        assert "image_path" not in kwargs
        assert not os.path.exists(kwargs["start_path"])  # removed after the run

    def test_without_references_the_loaded_model_generates(self, backend):
        image_mac, instance = backend
        model = MagicMock()
        model.generate_image.return_value = MagicMock(image=Image.new("RGB", (64, 64)))

        with patch.object(image_mac, "generate_with_references") as generate:
            instance.text_to_image(model, "p", 64, 64, 1, 4, 1.0)

        generate.assert_not_called()
        model.generate_image.assert_called_once()
