"""Test backend selection and load_model guard contracts."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.core.image_backend import ImageBackend


class TestBackendRegistryLookup:
    def test_get_backend_name_uses_dict_lookup_for_darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"darwin": ("mflux", lambda: MagicMock(spec=ImageBackend))})

        assert backends.get_backend_name() == "mflux"

    def test_get_backend_name_uses_dict_lookup_for_win32(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "win32")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"win32": ("diffusers", lambda: MagicMock(spec=ImageBackend))})

        assert backends.get_backend_name() == "diffusers"

    def test_get_backend_name_uses_dict_lookup_for_linux(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "linux")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"linux": ("diffusers", lambda: MagicMock(spec=ImageBackend))})

        assert backends.get_backend_name() == "diffusers"

    def test_get_backend_uses_dict_lookup_for_darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        backend = MagicMock(spec=ImageBackend)
        monkeypatch.setattr(backends, "BACKENDS", {})
        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"darwin": ("mflux", lambda: backend)})

        result = backends.get_backend()

        assert result is backend
        assert backends.BACKENDS["mflux"] is backend

    def test_get_backend_uses_dict_lookup_for_win32(self, monkeypatch):
        import zvisiongenerator.backends as backends

        backend = MagicMock(spec=ImageBackend)
        monkeypatch.setattr(backends, "BACKENDS", {})
        monkeypatch.setattr(backends.sys, "platform", "win32")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"win32": ("diffusers", lambda: backend)})

        result = backends.get_backend()

        assert result is backend
        assert backends.BACKENDS["diffusers"] is backend

    def test_get_backend_uses_dict_lookup_for_linux(self, monkeypatch):
        import zvisiongenerator.backends as backends

        backend = MagicMock(spec=ImageBackend)
        monkeypatch.setattr(backends, "BACKENDS", {})
        monkeypatch.setattr(backends.sys, "platform", "linux")
        monkeypatch.setattr(backends, "_IMAGE_BACKENDS_MAP", {"linux": ("diffusers", lambda: backend)})

        result = backends.get_backend()

        assert result is backend
        assert backends.BACKENDS["diffusers"] is backend

    def test_unsupported_platform_error_lists_supported_image_platforms(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "freebsd")
        monkeypatch.setattr(backends, "BACKENDS", {})

        with pytest.raises(RuntimeError, match="macOS, Windows, and Linux"):
            backends.get_backend()


# ---------------------------------------------------------------------------
# Guard: generation without load_model() must raise RuntimeError
# ---------------------------------------------------------------------------


class TestLoadModelGuard:
    """Backends must raise RuntimeError if generation is called without load_model()."""

    def test_mflux_text_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_mac import MfluxBackend

        backend = MfluxBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.text_to_image(
                model=MagicMock(),
                prompt="test",
                width=64,
                height=64,
                seed=42,
                steps=1,
                guidance=0.5,
            )

    def test_mflux_image_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_mac import MfluxBackend

        backend = MfluxBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=1,
                seed=42,
                guidance=0.5,
            )

    @pytest.mark.skipif(sys.platform == "darwin", reason="torch/CUDA not available on macOS CI")
    def test_diffusers_text_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_win import DiffusersBackend

        backend = DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.text_to_image(
                model=MagicMock(),
                prompt="test",
                width=64,
                height=64,
                seed=42,
                steps=1,
                guidance=0.5,
            )

    @pytest.mark.skipif(sys.platform == "darwin", reason="torch/CUDA not available on macOS CI")
    def test_diffusers_image_to_image_without_load_model(self):
        from zvisiongenerator.backends.image_win import DiffusersBackend

        backend = DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=1,
                seed=42,
                guidance=0.5,
            )


class _FakeModelComponent:
    def __init__(self, params):
        self._params = params
        self.updated = None

    def parameters(self):
        return self._params

    def update(self, params):
        self.updated = params


def _flatten_params(params):
    for value in params.values():
        if isinstance(value, Mapping):
            yield from _flatten_params(value)
        else:
            yield value


def _make_generated_image_result():
    result = MagicMock()
    result.image = Image.new("RGB", (64, 64))
    return result


@pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")
class TestIdeogram4MfluxBackend:
    def _import_backend_module(self):
        pytest.importorskip("mflux")
        pytest.importorskip("mlx.core")

        import zvisiongenerator.backends.image_mac as image_mac_module

        return image_mac_module

    def test_ideogram4_load_model_uses_fp8_config_without_quantize(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(image_mac_module, "detect_image_model", lambda _path: model_info)
            monkeypatch.setattr(image_mac_module, "_upcast_model_weights", MagicMock())
            monkeypatch.setattr(image_mac_module.ModelConfig, "ideogram4_fp8", MagicMock(return_value="ideogram4-config"))
            ideogram_ctor = MagicMock(return_value=MagicMock())
            monkeypatch.setattr(image_mac_module, "Ideogram4", ideogram_ctor)

            model, loaded_info = backend.load_model(
                "ideogram-ai/ideogram-4-fp8",
                quantize=4,
                lora_paths=["/tmp/style.safetensors"],
                lora_weights=[0.75],
            )

        assert loaded_info == model_info
        assert model is ideogram_ctor.return_value
        call_kwargs = ideogram_ctor.call_args.kwargs
        assert call_kwargs == {
            "model_path": "ideogram-ai/ideogram-4-fp8",
            "model_config": "ideogram4-config",
            "lora_paths": ["/tmp/style.safetensors"],
            "lora_scales": [0.75],
        }
        assert "quantize" not in call_kwargs

    def test_ideogram4_text_to_image_strips_scheduler_and_negative(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        image = backend.text_to_image(
            model=model,
            prompt='{"type":"caption","text":"keep verbatim"}',
            width=1024,
            height=1024,
            seed=123,
            steps=20,
            guidance=7.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=True,
            guidance_explicit=True,
        )

        assert image is not None
        model.generate_image.assert_called_once()
        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["prompt"] == '{"type":"caption","text":"keep verbatim"}'
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024
        assert kwargs["seed"] == 123
        assert kwargs["preset"] == "V4_DEFAULT_20"
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance"] == 7.0
        assert "scheduler" not in kwargs
        assert "negative_prompt" not in kwargs

    def test_ideogram4_text_to_image_wraps_plain_text_prompt(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        image = backend.text_to_image(
            model=model,
            prompt="a red sports car at sunset",
            width=1024,
            height=1024,
            seed=123,
            steps=20,
            guidance=7.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=True,
            guidance_explicit=True,
        )

        assert image is not None
        model.generate_image.assert_called_once()
        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["prompt"] == {
            "high_level_description": "a red sports car at sunset",
            "compositional_deconstruction": {
                "background": "a red sports car at sunset",
                "elements": [],
            },
        }
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024
        assert kwargs["seed"] == 123
        assert kwargs["preset"] == "V4_DEFAULT_20"
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance"] == 7.0
        assert "scheduler" not in kwargs
        assert "negative_prompt" not in kwargs

    def test_ideogram4_text_to_image_passes_json_prompt_through_unchanged(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()
        prompt = '{"high_level_description":"x","compositional_deconstruction":{"background":"x","elements":[]}}'

        image = backend.text_to_image(
            model=model,
            prompt=prompt,
            width=1024,
            height=1024,
            seed=123,
            steps=20,
            guidance=7.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=True,
            guidance_explicit=True,
        )

        assert image is not None
        model.generate_image.assert_called_once()
        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["prompt"] == prompt
        assert "scheduler" not in kwargs
        assert "negative_prompt" not in kwargs

    def test_ideogram4_text_to_image_passes_whitespace_prefixed_json_prompt_through_unchanged(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()
        prompt = '   {"high_level_description":"x","compositional_deconstruction":{"background":"x","elements":[]}}'

        image = backend.text_to_image(
            model=model,
            prompt=prompt,
            width=1024,
            height=1024,
            seed=123,
            steps=20,
            guidance=7.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=True,
            guidance_explicit=True,
        )

        assert image is not None
        model.generate_image.assert_called_once()
        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["prompt"] == prompt
        assert "scheduler" not in kwargs
        assert "negative_prompt" not in kwargs

    def test_wrap_ideogram4_prompt_wraps_plain_text(self):
        image_mac_module = self._import_backend_module()

        assert image_mac_module._wrap_ideogram4_prompt("a red sports car at sunset") == {
            "high_level_description": "a red sports car at sunset",
            "compositional_deconstruction": {
                "background": "a red sports car at sunset",
                "elements": [],
            },
        }

    def test_wrap_ideogram4_prompt_preserves_json_caption_text(self):
        image_mac_module = self._import_backend_module()
        prompt = '{"high_level_description":"x","compositional_deconstruction":{"background":"x","elements":[]}}'

        assert image_mac_module._wrap_ideogram4_prompt(prompt) == prompt

    def test_wrap_ideogram4_prompt_preserves_whitespace_prefixed_json_caption_text(self):
        image_mac_module = self._import_backend_module()
        prompt = '   {"high_level_description":"x","compositional_deconstruction":{"background":"x","elements":[]}}'

        assert image_mac_module._wrap_ideogram4_prompt(prompt) == prompt

    def test_ideogram4_text_to_image_omits_steps_and_guidance_when_not_explicit(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        backend.text_to_image(
            model=model,
            prompt="a lighthouse in fog",
            width=832,
            height=1216,
            seed=42,
            steps=20,
            guidance=7.0,
            steps_explicit=False,
            guidance_explicit=False,
        )

        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["preset"] == "V4_DEFAULT_20"
        assert "num_inference_steps" not in kwargs
        assert "guidance" not in kwargs

    @pytest.mark.parametrize(
        ("steps_explicit", "guidance_explicit", "expected_present", "expected_absent"),
        [
            (True, False, "num_inference_steps", "guidance"),
            (False, True, "guidance", "num_inference_steps"),
        ],
    )
    def test_ideogram4_text_to_image_treats_explicit_flags_independently(
        self,
        steps_explicit,
        guidance_explicit,
        expected_present,
        expected_absent,
    ):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        backend.text_to_image(
            model=model,
            prompt='{"type":"caption","text":"keep verbatim"}',
            width=832,
            height=1216,
            seed=42,
            steps=20,
            guidance=7.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=steps_explicit,
            guidance_explicit=guidance_explicit,
        )

        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["preset"] == "V4_DEFAULT_20"
        assert kwargs[expected_present] == (20 if expected_present == "num_inference_steps" else 7.0)
        assert expected_absent not in kwargs
        assert "scheduler" not in kwargs
        assert "negative_prompt" not in kwargs

    @pytest.mark.parametrize(
        ("first_sigma", "expected_effective_sigma"),
        [
            (1.006, 1.006),
            (None, 1.004),
        ],
    )
    def test_ideogram4_text_to_image_scopes_first_sigma_override_to_generate_image_call(self, first_sigma, expected_effective_sigma):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        captured_effective_sigmas: list[float | None] = []
        sentinel = image_mac_module._INITIAL_SIGMA_UNSET

        def _capture_generate_image(**kwargs):
            del kwargs
            captured_effective_sigmas.append(image_mac_module._effective_initial_sigma())
            return _make_generated_image_result()

        model.generate_image.side_effect = _capture_generate_image

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(image_mac_module, "IDEOGRAM4_INITIAL_SIGMA", 1.004)
            monkeypatch.setattr(image_mac_module, "_initial_sigma_override", sentinel)

            image = backend.text_to_image(
                model=model,
                prompt='{"type":"caption","text":"keep verbatim"}',
                width=1024,
                height=1024,
                seed=123,
                steps=20,
                guidance=7.0,
                steps_explicit=True,
                guidance_explicit=True,
                first_sigma=first_sigma,
            )

            assert image is not None
            model.generate_image.assert_called_once()
            assert captured_effective_sigmas[0] == pytest.approx(expected_effective_sigma)
            assert "first_sigma" not in model.generate_image.call_args.kwargs
            assert image_mac_module._initial_sigma_override is sentinel

    def test_ideogram4_image_to_image_raises(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)

        with pytest.raises(ValueError, match="(?i)img2img.*not supported"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=10,
                seed=7,
                guidance=7.0,
            )

    def test_zimage_text_to_image_ignores_explicit_flags_for_steps(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        model_info = image_mac_module.ImageModelInfo(family="zimage", is_distilled=False, size=None)
        generate_model = MagicMock()
        generate_model.generate_image.return_value = _make_generated_image_result()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(image_mac_module, "detect_image_model", lambda _path: model_info)
            monkeypatch.setattr(image_mac_module, "_upcast_model_weights", MagicMock())
            monkeypatch.setattr(image_mac_module.ModelConfig, "z_image", MagicMock(return_value="zimage-config"))
            zimage_ctor = MagicMock(return_value=generate_model)
            monkeypatch.setattr(image_mac_module, "ZImageTurbo", zimage_ctor)

            model, loaded_info = backend.load_model("Tongyi-MAI/Z-Image-Turbo", quantize=4)

        assert loaded_info == model_info
        assert model is generate_model

        backend.text_to_image(
            model=model,
            prompt="a red sports car",
            width=1024,
            height=1024,
            seed=123,
            steps=6,
            guidance=3.5,
            steps_explicit=False,
            guidance_explicit=False,
        )

        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["num_inference_steps"] == 6
        assert kwargs["guidance"] == 3.5


@pytest.mark.skipif(sys.platform != "darwin", reason="Selective MLX upcast only runs on macOS")
class TestSelectiveMacWeightUpcast:
    def test_upcast_model_weights_preserves_uint32_quantized_weights(self):
        mx = pytest.importorskip("mlx.core")

        from zvisiongenerator.backends.image_mac import _upcast_model_weights

        uint_weights = mx.array([1, 2, 3], dtype=mx.uint32)
        fp16_weights = mx.array([1.5, 2.5], dtype=mx.float16)
        bf16_weights = mx.array([3.0, 4.0], dtype=mx.bfloat16)
        float32_weights = mx.array([5.0], dtype=mx.float32)
        component = _FakeModelComponent(
            {
                "quantized": uint_weights,
                "nested": {
                    "fp16": fp16_weights,
                    "bf16": bf16_weights,
                    "fp32": float32_weights,
                },
            }
        )
        model = MagicMock(transformer=component)

        _upcast_model_weights(model, ["transformer"])

        assert component.updated is not None
        assert component.updated["quantized"] is uint_weights
        assert component.updated["quantized"].dtype == mx.uint32
        assert component.updated["nested"]["fp16"].dtype == mx.float32
        assert component.updated["nested"]["bf16"].dtype == mx.float32
        assert component.updated["nested"]["fp32"] is float32_weights
        assert all(value.dtype != mx.float16 for value in _flatten_params(component.updated))
        assert all(value.dtype != mx.bfloat16 for value in _flatten_params(component.updated))
