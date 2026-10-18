"""Test backend selection and load_model guard contracts."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest
from PIL import Image

from zvisiongenerator.core.image_backend import ImageBackend


class TestBackendRegistryLookup:
    def test_ltx_system_memory_is_estimated_only_by_the_cuda_backend(self, monkeypatch):
        import zvisiongenerator.backends as backends
        import zvisiongenerator.backends.cuda.video.memory_estimate as memory_estimate

        estimate = object()
        monkeypatch.setattr(memory_estimate, "cached_ltx_system_estimate", lambda config, model, quantize, *, find_local_dir: estimate)
        monkeypatch.setattr(backends.sys, "platform", "linux")
        assert backends.estimate_ltx_system_memory({}, "owner/ltx", 4, find_local_dir=lambda reference: None) is estimate
        monkeypatch.setattr(backends.sys, "platform", "darwin")
        assert backends.estimate_ltx_system_memory({}, "owner/ltx", 4, find_local_dir=lambda reference: None) is None

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

    @pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
    def test_every_supported_platform_stores_quants(self, monkeypatch, platform):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", platform)

        assert backends.supports_stored_quants() is True

    @pytest.mark.parametrize(("platform", "expected"), [("darwin", True), ("win32", True), ("linux", True), ("freebsd", False)])
    def test_every_supported_video_platform_writes_audio(self, monkeypatch, platform, expected):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", platform)

        assert backends.supports_video_audio() is expected

    @pytest.mark.parametrize(("platform", "expected"), [("darwin", True), ("win32", True), ("linux", True)])
    def test_every_ltx_video_backend_extends_a_clip(self, monkeypatch, platform, expected):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", platform)

        assert backends.supports_video_extend() is expected

    def test_a_failing_memory_probe_means_no_budget(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends, "get_accelerator_memory_budget", MagicMock(side_effect=RuntimeError("probe failed")))
        backends.accelerator_memory_budget_or_none.cache_clear()
        try:
            assert backends.accelerator_memory_budget_or_none() is None
        finally:
            backends.accelerator_memory_budget_or_none.cache_clear()

    def test_the_memory_budget_is_probed_once_per_process(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.utils.model_memory import MemoryBudget

        probe = MagicMock(return_value=MemoryBudget(10))
        monkeypatch.setattr(backends, "get_accelerator_memory_budget", probe)
        backends.accelerator_memory_budget_or_none.cache_clear()
        try:
            assert backends.accelerator_memory_budget_or_none() == backends.accelerator_memory_budget_or_none() == MemoryBudget(10)
        finally:
            backends.accelerator_memory_budget_or_none.cache_clear()

        probe.assert_called_once()

    def test_an_unregistered_platform_cannot_extend_a_clip(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "freebsd")

        assert backends.supports_video_extend() is False

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_cuda_memory_budget_is_vram_plus_system_memory(self, monkeypatch, platform):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.cuda import memory as memory_cuda
        from zvisiongenerator.utils.model_memory import MemoryBudget

        monkeypatch.setattr(backends.sys, "platform", platform)
        monkeypatch.setattr(memory_cuda, "memory_sizes", lambda: (10, 30))

        assert backends.get_accelerator_memory_budget() == MemoryBudget(10, 30)

    def test_mac_memory_budget_is_the_recommended_working_set_with_the_unified_memory(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.mac import memory as memory_mac
        from zvisiongenerator.utils.model_memory import MemoryBudget

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(memory_mac, "memory_budget_bytes", lambda: 24)
        monkeypatch.setattr(memory_mac, "memory_size_bytes", lambda: 32)

        assert backends.get_accelerator_memory_budget() == MemoryBudget(24, unified_bytes=32)

    def test_mac_without_a_working_set_size_has_no_budget(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.mac import memory as memory_mac

        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(memory_mac, "memory_budget_bytes", lambda: None)
        monkeypatch.setattr(memory_mac, "memory_size_bytes", lambda: 32)

        assert backends.get_accelerator_memory_budget() is None

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_cuda_platforms_release_finished_models(self, monkeypatch, platform):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.cuda import memory as memory_cuda

        released: list[bool] = []
        monkeypatch.setattr(backends.sys, "platform", platform)
        monkeypatch.setattr(memory_cuda, "release_memory", lambda: released.append(True))

        backends.release_accelerator_memory()

        assert released == [True]

    def test_gpu_memory_comes_from_nvidia_smi_without_starting_cuda(self, monkeypatch):
        import shutil
        import subprocess

        from zvisiongenerator.backends.cuda import memory as memory_cuda

        psutil_mod = MagicMock()
        psutil_mod.virtual_memory.return_value.total = 30
        monkeypatch.setitem(sys.modules, "psutil", psutil_mod)
        monkeypatch.setitem(sys.modules, "torch", None)  # importing torch would raise
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: MagicMock(stdout="10240\n"))

        assert memory_cuda.memory_sizes() == (10240 * 1024 * 1024, 30)

    def test_drop_cached_files_syncs_and_drops_every_file(self, monkeypatch, tmp_path):
        import os

        from zvisiongenerator.utils import page_cache

        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "weights.safetensors").write_bytes(b"w")
        (tmp_path / "config.json").write_text("{}")
        dropped: list[int] = []
        monkeypatch.setattr(os, "posix_fadvise", lambda fd, _offset, _length, advice: dropped.append(advice), raising=False)
        monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)

        page_cache.drop_cached_files(tmp_path)

        assert dropped == [4, 4]

    def test_release_memory_trims_the_heap(self, monkeypatch):
        from zvisiongenerator.backends.cuda import memory as memory_cuda

        trimmed: list[bool] = []
        monkeypatch.setitem(sys.modules, "torch", MagicMock())
        monkeypatch.setattr(memory_cuda, "_trim_heap", lambda: trimmed.append(True))

        memory_cuda.release_memory()

        assert trimmed == [True]

    def test_no_nvidia_smi_means_no_sizes(self, monkeypatch):
        import shutil

        from zvisiongenerator.backends.cuda import memory as memory_cuda

        monkeypatch.setattr(shutil, "which", lambda _name: None)

        assert memory_cuda.memory_sizes() is None

    def test_no_cuda_device_means_no_budget(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.cuda import memory as memory_cuda

        monkeypatch.setattr(backends.sys, "platform", "linux")
        monkeypatch.setattr(memory_cuda, "memory_sizes", lambda: None)

        assert backends.get_accelerator_memory_budget() is None

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
        pytest.importorskip("mflux")
        from zvisiongenerator.backends.mac.image import MfluxBackend

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
        pytest.importorskip("mflux")
        from zvisiongenerator.backends.mac.image import MfluxBackend

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
        from zvisiongenerator.backends.cuda.image.backend import DiffusersBackend

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
        from zvisiongenerator.backends.cuda.image.backend import DiffusersBackend

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

        import zvisiongenerator.backends.mac.image as image_mac_module

        return image_mac_module

    def test_ideogram4_load_model_uses_fp8_config_and_forwards_quantize(self):
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
                "/models/ideo",
                quantize=4,
                lora_paths=["/tmp/style.safetensors"],
                lora_weights=[0.75],
            )

        assert loaded_info == model_info
        assert model is ideogram_ctor.return_value
        assert ideogram_ctor.call_args.kwargs == {
            "quantize": 4,
            "model_path": "/models/ideo",
            "model_config": "ideogram4-config",
            "lora_paths": ["/tmp/style.safetensors"],
            "lora_scales": [0.75],
        }

    def test_ideogram4_loads_without_the_unconditional_component_and_restores_mflux(self):
        image_mac_module = self._import_backend_module()
        definition = image_mac_module.Ideogram4WeightDefinition
        initializer = image_mac_module.Ideogram4Initializer
        originals = (definition.get_components, initializer._init_models, initializer._apply_lora)

        with image_mac_module._instant_ideogram4():
            names = [component.name for component in definition.get_components()]
            assert initializer._apply_lora != originals[2]

        assert "unconditional_transformer" not in names and "conditional_transformer" in names
        assert (definition.get_components, initializer._init_models, initializer._apply_lora) == originals

    def test_ideogram4_loras_go_to_the_conditional_transformer_only(self):
        image_mac_module = self._import_backend_module()
        model = MagicMock()

        with pytest.MonkeyPatch.context() as monkeypatch:
            loader = MagicMock()
            loader.load_and_apply_lora.return_value = (["/l/a.safetensors"], [0.7])
            monkeypatch.setattr(image_mac_module, "LoRALoader", loader)
            with image_mac_module._instant_ideogram4():
                image_mac_module.Ideogram4Initializer._apply_lora(model, ["/l/a.safetensors"], [0.7], True)

        assert loader.load_and_apply_lora.call_args.kwargs["transformer"] is model.conditional_transformer
        loader._apply_single_lora.assert_not_called()
        assert (model.lora_paths, model.lora_scales) == (["/l/a.safetensors"], [0.7])

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
            steps=8,
            guidance=1.0,
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
        assert "preset" not in kwargs
        assert kwargs["num_inference_steps"] == 8
        assert kwargs["guidance"] == 1.0
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
            steps=8,
            guidance=1.0,
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
        assert "preset" not in kwargs
        assert kwargs["num_inference_steps"] == 8
        assert kwargs["guidance"] == 1.0
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
            steps=8,
            guidance=1.0,
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

    def test_ideogram4_text_to_image_keeps_a_whitespace_prefixed_json_caption_on_one_line(self):
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
            steps=8,
            guidance=1.0,
            scheduler="beta",
            negative_prompt="blurry",
            steps_explicit=True,
            guidance_explicit=True,
        )

        assert image is not None
        model.generate_image.assert_called_once()
        kwargs = model.generate_image.call_args.kwargs
        assert kwargs["prompt"] == prompt.strip()
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

        assert image_mac_module._wrap_ideogram4_prompt(prompt) == prompt.strip()

    def test_ideogram4_text_to_image_runs_the_given_steps_at_the_built_in_guidance(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        backend.text_to_image(model=model, prompt="a lighthouse in fog", width=832, height=1216, seed=42, steps=12, guidance=1.0)

        kwargs = model.generate_image.call_args.kwargs
        assert (kwargs["num_inference_steps"], kwargs["guidance"]) == (12, 1.0)
        assert "preset" not in kwargs

    def test_ideogram4_text_to_image_ignores_another_guidance_with_a_warning(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="ideogram4", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()

        with pytest.warns(UserWarning, match="guidance"):
            backend.text_to_image(model=model, prompt="a lighthouse in fog", width=832, height=1216, seed=42, steps=8, guidance=7.0, guidance_explicit=True)

        assert model.generate_image.call_args.kwargs["guidance"] == 1.0

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

        def _capture_generate_image(**kwargs):
            del kwargs
            captured_effective_sigmas.append(image_mac_module._effective_initial_sigma())
            return _make_generated_image_result()

        model.generate_image.side_effect = _capture_generate_image

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(image_mac_module, "IDEOGRAM4_INITIAL_SIGMA", 1.004)
            had_previous_value = hasattr(image_mac_module._initial_sigma_override, "value")
            previous_value = getattr(image_mac_module._initial_sigma_override, "value", image_mac_module._INITIAL_SIGMA_UNSET)
            if had_previous_value:
                delattr(image_mac_module._initial_sigma_override, "value")

            try:
                image = backend.text_to_image(
                    model=model,
                    prompt='{"type":"caption","text":"keep verbatim"}',
                    width=1024,
                    height=1024,
                    seed=123,
                    steps=8,
                    guidance=1.0,
                    steps_explicit=True,
                    guidance_explicit=True,
                    first_sigma=first_sigma,
                )
            finally:
                if had_previous_value:
                    image_mac_module._initial_sigma_override.value = previous_value
                elif hasattr(image_mac_module._initial_sigma_override, "value"):
                    delattr(image_mac_module._initial_sigma_override, "value")

            assert image is not None
            model.generate_image.assert_called_once()
            assert captured_effective_sigmas[0] == pytest.approx(expected_effective_sigma)
            assert "first_sigma" not in model.generate_image.call_args.kwargs
            assert image_mac_module._effective_initial_sigma() == 1.004

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


@pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")
class TestKrea2MfluxBackend:
    def _import_backend_module(self):
        pytest.importorskip("mflux")
        pytest.importorskip("mlx.core")

        import zvisiongenerator.backends.mac.image as image_mac_module

        return image_mac_module

    def test_krea2_load_model_forwards_quantize_and_loras(self):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        model_info = image_mac_module.ImageModelInfo(family="krea2", is_distilled=False, size=None)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(image_mac_module, "detect_image_model", lambda _path: model_info)
            monkeypatch.setattr(image_mac_module, "_upcast_model_weights", MagicMock())
            monkeypatch.setattr(image_mac_module, "materialize_weights", MagicMock())
            monkeypatch.setattr(image_mac_module, "apply_buffer_cache_policy", MagicMock())
            krea2_ctor = MagicMock(return_value=MagicMock())
            monkeypatch.setattr(image_mac_module, "Krea2", krea2_ctor)

            model, loaded_info = backend.load_model("unsloth/Krea-2-Turbo", quantize=4, lora_paths=["/tmp/style.safetensors"], lora_weights=[0.8])

        assert loaded_info == model_info
        assert model is krea2_ctor.return_value
        kwargs = krea2_ctor.call_args.kwargs
        assert {key: value for key, value in kwargs.items() if key != "model_config"} == {
            "quantize": 4,
            "model_path": "unsloth/Krea-2-Turbo",
            "lora_paths": ["/tmp/style.safetensors"],
            "lora_scales": [0.8],
        }
        assert kwargs["model_config"].model_name == image_mac_module.ModelConfig.krea2().model_name

    def test_krea2_turbo_config_pins_the_timestep_shift_at_every_size(self):
        image_mac_module = self._import_backend_module()
        from mflux.models.common.config.config import Config

        def first_img2img_sigma(model_config, width, height):
            config = Config(model_config=model_config, num_inference_steps=20, width=width, height=height, image_path="x.png", image_strength=0.8)
            return float(config.scheduler.sigmas[config.init_time_step])

        pinned = image_mac_module._krea2_turbo_config()

        # Krea trained Turbo at mu = 1.15: denoise 0.2 starts at sigma e^1.15 / (e^1.15 + 4) whatever the size.
        expected = math.exp(1.15) / (math.exp(1.15) + 4)
        assert first_img2img_sigma(pinned, 1024, 1024) == pytest.approx(expected, abs=1e-3)
        assert first_img2img_sigma(pinned, 1664, 2432) == pytest.approx(expected, abs=1e-3)
        # The shared mflux config keeps its size-dependent shift.
        assert first_img2img_sigma(image_mac_module.ModelConfig.krea2(), 1664, 2432) > expected + 0.1

    @pytest.mark.parametrize("method", ["text_to_image", "image_to_image"])
    def test_krea2_samples_with_euler(self, method):
        image_mac_module = self._import_backend_module()
        backend = image_mac_module.MfluxBackend()
        backend._model_info = image_mac_module.ImageModelInfo(family="krea2", is_distilled=False, size=None)
        model = MagicMock()
        model.generate_image.return_value = _make_generated_image_result()
        common = dict(model=model, prompt="a fox", seed=1, steps=8, guidance=1.0)

        if method == "text_to_image":
            backend.text_to_image(width=64, height=64, **common)
        else:
            backend.image_to_image(image=Image.new("RGB", (64, 64)), strength=0.4, **common)

        assert model.generate_image.call_args.kwargs["scheduler"] == "euler"


@pytest.mark.skipif(sys.platform != "darwin", reason="Selective MLX upcast only runs on macOS")
class TestSelectiveMacWeightUpcast:
    def test_upcast_model_weights_preserves_uint32_quantized_weights(self):
        mx = pytest.importorskip("mlx.core")

        from zvisiongenerator.backends.mac.image import _upcast_model_weights

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
