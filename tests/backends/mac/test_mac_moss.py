"""Tests for MOSS-Image on macOS: the staged load, prompt encoding, the denoising loop, the stored copy and the backend wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.core import JobPrompt
from zvisiongenerator.core.moss import MossSampler, img2img_run_steps

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")


@pytest.fixture
def mx():
    pytest.importorskip("mflux")
    return pytest.importorskip("mlx.core")


class _Transformer:
    """Record each pass and predict a constant velocity of the latents' shape."""

    def __init__(self, mx, events):
        self._mx = mx
        self.events = events
        self.calls: list[dict] = []

    def __call__(self, latents, timestep, embeds, skip_layers=frozenset()):
        self.calls.append({"timestep": timestep, "embeds": embeds, "skip_layers": skip_layers})
        self.events.append("denoise")
        return self._mx.ones(latents.shape)


class _Vae:
    """Encode to zeros and decode to a mid-grey image of the latents' size."""

    def __init__(self, mx):
        self._mx = mx
        self.bn = SimpleNamespace(running_mean=mx.zeros((128,)), running_var=mx.ones((128,)), eps=0.0)
        self.decoded: list = []

    def decode_packed_latents(self, latents, tiling_config=None):
        self.decoded.append(latents)
        _, _, height, width = latents.shape
        return self._mx.zeros((1, 3, height * 16, width * 16))


@pytest.fixture
def stages(mx, monkeypatch):
    """Replace mflux's loading, encoding and saving with recorders; return the shared event list and the fakes."""
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_prompt_encoder import MOSSImagePromptEncoder

    import zvisiongenerator.backends.mac.moss as module

    events: list = []
    fakes = SimpleNamespace(events=events, transformer=_Transformer(mx, events), vae=_Vae(mx), text_encoder=object(), saved=[], loras=[])

    def apply_and_quantize(weights, models, quantize_arg, weight_definition):
        assert weights == ("weights", tuple(component.name for component in weight_definition.get_components()))
        events.append(("load", tuple(models), quantize_arg))
        return quantize_arg

    def encode(text, tokenizer, text_encoder):
        assert text_encoder is fakes.text_encoder
        events.append(("encode", text))
        return mx.full((1, 3, 32), float(len(text)))

    def save_model(model, bits, base_path, weight_definition):
        parts = tuple(name for name in ("text_encoder", "transformer", "vae") if getattr(model, name) is not None)
        events.append(("save", parts, bits, base_path))

    components = {"text_encoder": fakes.text_encoder, "transformer": fakes.transformer, "vae": fakes.vae}
    monkeypatch.setattr(module, "_new_component", components.__getitem__)
    monkeypatch.setattr(WeightLoader, "load", staticmethod(lambda weight_definition, model_path: ("weights", tuple(component.name for component in weight_definition.get_components()))))
    monkeypatch.setattr(WeightApplier, "apply_and_quantize", staticmethod(apply_and_quantize))
    monkeypatch.setattr(module, "upcast_weights", lambda _parts, names: names and events.append(("upcast", tuple(names))))
    monkeypatch.setattr(module, "release_memory", lambda: events.append("free"))
    monkeypatch.setattr(module, "read_header", lambda _path: (0, {"diffusion_model.layers.0.proj.q.lora_down.weight": {}}))
    monkeypatch.setattr(TokenizerLoader, "load_all", staticmethod(lambda definitions, model_path: {"qwen3_5": object()}))
    monkeypatch.setattr(MOSSImagePromptEncoder, "encode", staticmethod(encode))
    monkeypatch.setattr(ModelSaver, "save_model", staticmethod(save_model))
    monkeypatch.setattr(LoRALoader, "load_and_apply_lora", staticmethod(lambda **kwargs: fakes.loras.append(kwargs) or ([], [])))

    def vae_encode(vae, image, tiling_config=None):
        events.append(("encode_image", tiling_config is not None))
        return mx.zeros((1, 32, image.shape[2] // 8, image.shape[3] // 8))

    monkeypatch.setattr(VAEUtil, "encode", staticmethod(vae_encode))
    return fakes


def _moss():
    import zvisiongenerator.backends.mac.moss as module

    return module


def _noise(*, seed):
    from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator

    return MOSSImageLatentCreator.create_noise(seed=seed, height=32, width=32)


def _sft(steps=4, guidance=3.0):
    return MossSampler(steps=steps, guidance=guidance, negative_prompt="", skip_layers=(22,))


def _turbo(steps=4):
    return MossSampler(steps=steps, guidance=1.0, negative_prompt=None, skip_layers=())


class TestStages:
    def test_the_text_encoder_is_freed_before_the_denoiser_loads(self, stages):
        module = _moss()
        model = module.load("/models/moss", 4, lora_paths=["/l/turbo.safetensors"], lora_scales=[1.0])

        assert module.encode_prompts(model, ["a", "b", "a"])
        module.ensure_denoiser(model)

        assert stages.events == [
            ("load", ("text_encoder",), 4),
            ("encode", "a"),
            ("encode", "b"),
            "free",
            ("load", ("transformer", "vae"), 4),
        ]
        assert model.text_encoder is None
        assert model.transformer is stages.transformer

    def test_the_loras_apply_unmerged_to_the_transformer(self, stages):
        module = _moss()
        model = module.load("/models/moss", None, lora_paths=["/l/turbo.safetensors", "/l/user.safetensors"], lora_scales=[1.0, 0.7])

        module.ensure_denoiser(model)

        (call,) = stages.loras
        assert call["transformer"] is stages.transformer
        assert call["lora_paths"] == ["/l/turbo.safetensors", "/l/user.safetensors"]
        assert call["lora_scales"] == [1.0, 0.7]
        assert call["bake_lora"] is False

    def test_the_upcast_components_are_upcast_as_their_stage_loads(self, stages):
        module = _moss()

        model = module.load("/models/moss", None, upcast=("vae",))
        module.ensure_denoiser(model)
        module.ensure_denoiser(model)

        assert [event for event in stages.events if event[0] in ("load", "upcast")] == [("load", ("text_encoder",), None), ("load", ("transformer", "vae"), None), ("upcast", ("vae",))]

    def test_a_prompt_missing_from_the_cache_frees_the_transformer_and_loads_the_encoder_again(self, stages):
        module = _moss()
        model = module.load("/models/moss", None)
        module.encode_prompts(model, ["a"])
        module.ensure_denoiser(model)
        stages.events.clear()

        module.encode_prompts(model, ["a", "new"])

        assert stages.events == ["free", ("load", ("text_encoder",), None), ("encode", "new"), "free"]
        assert model.transformer is None
        assert model.vae is stages.vae

    def test_a_cancel_keeps_the_encoder_loaded(self, stages):
        module = _moss()
        model = module.load("/models/moss", None)

        assert not module.encode_prompts(model, ["a"], cancelled=lambda: True)

        assert model.text_encoder is stages.text_encoder
        assert "free" not in stages.events


class TestJobTexts:
    def test_turbo_encodes_only_the_prompts(self):
        prompts = [JobPrompt("a", "bad"), JobPrompt("b", None), JobPrompt("a", None)]

        assert _moss().job_texts(prompts, turbo=True) == ("a", "b")

    def test_sft_also_encodes_the_negatives_and_the_empty_one(self):
        prompts = [JobPrompt("a", "bad"), JobPrompt("b", None)]

        assert _moss().job_texts(prompts, turbo=False) == ("a", "b", "bad", "")


class TestGenerate:
    def test_text_to_image_runs_every_step_with_the_guided_passes(self, mx, stages):
        module = _moss()
        model = module.load("/models/moss-sft", None)

        image = module.generate(model, "a", width=64, height=32, seed=1, sampler=_sft())

        assert image.size == (64, 32)
        conditional = stages.transformer.calls[0::2]
        unconditional = stages.transformer.calls[1::2]
        assert len(conditional) == len(unconditional) == 4
        assert all(call["skip_layers"] == frozenset() for call in conditional)
        assert all(call["skip_layers"] == frozenset({22}) for call in unconditional)
        assert np.allclose(np.array(unconditional[0]["embeds"]), 0.0)  # the empty negative prompt

    def test_turbo_runs_one_pass_per_step(self, stages):
        module = _moss()
        model = module.load("/models/moss", None)

        module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo())

        assert len(stages.transformer.calls) == 4

    def test_image_to_image_runs_the_tail_of_the_schedule_from_the_noised_image(self, mx, stages):
        from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler

        module = _moss()
        model = module.load("/models/moss", None)
        steps, strength = 8, 0.4
        started = []
        model.callbacks.register(SimpleNamespace(call_before_loop=lambda seed, prompt, latents, config, **_: started.append((latents, config))))

        module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo(steps), image=Image.new("RGB", (32, 32)), strength=strength)

        run_steps = img2img_run_steps(steps, strength)
        assert len(stages.transformer.calls) == run_steps
        sigmas = MOSSImageScheduler.compute_sigmas(steps)
        assert stages.transformer.calls[0]["timestep"] == pytest.approx(float(sigmas[steps - run_steps]) * 1000, rel=1e-5)
        latents, config = started[0]
        assert config.init_time_step == steps - run_steps
        noise = _noise(seed=1)
        assert np.allclose(np.array(latents), float(sigmas[steps - run_steps]) * np.array(noise), atol=1e-5)  # the image encodes to zeros

    def test_a_large_image_frees_the_transformer_before_the_decode_and_the_next_image_reloads_it(self, stages, monkeypatch):
        module = _moss()
        monkeypatch.setattr(module, "_DROP_TRANSFORMER_PIXELS", 32 * 32)
        model = module.load("/models/moss", None)

        module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo(1))
        assert model.transformer is stages.transformer
        stages.events.clear()

        module.generate(model, "a", width=64, height=32, seed=1, sampler=_turbo(1))
        assert stages.events == ["denoise", "free"]
        assert model.transformer is None
        stages.events.clear()

        module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo(1))
        assert stages.events == [("load", ("transformer",), None), "denoise"]  # the VAE stayed

    def test_a_large_start_image_is_encoded_in_tiles_without_the_transformer(self, stages, monkeypatch):
        module = _moss()
        monkeypatch.setattr(module, "_DROP_TRANSFORMER_PIXELS", 32 * 32)
        model = module.load("/models/moss", None)
        module.encode_prompts(model, ["a"])
        module.ensure_denoiser(model)
        stages.events.clear()

        module.generate(model, "a", width=64, height=32, seed=1, sampler=_turbo(2), image=Image.new("RGB", (64, 32)), strength=0.5)

        assert stages.events[:3] == ["free", ("encode_image", True), ("load", ("transformer",), None)]

    def test_a_small_start_image_is_encoded_whole_with_the_transformer_loaded(self, stages):
        module = _moss()
        model = module.load("/models/moss", None)
        module.encode_prompts(model, ["a"])
        module.ensure_denoiser(model)
        stages.events.clear()

        module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo(2), image=Image.new("RGB", (32, 32)), strength=0.5)

        assert stages.events[0] == ("encode_image", False)
        assert "free" not in stages.events

    def test_each_step_is_reported_and_a_callback_can_stop_the_run(self, stages):
        from mflux.utils.exceptions import StopImageGenerationException

        module = _moss()
        model = module.load("/models/moss", None)
        steps = []

        def in_loop(t, seed, prompt, latents, config, time_steps, **_):
            steps.append(t)
            if t == 1:
                raise StopImageGenerationException("stop")

        model.callbacks.register(SimpleNamespace(call_in_loop=in_loop))

        with pytest.raises(StopImageGenerationException):
            module.generate(model, "a", width=32, height=32, seed=1, sampler=_turbo())

        assert steps == [0, 1]
        assert stages.vae.decoded == []


class TestSave:
    def test_each_stage_is_written_on_its_own_without_loras_and_nothing_stays_loaded(self, stages):
        module = _moss()
        model = module.load("/models/moss", 8, lora_paths=["/l/turbo.safetensors"], lora_scales=[1.0])
        stages.events.clear()

        module.save(model, "/models/.moss@q8.partial")

        # The loaded text encoder is written as it is; the denoiser loads for its write without LoRAs.
        assert stages.events == [
            ("save", ("text_encoder",), 8, "/models/.moss@q8.partial"),
            "free",
            ("load", ("transformer", "vae"), 8),
            ("save", ("transformer", "vae"), 8, "/models/.moss@q8.partial"),
            "free",
        ]
        assert stages.loras == []
        assert model.text_encoder is model.transformer is model.vae is None

    def test_a_shared_copy_is_linked_into_the_new_one(self, tmp_path):
        module = _moss()
        shared = tmp_path / "moss@q4"
        for name in module._SAVED_FOLDERS:
            (shared / name).mkdir(parents=True)
            (shared / name / "0.safetensors").write_bytes(b"weights")

        module.link_saved_copy(shared, tmp_path / "partial")

        for name in module._SAVED_FOLDERS:
            assert (tmp_path / "partial" / name / "0.safetensors").stat().st_ino == (shared / name / "0.safetensors").stat().st_ino


@pytest.fixture
def backend(mx, monkeypatch, tmp_path):
    import zvisiongenerator.backends.mac.image as module

    monkeypatch.setattr(module.ModelConfig, "precision", module.ModelConfig.precision)
    monkeypatch.setattr(module, "upcast_weights", MagicMock())
    return module


def _use_moss(monkeypatch, module, *, turbo, built=True):
    monkeypatch.setattr(module, "detect_image_model", lambda _path: module.ImageModelInfo(family="moss", is_distilled=turbo, size=None))
    if built:  # otherwise the real check refuses the path
        monkeypatch.setattr(module, "require_built", lambda path: path)


class TestBackend:
    def test_turbo_loads_with_its_lora_first_at_full_strength(self, backend, monkeypatch, tmp_path):
        _use_moss(monkeypatch, backend, turbo=True)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: tmp_path / "turbo_lora.safetensors")
        load = MagicMock(return_value="model")
        monkeypatch.setattr(backend._moss, "load", load)

        model, info = backend.MfluxBackend().load_model("/models/moss", quantize=4, lora_paths=["/l/user.safetensors"], lora_weights=[0.5])

        assert model == "model"
        assert info.family == "moss"
        assert load.call_args.args == ("/models/moss", 4)
        assert load.call_args.kwargs["lora_paths"] == [str(tmp_path / "turbo_lora.safetensors"), "/l/user.safetensors"]
        assert load.call_args.kwargs["lora_scales"] == [1.0, 0.5]

    def test_a_folder_that_was_not_built_is_refused(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=True, built=False)
        load = MagicMock()
        monkeypatch.setattr(backend._moss, "load", load)

        with pytest.raises(RuntimeError, match="folders built on first use"):
            backend.MfluxBackend().load_model("/elsewhere/moss")

        load.assert_not_called()

    def test_a_quit_during_encoding_leaves_the_denoiser_unloaded(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=True)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value="model"))
        monkeypatch.setattr(backend._moss, "encode_prompts", lambda model, texts, cancelled: True)
        ensure = MagicMock()
        monkeypatch.setattr(backend._moss, "ensure_denoiser", ensure)
        mflux = backend.MfluxBackend()
        model, _info = mflux.load_model("/models/moss")

        mflux.prepare_prompts(model, [JobPrompt("a", None)], width=64, height=64, cancelled=lambda: True)

        ensure.assert_not_called()

    def test_sft_loads_only_the_user_loras(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=False)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        load = MagicMock()
        monkeypatch.setattr(backend._moss, "load", load)

        backend.MfluxBackend().load_model("/models/moss-sft", lora_paths=["/l/user.safetensors"])

        assert load.call_args.kwargs["lora_paths"] == ["/l/user.safetensors"]
        assert load.call_args.kwargs["lora_scales"] == [1.0]

    def test_prompts_are_encoded_before_the_denoiser_loads(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=False)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value="model"))
        calls = []
        monkeypatch.setattr(backend._moss, "encode_prompts", lambda model, texts, cancelled: calls.append(("encode", texts)) or True)
        monkeypatch.setattr(backend._moss, "ensure_denoiser", lambda model: calls.append(("denoiser",)))
        mflux = backend.MfluxBackend()
        model, _info = mflux.load_model("/models/moss-sft")

        mflux.prepare_prompts(model, [JobPrompt("a", "bad")], width=64, height=64)

        assert calls == [("encode", ("a", "bad", "")), ("denoiser",)]

    def test_image_to_image_floors_the_size_to_16_and_passes_the_strength(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=True)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        model = SimpleNamespace(callbacks=MagicMock())
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value=model))
        generate = MagicMock(return_value="image")
        monkeypatch.setattr(backend._moss, "generate", generate)
        mflux = backend.MfluxBackend()
        mflux.load_model("/models/moss")

        with pytest.warns(UserWarning, match="guidance 3 is ignored"):
            result = mflux.image_to_image(model, Image.new("RGB", (70, 50)), "a", strength=0.4, steps=5, seed=3, guidance=3.0)

        assert result == "image"
        kwargs = generate.call_args.kwargs
        assert (kwargs["width"], kwargs["height"], kwargs["strength"]) == (64, 48, 0.4)
        assert kwargs["sampler"].guidance == 1.0  # Turbo runs without guidance

    def test_a_skip_returns_none_and_the_callbacks_are_removed(self, backend, monkeypatch):
        from mflux.callbacks.callback_registry import CallbackRegistry
        from mflux.utils.exceptions import StopImageGenerationException

        _use_moss(monkeypatch, backend, turbo=True)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        model = SimpleNamespace(callbacks=CallbackRegistry())
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value=model))
        monkeypatch.setattr(backend._moss, "generate", MagicMock(side_effect=StopImageGenerationException("skip")))
        mflux = backend.MfluxBackend()
        mflux.load_model("/models/moss")

        result = mflux.text_to_image(model, "a", 64, 64, seed=1, steps=4, guidance=1.0, skip_signal=MagicMock(), step_callback=MagicMock())

        assert result is None
        assert model.callbacks.in_loop == [] and model.callbacks.before_loop == []

    def test_only_moss_needs_a_load_from_the_copy_after_a_save(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=False)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value="model"))
        mflux = backend.MfluxBackend()
        assert mflux.runs_after_save("model")  # nothing loaded yet

        mflux.load_model("/models/moss-sft", quantize=8)

        assert not mflux.runs_after_save("model")

    def test_a_stored_copy_is_saved_stage_by_stage(self, backend, monkeypatch):
        _use_moss(monkeypatch, backend, turbo=False)
        monkeypatch.setattr(backend, "turbo_lora_for", lambda path, turbo: None)
        model = SimpleNamespace(path="/models/moss-sft", quantize=8)
        monkeypatch.setattr(backend._moss, "load", MagicMock(return_value=model))
        save = MagicMock()
        monkeypatch.setattr(backend._moss, "save", save)
        mflux = backend.MfluxBackend()
        mflux.load_model("/models/moss-sft", quantize=8)

        mflux.save_quantized(model, "/models/.moss-sft@q8.partial")

        save.assert_called_once_with(model, "/models/.moss-sft@q8.partial")

    def test_a_copy_the_other_alias_shares_is_linked_from_files(self, backend, monkeypatch, tmp_path):
        shared = tmp_path / "moss@q8"
        monkeypatch.setattr(backend, "is_moss_dir", lambda path: True)
        monkeypatch.setattr(backend, "shared_quant_source", lambda source, bits, fmt: shared if bits == 8 else None)
        link = MagicMock()
        monkeypatch.setattr(backend._moss, "link_saved_copy", link)
        mflux = backend.MfluxBackend()

        assert mflux.quantizes_from_files(8, source="/models/moss-sft")
        assert not mflux.quantizes_from_files(4, source="/models/moss-sft")
        mflux.write_quantized_files("/models/moss-sft", "/models/.moss-sft@q8.partial", 8)

        link.assert_called_once_with(shared, backend.Path("/models/.moss-sft@q8.partial"))
        with pytest.raises(RuntimeError, match="No stored copy at q4"):
            mflux.write_quantized_files("/models/moss-sft", "/models/.moss-sft@q4.partial", 4)
