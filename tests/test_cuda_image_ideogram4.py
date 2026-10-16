"""Tests for Ideogram 4 on the diffusers/CUDA image backend, with mocked pipelines."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.core.ideogram4 import Ideogram4Sampler


@pytest.fixture()
def ideogram4():
    """Import the module with torch faked, as the other CUDA image tests do."""
    torch = MagicMock()
    with patch.dict(sys.modules, {"torch": torch}):
        sys.modules.pop("zvisiongenerator.backends.cuda.image.ideogram4", None)
        import zvisiongenerator.backends.cuda.image.ideogram4 as module

        yield module, torch
    sys.modules.pop("zvisiongenerator.backends.cuda.image.ideogram4", None)


class _Scheduler:
    """Records the sigmas the pipeline hands to ``set_timesteps``."""

    def __init__(self):
        self.received: list[list[float]] = []

    def set_timesteps(self, *, sigmas, device=None):
        self.received.append(list(sigmas))


class TestFirstSigmaOverride:
    def test_the_first_sigma_is_replaced_and_the_others_kept(self, ideogram4):
        module, _ = ideogram4
        scheduler = _Scheduler()

        with module.first_sigma_override(scheduler, 1.004):
            scheduler.set_timesteps(sigmas=[0.9998, 0.5, 0.1], device="cuda")

        assert scheduler.received == [[1.004, 0.5, 0.1]]

    def test_the_scheduler_is_restored_afterwards_also_after_an_error(self, ideogram4):
        module, _ = ideogram4
        scheduler = _Scheduler()

        with pytest.raises(RuntimeError):
            with module.first_sigma_override(scheduler, 1.004):
                raise RuntimeError("boom")
        scheduler.set_timesteps(sigmas=[0.9998, 0.5])

        assert scheduler.received == [[0.9998, 0.5]]
        assert "set_timesteps" not in vars(scheduler)

    def test_no_sigma_leaves_the_schedule_alone(self, ideogram4):
        module, _ = ideogram4
        scheduler = _Scheduler()

        with module.first_sigma_override(scheduler, None):
            scheduler.set_timesteps(sigmas=[0.9998, 0.5])

        assert scheduler.received == [[0.9998, 0.5]]


class _Pipeline:
    """A pipeline stand-in with Ideogram4Pipeline's call signature: an unknown keyword raises ``TypeError``."""

    def __init__(self, fail_with: Exception | None = None):
        self.scheduler = _Scheduler()
        self.text_encoder = MagicMock()
        self.calls: list[dict] = []
        self._fail_with = fail_with

    def __call__(
        self, prompt, height, width, num_inference_steps, guidance_scale, guidance_schedule, mu, std, max_sequence_length, generator, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "height": height,
                "width": width,
                "steps": num_inference_steps,
                "scale": guidance_scale,
                "schedule": guidance_schedule,
                "mu": mu,
                "std": std,
                "max": max_sequence_length,
                "callback": callback_on_step_end,
                "tensor_inputs": callback_on_step_end_tensor_inputs,
            }
        )
        if self._fail_with is not None:
            raise self._fail_with
        self.scheduler.set_timesteps(sigmas=[0.9998, 0.5, 0.1], device="cuda")
        return SimpleNamespace(images=["image"])


class TestGenerate:
    def _generate(self, module, pipeline, **overrides):
        options = {
            "width": 1024,
            "height": 768,
            "generator": object(),
            "sampler": Ideogram4Sampler(steps=20, guidance_schedule=(7.0,) * 18 + (3.0,) * 2, guidance_scale=None, mu=0.0, std=1.75),
            "first_sigma": None,
            "callback_kwargs": {},
            **overrides,
        }
        return module.generate(pipeline, "a red car", **options)

    def test_the_pipeline_gets_the_sampler_a_json_caption_and_the_default_first_sigma(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        result = self._generate(module, pipeline)

        assert result.images == ["image"]
        (call,) = pipeline.calls
        assert json.loads(call["prompt"])["high_level_description"] == "a red car"
        assert (call["width"], call["height"], call["steps"]) == (1024, 768, 20)
        assert call["schedule"] == [7.0] * 18 + [3.0] * 2 and call["scale"] is None
        assert (call["mu"], call["std"]) == (0.0, 1.75)
        assert pipeline.scheduler.received == [[1.004, 0.5, 0.1]]

    def test_a_per_run_first_sigma_wins_over_the_default(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        self._generate(module, pipeline, first_sigma=1.006)

        assert pipeline.scheduler.received == [[1.006, 0.5, 0.1]]

    def test_a_constant_guidance_is_passed_without_a_schedule(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        self._generate(module, pipeline, sampler=Ideogram4Sampler(steps=12, guidance_schedule=None, guidance_scale=5.0))

        (call,) = pipeline.calls
        assert (call["scale"], call["schedule"], call["steps"]) == (5.0, None, 12)

    def test_the_text_encoder_is_on_the_gpu_for_the_call_and_back_on_the_cpu_after_it(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()
        moves: list[str] = []
        pipeline.text_encoder.to = lambda device: moves.append("cpu" if device == "cpu" else "gpu")

        self._generate(module, pipeline)

        assert moves == ["gpu", "cpu"]

    def test_the_text_encoder_goes_back_to_the_cpu_and_the_scheduler_is_restored_when_the_call_fails(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline(fail_with=RuntimeError("out of memory"))
        moves: list[str] = []
        pipeline.text_encoder.to = lambda device: moves.append("cpu" if device == "cpu" else "gpu")

        with pytest.raises(RuntimeError, match="out of memory"):
            self._generate(module, pipeline)

        assert moves == ["gpu", "cpu"]
        assert "set_timesteps" not in vars(pipeline.scheduler)

    def test_the_progress_callback_is_passed_on(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        self._generate(module, pipeline, callback_kwargs={"callback_on_step_end": "callback", "callback_on_step_end_tensor_inputs": ["latents"]})

        assert (pipeline.calls[0]["callback"], pipeline.calls[0]["tensor_inputs"]) == ("callback", ["latents"])


class TestOffloadAfterEncoding:
    def _pipeline(self, hidden_states):
        pipeline = SimpleNamespace(text_encoder=MagicMock(), events=[])
        pipeline.text_encoder.to = lambda device: pipeline.events.append(f"encoder->{device}")
        pipeline._get_text_encoder_hidden_states = hidden_states
        return pipeline

    def test_the_text_encoder_leaves_the_gpu_as_soon_as_its_hidden_states_exist(self, ideogram4):
        module, _ = ideogram4

        def hidden_states(text_encoder, token_ids):
            pipeline.events.append("encoded")
            return ["layer-0", "layer-35"]

        pipeline = self._pipeline(hidden_states)
        module._offload_text_encoder_after_encoding(pipeline)

        result = pipeline._get_text_encoder_hidden_states("encoder", "tokens")

        assert result == ["layer-0", "layer-35"]  # the pipeline's own result, untouched
        assert pipeline.events == ["encoded", "encoder->cpu"]

    def test_it_also_leaves_when_encoding_fails(self, ideogram4):
        module, _ = ideogram4

        def hidden_states(text_encoder, token_ids):
            raise RuntimeError("out of memory")

        pipeline = self._pipeline(hidden_states)
        module._offload_text_encoder_after_encoding(pipeline)

        with pytest.raises(RuntimeError, match="out of memory"):
            pipeline._get_text_encoder_hidden_states("encoder", "tokens")

        assert pipeline.events == ["encoder->cpu"]


class TestRequireNf4:
    def _write_config(self, folder, config):
        (folder / "transformer").mkdir(parents=True)
        (folder / "transformer" / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return str(folder)

    def test_a_bitsandbytes_copy_is_accepted(self, ideogram4, tmp_path):
        module, _ = ideogram4

        module._require_nf4(self._write_config(tmp_path, {"quantization_config": {"quant_method": "bitsandbytes"}}))

    @pytest.mark.parametrize("config", [{}, {"quantization_config": {"quant_method": "fp8"}}])
    def test_a_copy_without_bitsandbytes_is_refused_with_the_repo_to_use(self, ideogram4, tmp_path, config):
        module, _ = ideogram4

        with pytest.raises(RuntimeError, match="ideogram-4-nf4-diffusers"):
            module._require_nf4(self._write_config(tmp_path, config))

    def test_a_hub_repo_is_checked_through_its_cached_config(self, ideogram4, tmp_path, monkeypatch):
        module, _ = ideogram4
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"quantization_config": {"quant_method": "bitsandbytes"}}), encoding="utf-8")
        asked: list[tuple] = []
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, filename: asked.append((repo, filename)) or str(config_path))

        module._require_nf4("owner/ideogram-nf4")

        assert asked == [("owner/ideogram-nf4", "transformer/config.json")]


class TestLoadPipeline:
    def test_each_component_leaves_the_gpu_before_the_next_loads_and_the_pipeline_is_placed(self, ideogram4, monkeypatch):
        module, torch = ideogram4
        events: list[str] = []

        def _component(name):
            component = MagicMock(name=name)
            component.to.side_effect = lambda device: events.append(f"{name}->{device}")
            return component

        text_encoder, transformer, unconditional = _component("text_encoder"), _component("transformer"), _component("unconditional_transformer")
        by_subfolder = {"text_encoder": text_encoder, "transformer": transformer, "unconditional_transformer": unconditional}

        def _load(path, subfolder, **kwargs):
            events.append(f"load {subfolder}")
            return by_subfolder[subfolder]

        pipeline = MagicMock()
        diffusers = MagicMock()
        diffusers.AutoModel.from_pretrained.side_effect = _load
        diffusers.Ideogram4Pipeline.from_pretrained.return_value = pipeline
        transformers = MagicMock()
        transformers.AutoModel.from_pretrained.side_effect = _load
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setattr(module, "_require_nf4", lambda path: events.append("checked"))
        monkeypatch.setattr(module, "_place", lambda pipe, device: events.append("placed"))

        result = module.load_pipeline("owner/nf4", "bf16")

        assert result is pipeline
        assert events == [
            "checked",
            "load text_encoder",
            "text_encoder->cpu",
            "load transformer",
            "transformer->cpu",
            "load unconditional_transformer",
            "unconditional_transformer->cpu",
            "placed",
        ]
        kwargs = diffusers.Ideogram4Pipeline.from_pretrained.call_args.kwargs
        assert (kwargs["text_encoder"], kwargs["transformer"], kwargs["unconditional_transformer"]) == (text_encoder, transformer, unconditional)

    def test_placement_streams_both_transformers_and_sends_the_text_encoder_away_before_the_first_step(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        pipeline = SimpleNamespace(vae=MagicMock(), transformer=MagicMock(), unconditional_transformer=MagicMock(), text_encoder=MagicMock())
        streamed, outside, unloaded = [], [], []
        monkeypatch.setattr(module, "stream_blocks", lambda blocks, device: streamed.append(blocks))
        monkeypatch.setattr(module, "move_outside", lambda model, blocks, device: outside.append(model))
        monkeypatch.setattr(module, "unload_before", lambda trigger, parts: unloaded.append((trigger, list(parts))))
        monkeypatch.setattr(module, "_offload_text_encoder_after_encoding", lambda pipe: unloaded.append(("offload hook", pipe)))

        module._place(pipeline, "cuda")

        pipeline.vae.to.assert_called_once_with("cuda")
        assert streamed == [pipeline.transformer.layers, pipeline.unconditional_transformer.layers]
        assert outside == [pipeline.transformer, pipeline.unconditional_transformer]
        assert unloaded == [(pipeline.transformer, [pipeline.text_encoder]), ("offload hook", pipeline)]
