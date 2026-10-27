"""Tests for Ideogram 4 Instant on the diffusers/CUDA image backend, with mocked pipelines."""

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

    def test_a_call_without_sigmas_passes_through(self, ideogram4):
        module, _ = ideogram4
        calls: list[tuple] = []

        class _StepScheduler:
            def set_timesteps(self, num_inference_steps, device=None):
                calls.append((num_inference_steps, device))

        scheduler = _StepScheduler()
        with module.first_sigma_override(scheduler, 1.004):
            scheduler.set_timesteps(20, device="cuda")

        assert calls == [(20, "cuda")]


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
            "sampler": Ideogram4Sampler(steps=8, mu=0.0, std=1.75),
            "first_sigma": None,
            "callback_kwargs": {},
            **overrides,
        }
        return module.generate(pipeline, "a red car", **options)

    def test_the_pipeline_gets_the_sampler_built_in_guidance_a_json_caption_and_the_default_first_sigma(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        result = self._generate(module, pipeline)

        assert result.images == ["image"]
        (call,) = pipeline.calls
        assert json.loads(call["prompt"])["high_level_description"] == "a red car"
        assert (call["width"], call["height"], call["steps"]) == (1024, 768, 8)
        assert (call["scale"], call["schedule"]) == (1.0, None)
        assert (call["mu"], call["std"]) == (0.0, 1.75)
        assert pipeline.scheduler.received == [[1.004, 0.5, 0.1]]

    def test_a_per_run_first_sigma_wins_over_the_default(self, ideogram4):
        module, _ = ideogram4
        pipeline = _Pipeline()

        self._generate(module, pipeline, first_sigma=1.006)

        assert pipeline.scheduler.received == [[1.006, 0.5, 0.1]]

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


class TestZeroUnconditionalTransformer:
    def test_it_has_no_weights_reports_its_dtype_and_returns_zeros(self):
        torch = pytest.importorskip("torch")
        sys.modules.pop("zvisiongenerator.backends.cuda.image.ideogram4", None)
        try:
            from zvisiongenerator.backends.cuda.image.ideogram4 import zero_unconditional_transformer

            stand_in = zero_unconditional_transformer(torch.bfloat16)
            hidden = torch.ones(1, 4, 8)

            (output,) = stand_in(hidden_states=hidden, timestep=torch.zeros(1), encoder_hidden_states=None)
        finally:
            sys.modules.pop("zvisiongenerator.backends.cuda.image.ideogram4", None)

        assert list(stand_in.parameters()) == []
        assert stand_in.dtype == torch.bfloat16
        assert torch.equal(output, torch.zeros_like(hidden))


def _write_index(folder, unconditional):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline", "unconditional_transformer": unconditional}), encoding="utf-8")
    return str(folder)


class TestRequireInstant:
    def test_an_instant_folder_is_accepted(self, ideogram4, tmp_path):
        module, _ = ideogram4

        module._require_instant(_write_index(tmp_path / "ideo", [None, None]))

    def test_the_base_model_is_refused_with_the_alias_to_use(self, ideogram4, tmp_path):
        module, _ = ideogram4

        with pytest.raises(RuntimeError, match="'ideo'"):
            module._require_instant(_write_index(tmp_path / "base", ["diffusers", "Ideogram4Transformer2DModel"]))

    def test_a_hub_repo_is_refused(self, ideogram4):
        module, _ = ideogram4

        with pytest.raises(RuntimeError, match="'ideo'"):
            module._require_instant("unsloth/ideogram-4-nf4-diffusers")


class TestLoadPipeline:
    def _fakes(self, module, monkeypatch):
        events: list[str] = []

        def _component(name):
            component = MagicMock(name=name)
            component.to.side_effect = lambda device: events.append(f"{name}->{device}")
            return component

        text_encoder, transformer = _component("text_encoder"), _component("transformer")
        pipeline = SimpleNamespace()
        diffusers, transformers = MagicMock(), MagicMock()
        diffusers.AutoModel.from_pretrained.side_effect = lambda path, subfolder, **kwargs: events.append(f"bf16 {subfolder}") or transformer
        diffusers.Ideogram4Pipeline.from_pretrained.side_effect = lambda path, **kwargs: events.append(f"pipeline unconditional={kwargs['unconditional_transformer']}") or pipeline
        transformers.AutoModel.from_pretrained.side_effect = lambda path, subfolder, **kwargs: events.append(f"load {subfolder}") or text_encoder
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setattr(module, "_require_instant", lambda path: events.append("checked"))
        monkeypatch.setattr(module, "_place", lambda pipe, device, nf4: events.append(f"placed nf4={nf4}"))
        monkeypatch.setattr(module, "zero_unconditional_transformer", lambda dtype: "zero")
        monkeypatch.setattr(module, "load_nf4_transformer", lambda path, dtype, prequantized: events.append(f"nf4 prequantized={prequantized}") or transformer)
        monkeypatch.setattr(module, "load_fp8_transformer", lambda path, dtype: events.append("fp8") or transformer)
        return events, pipeline, transformer

    def test_bf16_loads_the_text_encoder_off_the_gpu_then_one_transformer_and_the_zero_stand_in(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        events, pipeline, _ = self._fakes(module, monkeypatch)

        result = module.load_pipeline("/models/ideo", "bf16", None, prequantized=False)

        assert result is pipeline
        assert events == ["checked", "load text_encoder", "text_encoder->cpu", "bf16 transformer", "pipeline unconditional=None", "placed nf4=False"]
        assert pipeline.unconditional_transformer == "zero"

    def test_q4_loads_an_nf4_transformer_and_moves_it_off_the_gpu(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        events, _, _ = self._fakes(module, monkeypatch)

        module.load_pipeline("/models/ideo@q4", "bf16", 4, prequantized=True)

        assert events[3:] == ["nf4 prequantized=True", "transformer->cpu", "pipeline unconditional=None", "placed nf4=True"]

    def test_q8_streams_an_fp8_transformer(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        events, _, _ = self._fakes(module, monkeypatch)

        module.load_pipeline("/models/ideo", "bf16", 8, prequantized=False)

        assert events[3:] == ["fp8", "pipeline unconditional=None", "placed nf4=False"]


class TestPlace:
    def _pipeline(self):
        return SimpleNamespace(vae=MagicMock(), transformer=MagicMock(), unconditional_transformer=MagicMock(), text_encoder=MagicMock())

    def _record(self, module, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(module, "stream_blocks", lambda blocks, device: calls.append(("stream", blocks)))
        monkeypatch.setattr(module, "move_outside", lambda model, blocks, device: calls.append(("outside", model)))
        monkeypatch.setattr(module, "group_offload", lambda model, device, kind: calls.append(("group", model, kind)))
        monkeypatch.setattr(module, "unload_before", lambda trigger, parts: calls.append(("unload", trigger, list(parts))))
        monkeypatch.setattr(module, "_offload_text_encoder_after_encoding", lambda pipe: calls.append(("offload hook", pipe)))
        return calls

    def test_an_nf4_transformer_streams_by_parameter_swap(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        pipeline = self._pipeline()
        calls = self._record(module, monkeypatch)

        module._place(pipeline, "cuda", nf4=True)

        pipeline.vae.to.assert_called_once_with("cuda")
        pipeline.unconditional_transformer.to.assert_called_once_with("cuda")
        assert calls == [
            ("stream", pipeline.transformer.layers),
            ("outside", pipeline.transformer),
            ("unload", pipeline.transformer, [pipeline.text_encoder]),
            ("offload hook", pipeline),
        ]

    def test_bf16_and_fp8_transformers_stream_with_group_offloading(self, ideogram4, monkeypatch):
        module, _ = ideogram4
        pipeline = self._pipeline()
        calls = self._record(module, monkeypatch)

        module._place(pipeline, "cuda", nf4=False)

        assert calls[0] == ("group", pipeline.transformer, "block_level")
        assert ("stream", pipeline.transformer.layers) not in calls


class TestSaveQuantized:
    def test_the_transformer_is_saved_and_the_shared_components_are_linked_from_the_source(self, ideogram4, tmp_path):
        module, _ = ideogram4
        source = tmp_path / "ideo"
        for name in ("scheduler", "text_encoder", "tokenizer", "vae"):
            (source / name).mkdir(parents=True)
            (source / name / "config.json").write_text(name, encoding="utf-8")
        pipeline = SimpleNamespace(transformer=MagicMock(), _ziv_source_path=str(source))
        target = tmp_path / ".ideo@q4.partial"

        module.save_quantized(pipeline, str(target))

        pipeline.transformer.save_pretrained.assert_called_once_with(str(target / "transformer"))
        linked = target / "text_encoder" / "config.json"
        assert linked.read_text(encoding="utf-8") == "text_encoder"
        assert linked.stat().st_ino == (source / "text_encoder" / "config.json").stat().st_ino
