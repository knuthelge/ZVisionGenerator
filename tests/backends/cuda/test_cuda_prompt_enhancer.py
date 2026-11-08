"""Tests for the presence penalty and the decode path choice of the transformers prompt enhancer."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda import prompt_enhancer  # noqa: E402

_PROMPT = [5, 6, 7]


class _TextStreamerStandIn:
    """Hand every put token's decoded text to ``on_finalized_text``, as the real streamer does for a finished word."""

    ends = 0

    def __init__(self, tokenizer, skip_special_tokens=False):
        self.tokenizer = tokenizer

    def put(self, value):
        self.on_finalized_text(self.tokenizer.decode(value.tolist()))

    def end(self):
        _TextStreamerStandIn.ends += 1
        self.on_finalized_text("")


@pytest.fixture
def fake_transformers(monkeypatch):
    """Install a stand-in ``transformers`` module; importing the real one costs seconds and only its base classes are used."""
    module = types.ModuleType("transformers")
    module.LogitsProcessor = type("LogitsProcessor", (), {})
    module.LogitsProcessorList = list
    module.StoppingCriteria = type("StoppingCriteria", (), {})
    module.StoppingCriteriaList = list
    module.TextIteratorStreamer = MagicMock()
    module.TextStreamer = _TextStreamerStandIn
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=MagicMock())
    module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


@pytest.fixture
def enhancer_and_model(fake_transformers, monkeypatch):
    """A prompt enhancer over a mocked tokenizer and model; the prompt is three tokens long."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = 7
    tokenizer.eos_token_id = 3
    tokenizer.unk_token_id = 0
    tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([_PROMPT])}
    model = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    return prompt_enhancer.TransformersPromptEnhancer("owner/repo", None), model


def _processor():
    (processor,) = prompt_enhancer._make_presence_penalty(len(_PROMPT))
    return processor


class TestPresencePenalty:
    def test_generate_passes_the_penalty_to_the_model(self, enhancer_and_model):
        enhancer, model = enhancer_and_model
        list(enhancer.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.7))
        processors = model.generate.call_args.kwargs["logits_processor"]
        assert len(processors) == 1
        # A token generated once is lowered by the penalty while the prompt is untouched.
        input_ids = torch.tensor([[*_PROMPT, 2]])
        assert processors[0](input_ids, torch.zeros(1, 8))[0].tolist() == [0, 0, -1.5, 0, 0, 0, 0, 0]

    def test_only_generated_tokens_are_penalised_once_each(self):
        scores = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        # The prompt holds 5, 6 and 7; the output so far holds 1 three times and 2 once.
        input_ids = torch.tensor([[*_PROMPT, 1, 2, 1, 1]])
        out = _processor()(input_ids, scores.clone())
        expected = scores.clone()
        expected[0, 1] -= 1.5
        expected[0, 2] -= 1.5
        assert torch.equal(out, expected)

    def test_prompt_tokens_repeated_in_the_output_are_penalised(self):
        out = _processor()(torch.tensor([[*_PROMPT, 5]]), torch.zeros(1, 8))
        assert out[0].tolist() == [0, 0, 0, 0, 0, -1.5, 0, 0]

    def test_nothing_is_penalised_before_the_first_token(self):
        scores = torch.randn(1, 8)
        assert torch.equal(_processor()(torch.tensor([_PROMPT]), scores.clone()), scores)


_PINNED_DIFF = {"_from_model_config": True, "eos_token_id": 248044, "transformers_version": "5.21.0", "use_cache": True}


def _generation_config(**diff):
    return types.SimpleNamespace(to_diff_dict=lambda: {**_PINNED_DIFF, **diff})


class TestGraphDecodeGate:
    def test_pinned_models_use_the_graph(self):
        assert prompt_enhancer._uses_graph_decode("qwen3_5_text", False, _generation_config()) is True

    def test_repo_sampling_settings_keep_generate(self):
        assert prompt_enhancer._uses_graph_decode("qwen3_5_text", False, _generation_config(top_k=20)) is False

    def test_other_model_types_and_quantized_repos_keep_generate(self):
        assert prompt_enhancer._uses_graph_decode("qwen3_5", False, _generation_config()) is False
        assert prompt_enhancer._uses_graph_decode("qwen3_5_text", True, _generation_config()) is False


def _graph_tokens(tokens, pulled):
    def fake(*args, **kwargs):
        for token in tokens:
            pulled.append(token)
            yield token

    return fake


def _use_graph(enhancer, monkeypatch, tokens, pulled):
    enhancer._graph_decode = True
    enhancer._tokenizer.decode.side_effect = lambda ids: "".join(chr(96 + i) for i in ids)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.enhancer_decode.graph_tokens", _graph_tokens(tokens, pulled))


class TestGraphStreaming:
    def test_eos_ends_the_stream_without_being_passed_on(self, enhancer_and_model, monkeypatch):
        enhancer, model = enhancer_and_model
        pulled: list[int] = []
        _TextStreamerStandIn.ends = 0
        _use_graph(enhancer, monkeypatch, [1, 2, 7, 4], pulled)
        deltas = list(enhancer.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.7))
        assert "".join(deltas) == "ab"
        assert pulled == [1, 2, 7]
        assert _TextStreamerStandIn.ends == 1
        model.generate.assert_not_called()

    def test_cancel_stops_after_the_token_that_saw_it(self, enhancer_and_model, monkeypatch):
        enhancer, _ = enhancer_and_model
        pulled: list[int] = []
        _use_graph(enhancer, monkeypatch, [1, 2, 4], pulled)
        deltas = list(enhancer.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.7, cancelled=lambda: len(pulled) >= 2))
        assert "".join(deltas) == "ab"
        assert pulled == [1, 2]

    def test_a_decode_failure_is_raised_without_falling_back_to_generate(self, enhancer_and_model, monkeypatch):
        enhancer, model = enhancer_and_model
        boom = RuntimeError("CUDA out of memory")

        def failing(*args, **kwargs):
            yield 1
            raise boom

        enhancer._graph_decode = True
        enhancer._tokenizer.decode.side_effect = lambda ids: "a"
        monkeypatch.setattr("zvisiongenerator.backends.cuda.enhancer_decode.graph_tokens", failing)
        with pytest.raises(RuntimeError) as caught:
            list(enhancer.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.7))
        assert caught.value.__cause__ is boom
        model.generate.assert_not_called()
