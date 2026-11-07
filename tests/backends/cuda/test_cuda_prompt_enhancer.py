"""Tests for the presence penalty on the transformers prompt enhancer."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda import prompt_enhancer  # noqa: E402

_PROMPT = [5, 6, 7]


@pytest.fixture
def fake_transformers(monkeypatch):
    """Install a stand-in ``transformers`` module; importing the real one costs seconds and only its base classes are used."""
    module = types.ModuleType("transformers")
    module.LogitsProcessor = type("LogitsProcessor", (), {})
    module.LogitsProcessorList = list
    module.StoppingCriteria = type("StoppingCriteria", (), {})
    module.StoppingCriteriaList = list
    module.TextIteratorStreamer = MagicMock()
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
