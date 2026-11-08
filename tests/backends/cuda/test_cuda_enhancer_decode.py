"""Tests for the CUDA-graph decode loop of the transformers prompt enhancer (CPU tensors, no model, no GPU)."""

from __future__ import annotations

import contextlib
import sys
import types
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda import enhancer_decode  # noqa: E402

_VOCAB = 100


class TestChooseToken:
    def test_greedy_picks_the_highest_penalised_score(self):
        logits = torch.zeros(_VOCAB)
        logits[3], logits[5] = 5.0, 4.5
        no_history = torch.zeros(_VOCAB)
        assert enhancer_decode._choose_token(logits, no_history, 0.0, penalty=1.5).item() == 3
        # The former leader loses once it is penalised, to a token only 1.0 below it.
        logits[5] = 4.0
        seen = torch.zeros(_VOCAB)
        seen[3] = 1.0
        assert enhancer_decode._choose_token(logits, seen, 0.0, penalty=1.5).item() == 5
        # A tiny temperature samples the same token as greedy.
        assert enhancer_decode._choose_token(logits, seen, 1e-4, penalty=1.5).item() == 5

    def test_sampling_stays_in_the_top_50_and_repeats_under_a_seed(self):
        logits = torch.randn(_VOCAB)
        seen = torch.zeros(_VOCAB)
        seen[logits.argmax()] = 1.0
        top_ids = set((logits - 1.5 * seen).topk(50).indices.tolist())

        def draw() -> list[int]:
            torch.manual_seed(7)
            return [enhancer_decode._choose_token(logits, seen, 1.0, penalty=1.5).item() for _ in range(300)]

        picks = draw()
        assert set(picks) <= top_ids
        assert len(set(picks)) > 1
        assert draw() == picks

        # A high temperature flattens the draw: it reaches rank 49 of ordered logits and nothing beyond it.
        ordered = torch.linspace(1.0, 0.0, _VOCAB)
        torch.manual_seed(3)
        flat = {enhancer_decode._choose_token(ordered, torch.zeros(_VOCAB), 100.0, penalty=1.5).item() for _ in range(2000)}
        assert max(flat) == 49


class _FakeCache:
    instances: list[_FakeCache] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeCache.instances.append(self)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Replace the CUDA graph machinery and the static cache so the decode loop runs on CPU tensors."""
    _FakeCache.instances = []
    transformers = types.ModuleType("transformers")
    transformers.StaticCache = _FakeCache
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    graph_class = MagicMock()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_class)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    return graph_class


def _model_with_token(token: int) -> MagicMock:
    logits = torch.zeros(1, 1, _VOCAB)
    logits[0, 0, token] = 10.0
    model = MagicMock()
    model.return_value.logits = logits
    return model


class TestGraphTokens:
    def test_yields_max_tokens_and_sizes_the_cache_for_prompt_plus_budget(self, fake_cuda):
        prompt = torch.tensor([[1, 2, 3, 4]])
        tokens = list(enhancer_decode.graph_tokens(_model_with_token(9), prompt, max_tokens=5, temperature=0.0, penalty=1.5))
        assert len(tokens) == 5
        assert _FakeCache.instances[0].kwargs["max_cache_len"] == 4 + 5
