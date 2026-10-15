"""Tests for running token-wise layers chunk by chunk along the token axis."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda.video.token_chunking import chunk_tokens  # noqa: E402


class _CountingFeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8))
        self.calls: list[int] = []

    def forward(self, hidden_states, *args, **kwargs):
        self.calls.append(hidden_states.shape[1])
        return self.net(hidden_states)


def test_chunked_output_equals_the_whole_sequence_output():
    torch.manual_seed(0)
    module = _CountingFeedForward()
    x = torch.randn(2, 10, 8)
    expected = module(x)
    module.calls.clear()

    chunk_tokens([module], tokens_per_chunk=4)
    out = module(x)

    assert torch.allclose(out, expected, atol=1e-6)
    assert module.calls == [4, 4, 2]


def test_parameter_names_are_unchanged():
    module = _CountingFeedForward()
    names = [name for name, _ in module.named_parameters()]

    chunk_tokens([module], tokens_per_chunk=4)

    assert [name for name, _ in module.named_parameters()] == names


def test_inputs_not_shaped_batch_tokens_dim_run_whole():
    module = _CountingFeedForward()
    chunk_tokens([module], tokens_per_chunk=2)

    module(torch.randn(10, 8))

    assert module.calls == [8]
