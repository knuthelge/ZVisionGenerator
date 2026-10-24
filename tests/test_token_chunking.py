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


class _SwiGLUFeedForward(torch.nn.Module):
    """A tiny stand-in for a transformer block's SwiGLU feed-forward."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(16, 64)
        self.down = torch.nn.Linear(32, 16)

    def forward(self, hidden_states):
        value, gate = self.up(hidden_states).chunk(2, dim=-1)
        return self.down(value * torch.nn.functional.silu(gate))


@pytest.mark.parametrize(("dtype", "atol"), [(torch.float32, 1e-6), (torch.bfloat16, 2e-2)])
def test_a_chunked_feed_forward_matches_the_unchunked_one(dtype, atol):
    torch.manual_seed(0)
    module = _SwiGLUFeedForward().to(dtype)
    x = torch.randn(1, 4096 + 1000, 16, dtype=dtype)
    expected = module(x)

    chunk_tokens([module], tokens_per_chunk=4096)

    assert torch.allclose(module(x), expected, atol=atol)
