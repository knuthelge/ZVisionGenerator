"""Tests for the attention choice of YuE2's CUDA-graph decoder, run without a GPU."""

from __future__ import annotations

import pytest
import torch
from yue2 import cuda_graph

from zvisiongenerator.backends.audio import yue2_attention
from zvisiongenerator.backends.audio.yue2_attention import graph_attention, graph_attention_backend


class _Graph:
    """Stands in for the kit's decoder and records the attention it was asked for."""

    def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto", fuse_projections=False):
        self.attention_backend = attention_backend


@pytest.fixture
def kit_graph(monkeypatch):
    monkeypatch.setattr(cuda_graph, "GraphAR", _Graph)
    return _Graph


def _flash(monkeypatch, available, capability=(8, 6)):
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)


class TestGraphAttentionBackend:
    def test_a_build_with_flash_attention_keeps_the_kits_choice(self, monkeypatch):
        _flash(monkeypatch, True)

        assert graph_attention_backend() == "auto"

    def test_a_build_without_flash_attention_asks_for_sdpa(self, monkeypatch):
        _flash(monkeypatch, False)

        assert graph_attention_backend() == "sdpa"

    @pytest.mark.parametrize(("capability", "expected"), [((7, 5), "sdpa"), ((8, 0), "auto"), ((8, 6), "auto")])
    def test_a_card_older_than_ampere_asks_for_sdpa_even_where_the_build_has_flash_attention(self, monkeypatch, capability, expected):
        _flash(monkeypatch, True, capability)

        assert graph_attention_backend() == expected


class TestGraphAttention:
    def test_without_flash_attention_the_decoder_is_built_with_sdpa(self, monkeypatch, kit_graph):
        _flash(monkeypatch, False)

        with graph_attention("cuda"):
            graph = cuda_graph.GraphAR(object(), [[1]], 8)

        assert graph.attention_backend == "sdpa"
        assert cuda_graph.GraphAR is kit_graph

    def test_with_flash_attention_the_decoder_is_left_alone(self, monkeypatch, kit_graph):
        _flash(monkeypatch, True)

        with graph_attention("cuda"):
            assert cuda_graph.GraphAR is kit_graph
            assert cuda_graph.GraphAR(object(), [[1]], 8).attention_backend == "auto"

    @pytest.mark.parametrize("device", ["mps", "cpu"])
    def test_a_device_without_the_graph_decoder_is_left_alone(self, monkeypatch, kit_graph, device):
        monkeypatch.setattr(yue2_attention, "graph_attention_backend", lambda: pytest.fail("asked the CUDA build"))

        with graph_attention(device):
            assert cuda_graph.GraphAR is kit_graph

    def test_an_explicit_attention_wins(self, monkeypatch, kit_graph):
        _flash(monkeypatch, False)

        with graph_attention("cuda"):
            graph = cuda_graph.GraphAR(object(), [[1]], 8, attention_backend="cudnn")

        assert graph.attention_backend == "cudnn"

    def test_the_decoder_is_restored_when_the_block_raises(self, monkeypatch, kit_graph):
        _flash(monkeypatch, False)

        with pytest.raises(RuntimeError):
            with graph_attention("cuda"):
                raise RuntimeError("stopped")

        assert cuda_graph.GraphAR is kit_graph
