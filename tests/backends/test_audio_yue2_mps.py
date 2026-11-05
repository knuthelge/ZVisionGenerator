"""Tests for the exact YuE2 speed-ups on MPS: folded decode attention and the song-token head, run on CPU tensors."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from yue2 import modeling_yue2
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, Sampling
from yue2.sampling import distribution

from zvisiongenerator.backends.audio.yue2_mps import _fold_grouped_decode, folded_decode_attention, semantic_head


def _grouped(heads: int = 4, kv_heads: int = 2, keys: int = 9, queries: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(1, heads, queries, 8, generator=generator),
        torch.randn(1, kv_heads, keys, 8, generator=generator),
        torch.randn(1, kv_heads, keys, 8, generator=generator),
    )


class TestFoldGroupedDecode:
    def test_matches_attention_over_repeated_key_and_value_heads(self):
        query, key, value = _grouped()
        repeated = F.scaled_dot_product_attention(query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1))

        torch.testing.assert_close(_fold_grouped_decode(query, key, value), repeated)


class TestFoldedDecodeAttention:
    def test_single_unmasked_grouped_queries_are_answered_without_the_kit_attention(self, monkeypatch):
        query, key, value = _grouped()
        repeated = F.scaled_dot_product_attention(query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1))
        monkeypatch.setattr(modeling_yue2, "sdpa", lambda *args, **kwargs: pytest.fail("reached the kit attention"))

        with folded_decode_attention():
            result = modeling_yue2.sdpa(query, key, value)

        torch.testing.assert_close(result, repeated)

    @pytest.mark.parametrize(
        ("shape", "kwargs"),
        [
            ({"queries": 3}, {}),
            ({}, {"attn_mask": torch.ones(1, 1, 1, 9, dtype=torch.bool)}),
            ({}, {"is_causal": True}),
            ({"kv_heads": 4}, {}),
        ],
        ids=["prefill", "masked", "causal", "not grouped"],
    )
    def test_other_calls_reach_the_kit_attention(self, monkeypatch, shape, kwargs):
        seen = []
        monkeypatch.setattr(modeling_yue2, "sdpa", lambda *args, **kw: seen.append(kw) or "kit")
        query, key, value = _grouped(**shape)

        with folded_decode_attention():
            result = modeling_yue2.sdpa(query, key, value, **kwargs)

        assert result == "kit" and len(seen) == 1

    def test_the_kit_attention_is_back_after_the_block_even_on_an_error(self):
        original = modeling_yue2.sdpa

        with pytest.raises(RuntimeError), folded_decode_attention():
            raise RuntimeError("take failed")

        assert modeling_yue2.sdpa is original


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, CODEC_OFFSET + CODEC_SIZE, bias=False)


class TestSemanticHead:
    @pytest.mark.parametrize("cfg_scale", [1.0, 1.5, 3.0])
    def test_the_sampler_scores_match_the_full_heads_with_and_without_guidance(self, cfg_scale):
        model = _Model()
        conditional, unconditional = torch.randn(1, 4), torch.randn(1, 4)
        sampling = Sampling(temperature=1.0, top_p=0.95, top_k=50, repetition_penalty=1.1, penalty_window=50, min_tokens=0, max_tokens=10)

        def scores():
            positive, negative = model.lm_head(conditional), model.lm_head(unconditional)
            return distribution(negative + cfg_scale * (positive - negative), sampling, [CODEC_OFFSET + 3], 1, "semantic")

        expected = scores()
        with semantic_head(model):
            narrowed = scores()

        assert not narrowed.isnan().any()
        torch.testing.assert_close(narrowed, expected, rtol=0, atol=0)

    def test_the_full_head_is_back_after_the_block_even_on_an_error(self):
        model = _Model()
        original = model.lm_head

        with pytest.raises(RuntimeError), semantic_head(model):
            raise RuntimeError("take failed")

        assert model.lm_head is original
