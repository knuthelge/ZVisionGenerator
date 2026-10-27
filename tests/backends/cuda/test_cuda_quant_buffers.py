"""Tests that ``empty_diffusers_model`` keeps non-persistent buffers at the values the constructor computes (real torch, tiny models)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")
from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel  # noqa: E402

from zvisiongenerator.backends.cuda.quant import empty_diffusers_model  # noqa: E402

_VAE_CONFIG = {
    "latent_channels": 4,
    "block_out_channels": (4, 4, 4, 4, 4, 4),
    "layers_per_block": 1,
    "norm_num_groups": 2,
    "decoder_num_layers": 1,
    "decoder_num_attention_heads": 1,
    "decoder_attention_head_dim": 24,
    "decoder_num_register_tokens": 1,
    "decoder_ffn_mult": 1,
}


def test_the_h3_vae_rotary_frequencies_equal_the_stock_models() -> None:
    stock = AutoencoderKLMiniMaxH3(**_VAE_CONFIG)
    model = empty_diffusers_model(dict(stock.config), torch.bfloat16, AutoencoderKLMiniMaxH3)
    got, want = model.decoder.rope.inv_freq, stock.decoder.rope.inv_freq
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    assert "decoder.rope.inv_freq" not in model.state_dict()  # not persistent: nothing else would restore it
    assert not torch.equal(want.to(torch.bfloat16).float(), want)  # the tiny config does expose the bf16 rounding


_TRANSFORMER_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "ffn_dim": 128,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 48,
    "freq_dim": 32,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 32,
    "rope_freq_dim": 4,
}


def test_the_h3_transformer_rotary_frequencies_equal_the_stock_models() -> None:
    stock = MiniMaxH3Transformer3DModel(**_TRANSFORMER_CONFIG)
    model = empty_diffusers_model(dict(stock.config), torch.bfloat16, MiniMaxH3Transformer3DModel)
    assert model.rope.inv_freq.dtype == torch.float32
    assert torch.equal(model.rope.inv_freq, stock.rope.inv_freq)
    assert model.rope.inv_freq.data_ptr() != stock.rope.inv_freq.data_ptr()  # built for this model, not shared


class _TinyModel(torch.nn.Module):
    """A diffusers-free stand-in with no non-persistent buffers, like the LTX models."""

    _keep_in_fp32_modules = ("norm",)

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.register_buffer("stat", torch.ones(4))  # persistent: cast with the model


class _TinyConfigClass:
    @staticmethod
    def from_config(_config: dict) -> _TinyModel:
        return _TinyModel()


def test_a_model_without_non_persistent_buffers_is_cast_as_before() -> None:
    model = empty_diffusers_model({}, torch.bfloat16, _TinyConfigClass)
    assert model.proj.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32
    assert model.stat.dtype == torch.bfloat16
    assert all(parameter.is_meta for parameter in model.parameters())
