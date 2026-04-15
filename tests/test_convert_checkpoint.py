"""Unit tests for zvisiongenerator.converters.convert_checkpoint."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.converters.convert_checkpoint import (  # noqa: E402
    TRANSFORMER_PREFIX,
    convert_transformer_keys,
    convert_flux2_transformer_keys,
)


# ── ZImage key conversion ────────────────────────────────────────────────────


class TestConvertTransformerKeys:
    """Tests for ZImage (safetensors → diffusers) key conversion."""

    def test_prefixed_key_conversion(self):
        """Known safetensors prefixed key → expected output key."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}final_layer.foo": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert "all_final_layer.2-1.foo" in result

    def test_x_embedder_replacement(self):
        """x_embedder prefix replacement."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}x_embedder.weight": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert "all_x_embedder.2-1.weight" in result

    def test_attention_out_weight_rename(self):
        """Attention out weight key rename."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}block.0.attention.out.weight": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert "block.0.attention.to_out.0.weight" in result

    def test_attention_norm_renames(self):
        """Attention q_norm/k_norm renames."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}block.0.attention.q_norm.weight": torch.zeros(4),
            f"{TRANSFORMER_PREFIX}block.0.attention.k_norm.weight": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert "block.0.attention.norm_q.weight" in result
        assert "block.0.attention.norm_k.weight" in result

    def test_unprefixed_keys_ignored(self):
        """Keys without the transformer prefix are dropped."""
        state_dict = {
            "text_encoders.foo": torch.zeros(4),
            "vae.bar": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert len(result) == 0

    def test_norm_final_dropped(self):
        """norm_final.weight is explicitly dropped."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}norm_final.weight": torch.zeros(4),
        }
        result = convert_transformer_keys(state_dict)
        assert len(result) == 0

    def test_qkv_tensor_split(self):
        """A concatenated QKV tensor is correctly split into separate q, k, v keys."""
        # 3 chunks of 4 rows each
        qkv = torch.arange(12).reshape(12, 1).float()
        state_dict = {
            f"{TRANSFORMER_PREFIX}block.0.attention.qkv.weight": qkv,
        }
        result = convert_transformer_keys(state_dict)
        assert "block.0.attention.to_q.weight" in result
        assert "block.0.attention.to_k.weight" in result
        assert "block.0.attention.to_v.weight" in result
        # Verify values
        assert torch.equal(result["block.0.attention.to_q.weight"], qkv[:4])
        assert torch.equal(result["block.0.attention.to_k.weight"], qkv[4:8])
        assert torch.equal(result["block.0.attention.to_v.weight"], qkv[8:12])


# ── FLUX.2 Klein key conversion ──────────────────────────────────────────────


class TestConvertFlux2TransformerKeys:
    """Tests for FLUX.2 Klein (safetensors → diffusers) key conversion."""

    def test_prefixed_double_block_rename(self):
        """Known prefixed double_blocks key → transformer_blocks rename."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}double_blocks.0.img_mlp.0.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "transformer_blocks.0.ff.linear_in.weight" in result

    def test_prefixed_single_block_rename(self):
        """Known prefixed single_blocks key → single_transformer_blocks rename."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}single_blocks.3.linear2.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "single_transformer_blocks.3.attn.to_out.weight" in result

    def test_prefixed_top_level_rename(self):
        """Known top-level rename: img_in → x_embedder."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}img_in.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "x_embedder.weight" in result

    def test_double_block_img_qkv_split(self):
        """Double block image attention QKV split."""
        qkv = torch.arange(12).reshape(12, 1).float()
        state_dict = {
            f"{TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv.weight": qkv,
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "transformer_blocks.0.attn.to_q.weight" in result
        assert "transformer_blocks.0.attn.to_k.weight" in result
        assert "transformer_blocks.0.attn.to_v.weight" in result
        assert torch.equal(result["transformer_blocks.0.attn.to_q.weight"], qkv[:4])
        assert torch.equal(result["transformer_blocks.0.attn.to_k.weight"], qkv[4:8])
        assert torch.equal(result["transformer_blocks.0.attn.to_v.weight"], qkv[8:12])

    def test_double_block_txt_qkv_split(self):
        """Double block text attention QKV split."""
        qkv = torch.arange(6).reshape(6, 1).float()
        state_dict = {
            f"{TRANSFORMER_PREFIX}double_blocks.1.txt_attn.qkv.weight": qkv,
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "transformer_blocks.1.attn.add_q_proj.weight" in result
        assert "transformer_blocks.1.attn.add_k_proj.weight" in result
        assert "transformer_blocks.1.attn.add_v_proj.weight" in result

    def test_diffusers_format_passthrough(self):
        """State dict already in diffusers format passes through as-is."""
        state_dict = {
            "transformer_blocks.0.attn.to_q.weight": torch.zeros(4),
            "single_transformer_blocks.0.attn.to_out.weight": torch.zeros(4),
            "x_embedder.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert len(result) == 3
        assert "transformer_blocks.0.attn.to_q.weight" in result
        assert "single_transformer_blocks.0.attn.to_out.weight" in result
        assert "x_embedder.weight" in result

    def test_unprefixed_safetensors_format_detected(self):
        """Unprefixed safetensors keys are detected and converted."""
        state_dict = {
            "double_blocks.0.img_mlp.0.weight": torch.zeros(4),
            "single_blocks.0.linear2.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        # Should have been converted to diffusers format
        assert "transformer_blocks.0.ff.linear_in.weight" in result
        assert "single_transformer_blocks.0.attn.to_out.weight" in result

    def test_prefixed_safetensors_format_detected(self):
        """Prefixed safetensors keys are detected and converted."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}double_blocks.0.img_mlp.2.weight": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "transformer_blocks.0.ff.linear_out.weight" in result

    def test_scale_to_weight_rename(self):
        """'.scale' suffix is renamed to '.weight' in norm contexts."""
        state_dict = {
            f"{TRANSFORMER_PREFIX}double_blocks.0.img_attn.norm.query_norm.scale": torch.zeros(4),
        }
        result = convert_flux2_transformer_keys(state_dict)
        assert "transformer_blocks.0.attn.norm_q.weight" in result
