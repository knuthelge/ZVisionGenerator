"""Tests for the Higgs TTS 3 network: the weight map, the strict loader on a tiny checkpoint, the buffer check and the fused modules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Model

from tests.backends._helpers import TINY, checkpoint_tensors, write_checkpoint
from zvisiongenerator.backends.audio import higgs3_model
from zvisiongenerator.backends.audio.higgs3_model import FusedCodebookEmbedding, FusedCodebookHead, Higgs3Model, _qwen3_config, checkpoint_target, load_higgs3_model
from zvisiongenerator.backends.audio.higgs3_prompt import AUDIO_SLOT
from zvisiongenerator.utils import safetensors_stream

ROTARY = {"backbone.rotary_emb.inv_freq", "backbone.rotary_emb.original_inv_freq"}


class TestCheckpointTarget:
    @pytest.mark.parametrize(
        "name",
        [
            "tied.head.modality_heads.0.weight",
            "tied.head.text_head.weight",
            "tied.embedding.modality_embeddings.0.model.encoder.conv.weight",
            "tied.embedding.modality_embeddings.0.model.quantizer.codebook.embed",
        ],
    )
    def test_the_tensors_the_model_does_not_use_map_to_none(self, name: str) -> None:
        assert checkpoint_target(name) is None

    @pytest.mark.parametrize("name", ["lm_head.weight", "tied.embedding.modality_embeddings.1.embedding.weight", "model.layers.0.mlp.up_proj.weight"])
    def test_an_unknown_name_is_refused_naming_it(self, name: str) -> None:
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            checkpoint_target(name)


class ReadCounter:
    """Wraps ``safetensors_stream._read_tensor`` to record the names and bytes read."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.bytes = 0

    def __enter__(self) -> ReadCounter:
        real = safetensors_stream._read_tensor

        def counting(handle, path, data_start, name, info):  # noqa: ANN001, ANN202
            self.names.append(name)
            self.bytes += info["data_offsets"][1] - info["data_offsets"][0]
            return real(handle, path, data_start, name, info)

        self._patch = patch.object(safetensors_stream, "_read_tensor", counting)
        self._patch.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._patch.stop()


class TestLoad:
    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
        path = tmp_path / "model.safetensors"
        return path, write_checkpoint(path)

    def test_every_parameter_equals_its_file_tensor(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        path, tensors = checkpoint

        model = load_higgs3_model(TINY, path, "cpu")

        parameters = dict(model.named_parameters())
        for name, tensor in tensors.items():
            target = checkpoint_target(name)
            if target is not None:
                assert torch.equal(parameters[target], tensor), name
        assert set(parameters) == {checkpoint_target(name) for name in tensors if checkpoint_target(name)}

    def test_the_stored_head_copy_is_not_used(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        path, tensors = checkpoint

        model = load_higgs3_model(TINY, path, "cpu")

        assert torch.equal(model.audio_head.weight, tensors["tied.embedding.modality_embeddings.0.embedding.weight"])
        assert not torch.equal(model.audio_head.weight, tensors["tied.head.modality_heads.0.weight"])

    def test_the_rotary_buffers_equal_a_plain_cpu_init(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        model = load_higgs3_model(TINY, checkpoint[0], "cpu")
        reference = Qwen3Model(_qwen3_config(TINY)).rotary_emb

        assert dict(model.named_buffers()).keys() == ROTARY
        assert model.backbone.rotary_emb.inv_freq.dtype == torch.float32
        assert torch.equal(model.backbone.rotary_emb.inv_freq, reference.inv_freq)
        assert torch.equal(model.backbone.rotary_emb.original_inv_freq, reference.original_inv_freq)

    def test_the_model_is_bf16_frozen_and_in_eval_mode(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        model = load_higgs3_model(TINY, checkpoint[0], "cpu")

        assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert not model.training
        assert model.backbone.config._attn_implementation == "sdpa"

    def test_only_the_used_tensors_are_read(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        path, tensors = checkpoint
        used = {name for name in tensors if checkpoint_target(name) is not None}

        with ReadCounter() as counter:
            load_higgs3_model(TINY, path, "cpu")

        assert sorted(counter.names) == sorted(used)
        assert len(counter.names) == len(set(counter.names)) == 2 * 11 + 3
        assert counter.bytes == sum(tensors[name].numel() * 2 for name in used)

    def test_a_missing_tensor_is_refused_naming_its_parameter(self, tmp_path: Path) -> None:
        tensors = checkpoint_tensors()
        del tensors["body.layers.1.mlp.up_proj.weight"]
        path = tmp_path / "model.safetensors"
        write_checkpoint(path, tensors)

        with pytest.raises(RuntimeError, match=r"backbone\.layers\.1\.mlp\.up_proj\.weight"):
            load_higgs3_model(TINY, path, "cpu")
        assert torch.get_default_dtype() == torch.float32

    def test_a_shape_mismatch_is_refused_naming_the_tensor(self, tmp_path: Path) -> None:
        tensors = checkpoint_tensors()
        tensors["body.layers.0.self_attn.q_proj.weight"] = torch.zeros(31, 32, dtype=torch.bfloat16)
        path = tmp_path / "model.safetensors"
        write_checkpoint(path, tensors)

        with pytest.raises(ValueError, match=r"body\.layers\.0\.self_attn\.q_proj\.weight"):
            load_higgs3_model(TINY, path, "cpu")

    def test_an_unexpected_tensor_is_refused_naming_it(self, tmp_path: Path) -> None:
        tensors = checkpoint_tensors()
        tensors["lm_head.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
        path = tmp_path / "model.safetensors"
        write_checkpoint(path, tensors)

        with pytest.raises(ValueError, match=r"lm_head\.weight"):
            load_higgs3_model(TINY, path, "cpu")

    def test_a_failure_while_building_restores_the_default_dtype(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        def boom(*_args: object) -> None:
            raise RuntimeError("no memory")

        with patch.object(higgs3_model, "Higgs3Model", boom), pytest.raises(RuntimeError, match="no memory"):
            load_higgs3_model(TINY, checkpoint[0], "cpu")

        assert torch.get_default_dtype() == torch.float32

    def test_a_buffer_the_loader_does_not_write_fails_the_load(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        class Stray(Higgs3Model):
            def __init__(self, config: object) -> None:
                super().__init__(config)  # type: ignore[arg-type]
                self.register_buffer("stray_state", torch.zeros(1))

        with patch.object(higgs3_model, "Higgs3Model", Stray), pytest.raises(RuntimeError, match="stray_state"):
            load_higgs3_model(TINY, checkpoint[0], "cpu")
        assert torch.get_default_dtype() == torch.float32

    def test_the_rotary_frequencies_follow_the_configured_theta(self, checkpoint: tuple[Path, dict[str, torch.Tensor]]) -> None:
        config = replace(TINY, rope_theta=500000.0)

        model = load_higgs3_model(config, checkpoint[0], "cpu")

        expected = 1.0 / (500000.0 ** (torch.arange(0, TINY.head_dim, 2, dtype=torch.int64).float() / TINY.head_dim))
        assert torch.allclose(model.backbone.rotary_emb.inv_freq, expected, rtol=1e-6)
        assert not torch.allclose(model.backbone.rotary_emb.inv_freq, load_higgs3_model(TINY, checkpoint[0], "cpu").backbone.rotary_emb.inv_freq)


class TestQwen3Config:
    def test_every_size_reaches_the_backbone_config(self) -> None:
        config = replace(
            TINY,
            hidden_size=48,
            intermediate_size=96,
            num_layers=3,
            num_heads=6,
            num_kv_heads=3,
            head_dim=8,
            rms_norm_eps=3e-6,
            rope_theta=777777.0,
            text_vocab_size=77,
            max_positions=999,
        )

        qwen = _qwen3_config(config)

        assert (qwen.hidden_size, qwen.intermediate_size, qwen.num_hidden_layers) == (48, 96, 3)
        assert (qwen.num_attention_heads, qwen.num_key_value_heads, qwen.head_dim) == (6, 3, 8)
        assert (qwen.rms_norm_eps, qwen.vocab_size, qwen.max_position_embeddings) == (3e-6, 77, 999)
        assert qwen.rope_parameters["rope_theta"] == 777777.0
        assert qwen.rope_parameters["rope_type"] == "default"
        assert qwen.pad_token_id is None
        assert qwen._attn_implementation == "sdpa"


class TestFusedModules:
    def test_the_embedding_sums_the_offset_rows(self) -> None:
        embedding = FusedCodebookEmbedding(num_codebooks=3, codebook_size=6, hidden_size=4)
        embedding.weight.data = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 4)

        out = embedding(torch.tensor([[1, 0, 5], [2, 2, 2]]))

        weight = embedding.weight.data
        assert torch.equal(out[0], weight[1] + weight[6] + weight[17])
        assert torch.equal(out[1], weight[2] + weight[8] + weight[14])
        assert out.shape == (2, 4)

    def test_the_embedding_sums_in_one_reduction_in_bf16(self) -> None:
        embedding = FusedCodebookEmbedding(num_codebooks=8, codebook_size=64, hidden_size=64)
        torch.manual_seed(3)
        embedding.weight.data = torch.randn(8 * 64, 64).to(torch.bfloat16)
        codes = torch.randint(0, 64, (50, 8))
        offsets = torch.arange(8) * 64
        gathered = F.embedding(codes + offsets, embedding.weight)
        sequential = gathered[:, 0]
        for codebook in range(1, 8):
            sequential = sequential + gathered[:, codebook]
        assert not torch.equal(sequential, gathered.sum(dim=-2))  # the data tells one reduction from a chain of adds apart

        out = embedding(codes)

        assert torch.equal(out, gathered.sum(dim=-2))

    def test_the_head_returns_one_row_of_logits_per_codebook(self) -> None:
        head = FusedCodebookHead(num_codebooks=3, codebook_size=6, hidden_size=4)
        head.weight.data = torch.randn(18, 4)
        hidden = torch.randn(2, 4)

        logits = head(hidden)

        assert logits.shape == (2, 3, 6)
        assert torch.allclose(logits.reshape(2, 18), hidden @ head.weight.T)


class TestModelMethods:
    @pytest.fixture
    def model(self, tmp_path: Path) -> Higgs3Model:
        path = tmp_path / "model.safetensors"
        write_checkpoint(path)
        return load_higgs3_model(TINY, path, "cpu")

    def test_embed_prompt_puts_audio_rows_at_the_slots_and_text_elsewhere(self, model: Higgs3Model) -> None:
        rows = torch.tensor([[1, 0, 3], [4, 4, 4]])
        prompt = [7, AUDIO_SLOT, AUDIO_SLOT, 9]

        with torch.no_grad():
            embeds = model.embed_prompt(prompt, rows)
            expected_audio = model.audio_embedding(rows)

        text = model.backbone.embed_tokens.weight
        assert embeds.shape == (1, 4, 32)
        assert torch.equal(embeds[0, 0], text[7])
        assert torch.equal(embeds[0, 3], text[9])
        assert torch.equal(embeds[0, 1:3], expected_audio.to(embeds.dtype))

    def test_embed_prompt_without_slots_is_the_text_embedding(self, model: Higgs3Model) -> None:
        embeds = model.embed_prompt([3, 4, 5], None)

        assert torch.equal(embeds[0], model.backbone.embed_tokens.weight[[3, 4, 5]])
