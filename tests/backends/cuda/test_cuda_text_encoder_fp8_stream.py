"""Tests for streaming a transformers text encoder into FP8 weight storage, against ``from_pretrained`` plus ``cast_to_fp8`` (tiny real models on the CPU)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends.cuda import quant  # noqa: E402

_FP8 = torch.float8_e4m3fn


def _qwen3(*, tie: bool):
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=tie)
    return transformers.Qwen3ForCausalLM(config)


def _qwen3_vl():
    text = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8}
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32, "deepstack_visual_indexes": [0]}
    return transformers.Qwen3VLModel(transformers.Qwen3VLConfig(text_config=text, vision_config=vision))


def _save(model, folder, dtype=torch.bfloat16):
    model.to(dtype).save_pretrained(str(folder), max_shard_size="20KB")
    return folder


def _reference(folder):
    """Load *folder* the way the loader did before streaming."""
    return quant.cast_to_fp8(transformers.AutoModel.from_pretrained(str(folder), dtype=torch.bfloat16), torch.bfloat16)


def _streamed(folder):
    model = quant._stream_fp8_text_encoder(folder, torch.bfloat16)
    assert model is not None
    return model


def _assert_same(model, reference, *, config=True):
    assert type(model) is type(reference)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, expected in reference.state_dict().items():
        got = model.state_dict()[key]
        assert got.dtype == expected.dtype, key
        assert torch.equal(got.to(torch.float32), expected.to(torch.float32)), key
    for (name, buffer), (_, expected) in zip(model.named_buffers(), reference.named_buffers(), strict=True):
        assert buffer.dtype == expected.dtype, name
        assert torch.equal(buffer, expected), name
    assert not model.training
    if config:
        assert model.config.to_dict() == reference.config.to_dict()
    for name, module in reference.named_modules():
        assert getattr(model.get_submodule(name), "compute_dtype", None) == getattr(module, "compute_dtype", None), name
    assert model.get_submodule("layers.0.mlp.down_proj" if hasattr(model, "layers") else "language_model.layers.0.mlp.down_proj").weight.dtype == _FP8


class TestStreamFp8TextEncoder:
    @pytest.mark.parametrize("tie", [False, True])
    def test_a_causal_lm_checkpoint_loads_like_from_pretrained(self, tmp_path, tie):
        folder = _save(_qwen3(tie=tie), tmp_path / "text_encoder")

        _assert_same(_streamed(folder), _reference(folder))

    def test_a_vision_language_checkpoint_loads_like_from_pretrained(self, tmp_path):
        folder = _save(_qwen3_vl(), tmp_path / "text_encoder")

        _assert_same(_streamed(folder), _reference(folder))

    def test_never_reads_the_language_model_head(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "text_encoder")
        read: list[str] = []
        real = quant.iter_tensors

        def spy(path, wanted=None):
            for name, tensor in real(path, wanted):
                read.append(name)
                yield name, tensor

        with patch.object(quant, "iter_tensors", spy):
            _streamed(folder)

        assert read
        assert not any("lm_head" in name for name in read)

    def test_weights_saved_in_float32_round_through_bfloat16_like_from_pretrained(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "text_encoder", dtype=torch.float32)

        _assert_same(_streamed(folder), _reference(folder))

    def test_a_stored_fp8_copy_reloads_unchanged(self, tmp_path):
        source = _save(_qwen3(tie=True), tmp_path / "source")
        stored = tmp_path / "stored"
        _streamed(source).save_pretrained(str(stored), max_shard_size="20KB")

        again = _streamed(stored)

        _assert_same(again, _reference(stored))
        _assert_same(again, _reference(source), config=False)  # the saved copy names itself and its class in its config

    def test_weights_are_not_held_at_full_precision_beside_the_fp8_ones(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "text_encoder")

        model = _streamed(folder)

        assert not any(tensor.dtype == torch.bfloat16 and tensor.ndim == 2 and "proj" in key for key, tensor in model.state_dict().items())

    def test_a_quantized_checkpoint_does_not_stream(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "text_encoder")
        config = transformers.AutoConfig.from_pretrained(str(folder))
        config.quantization_config = {"quant_method": "bitsandbytes"}
        config.save_pretrained(str(folder))

        assert quant._stream_fp8_text_encoder(folder, torch.bfloat16) is None

    def test_weights_in_another_format_do_not_stream(self, tmp_path):
        folder = tmp_path / "text_encoder"
        _qwen3(tie=False).config.save_pretrained(str(folder))

        assert quant._stream_fp8_text_encoder(folder, torch.bfloat16) is None

    def test_files_that_do_not_cover_the_model_do_not_stream(self, tmp_path):
        folder = tmp_path / "text_encoder"
        _qwen3(tie=False).config.save_pretrained(str(folder))
        weights = {key: value.to(torch.bfloat16).contiguous() for key, value in _qwen3(tie=False).state_dict().items() if "layers.1" not in key}
        save_file(weights, str(folder / "model.safetensors"))

        assert quant._stream_fp8_text_encoder(folder, torch.bfloat16) is None

    def test_a_checkpoint_transformers_fuses_while_loading_does_not_stream(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "text_encoder")
        converter = transformers.core_model_loading.WeightConverter(source_patterns="q_proj.weight", target_patterns="q_proj.weight", operations=[transformers.core_model_loading.Concatenate(dim=0)])

        with patch("transformers.conversion_mapping.get_model_conversion_mapping", return_value=[converter]):
            assert quant._stream_fp8_text_encoder(folder, torch.bfloat16) is None


class TestFp8TextEncoder:
    def test_streams_when_it_can(self, tmp_path):
        folder = _save(_qwen3(tie=False), tmp_path / "zit" / "text_encoder")

        with patch.object(quant, "_load_text_encoder") as mock_load:
            model = quant._fp8_text_encoder(str(tmp_path / "zit"), torch.bfloat16)

        mock_load.assert_not_called()
        _assert_same(model, _reference(folder))

    def test_loads_and_casts_what_cannot_stream(self, tmp_path):
        with (
            patch.object(quant, "_component_dir", return_value=tmp_path),
            patch.object(quant, "_stream_fp8_text_encoder", return_value=None),
            patch.object(quant, "_load_text_encoder", return_value="loaded") as mock_load,
        ):
            with patch.object(quant, "cast_to_fp8", side_effect=lambda component, _dtype: f"cast {component}"):
                assert quant._fp8_text_encoder("/models/zit", torch.bfloat16) == "cast loaded"

        mock_load.assert_called_once_with("/models/zit", torch.bfloat16)
