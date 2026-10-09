"""Tests for streaming a diffusers transformer into FP8 weight storage, with a tiny real model on the CPU."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402
from diffusers.models.modeling_utils import ModelMixin  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends import image_win_quant as quant  # noqa: E402


class _TinyTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 8):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(self.layer(x))


def _save(component_dir, tensors):
    component_dir.mkdir(parents=True, exist_ok=True)
    (component_dir / "config.json").write_text(json.dumps({"_class_name": "_TinyTransformer", "width": 8}))
    save_file(tensors, str(component_dir / "diffusion_pytorch_model.safetensors"))


def _source_tensors():
    source = _TinyTransformer().to(torch.bfloat16)
    return {key: value.contiguous() for key, value in source.state_dict().items()}


def _stream(component_dir):
    with patch("diffusers.AutoModel.from_config", side_effect=lambda config: _TinyTransformer(config["width"])):
        return quant._stream_fp8_transformer(component_dir, torch.bfloat16)


class TestStreamFp8Transformer:
    def test_layer_weights_land_in_fp8_and_norms_stay_bfloat16(self, tmp_path):
        tensors = _source_tensors()
        _save(tmp_path / "transformer", tensors)

        model = _stream(tmp_path / "transformer")

        assert model.layer.weight.dtype == torch.float8_e4m3fn
        assert model.norm.weight.dtype == torch.bfloat16
        assert torch.equal(model.layer.weight, tensors["layer.weight"].to(torch.float8_e4m3fn))
        assert torch.equal(model.norm.weight, tensors["norm.weight"])
        assert model.layer.compute_dtype == torch.bfloat16  # LoRA adapters load in bfloat16

    def test_streamed_model_runs_in_the_compute_dtype(self, tmp_path):
        _save(tmp_path / "transformer", _source_tensors())

        out = _stream(tmp_path / "transformer")(torch.randn(2, 8, dtype=torch.bfloat16))

        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 8)

    def test_a_stored_fp8_copy_streams_back_unchanged(self, tmp_path):
        _save(tmp_path / "source", _source_tensors())
        first = _stream(tmp_path / "source")
        stored = {key: value.contiguous() for key, value in first.state_dict().items()}
        _save(tmp_path / "stored", stored)

        second = _stream(tmp_path / "stored")

        assert all(torch.equal(second.state_dict()[key], value) for key, value in stored.items())

    def test_missing_tensors_raise(self, tmp_path):
        tensors = _source_tensors()
        del tensors["norm.weight"]
        _save(tmp_path / "transformer", tensors)

        with pytest.raises(RuntimeError):
            _stream(tmp_path / "transformer")

    def test_each_weight_file_leaves_the_page_cache_once_read(self, tmp_path):
        _save(tmp_path / "transformer", _source_tensors())

        with patch.object(quant, "drop_cached_file") as mock_drop:
            _stream(tmp_path / "transformer")

        mock_drop.assert_called_once_with(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors")
