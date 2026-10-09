"""Tests for the diffusers backend's NF4 and FP8 weights, with torch, diffusers and transformers mocked."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.backends import image_win_quant as quant


@pytest.fixture
def libs():
    """Install fake torch, diffusers, transformers, accelerate and safetensors modules for the lazy imports."""
    torch_mod = MagicMock()
    torch_mod.float8_e4m3fn = "fp8-sentinel"
    diffusers_mod = MagicMock()
    transformers_mod = MagicMock()
    accelerate_mod = MagicMock()
    accelerate_mod.init_empty_weights = contextlib.nullcontext
    safetensors_torch = MagicMock()
    fakes = {
        "torch": torch_mod,
        "diffusers": diffusers_mod,
        "diffusers.hooks": diffusers_mod.hooks,
        "transformers": transformers_mod,
        "accelerate": accelerate_mod,
        "safetensors": MagicMock(torch=safetensors_torch),
        "safetensors.torch": safetensors_torch,
    }
    with patch.dict(sys.modules, fakes):
        yield SimpleNamespace(torch=torch_mod, diffusers=diffusers_mod, transformers=transformers_mod, safetensors=safetensors_torch)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "zit"
    for name in ("text_encoder", "transformer", "vae", "scheduler"):
        (source / name).mkdir(parents=True)
    index = {"_class_name": "ZImagePipeline", "safety_checker": [None, None], **{name: ["diffusers", "X"] for name in ("text_encoder", "transformer", "vae", "scheduler")}}
    (source / "model_index.json").write_text(json.dumps(index))
    (source / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (source / "scheduler" / "scheduler_config.json").write_text("{}")
    (source / "single_file_checkpoint.safetensors").write_bytes(b"not part of the pipeline")
    return source


class TestCastToFp8:
    def test_diffusers_models_use_their_own_skip_list(self, libs):
        model = MagicMock()

        quant.cast_to_fp8(model, "bf16")

        model.enable_layerwise_casting.assert_called_once_with(storage_dtype="fp8-sentinel", compute_dtype="bf16")
        libs.diffusers.hooks.apply_layerwise_casting.assert_not_called()

    def test_fp8_layers_declare_the_compute_dtype_for_lora_adapters(self, libs):
        fp8_layer = SimpleNamespace(weight=SimpleNamespace(dtype="fp8-sentinel"))
        norm = SimpleNamespace(weight=SimpleNamespace(dtype="bf16"))
        container = SimpleNamespace()
        model = MagicMock()
        model.modules.return_value = [container, fp8_layer, norm]

        quant.cast_to_fp8(model, "bf16")

        assert fp8_layer.compute_dtype == "bf16"
        assert not hasattr(norm, "compute_dtype")
        assert not hasattr(container, "compute_dtype")

    def test_text_encoders_keep_norms_and_embeddings_in_the_compute_dtype(self, libs):
        text_encoder = MagicMock(spec=["forward", "modules"])

        quant.cast_to_fp8(text_encoder, "bf16")

        kwargs = libs.diffusers.hooks.apply_layerwise_casting.call_args.kwargs
        assert kwargs["storage_dtype"] == "fp8-sentinel"
        assert kwargs["compute_dtype"] == "bf16"
        assert {"norm", "embed", "lm_head"} <= set(kwargs["skip_modules_pattern"])


class TestLoadFp8Components:
    def test_text_encoder_is_cast_before_the_transformer_streams_in(self, libs):
        events: list[str] = []
        libs.transformers.AutoModel.from_pretrained.side_effect = lambda *_a, **_k: events.append("load text_encoder") or "te"

        with (
            patch.object(quant, "cast_to_fp8", side_effect=lambda component, _dtype: events.append(f"cast {component}") or component),
            patch.object(quant, "_component_dir", side_effect=lambda path, name: Path(path) / name),
            patch.object(quant, "_stream_fp8_transformer", side_effect=lambda path, _dtype: events.append(f"stream {path.name}") or "tx"),
        ):
            components = quant.load_fp8_components("/models/zit", "bf16")

        assert components == {"text_encoder": "te", "transformer": "tx"}
        assert events == ["load text_encoder", "cast te", "stream transformer"]


class TestWriteFp8Copy:
    def test_saves_the_cast_components_and_links_the_rest(self, libs, tmp_path):
        source = _source(tmp_path)
        target = tmp_path / ".zit@q8.partial"
        text_encoder, transformer = MagicMock(), MagicMock()
        libs.transformers.AutoModel.from_pretrained.return_value = text_encoder

        with patch.object(quant, "cast_to_fp8", side_effect=lambda component, _dtype: component) as mock_cast, patch.object(quant, "_stream_fp8_transformer", return_value=transformer) as mock_stream:
            quant.write_fp8_copy(source, target, "bf16")

        mock_cast.assert_called_once_with(text_encoder, "bf16")
        assert mock_stream.call_args.args[0] == source / "transformer"
        text_encoder.save_pretrained.assert_called_once_with(str(target / "text_encoder"))
        transformer.save_pretrained.assert_called_once_with(str(target / "transformer"))
        linked = target / "vae" / "diffusion_pytorch_model.safetensors"
        assert linked.read_bytes() == b"vae"
        assert linked.stat().st_ino == (source / "vae" / "diffusion_pytorch_model.safetensors").stat().st_ino
        assert (target / "model_index.json").is_file()
        assert (target / "scheduler" / "scheduler_config.json").is_file()
        assert not (target / "single_file_checkpoint.safetensors").exists()  # not a component model_index.json lists

    def test_releases_memory_and_drops_files_after_each_component(self, libs, tmp_path):
        source = _source(tmp_path)
        target = tmp_path / ".zit@q8.partial"
        events: list[str] = []

        with (
            patch.object(quant, "cast_to_fp8", side_effect=lambda component, _dtype: component),
            patch.object(quant, "_stream_fp8_transformer", return_value=MagicMock()),
            patch.object(quant, "release_memory", side_effect=lambda: events.append("release")),
            patch.object(quant, "drop_cached_files", side_effect=lambda path: events.append(f"drop {path.parent.name}/{path.name}")),
        ):
            quant.write_fp8_copy(source, target, "bf16")

        assert events == [
            "release",
            f"drop {source.name}/text_encoder",
            f"drop {target.name}/text_encoder",
            "release",
            f"drop {source.name}/transformer",
            f"drop {target.name}/transformer",
        ]

    def test_stops_between_components_when_cancelled(self, libs, tmp_path):
        source = _source(tmp_path)
        answers = iter([False, True])

        with patch.object(quant, "cast_to_fp8"), patch.object(quant, "_stream_fp8_transformer") as mock_stream:
            quant.write_fp8_copy(source, tmp_path / ".zit@q8.partial", "bf16", cancelled=lambda: next(answers))

        libs.transformers.AutoModel.from_pretrained.assert_called_once()
        mock_stream.assert_not_called()
        assert not (tmp_path / ".zit@q8.partial" / "vae").exists()


class TestLoadNf4Components:
    def test_quantizes_at_load(self, libs):
        quant.load_nf4_components("/models/zit", "bf16", prequantized=False, text_encoder_to_cpu=False)

        assert libs.transformers.AutoModel.from_pretrained.call_args.kwargs["quantization_config"] is libs.transformers.BitsAndBytesConfig.return_value
        assert libs.diffusers.AutoModel.from_pretrained.call_args.kwargs["quantization_config"] is libs.diffusers.BitsAndBytesConfig.return_value

    def test_stored_copy_carries_its_own_quantization(self, libs):
        quant.load_nf4_components("/models/zit@q4", "bf16", prequantized=True, text_encoder_to_cpu=False)

        assert "quantization_config" not in libs.transformers.AutoModel.from_pretrained.call_args.kwargs
        assert "quantization_config" not in libs.diffusers.AutoModel.from_pretrained.call_args.kwargs

    def test_text_encoder_can_leave_the_gpu_before_the_transformer_loads(self, libs):
        text_encoder = libs.transformers.AutoModel.from_pretrained.return_value

        quant.load_nf4_components("/models/krea", "bf16", prequantized=False, text_encoder_to_cpu=True)

        text_encoder.to.assert_called_once_with("cpu")
