"""Tests for the LTX weights of the CUDA video backend, with tiny real models on the CPU."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402
from diffusers.models.modeling_utils import ModelMixin  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from zvisiongenerator.backends.cuda.video import weights  # noqa: E402
from zvisiongenerator.utils.stored_quant import MANIFEST_NAME  # noqa: E402

_PREFIX = "model.diffusion_model."
_WIDTH = 64


class _TinyTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = _WIDTH):
        super().__init__()
        self.proj_in = torch.nn.Linear(width, width)
        self.transformer_blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width)) for _ in range(2))

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _reference() -> _TinyTransformer:
    torch.manual_seed(0)
    return _TinyTransformer().to(torch.bfloat16)


def _write_checkpoint(path, model, *, drop: str | None = None):
    """Save *model* as a single-file LTX-2 checkpoint, with other components beside the transformer."""
    tensors = {_PREFIX + name: tensor.contiguous() for name, tensor in model.state_dict().items() if name != drop}
    tensors["vae.decoder.conv.weight"] = torch.zeros(2)
    tensors[_PREFIX + "video_embeddings_connector.registers"] = torch.zeros(2)
    save_file(tensors, str(path))
    return path


def _arange_tensor(shape, salt, dtype=torch.bfloat16):
    """Return a tensor of *shape* whose values come from ``torch.arange`` (no RNG): distinct per *salt*, within bfloat16 range."""
    values = ((torch.arange(math.prod(shape)) * 7 + salt) % 251 - 125).to(torch.float32) / 32
    return values.reshape(shape).to(dtype)


def _write_arange_checkpoint(path, dtype=torch.bfloat16):
    """Save a single-file LTX-2 checkpoint of the tiny transformer with arange values, and other components beside it."""
    tensors = {_PREFIX + name: _arange_tensor(list(tensor.shape), salt, dtype) for salt, (name, tensor) in enumerate(_TinyTransformer().state_dict().items())}
    tensors["vae.decoder.conv.weight"] = torch.zeros(2)
    save_file(tensors, str(path))
    return path


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _read_folder(folder):
    """Return ``{name: (dtype, shape, sha256 of the raw bytes)}`` for every tensor in the safetensors files of *folder*."""
    return {name: (str(tensor.dtype), list(tensor.shape), _digest(tensor)) for path in sorted(folder.glob("*.safetensors")) for name, tensor in load_file(str(path)).items()}


@pytest.fixture
def tiny_models():
    with patch("diffusers.AutoModel.from_config", side_effect=lambda config: _TinyTransformer(config.get("width", _WIDTH))):
        yield


class TestTransformerNameMap:
    def test_checkpoint_names_become_diffusers_names_and_other_components_are_left_out(self):
        names = [
            _PREFIX + "transformer_blocks.0.attn1.q_norm.weight",
            _PREFIX + "adaln_single.linear.weight",
            _PREFIX + "prompt_adaln_single.linear.weight",
            _PREFIX + "audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias",
            _PREFIX + "patchify_proj.weight",
            _PREFIX + "video_embeddings_connector.learnable_registers",
            "vae.decoder.conv_in.conv.weight",
            "text_embedding_projection.video_aggregate_embed.weight",
        ]

        mapped = weights.transformer_name_map(names)

        assert mapped == {
            _PREFIX + "transformer_blocks.0.attn1.q_norm.weight": "transformer_blocks.0.attn1.norm_q.weight",
            _PREFIX + "adaln_single.linear.weight": "time_embed.linear.weight",
            _PREFIX + "prompt_adaln_single.linear.weight": "prompt_adaln.linear.weight",
            _PREFIX + "audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias": "audio_prompt_adaln.emb.timestep_embedder.linear_1.bias",
            _PREFIX + "patchify_proj.weight": "proj_in.weight",
        }


class TestElementCounts:
    def test_a_checkpoint_splits_into_block_matrices_and_the_rest(self, tmp_path):
        checkpoint = _write_checkpoint(tmp_path / "ltx.safetensors", _reference())

        blocks, rest = weights.transformer_elements(checkpoint)

        assert blocks == 2 * _WIDTH * _WIDTH  # the two block linears' weights
        assert rest == _WIDTH * _WIDTH + _WIDTH + 2 * (_WIDTH + 2 * _WIDTH)  # proj_in, and the block biases and norms

    def test_a_component_folder_counts_every_shard(self, tmp_path):
        save_file({"a": torch.zeros(3, 4)}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"b": torch.zeros(5)}, str(tmp_path / "model-00002-of-00002.safetensors"))
        index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        assert weights.component_elements(tmp_path) == 17

    def test_the_text_encoder_counts_its_bfloat16_embeddings_and_norms_twice(self, tmp_path):
        tensors = {"model.embed_tokens.weight": torch.zeros(10, 4), "model.layers.0.input_layernorm.weight": torch.zeros(4), "model.layers.0.mlp.up_proj.weight": torch.zeros(4, 4)}
        save_file(tensors, str(tmp_path / "model.safetensors"))

        assert weights.text_encoder_bytes(tmp_path) == 2 * 40 + 2 * 4 + 16

    def test_the_transformer_count_handles_scalars_zero_dims_and_other_ranks(self):
        shapes = [
            ("transformer_blocks.0.attn.weight", [4, 3]),  # block matrix
            ("transformer_blocks.0.empty.weight", [0, 5]),  # block matrix without elements
            ("transformer_blocks.0.bias", [7]),  # block vector
            ("transformer_blocks.0.conv.weight", [2, 3, 4]),  # block tensor of rank 3
            ("transformer_blocks.0.scale", []),  # scalar
            ("proj_in.weight", [6, 2]),  # matrix outside the blocks
            ("proj_in.empty", [2, 0, 3]),
        ]

        assert weights.count_transformer_elements(shapes) == (12, 7 + 24 + 1 + 12)

    def test_a_checkpoint_counts_as_it_does_under_its_diffusers_names(self, tmp_path):
        # One name per top-level group of the stock LTX-2.3 checkpoint, the renamed in-block names, and other components.
        shapes = {
            **{_PREFIX + f"{group}.linear.weight": [8, 4] for group in ("adaln_single", "audio_adaln_single", "prompt_adaln_single", "audio_prompt_adaln_single")},
            **{_PREFIX + f"av_ca_{kind}_adaln_single.linear.weight": [6, 4] for kind in ("video_scale_shift", "a2v_gate", "audio_scale_shift", "v2a_gate")},
            _PREFIX + "patchify_proj.weight": [4, 2],
            _PREFIX + "audio_patchify_proj.weight": [4, 3],
            _PREFIX + "proj_out.weight": [2, 4],
            _PREFIX + "audio_proj_out.bias": [3],
            _PREFIX + "scale_shift_table": [6, 4],
            _PREFIX + "audio_scale_shift_table": [6, 3],
            _PREFIX + "video_embeddings_connector.learnable_registers": [16, 4],
            _PREFIX + "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": [4, 4],
            _PREFIX + "transformer_blocks.0.attn1.to_q.weight": [4, 4],
            _PREFIX + "transformer_blocks.0.attn1.q_norm.weight": [4],
            _PREFIX + "transformer_blocks.0.attn1.k_norm.weight": [4],
            _PREFIX + "transformer_blocks.0.scale_shift_table": [6, 4],
            _PREFIX + "transformer_blocks.0.scale_shift_table_a2v_ca_video": [5, 4],
            _PREFIX + "transformer_blocks.0.scale_shift_table_a2v_ca_audio": [5, 3],
            "vae.decoder.conv_in.conv.weight": [3, 3],
            "text_embedding_projection.video_aggregate_embed.weight": [4, 4],
        }
        header = json.dumps({name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in shapes.items()}).encode()
        checkpoint = tmp_path / "ltx.safetensors"
        checkpoint.write_bytes(len(header).to_bytes(8, "little") + header)

        renamed = weights.count_transformer_elements((name, shapes[original]) for original, name in weights.transformer_name_map(shapes).items())

        assert weights.transformer_elements(checkpoint) == renamed


class TestEnsurePrepared:
    def _source(self, tmp_path, content=b"weights"):
        source = tmp_path / "source.safetensors"
        source.write_bytes(content)
        return source

    def test_a_changed_source_or_level_writes_a_new_copy(self, tmp_path):
        source = self._source(tmp_path)
        target = tmp_path / "model@q8"
        weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("old"))
        source.write_bytes(b"newer weights")

        weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("new"))

        assert (target / "w").read_text() == "new"


class TestPreparedManifestBytes:
    """The manifest of a prepared copy made from Hugging Face cache files is pinned byte for byte: a change would make every installed copy look stale and rebuild it."""

    _BLOB = "cd625eb8fb67fb0921e955952c9b718307c93d6656f7eadf1bfd8064aeeabf1d"
    _OTHER_BLOB = "6d948ca9be609363a9adc0a829aa4c43d0edce94694704327ede5f571ce3d258"

    def _sources(self, tmp_path):
        (tmp_path / "blobs").mkdir()
        real = tmp_path / "blobs" / self._BLOB
        real.write_bytes(b"1234567")
        link = tmp_path / "link.safetensors"
        link.symlink_to(real)
        other = tmp_path / "blobs" / self._OTHER_BLOB
        other.write_bytes(b"0123456789a")
        return [link, other]

    @pytest.mark.parametrize(
        ("bits", "storage_format"),
        [(8, "fp8-e4m3"), (4, "nf4-bitsandbytes-0.99")],
    )
    def test_the_manifest_bytes_are_pinned(self, monkeypatch, tmp_path, bits, storage_format):
        monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.99.0")  # a version no real install has
        sources = self._sources(tmp_path)

        target = weights.ensure_prepared(tmp_path / "copy", sources, bits, lambda folder: (folder / "w").write_text("x"))

        expected = (
            "{\n"
            '  "layout": 1,\n'
            f'  "bits": {bits},\n'
            f'  "format": "{storage_format}",\n'
            '  "sources": [\n'
            "    {\n"
            f'      "name": "{self._BLOB}",\n'
            '      "bytes": 7\n'
            "    },\n"
            "    {\n"
            f'      "name": "{self._OTHER_BLOB}",\n'
            '      "bytes": 11\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
        assert (target / MANIFEST_NAME).read_bytes() == expected.encode("utf-8")


class TestTransformerWeights:
    def test_the_fp8_copy_stores_layer_weights_in_fp8_and_runs_close_to_the_source(self, tmp_path, tiny_models):
        reference = _reference()
        checkpoint = _write_checkpoint(tmp_path / "ltx.safetensors", reference)
        weights.write_transformer(checkpoint, {"width": _WIDTH}, tmp_path / "q8", 8, device="cpu")

        model = weights.load_transformer({"width": _WIDTH}, prepared=tmp_path / "q8", bits=8, compute_dtype=torch.bfloat16)

        assert model.transformer_blocks[0][0].weight.dtype == torch.float8_e4m3fn
        assert model.transformer_blocks[0][1].weight.dtype == torch.bfloat16  # norms keep the compute dtype
        assert json.loads((tmp_path / "q8" / "config.json").read_text()) == {"width": _WIDTH}
        x = torch.randn(3, _WIDTH, dtype=torch.bfloat16)
        assert torch.allclose(model(x).float(), reference(x).float(), atol=0.25)

    def test_a_checkpoint_missing_a_transformer_tensor_is_refused(self, tmp_path, tiny_models):
        checkpoint = _write_checkpoint(tmp_path / "ltx.safetensors", _reference(), drop="proj_in.bias")

        with pytest.raises(RuntimeError, match="do not cover"):
            weights.write_transformer(checkpoint, {"width": _WIDTH}, tmp_path / "q8", 8, device="cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes packs NF4 on a CUDA GPU")
    def test_the_nf4_copy_packs_block_linears_and_loads_them_on_the_cpu(self, tmp_path, tiny_models):
        bnb = pytest.importorskip("bitsandbytes")
        reference = _reference()
        checkpoint = _write_checkpoint(tmp_path / "ltx.safetensors", reference)
        weights.write_transformer(checkpoint, {"width": _WIDTH}, tmp_path / "q4", 4, device=torch.device("cuda"))

        model = weights.load_transformer({"width": _WIDTH}, prepared=tmp_path / "q4", bits=4, compute_dtype=torch.bfloat16)

        block_linear = model.transformer_blocks[0][0]
        assert isinstance(block_linear, bnb.nn.Linear4bit)
        assert block_linear.weight.device.type == "cpu"
        assert type(model.proj_in) is torch.nn.Linear  # layers outside the blocks stay bfloat16
        x = torch.randn(3, _WIDTH, dtype=torch.bfloat16, device="cuda")
        out = model.to("cuda")(x)
        assert torch.allclose(out.float(), reference.to("cuda")(x).float(), atol=0.5)


_FP8 = "torch.float8_e4m3fn"
_BF16 = "torch.bfloat16"


class TestFp8TransformerLayout:
    """The FP8 copy's files, index, config and tensors are pinned: a change would orphan every installed copy."""

    def test_the_copy_has_exactly_this_layout(self, tmp_path, tiny_models):
        checkpoint = _write_arange_checkpoint(tmp_path / "ltx.safetensors")

        weights.write_transformer(checkpoint, {"width": _WIDTH}, tmp_path / "q8", 8, device="cpu")

        out = tmp_path / "q8"
        assert sorted(path.name for path in out.iterdir()) == ["config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"]
        assert json.loads((out / "model.safetensors.index.json").read_text()) == {
            "metadata": {"total_size": 17152},
            "weight_map": dict.fromkeys(
                [
                    "proj_in.bias",
                    "proj_in.weight",
                    "transformer_blocks.0.0.bias",
                    "transformer_blocks.0.0.weight",
                    "transformer_blocks.0.1.bias",
                    "transformer_blocks.0.1.weight",
                    "transformer_blocks.1.0.bias",
                    "transformer_blocks.1.0.weight",
                    "transformer_blocks.1.1.bias",
                    "transformer_blocks.1.1.weight",
                ],
                "model-00001-of-00001.safetensors",
            ),
        }
        assert (out / "config.json").read_bytes() == b'{\n  "width": 64\n}\n'
        assert _read_folder(out) == {
            "proj_in.bias": (_BF16, [64], "13d545059a9b9602fcb157bf3069698b8bc60546875fcf8ecacdbed1811b306d"),
            "proj_in.weight": (_BF16, [64, 64], "14dfcbbf46c102e36bc2556ff8d38fb9479e6beecd7d7e5a45387fda10f4d9ba"),
            "transformer_blocks.0.0.bias": (_FP8, [64], "3d0cdf0a66f3981eee7381e6672b6dd479dab6bc541ebd94be64433fa41b729e"),
            "transformer_blocks.0.0.weight": (_FP8, [64, 64], "a6fc9a66b51351949f6f43a8670c4526db1c9c52f2ff535d09d12a67275148bc"),
            "transformer_blocks.0.1.bias": (_BF16, [64], "2107ea243587f8d79124dd5ffc72265c55d7bad31e2e56d2c852de5c5d443a5c"),
            "transformer_blocks.0.1.weight": (_BF16, [64], "f0a1bac823f36f546f702dd4e367e3f2c9d74b22eb0c9a117de477f91ae71fcd"),
            "transformer_blocks.1.0.bias": (_FP8, [64], "f1760224663e212bc10b10da20919be327c4a377607506575b0ff4484165be9c"),
            "transformer_blocks.1.0.weight": (_FP8, [64, 64], "e9d25977fd9d691356eef1f94a839a982b3ba1c07db367c42304e5d3e6d10bbf"),
            "transformer_blocks.1.1.bias": (_BF16, [64], "cfe69753b24cda93136133024a249d82e6acaeaebf3a7195ea6ecdf9ed3bd935"),
            "transformer_blocks.1.1.weight": (_BF16, [64], "16decb67ab0ced30e5669306db5d64e1f49834536708d0ca1548825760954157"),
        }


_NF4_PART_NAMES = ("absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4")
_NF4_BLOCK_LINEARS = ("transformer_blocks.0.0", "transformer_blocks.1.0")


class _FakeLinear4bit(torch.nn.Module):
    """Stands in for ``bitsandbytes.nn.Linear4bit``: records how it was built and keeps a (meta) bias like the real layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs
        self.bias = torch.nn.Parameter(torch.empty(args[1], device="meta", dtype=torch.bfloat16)) if kwargs["bias"] else None


class _FakeQuantState:
    def __init__(self, parts):
        self._parts = parts

    def as_dict(self, packed=False):
        assert packed is True
        return dict(self._parts)


class _FakeBitsandbytes:
    """Fake ``bitsandbytes`` modules (entered through ``sys.modules``) and a record of what the code under test asked of them."""

    def __init__(self):
        self.quantize_calls: list[dict] = []
        self.prequantized_calls: list[dict] = []

        functional = ModuleType("bitsandbytes.functional")
        functional.quantize_4bit = self._quantize_4bit
        params = type("Params4bit", (), {"from_prequantized": staticmethod(self._from_prequantized)})
        nn = ModuleType("bitsandbytes.nn")
        nn.Linear4bit = _FakeLinear4bit
        nn.Params4bit = params
        package = ModuleType("bitsandbytes")
        package.functional, package.nn = functional, nn
        self.modules = {"bitsandbytes": package, "bitsandbytes.functional": functional, "bitsandbytes.nn": nn}

    def _quantize_4bit(self, weight, **kwargs):
        self.quantize_calls.append({"dtype": weight.dtype, "device": weight.device.type, "shape": list(weight.shape), **kwargs})
        rows = weight.shape[0] * weight.shape[1] // 2
        packed = (torch.arange(rows) % 256).to(torch.uint8).reshape(-1, 1)
        parts = {
            "absmax": torch.arange(weight.numel() // 64, dtype=torch.float32),
            "quant_map": torch.arange(16, dtype=torch.float32),
            "nested_absmax": torch.ones(1, dtype=torch.float32),
            "nested_quant_map": torch.arange(256, dtype=torch.float32),
            "quant_state.bitsandbytes__nf4": torch.arange(5, dtype=torch.uint8),
        }
        return packed, _FakeQuantState(parts)

    def _from_prequantized(self, **kwargs):
        self.prequantized_calls.append({**kwargs, "quantized_stats": dict(kwargs["quantized_stats"])})
        return torch.nn.Parameter(kwargs["data"], requires_grad=False)


@pytest.fixture
def fake_bitsandbytes(monkeypatch):
    fake = _FakeBitsandbytes()
    for name, module in fake.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake


class TestNf4TransformerLayout:
    """The NF4 copy's layout and load, with a fake ``bitsandbytes`` so they run on the CPU."""

    def _write(self, tmp_path):
        checkpoint = _write_arange_checkpoint(tmp_path / "ltx.safetensors", torch.float32)
        weights.write_transformer(checkpoint, {"width": _WIDTH}, tmp_path / "q4", 4, device=torch.device("meta"))
        return tmp_path / "q4"

    def test_the_copy_holds_each_block_linear_weight_with_its_five_parts(self, tmp_path, tiny_models, fake_bitsandbytes):
        out = self._write(tmp_path)

        weight_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
        assert sorted(weight_map) == sorted(
            [
                "proj_in.bias",
                "proj_in.weight",
                "transformer_blocks.0.0.bias",
                "transformer_blocks.0.0.weight",
                *(f"transformer_blocks.0.0.weight.{part}" for part in _NF4_PART_NAMES),
                "transformer_blocks.0.1.bias",
                "transformer_blocks.0.1.weight",
                "transformer_blocks.1.0.bias",
                "transformer_blocks.1.0.weight",
                *(f"transformer_blocks.1.0.weight.{part}" for part in _NF4_PART_NAMES),
                "transformer_blocks.1.1.bias",
                "transformer_blocks.1.1.weight",
            ]
        )
        stored = _read_folder(out)
        plain = {name: stored[name] for name in stored if not name.startswith(tuple(f"{linear}.weight" for linear in _NF4_BLOCK_LINEARS))}
        assert {name: (dtype, shape) for name, (dtype, shape, _) in plain.items()} == {
            "proj_in.bias": (_BF16, [64]),
            "proj_in.weight": (_BF16, [64, 64]),
            "transformer_blocks.0.0.bias": (_BF16, [64]),
            "transformer_blocks.0.1.bias": (_BF16, [64]),
            "transformer_blocks.0.1.weight": (_BF16, [64]),
            "transformer_blocks.1.0.bias": (_BF16, [64]),
            "transformer_blocks.1.1.bias": (_BF16, [64]),
            "transformer_blocks.1.1.weight": (_BF16, [64]),
        }
        for linear in _NF4_BLOCK_LINEARS:
            assert stored[f"{linear}.weight"][:2] == ("torch.uint8", [2048, 1])
            assert {part: stored[f"{linear}.weight.{part}"][:2] for part in _NF4_PART_NAMES} == {
                "absmax": ("torch.float32", [64]),
                "quant_map": ("torch.float32", [16]),
                "nested_absmax": ("torch.float32", [1]),
                "nested_quant_map": ("torch.float32", [256]),
                "quant_state.bitsandbytes__nf4": ("torch.uint8", [5]),
            }

    def test_each_block_linear_is_packed_once_from_a_bfloat16_weight_on_the_given_device(self, tmp_path, tiny_models, fake_bitsandbytes):
        self._write(tmp_path)

        assert (
            fake_bitsandbytes.quantize_calls
            == [{"dtype": torch.bfloat16, "device": "meta", "shape": [64, 64], "blocksize": 64, "compress_statistics": True, "quant_type": "nf4", "quant_storage": torch.uint8}] * 2
        )

    def test_the_copy_loads_block_linears_as_linear4bit_and_leaves_the_rest(self, tmp_path, tiny_models, fake_bitsandbytes):
        out = self._write(tmp_path)

        model = weights.load_transformer({"width": _WIDTH}, prepared=out, bits=4, compute_dtype=torch.bfloat16)

        swapped = {name for name, module in model.named_modules() if isinstance(module, _FakeLinear4bit)}
        assert swapped == set(_NF4_BLOCK_LINEARS)
        assert type(model.proj_in) is torch.nn.Linear
        assert model.get_submodule(_NF4_BLOCK_LINEARS[0]).weight.dtype == torch.uint8


class TestConnectors:
    def _save(self, folder):
        reference = _reference()
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"width": _WIDTH}))
        save_file({name: tensor.contiguous() for name, tensor in reference.state_dict().items()}, str(folder / "diffusion_pytorch_model.safetensors"))
        return reference

    def test_connectors_load_with_fp8_layers(self, tmp_path):
        layer_dtype = torch.float8_e4m3fn
        reference = self._save(tmp_path / "connectors")
        with patch("diffusers.pipelines.ltx2.connectors.LTX2TextConnectors", _TinyTransformer):
            model = weights.load_connectors(tmp_path / "connectors", compute_dtype=torch.bfloat16)

        layer, source = model.transformer_blocks[0][0], reference.transformer_blocks[0][0]
        assert layer.weight.dtype == layer_dtype
        assert torch.equal(layer.weight, source.weight.to(layer_dtype))


class _TinyTextEncoder(torch.nn.Module):
    """Gemma 3's layout in miniature: a nested language model and an output head tied to its embedding."""

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(10, 8)
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.lm_head = torch.nn.Linear(8, 10, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def _empty_text_encoder(_source_dir, compute_dtype):
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = _TinyTextEncoder()
    for param in model.parameters():
        param.data = param.data.to(compute_dtype)
    return model


def _write_text_encoder_repo(repo):
    """Save a tiny text encoder under the older Gemma 3 names, as the diffusers repo does."""
    torch.manual_seed(1)
    source = _TinyTextEncoder()
    legacy = {name.replace("model.language_model.", "language_model.model."): tensor for name, tensor in source.state_dict().items() if name != "lm_head.weight"}
    (repo / "text_encoder").mkdir(parents=True)
    save_file(legacy, str(repo / "text_encoder" / "model-00001-of-00001.safetensors"))
    index = {"weight_map": dict.fromkeys(legacy, "model-00001-of-00001.safetensors")}
    (repo / "text_encoder" / "model.safetensors.index.json").write_text(json.dumps(index))
    return source


class TestTextEncoderWeights:
    def test_older_gemma_names_are_renamed_to_the_models(self):
        known = {"model.language_model.layers.0.weight", "lm_head.weight"}

        assert weights._text_encoder_name("language_model.model.layers.0.weight", known) == "model.language_model.layers.0.weight"
        assert weights._text_encoder_name("vision_tower.vision_model.encoder.x", known) == "model.vision_tower.encoder.x"
        assert weights._text_encoder_name("multi_modal_projector.norm.weight", known) == "model.multi_modal_projector.norm.weight"
        assert weights._text_encoder_name("lm_head.weight", known) == "lm_head.weight"

    def test_the_fp8_copy_loads_back_with_fp8_layers_and_a_tied_head(self, tmp_path):
        source = _write_text_encoder_repo(tmp_path / "repo")
        with patch.object(weights, "_empty_text_encoder", _empty_text_encoder):
            weights.write_text_encoder(tmp_path / "repo", tmp_path / "te")
            model = weights.load_text_encoder(tmp_path / "repo", prepared=tmp_path / "te", compute_dtype=torch.bfloat16)

        layer = model.model.language_model.layers[0]
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert torch.equal(layer.weight, source.model.language_model.layers[0].weight.to(torch.float8_e4m3fn))
        embed = model.model.language_model.embed_tokens.weight
        assert embed.dtype == torch.bfloat16  # embeddings keep the compute dtype
        assert model.lm_head.weight is embed


# The tensors of the LTX-2.5 text encoder that are not language-model weights, as the checkpoint names them (read from the
# header of unsloth/LTX-2.5-FP8), and the Gemma4UnifiedForConditionalGeneration names transformers 5.19 gives them.
_GEMMA4_CHECKPOINT_TO_MODEL = {
    "model.embed_audio.embedding_projection.weight": "model.embed_audio.embedding_projection.weight",
    "model.embed_vision.embedding_projection.weight": "model.embed_vision.multimodal_embedder.embedding_projection.weight",
    "model.vision_embedder.patch_dense.bias": "model.embed_vision.patch_dense.bias",
    "model.vision_embedder.patch_dense.weight": "model.embed_vision.patch_dense.weight",
    "model.vision_embedder.patch_ln1.bias": "model.embed_vision.patch_ln1.bias",
    "model.vision_embedder.patch_ln1.weight": "model.embed_vision.patch_ln1.weight",
    "model.vision_embedder.patch_ln2.bias": "model.embed_vision.patch_ln2.bias",
    "model.vision_embedder.patch_ln2.weight": "model.embed_vision.patch_ln2.weight",
    "model.vision_embedder.pos_embedding": "model.embed_vision.pos_embedding",
    "model.vision_embedder.pos_norm.bias": "model.embed_vision.pos_norm.bias",
    "model.vision_embedder.pos_norm.weight": "model.embed_vision.pos_norm.weight",
}
_GEMMA4_CLASS = "Gemma4UnifiedForConditionalGeneration"


def _write_model_index(repo, text_encoder_class):
    repo.mkdir(parents=True, exist_ok=True)
    index = {"_class_name": "LTX2Pipeline", "text_encoder": ["transformers", text_encoder_class]}
    (repo / "model_index.json").write_text(json.dumps(index))


def _written_names(repo, known):
    return sorted(name for name, _ in weights._source_text_encoder_tensors(repo, known))


class TestGemma4TextEncoderNames:
    _renames = weights._TEXT_ENCODER_RENAMES[_GEMMA4_CLASS]

    def test_the_vision_embedder_and_projection_are_renamed_and_known_names_stay(self):
        known = {"model.embed_vision.c", "model.embed_vision.a", "model.language_model.layers.0.weight"}

        assert weights._text_encoder_name("model.vision_embedder.a", known, self._renames) == "model.embed_vision.a"
        assert weights._text_encoder_name("model.embed_vision.b", known, self._renames) == "model.embed_vision.multimodal_embedder.b"
        assert weights._text_encoder_name("model.embed_vision.c", known, self._renames) == "model.embed_vision.c"
        assert weights._text_encoder_name("model.language_model.layers.0.weight", known, self._renames) == "model.language_model.layers.0.weight"
        assert weights._text_encoder_name("language_model.model.layers.0.weight", known, self._renames) == "model.language_model.layers.0.weight"

    def test_renaming_is_a_single_pass(self):
        # vision_embedder -> embed_vision must not be renamed again by the broad embed_vision rule.
        assert weights._text_encoder_name("model.vision_embedder.a", set(), self._renames) == "model.embed_vision.a"

    def test_the_default_table_leaves_the_gemma4_names_alone(self):
        assert weights._text_encoder_name("model.vision_embedder.a", set()) == "model.vision_embedder.a"
        assert weights._text_encoder_name("model.embed_vision.b", set()) == "model.embed_vision.b"

    def test_the_ltx25_checkpoint_covers_the_model_exactly(self):
        language = {f"model.language_model.layers.{i}.mlp.up_proj.weight" for i in range(3)}
        known = language | set(_GEMMA4_CHECKPOINT_TO_MODEL.values()) | {"lm_head.weight"}
        checkpoint = language | set(_GEMMA4_CHECKPOINT_TO_MODEL)

        mapped = {weights._text_encoder_name(name, known, self._renames) for name in checkpoint}

        assert known - mapped == {"lm_head.weight"}  # the tied head is not stored
        assert mapped - known == set()
        assert {name: weights._text_encoder_name(name, known, self._renames) for name in _GEMMA4_CHECKPOINT_TO_MODEL} == _GEMMA4_CHECKPOINT_TO_MODEL


class TestTextEncoderTableChoice:
    def _repo_with(self, repo, names):
        (repo / "text_encoder").mkdir(parents=True)
        save_file({name: torch.zeros(1) for name in names}, str(repo / "text_encoder" / "model-00001-of-00001.safetensors"))
        index = {"weight_map": dict.fromkeys(names, "model-00001-of-00001.safetensors")}
        (repo / "text_encoder" / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_a_gemma4_index_selects_the_gemma4_table(self, tmp_path):
        self._repo_with(tmp_path, ["model.vision_embedder.a", "model.embed_vision.b", "language_model.model.x"])
        _write_model_index(tmp_path, _GEMMA4_CLASS)

        names = _written_names(tmp_path, {"model.embed_vision.a", "model.language_model.x"})

        assert names == ["model.embed_vision.a", "model.embed_vision.multimodal_embedder.b", "model.language_model.x"]

    def test_a_gemma3_index_selects_the_legacy_table(self, tmp_path):
        self._repo_with(tmp_path, ["model.vision_embedder.a", "model.embed_vision.b", "language_model.model.x"])
        _write_model_index(tmp_path, "Gemma3ForConditionalGeneration")

        assert _written_names(tmp_path, set()) == ["model.embed_vision.b", "model.language_model.x", "model.vision_embedder.a"]

    def test_a_missing_index_selects_the_legacy_table(self, tmp_path):
        self._repo_with(tmp_path, ["model.vision_embedder.a", "model.embed_vision.b", "language_model.model.x"])

        assert _written_names(tmp_path, set()) == ["model.embed_vision.b", "model.language_model.x", "model.vision_embedder.a"]
