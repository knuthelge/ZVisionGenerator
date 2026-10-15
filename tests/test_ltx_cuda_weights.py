"""Tests for the LTX weights of the CUDA video backend, with tiny real models on the CPU."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402
from diffusers.models.modeling_utils import ModelMixin  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends import ltx_cuda_weights as weights  # noqa: E402
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


class TestEnsurePrepared:
    def _source(self, tmp_path, content=b"weights"):
        source = tmp_path / "source.safetensors"
        source.write_bytes(content)
        return source

    def test_a_missing_copy_is_written_and_marked_complete(self, tmp_path):
        source = self._source(tmp_path)
        target = tmp_path / "cache" / "model@q8"

        result = weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("x"))

        assert result == target
        assert (target / "w").read_text() == "x"
        assert json.loads((target / MANIFEST_NAME).read_text())["bits"] == 8

    def test_a_current_copy_is_reused_without_writing(self, tmp_path):
        source = self._source(tmp_path)
        target = tmp_path / "model@q8"
        weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("x"))
        writes: list[object] = []

        weights.ensure_prepared(target, [source], 8, writes.append)

        assert writes == []

    def test_a_changed_source_or_level_writes_a_new_copy(self, tmp_path):
        source = self._source(tmp_path)
        target = tmp_path / "model@q8"
        weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("old"))
        source.write_bytes(b"newer weights")

        weights.ensure_prepared(target, [source], 8, lambda folder: (folder / "w").write_text("new"))

        assert (target / "w").read_text() == "new"

    def test_a_current_copy_elsewhere_is_hard_linked_instead_of_written(self, tmp_path):
        source = self._source(tmp_path)
        first = tmp_path / "ltx-8@q8" / "text_encoder"
        weights.ensure_prepared(first, [source], 8, lambda folder: (folder / "w").write_text("x"))
        writes: list[object] = []

        second = weights.ensure_prepared(tmp_path / "ltx-4@q4" / "text_encoder", [source], 8, writes.append, reuse=[first])

        assert writes == []
        assert (second / "w").stat().st_ino == (first / "w").stat().st_ino  # one file on disk
        assert json.loads((second / MANIFEST_NAME).read_text())["bits"] == 8

    def test_a_stale_copy_elsewhere_is_not_reused(self, tmp_path):
        source = self._source(tmp_path)
        other = tmp_path / "ltx-8@q8" / "text_encoder"
        weights.ensure_prepared(other, [source], 8, lambda folder: (folder / "w").write_text("old"))
        source.write_bytes(b"newer weights")

        target = weights.ensure_prepared(tmp_path / "ltx-4@q4" / "text_encoder", [source], 8, lambda folder: (folder / "w").write_text("new"), reuse=[other])

        assert (target / "w").read_text() == "new"

    def test_a_failed_write_leaves_no_copy_behind(self, tmp_path):
        source = self._source(tmp_path)
        target = tmp_path / "model@q4"

        def _fail(folder):
            (folder / "w").write_text("half")
            raise RuntimeError("disk full")

        with pytest.raises(RuntimeError, match="disk full"):
            weights.ensure_prepared(target, [source], 4, _fail)

        assert not target.exists()
        assert list(tmp_path.glob(".model@q4.*.partial")) == []


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
