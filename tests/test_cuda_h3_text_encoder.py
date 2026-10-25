"""Tests for the H3 text encoder: the cut Qwen3-VL, its filters and loader, and contracts against diffusers' encoder steps.

Everything runs on a tiny Qwen3-VL (52 decoder layers, a four-block vision tower) with a byte-level tokenizer and a
processor written to ``tmp_path``; nothing real is loaded.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("torchvision")  # Qwen3-VL's processor needs it; macOS does not install it
safetensors_torch = pytest.importorskip("safetensors.torch")
PIL_Image = pytest.importorskip("PIL.Image")

from zvisiongenerator.backends.cuda.video.h3 import text_encoder  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.text_encoder import (  # noqa: E402
    TEXT_TAG,
    VIDEO_TAG,
    empty_text_encoder,
    encode,
    load_text_encoder,
    tokenize_presentation,
    wanted_tensor,
)

_SPECIAL = ["<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
_LAYERS = 52  # the tiny checkpoint has more layers than the 50 the encoder keeps, as the real one does
_PROMPT = "a red cat walks"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Return a tiny checkpoint: tokenizer and processor directories, the config, and a random float32 reference model."""
    from tokenizers import pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor

    root = tmp_path_factory.mktemp("tiny_h3_te")
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Qwen2TokenizerFast(vocab=vocab, merges=[], unk_token="<|endoftext|>", pad_token="<|endoftext|>", additional_special_tokens=_SPECIAL)
    tokenizer.save_pretrained(root / "tokenizer")
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer"))
    images = Qwen2VLImageProcessor(patch_size=4, merge_size=2, temporal_patch_size=2, size={"shortest_edge": 64, "longest_edge": 4096})
    processor = Qwen3VLProcessor(image_processor=images, tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor())
    processor.save_pretrained(root / "processor")
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in _SPECIAL}
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": _LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [1, 2, 3],
        },
        image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
    )
    config.save_pretrained(root / "config")
    torch.manual_seed(0)
    reference = Qwen3VLForConditionalGeneration(config).eval()
    return types.SimpleNamespace(root=root, config=config, reference=reference, tokenizer=tokenizer, processor=processor)


def _cut_model(tiny, *, vision: bool = True):
    """Return the 50-layer encoder in float32 holding the reference model's weights."""
    model = empty_text_encoder(tiny.config, layers=50, vision=vision, dtype=torch.float32)
    state = {name[len("model.") :]: tensor for name, tensor in tiny.reference.state_dict().items() if wanted_tensor(name, 50, vision)}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def _keyframes(count: int):
    colors = [(10, 200, 30), (200, 40, 90)]
    return [PIL_Image.new("RGB", (40, 24), colors[index]) for index in range(count)]


def _diffusers_step(tiny, step_class, **inputs):
    """Run diffusers' encoder step on the reference model; return its ``(prompt_embeds, text_token_tags)``."""
    from diffusers.modular_pipelines import PipelineState

    components = types.SimpleNamespace(
        text_encoder=tiny.reference,
        tokenizer=tiny.tokenizer,
        processor=tiny.processor,
        text_encoder_layer=50,
        _execution_device=torch.device("cpu"),
        text_tag=TEXT_TAG,
        video_tag=VIDEO_TAG,
    )
    state = PipelineState()
    for name, value in inputs.items():
        state.set(name, value)
    _, state = step_class()(components, state)
    return state.get("prompt_embeds"), state.get("text_token_tags")


# --- filters -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vision", "wanted"),
    [
        ("model.language_model.embed_tokens.weight", True, True),
        ("model.language_model.layers.0.self_attn.q_proj.weight", True, True),
        ("model.language_model.layers.49.mlp.down_proj.weight.absmax", True, True),
        ("model.language_model.layers.50.self_attn.q_proj.weight", True, False),
        ("model.language_model.layers.63.input_layernorm.weight", True, False),
        ("model.language_model.norm.weight", True, False),
        ("lm_head.weight", True, False),
        ("model.visual.blocks.0.attn.qkv.weight", True, True),
        ("model.visual.merger.linear_fc2.bias", True, True),
        ("model.visual.blocks.0.attn.qkv.weight", False, False),
        ("model.language_model.layers.0.self_attn.q_proj.weight", False, True),
    ],
)
def test_wanted_tensor(name, vision, wanted):
    assert wanted_tensor(name, 50, vision) is wanted


# --- the cut model -----------------------------------------------------------------------------------------------


def test_empty_encoder_keeps_the_layers_with_an_identity_norm(tiny):
    model = empty_text_encoder(tiny.config, layers=50, vision=True)

    assert len(model.language_model.layers) == 50
    assert isinstance(model.language_model.norm, torch.nn.Identity)
    assert model.visual is not None
    assert {param.device.type for param in model.parameters()} == {"meta"}
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}
    assert tiny.config.text_config.num_hidden_layers == _LAYERS  # the checkpoint's config is left alone


def test_empty_encoder_without_vision_has_no_vision_tower(tiny):
    model = empty_text_encoder(tiny.config, layers=50, vision=False)

    assert model.visual is None
    assert not any(name.startswith("visual.") for name in model.state_dict())


@pytest.mark.parametrize("layers", [0, _LAYERS + 1])
def test_empty_encoder_refuses_a_layer_count_outside_the_checkpoint(tiny, layers):
    with pytest.raises(ValueError, match="between 1 and 52"):
        empty_text_encoder(tiny.config, layers=layers)


def test_the_tap_is_the_raw_output_of_the_last_kept_layer(tiny):
    model = _cut_model(tiny)
    captured = []
    handle = model.language_model.layers[-1].register_forward_hook(lambda _module, _args, output: captured.append(output))
    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", [])

    result = encode(model, presentation, "cpu")
    handle.remove()

    assert torch.equal(result, captured[0])
    # With the stock final norm the last hidden state is the normalised one, which is not the conditioning.
    stock = _cut_model(tiny)
    stock.language_model.norm = torch.nn.RMSNorm(32)
    assert not torch.allclose(encode(stock, presentation, "cpu"), result)


def test_encode_refuses_keyframes_without_a_vision_tower(tiny):
    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", _keyframes(1))

    with pytest.raises(ValueError, match="vision tower"):
        encode(_cut_model(tiny, vision=False), presentation, "cpu")


# --- contracts against diffusers' encoder steps ------------------------------------------------------------------


def test_tags_match_diffusers():
    from diffusers.modular_pipelines.minimax_h3 import modular_pipeline

    assert (TEXT_TAG, VIDEO_TAG) == (modular_pipeline.MINIMAX_H3_TEXT_TAG, modular_pipeline.MINIMAX_H3_VIDEO_TAG)


def test_text_only_presentation_matches_the_t2va_step(tiny):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep

    expected, expected_tags = _diffusers_step(tiny, MiniMaxH3TextEncoderStep, prompt=_PROMPT)

    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", [])
    embeds = encode(_cut_model(tiny), presentation, "cpu")

    assert list(presentation.token_ids) == tiny.tokenizer(_PROMPT, add_special_tokens=False)["input_ids"]
    assert torch.equal(torch.tensor(presentation.token_tags), expected_tags)
    assert not presentation.vision_inputs
    assert embeds.shape == expected.shape == (1, len(presentation.token_ids), 32)
    torch.testing.assert_close(embeds, expected, rtol=1e-5, atol=1e-6)


def test_the_t2va_presentation_does_not_read_the_processor(tiny, tmp_path):
    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tmp_path / "missing", [])

    assert len(presentation.token_ids) == len(presentation.token_tags) > 0


@pytest.mark.parametrize("count", [1, 2])
def test_keyframe_presentation_matches_the_fl2va_step(tiny, count):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3FL2VATextEncoderStep

    images = _keyframes(count)
    expected, expected_tags = _diffusers_step(tiny, MiniMaxH3FL2VATextEncoderStep, prompt=_PROMPT, keyframes=images)

    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", images)
    embeds = encode(_cut_model(tiny), presentation, "cpu")

    assert torch.equal(torch.tensor(presentation.token_tags), expected_tags)
    assert VIDEO_TAG in presentation.token_tags
    assert list(presentation.mm_token_type_ids) == tiny.processor.create_mm_token_type_ids([list(presentation.token_ids)])[0]
    assert set(presentation.mm_token_type_ids) == {0, 1}
    pads = presentation.token_ids.count(tiny.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    assert pads == count * 15  # a 40x24 image is a 10x6 patch grid, 15 merged positions
    assert embeds.shape == expected.shape == (1, len(presentation.token_ids), 32)
    torch.testing.assert_close(embeds, expected, rtol=1e-5, atol=1e-6)


# --- the NF4 loader ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def packed_tensors(tiny):
    """Return the reference model's tensors as a checkpoint holds them: NF4 decoder linears, everything else bfloat16 (packed once, it is slow)."""
    from zvisiongenerator.backends.cuda import quant

    pytest.importorskip("bitsandbytes")
    tensors = {}
    for name, tensor in tiny.reference.state_dict().items():
        if name.startswith("model.language_model.layers.") and name.endswith("_proj.weight"):
            tensors.update(quant.pack_nf4(name, tensor, "cpu"))
        else:
            tensors[name] = tensor.to(torch.bfloat16)
    return {name: tensor.contiguous() for name, tensor in tensors.items()}


def _write_shards(packed, directory, *, drop=lambda _name: False):
    """Write *packed* (less the tensors *drop* accepts) as two shards in *directory*; return their paths."""
    names = sorted(name for name in packed if not drop(name))
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "model-00001-of-00002.safetensors", directory / "model-00002-of-00002.safetensors"]
    safetensors_torch.save_file({name: packed[name] for name in names[::2]}, str(paths[0]))
    safetensors_torch.save_file({name: packed[name] for name in names[1::2]}, str(paths[1]))
    return paths


@pytest.fixture
def streaming_spies(monkeypatch):
    """Replace the block streaming (it pins memory, which needs a GPU) with recorders; return their call lists."""
    streamed, moved = [], []
    monkeypatch.setattr(text_encoder, "stream_blocks", lambda blocks, device: streamed.append((list(blocks), device)))
    monkeypatch.setattr(text_encoder, "move_outside", lambda model, blocks, device: moved.append((model, blocks, device)))
    return types.SimpleNamespace(streamed=streamed, moved=moved)


def test_load_builds_the_nf4_tower_and_the_vision_tower(tiny, packed_tensors, tmp_path, streaming_spies):
    from bitsandbytes.nn import Linear4bit

    paths = _write_shards(packed_tensors, tmp_path / "shards")

    model = load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=True)

    layers = model.language_model.layers
    assert len(layers) == 50
    assert isinstance(layers[0].self_attn.q_proj, Linear4bit)
    assert isinstance(layers[49].mlp.down_proj, Linear4bit)
    assert layers[0].self_attn.q_proj.weight.quant_state is not None
    assert not isinstance(model.visual.blocks[0].attn.qkv, Linear4bit)
    assert isinstance(model.language_model.norm, torch.nn.Identity)
    assert {param.dtype for param in model.visual.parameters()} == {torch.bfloat16}
    torch.testing.assert_close(model.language_model.embed_tokens.weight, tiny.reference.model.language_model.embed_tokens.weight.to(torch.bfloat16))
    torch.testing.assert_close(model.visual.blocks[0].attn.qkv.weight, tiny.reference.model.visual.blocks[0].attn.qkv.weight.to(torch.bfloat16))
    torch.testing.assert_close(layers[3].input_layernorm.weight, tiny.reference.model.language_model.layers[3].input_layernorm.weight.to(torch.bfloat16))
    assert not model.training


def test_load_streams_the_layers_and_keeps_the_rest_on_the_device(tiny, packed_tensors, tmp_path, streaming_spies):
    paths = _write_shards(packed_tensors, tmp_path / "shards")

    model = load_text_encoder(tiny.root / "config", paths, "cuda:7", layers=50, vision=True)

    ((blocks, streamed_device),) = streaming_spies.streamed
    assert blocks == list(model.language_model.layers)
    assert streamed_device == "cuda:7"
    ((moved_model, moved_blocks, moved_device),) = streaming_spies.moved
    assert moved_model is model
    assert moved_blocks is model.language_model.layers
    assert moved_device == "cuda:7"


def test_load_without_vision_leaves_the_vision_tower_out(tiny, packed_tensors, tmp_path, streaming_spies):
    paths = _write_shards(packed_tensors, tmp_path / "shards")

    model = load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=False)

    assert model.visual is None
    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", [])
    embeds = encode(model, presentation, "cpu")
    assert embeds.shape == (1, len(presentation.token_ids), 32)
    assert embeds.dtype == torch.bfloat16
    assert torch.isfinite(embeds.float()).all()


def test_a_loaded_encoder_encodes_keyframes(tiny, packed_tensors, tmp_path, streaming_spies):
    paths = _write_shards(packed_tensors, tmp_path / "shards")
    model = load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=True)
    presentation = tokenize_presentation(_PROMPT, tiny.root / "tokenizer", tiny.root / "processor", _keyframes(2))

    embeds = encode(model, presentation, "cpu")

    assert embeds.shape == (1, len(presentation.token_ids), 32)
    assert torch.isfinite(embeds.float()).all()


def test_load_reads_only_what_the_cut_model_needs(tiny, packed_tensors, tmp_path, streaming_spies, monkeypatch):
    paths = _write_shards(packed_tensors, tmp_path / "shards")
    read = []
    original = text_encoder.iter_tensors

    def spy(path, wanted=None):
        for name, tensor in original(path, wanted):
            read.append(name)
            yield name, tensor

    monkeypatch.setattr(text_encoder, "iter_tensors", spy)

    load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=False)

    assert read
    assert not any(".layers.50." in name or ".layers.51." in name for name in read)
    assert not any(name.startswith(("lm_head.", "model.visual.", "model.language_model.norm.")) for name in read)
    assert "model.language_model.embed_tokens.weight" in read


def test_load_refuses_shards_that_miss_a_tensor_before_reading_any(tiny, packed_tensors, tmp_path, streaming_spies, monkeypatch):
    paths = _write_shards(packed_tensors, tmp_path / "shards", drop=lambda name: name == "model.language_model.layers.7.self_attn.k_proj.weight")
    monkeypatch.setattr(text_encoder, "iter_tensors", lambda *_args: pytest.fail("weights were read"))

    with pytest.raises(RuntimeError, match=r"do not hold every tensor.*layers\.7\.self_attn\.k_proj\.weight"):
        load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=True)


def test_load_refuses_an_incomplete_nf4_weight(tiny, packed_tensors, tmp_path, streaming_spies):
    paths = _write_shards(packed_tensors, tmp_path / "shards", drop=lambda name: name == "model.language_model.layers.2.mlp.up_proj.weight.nested_absmax")

    with pytest.raises(RuntimeError, match=r"layers\.2\.mlp\.up_proj\.weight is incomplete"):
        load_text_encoder(tiny.root / "config", paths, "cpu", layers=50, vision=True)


def test_load_refuses_layers_outside_the_checkpoint(tiny, packed_tensors, tmp_path, streaming_spies):
    paths = _write_shards(packed_tensors, tmp_path / "shards")

    with pytest.raises(ValueError, match="between 1 and 52"):
        load_text_encoder(tiny.root / "config", paths, "cpu", layers=60, vision=True)
