"""Pin the LTX-2 layer names ``normalise_lora_keys`` writes for diffusers' LoRA converter.

Builds a tiny ``LTX2VideoTransformer3DModel`` on the meta device (no weights) for its real module names, and runs the real
``LTX2LoraLoaderMixin.lora_state_dict`` on names the normaliser wrote. A failure after a diffusers bump means a layer a LoRA
adapts is renamed or dropped by the converter, so the root sets in ``utils/lora_keys.py`` need to follow.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers.loaders.lora_pipeline import LTX2LoraLoaderMixin  # noqa: E402

from zvisiongenerator.utils.lora_keys import normalise_lora_keys  # noqa: E402

# The trainer's (ltx-core) top-level modules and the diffusers module each becomes, from the converter's own renames. The
# projections are Linear layers themselves; the others hold one as ``.linear``.
_PROJECTIONS = {"patchify_proj": "proj_in", "audio_patchify_proj": "audio_proj_in"}
_NATIVE_TO_DIFFUSERS = {
    "adaln_single": "time_embed",
    "audio_adaln_single": "audio_time_embed",
    "prompt_adaln_single": "prompt_adaln",
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
}
_TINY = {
    "attention_head_dim": 8,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "audio_in_channels": 8,
    "audio_num_attention_heads": 2,
    "audio_out_channels": 8,
    "caption_channels": 16,
    "cross_attention_dim": 16,
    "in_channels": 8,
    "num_attention_heads": 2,
    "num_layers": 1,
    "out_channels": 8,
    "cross_attn_mod": True,
    "audio_cross_attn_mod": True,
    "use_prompt_embeddings": True,
}


@pytest.fixture(scope="module")
def linear_modules() -> list[str]:
    """Return the name of every Linear layer of a tiny LTX-2 transformer: the layers a LoRA can adapt."""
    with torch.device("meta"):
        model = diffusers.LTX2VideoTransformer3DModel(**_TINY)
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]


def _converted(keys: list[str]) -> set[str]:
    """Return the keys diffusers' converter gives for LoRA tensors named *keys*, after the normaliser's renames."""
    renames = normalise_lora_keys(keys, "ltx")
    return set(LTX2LoraLoaderMixin.lora_state_dict({renames.get(key, key): torch.zeros(1) for key in keys}))


def _spellings(module: str) -> dict[str, list[str]]:
    plain = [f"{module}.lora_A.weight", f"{module}.lora_B.weight"]
    return {
        "bare": plain,
        "peft": [f"base_model.model.{module}.lora_A.default.weight", f"base_model.model.{module}.lora_B.default.weight"],
        "native prefix": [f"diffusion_model.{module}.lora_A.weight", f"diffusion_model.{module}.lora_B.weight"],
        "diffusers prefix": [f"transformer.{module}.lora_A.weight", f"transformer.{module}.lora_B.weight"],
    }


@pytest.mark.parametrize("spelling", ["bare", "peft", "native prefix", "diffusers prefix"])
def test_every_adaptable_layer_of_the_model_reaches_the_transformer_under_its_own_name(linear_modules, spelling):
    for module in linear_modules:
        keys = _spellings(module)[spelling]

        assert _converted(keys) == {f"transformer.{module}.lora_A.weight", f"transformer.{module}.lora_B.weight"}, module


@pytest.mark.parametrize("spelling", ["bare", "peft", "native prefix"])
def test_every_top_level_layer_of_the_trainers_model_reaches_a_layer_of_the_diffusers_model(linear_modules, spelling):
    layers = {native: (diffusers_name, "") for native, diffusers_name in _PROJECTIONS.items()}
    layers |= {native: (diffusers_name, ".linear") for native, diffusers_name in _NATIVE_TO_DIFFUSERS.items()}
    for native, (diffusers_name, inner) in layers.items():
        assert f"{diffusers_name}{inner}" in linear_modules

        assert _converted(_spellings(f"{native}{inner}")[spelling]) == {f"transformer.{diffusers_name}{inner}.lora_A.weight", f"transformer.{diffusers_name}{inner}.lora_B.weight"}, native


def test_a_file_that_mixes_the_trainers_layers_with_blocks_keeps_all_of_them():
    keys = []
    for module in ("transformer_blocks.0.attn1.to_q", "patchify_proj", "audio_patchify_proj", "adaln_single.linear", "av_ca_a2v_gate_adaln_single.linear", "prompt_adaln_single.linear", "proj_out"):
        keys += _spellings(module)["bare"]

    assert len(_converted(keys)) == len(keys)
