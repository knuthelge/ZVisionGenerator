"""Shared fixtures for the converted-H3 tests: a tiny H3 config and a tiny checkpoint under diffusers' and MiniMax's names."""

from __future__ import annotations

import torch

from zvisiongenerator.converters import h3_checkpoint

_TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 4,
    "hidden_size": 8,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 16,
    "in_channels": 2,
    "audio_in_channels": 2,
    "patch_size": [1, 2, 2],
    "text_dim": 4,
    "freq_dim": 8,
    "time_embed_hidden_dim": 8,
    "time_embed_dim": 6,
}


def _diffusers_state(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Return a random tiny H3 transformer under diffusers' names, in the shapes the reader expects."""
    return {name: torch.randn(shape, generator=generator) for name, shape in h3_checkpoint._expected_shapes(None).items()}


def _native(state: dict[str, torch.Tensor], *, interleave: bool) -> dict[str, torch.Tensor]:
    """Return *state* under MiniMax's names: QKV fused (per head when *interleave*), fc1 halves as ``[gate; value]``."""
    heads, head_dim = _TINY["num_attention_heads"], _TINY["attention_head_dim"]
    renames = {new: old for old, new in h3_checkpoint._RENAMES}
    native: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        source = name
        for new, old in renames.items():
            if source.startswith(new):
                source = old + source[len(new) :]
                break
        for old, new in h3_checkpoint._INNER_RENAMES:
            source = source.replace(new, old)
        if name.endswith(".attn.to_q.weight"):
            stem = name.removesuffix("to_q.weight")
            q, k, v = (state[f"{stem}{part}.weight"] for part in ("to_q", "to_k", "to_v"))
            if interleave:
                fused = torch.stack([t.reshape(heads, head_dim, -1) for t in (q, k, v)], dim=1).reshape(3 * heads * head_dim, -1)
            else:
                fused = torch.cat([q, k, v])
            native[source.replace("to_q.weight", "qkv_proj.weight")] = fused
        elif name.endswith(".ff.net.0.proj.weight"):
            value, gate = tensor.chunk(2)
            native[source.replace(".ff.net.0.proj.weight", ".mlp.fc1.weight")] = torch.cat([gate, value])
        else:
            native[source] = tensor
    native["rope.inv_freq"] = torch.ones(2)
    return native
