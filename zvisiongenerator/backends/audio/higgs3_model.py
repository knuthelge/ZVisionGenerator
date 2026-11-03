"""The Higgs TTS 3 network: a Qwen3 backbone with a fused multi-codebook audio embedding and head, and its weight loader.

The fused codebook embedding and head are adapted from sgl-project/sglang-omni (``sglang_omni/models/higgs_tts/modeling.py``,
Apache License 2.0, Copyright 2025-2026 SGLang-Omni Team) at upstream commit ``d80161dc34d7b900e7a0869481fa79d35ed618d9``.
Changed: the classes are renamed and the head computes in ``forward``, so ``module(x)`` runs it and forward hooks fire. See
THIRD_PARTY_LICENSES.md. The backbone is transformers' Apache-2.0 ``Qwen3Model``, used as a library class.

The loader builds the network on the meta device, so nothing is allocated twice, and fills it from the official
``model.safetensors`` one tensor at a time through a strict name map. Tensors the model does not use are never read.

This module imports torch and transformers at module level, so the backend imports it lazily.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from zvisiongenerator.backends.audio.higgs3_prompt import AUDIO_SLOT
from zvisiongenerator.utils.safetensors_stream import iter_tensors

if TYPE_CHECKING:
    from transformers import Cache

    from zvisiongenerator.backends.audio.higgs3_config import Higgs3Config

__all__ = ["FusedCodebookEmbedding", "FusedCodebookHead", "Higgs3Model", "checkpoint_target", "load_higgs3_model"]

# The two buffers of the backbone's rotary embedding. They are not in the checkpoint; the loader computes them.
_ROTARY_BUFFERS = frozenset({"backbone.rotary_emb.inv_freq", "backbone.rotary_emb.original_inv_freq"})
_TEXT_EMBEDDING = "tied.embedding.text_embedding.weight"
_AUDIO_EMBEDDING = "tied.embedding.modality_embeddings.0.embedding.weight"
# Checkpoint tensors the model does not use: the codec copy bundled in the file, the text head, and the stored copy of the
# audio head (the head shares the audio embedding's weight).
_SKIPPED_PREFIXES = ("tied.embedding.modality_embeddings.0.model.",)
_SKIPPED_NAMES = frozenset({"tied.head.text_head.weight", "tied.head.modality_heads.0.weight"})


class FusedCodebookEmbedding(nn.Module):
    """One embedding table of ``N * V`` rows that serves N codebooks of V values, summing the N rows of a frame."""

    def __init__(self, num_codebooks: int, codebook_size: int, hidden_size: int) -> None:
        """Build the table.

        Args:
            num_codebooks: N, the number of codebooks.
            codebook_size: V, the values per codebook.
            hidden_size: The width of an embedding.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * codebook_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Return the embedding of *codes* ``[..., N]`` as ``[..., D]``: codebook c reads row ``code + c * V``, and the N rows are summed."""
        offsets = torch.arange(self.num_codebooks, device=codes.device, dtype=codes.dtype) * self.codebook_size
        return F.embedding(codes + offsets, self.weight).sum(dim=-2)


class FusedCodebookHead(nn.Module):
    """One linear map from a hidden state to the logits of all N codebooks."""

    def __init__(self, num_codebooks: int, codebook_size: int, hidden_size: int) -> None:
        """Build the head; its weight is tied to the audio embedding's after loading.

        Args:
            num_codebooks: N, the number of codebooks.
            codebook_size: V, the values per codebook.
            hidden_size: The width of a hidden state.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * codebook_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the logits ``[L, N, V]`` of *hidden* ``[L, D]``."""
        return F.linear(hidden, self.weight).reshape(hidden.shape[0], self.num_codebooks, self.codebook_size)


class Higgs3Model(nn.Module):
    """The Higgs TTS 3 network: a Qwen3 backbone, a fused audio embedding, and a fused audio head tied to it.

    The backbone's own token embedding holds the text and the structural tokens of the prompt; the model never predicts text.
    """

    def __init__(self, config: Higgs3Config) -> None:
        """Build the network with uninitialised weights.

        Args:
            config: The network sizes.
        """
        super().__init__()
        self.backbone = Qwen3Model(_qwen3_config(config))
        self.audio_embedding = FusedCodebookEmbedding(config.num_codebooks, config.codebook_size, config.hidden_size)
        self.audio_head = FusedCodebookHead(config.num_codebooks, config.codebook_size, config.hidden_size)

    def embed_prompt(self, prompt_ids: list[int], reference_rows: torch.Tensor | None) -> torch.Tensor:
        """Return the prompt's embeddings ``[1, S, D]``: text embeddings, with the reference rows' audio embeddings at the slots.

        Args:
            prompt_ids: The prompt, with ``AUDIO_SLOT`` where a reference row goes.
            reference_rows: The delayed reference codes ``[R, N]``, one row per slot; ``None`` when the prompt has no slots.

        Raises:
            ValueError: If the number of slots is not the number of reference rows.
        """
        device = self.audio_embedding.weight.device
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        slots = ids == AUDIO_SLOT
        embeds = self.backbone.embed_tokens(torch.where(slots, torch.zeros_like(ids), ids))
        wanted = int(slots.sum().item())
        rows = 0 if reference_rows is None else reference_rows.shape[0]
        if wanted != rows:
            raise ValueError(f"The prompt has {wanted} audio slots but {rows} reference rows were given.")
        if reference_rows is not None and wanted:
            audio = self.audio_embedding(reference_rows.to(device))
            embeds[slots] = audio.to(embeds.dtype)
        return embeds.unsqueeze(0)

    def prefill(self, embeds: torch.Tensor) -> tuple[torch.Tensor, Cache]:
        """Run the prompt's *embeds* ``[1, S, D]`` through the backbone and return the last position's hidden state ``[1, D]`` and the cache."""
        output = self.backbone(inputs_embeds=embeds, use_cache=True)
        return output.last_hidden_state[:, -1, :], output.past_key_values

    def step(self, codes: torch.Tensor, cache: Cache) -> torch.Tensor:
        """Run one frame of *codes* ``[N]`` (on the model's device) through the backbone on *cache* and return its hidden state ``[1, D]``."""
        embeds = self.audio_embedding(codes.unsqueeze(0)).unsqueeze(1)
        output = self.backbone(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        return output.last_hidden_state[:, -1, :]


def _qwen3_config(config: Higgs3Config) -> Qwen3Config:
    """Return the transformers config of the backbone that *config* describes."""
    return Qwen3Config(
        vocab_size=config.text_vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        max_position_embeddings=config.max_positions,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        pad_token_id=None,
        use_cache=True,
        attn_implementation="sdpa",
    )


def checkpoint_target(name: str) -> str | None:
    """Return the name of the model parameter that the checkpoint tensor *name* fills, or ``None`` for a tensor the model does not use.

    Raises:
        ValueError: If *name* is not a tensor of the official checkpoint.
    """
    if name == _TEXT_EMBEDDING:
        return "backbone.embed_tokens.weight"
    if name == _AUDIO_EMBEDDING:
        return "audio_embedding.weight"
    if name in _SKIPPED_NAMES or name.startswith(_SKIPPED_PREFIXES):
        return None
    if name.startswith("body."):
        return "backbone." + name.removeprefix("body.")
    raise ValueError(f"The checkpoint holds a tensor this model has no place for: {name}")


@contextmanager
def _bf16_default() -> Iterator[None]:
    """Make new floating-point parameters bf16 for the length of the block, then restore the previous default dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_higgs3_model(config: Higgs3Config, checkpoint: Path, device: str) -> Higgs3Model:
    """Return the network for *config* on *device*, filled from the official safetensors file *checkpoint*, in bf16 and ready to run.

    The network is built on the meta device, allocated empty on *device*, and filled one tensor at a time, so peak host
    memory is about one tensor. Tensors the model does not use are never read.

    Args:
        config: The network sizes.
        checkpoint: The official ``model.safetensors``.
        device: The torch device to load onto.

    Raises:
        ValueError: If a checkpoint tensor has no place in the model, or its shape or dtype is not the model's.
        RuntimeError: If a parameter is filled twice or never, or a buffer is missing, extra or never written.
    """
    with torch.device("meta"), _bf16_default():
        model = Higgs3Model(config)
    model.to_empty(device=device)
    with torch.no_grad():
        _fill_parameters(model, checkpoint)
        model.audio_head.weight = model.audio_embedding.weight
        written = _write_rotary_buffers(model)
    _check_buffers(model, written)
    wrong = [name for name, parameter in model.named_parameters() if parameter.dtype != torch.bfloat16]
    if wrong:
        raise RuntimeError(f"These parameters are not bf16: {', '.join(wrong)}")
    return model.eval().requires_grad_(False)


def _fill_parameters(model: Higgs3Model, checkpoint: Path) -> None:
    """Copy each used tensor of *checkpoint* into its parameter, and check that every parameter but the tied head was filled exactly once."""
    parameters = dict(model.named_parameters())
    filled: set[str] = set()
    for name, tensor in iter_tensors(checkpoint, wanted=lambda tensor_name: checkpoint_target(tensor_name) is not None):
        target = checkpoint_target(name)
        destination = parameters.get(target) if target is not None else None
        if destination is None:
            raise ValueError(f"The checkpoint tensor {name} maps to {target}, which this model does not have.")
        if target in filled:
            raise RuntimeError(f"The checkpoint holds {name} more than once, or two tensors map to {target}.")
        if tensor.shape != destination.shape or tensor.dtype != destination.dtype:
            raise ValueError(f"The checkpoint tensor {name} is {tuple(tensor.shape)} {tensor.dtype}; the model needs {tuple(destination.shape)} {destination.dtype}.")
        destination.copy_(tensor)
        filled.add(target)
    missing = sorted(set(parameters) - filled - {"audio_head.weight"})
    if missing:
        raise RuntimeError(f"The checkpoint has no tensor for: {', '.join(missing)}")


def _write_rotary_buffers(model: Higgs3Model) -> set[str]:
    """Compute the rotary frequencies on the CPU, as transformers does at construction, copy them into the backbone's two buffers, and return the buffer names written."""
    rotary = model.backbone.rotary_emb
    inv_freq, _ = Qwen3RotaryEmbedding.compute_default_rope_parameters(rotary.config)
    rotary.inv_freq.copy_(inv_freq)
    rotary.original_inv_freq.copy_(inv_freq)
    return {"backbone.rotary_emb.inv_freq", "backbone.rotary_emb.original_inv_freq"}


def _check_buffers(model: nn.Module, written: set[str]) -> None:
    """Check that the buffers of *model* are exactly the two rotary buffers, and that both are in *written*.

    A buffer left over from ``to_empty`` holds garbage rather than failing loudly, so every buffer must be accounted for.

    Raises:
        RuntimeError: If a buffer is extra, missing or not written; the message names it.
    """
    names = {name for name, _ in model.named_buffers()}
    extra = sorted(names - _ROTARY_BUFFERS)
    if extra:
        raise RuntimeError(f"The model has buffers the loader does not write: {', '.join(extra)}")
    missing = sorted(_ROTARY_BUFFERS - names)
    if missing:
        raise RuntimeError(f"The model has no buffer: {', '.join(missing)}")
    unwritten = sorted(_ROTARY_BUFFERS - written)
    if unwritten:
        raise RuntimeError(f"The loader did not write the buffer: {', '.join(unwritten)}")
