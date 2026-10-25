"""Converted MiniMax H3 folders on macOS: write one from a fine-tune.

A converted folder is a local H3 folder in the layout the MLX port loads:

- ``transformer/``: the fine-tune's transformer under MiniMax's own names (fused QKV interleaved per head, the gated
  feed-forward as ``[gate; value]``), quantized with ``mx.quantize`` to the shipped recipe (8-bit, groups of 32, the
  AdaLN projections at 8-bit too; the float32 modules kept as they are), with ``config.json``, ``quant_config.json``
  and a shard index. One tensor is held at a time (the three of a QKV while it is fused).
- ``FL2VA/``: the stock components (VAEs, tokenizer, processor), hard-linked from the stock download. The text encoder is
  read from the stock download itself at load time, so stock and every converted model share one text-encoder copy.
- the folder manifest (:mod:`zvisiongenerator.utils.h3_converted`), written last.

The turbo LoRA is merged before quantizing unless the checkpoint is already step-distilled, so a converted model never
gets it at run time. Quantization is plain round-to-nearest: the shipped transformer's calibration needs recorded
activations and a CUDA device.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.converters.h3_checkpoint import STOCK_CONFIG, diffusers_tensors, native_name, stock_time_embedder
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.h3_converted import write_converted_h3
from zvisiongenerator.utils.h3_lora import LoraMerger, load_pairs
from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora
from zvisiongenerator.utils.hf_snapshot import snapshot_download as hub_snapshot_download
from zvisiongenerator.utils.hf_snapshot import snapshot_with
from zvisiongenerator.utils.safetensors_stream import ShardWriter

if TYPE_CHECKING:
    import torch

    from zvisiongenerator.converters.h3_checkpoint import H3Checkpoint

__all__ = ["STORAGE", "convert", "native_config", "native_tensors", "write_converted"]

STORAGE = "mlx-int8"
_BITS = 8
_GROUP_SIZE = 32
_TRANSFORMER_DIR = "transformer"
_COMPONENTS_DIR = "FL2VA"
_TEXT_ENCODER_DIR = "text_encoder"
_INDEX_NAME = "model.safetensors.index.json"
# The float32 modules of the release; everything else is stored in bfloat16 (as the fork's ``load.FP32_PREFIXES``).
_FP32_PREFIXES = ("video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.video_out.", "final_layer.audio_out.")
_QKV = ("to_q", "to_k", "to_v")
# What the shipped recipe quantizes (the fork's ``quantize.CORE_LINEARS``, plus the block AdaLN projections at 8 bits).
_CORE_LINEARS = (".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1", ".mlp.fc2")
_BLOCK_ADALN = re.compile(r"blocks\.\d+\.adaln_proj\.linear")

Report = Callable[[str, int, int], None]
# Quantizes a weight into MLX's ``(packed weight, scales, biases)`` as torch tensors (uint32, bfloat16, bfloat16).
Quantize = Callable[["torch.Tensor"], tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]


def convert(checkpoint: H3Checkpoint, target: Path, *, turbo: bool, config: dict[str, Any], report: Report) -> None:
    """Write the MLX folder of *checkpoint* into *target*, with the stock components.

    Raises:
        RuntimeError: When a stock file can't be downloaded.
    """
    from zvisiongenerator.backends.mac.memory import mlx_cache_limit

    preset = h3_mlx_preset(config)
    report("download", 0, 1)
    components = snapshot_with(preset.components, hub_snapshot_download) / _COMPONENTS_DIR
    lora_file, scale = None, 0.0
    if not turbo:
        lora, scale = h3_turbo_lora(config)
        lora_file = snapshot_with(lora, hub_snapshot_download) / lora.files[0]
    # Stock's time embedder is unquantized in the shipped INT8 transformer: its index is fetched, then only its tensors are read by range from the shard holding it.
    time_embedder = stock_time_embedder(preset.transformer.repo_id, preset.transformer.revision, index=_INDEX_NAME, native=True) if checkpoint.curve_rank is not None else None
    # No MLX buffer cache: each tensor is quantized once, and cached buffers would add up to the whole transformer.
    with mlx_cache_limit(0):
        write_converted(checkpoint, target, components=components, lora_file=lora_file, lora_scale=scale, time_embedder=time_embedder, quantize=_mlx_quantize, report=report)


def write_converted(
    checkpoint: H3Checkpoint,
    target: Path,
    *,
    components: Path,
    lora_file: Path | None,
    lora_scale: float,
    time_embedder: dict[str, Any] | None,
    quantize: Quantize,
    report: Report,
) -> None:
    """Write the MLX folder of *checkpoint* into the new folder *target*.

    Args:
        checkpoint: The checked checkpoint.
        target: The folder to write (a partial folder; the caller moves it into place).
        components: The stock ``FL2VA`` folder the other components are linked from.
        lora_file: The turbo LoRA to merge before quantizing, or ``None`` for an already step-distilled checkpoint.
        lora_scale: The turbo merge factor, which replaces the LoRA's ``alpha / rank``.
        time_embedder: Stock's time embedder, which un-prunes a pruned checkpoint.
        quantize: Quantizes one weight (``mx.quantize`` at 8 bits, groups of 32).
        report: Called with ``(stage, done, total)`` as the conversion advances.
    """
    merger = LoraMerger(load_pairs(lora_file) if lora_file is not None else {}, lora_scale)
    transformer = target / _TRANSFORMER_DIR
    transformer.mkdir(parents=True)
    tensors = diffusers_tensors(checkpoint, progress=lambda done, total: report("transformer", done, total), time_embedder=time_embedder)
    writer = ShardWriter(transformer, "model", _INDEX_NAME)
    for name, tensor in native_tensors(_merged(tensors, merger)):
        module = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and tensor.ndim == 2 and _quantized(module):
            weight, scales, biases = quantize(tensor)
            writer.add({name: weight, f"{module}.scales": scales, f"{module}.biases": biases})
        else:
            writer.add({name: _stored(name, tensor)})
    if missed := merger.unmerged():
        raise RuntimeError(f"The turbo LoRA targets {len(missed)} modules the transformer does not have, e.g. {missed[0]}.")
    writer.close()
    (transformer / "config.json").write_text(json.dumps(native_config(), indent=2) + "\n", encoding="utf-8")
    quant = {"bits": _BITS, "group_size": _GROUP_SIZE, "mode": "affine", "quantize_adaln": True, "adaln_bits": _BITS, "bf16_layers": []}
    (transformer / "quant_config.json").write_text(json.dumps(quant, indent=2) + "\n", encoding="utf-8")
    report("components", 0, 1)
    (target / _COMPONENTS_DIR).mkdir()
    for source in sorted(components.iterdir()):
        if source.name not in (_TRANSFORMER_DIR, _TEXT_ENCODER_DIR):
            link_tree(source, target / _COMPONENTS_DIR / source.name)
    write_converted_h3(
        target,
        {
            "storage": STORAGE,
            "turbo": "merged" if lora_file is not None else "distilled",
            "turbo_lora": lora_file.name if lora_file is not None else None,
            "turbo_scale": lora_scale if lora_file is not None else None,
            "pruned_rank": checkpoint.curve_rank,
            "source": checkpoint.path.name,
            "source_bytes": sum(path.stat().st_size for path in checkpoint.files),
        },
    )


def native_config() -> dict[str, Any]:
    """Return the ``config.json`` of the MLX port's transformer, in the release's names, from stock's config."""
    config = STOCK_CONFIG
    return {
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "token_refiner_num_layers": config["num_refiner_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "attention_head_dim": config["attention_head_dim"],
        "ffn_hidden_size": config["ffn_dim"],
        "latents_dim": config["in_channels"],
        "audio_latents_dim": config["audio_in_channels"],
        "patch_size": config["patch_size"],
        "text_dim": config["text_dim"],
        "timestep_input_dim": config["freq_dim"],
        "time_embed_hidden_size": config["time_embed_hidden_dim"],
        "time_embed_dim": config["time_embed_dim"],
        "adaln_out_features": 6 * 3 * config["hidden_size"],
        "final_adaln_out_features": 2 * config["hidden_size"],
        "rope_inv_freq_len": 16,
        "norm_eps": 1e-05,
        "qk_norm_eps": 1e-05,
        "final_norm_eps": 1e-05,
    }


def native_tensors(tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield diffusers-named H3 tensors under MiniMax's names: each block's ``to_q``/``to_k``/``to_v`` fused per head once all three passed, the feed-forward's halves swapped back.

    Raises:
        RuntimeError: When a QKV is incomplete at the end.
    """
    import torch

    heads, head_dim = STOCK_CONFIG["num_attention_heads"], STOCK_CONFIG["attention_head_dim"]
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        stem, _, leaf = name.rpartition(".")
        if leaf == "weight" and stem.rpartition(".")[2] in _QKV:
            attention = stem.rpartition(".")[0]
            parts = pending.setdefault(attention, {})
            parts[stem.rpartition(".")[2]] = tensor
            if parts.keys() == set(_QKV):
                del pending[attention]
                grouped = torch.stack([parts[part].reshape(heads, head_dim, -1) for part in _QKV], dim=1)
                yield native_name(f"{attention}.qkv_proj.weight"), grouped.reshape(3 * heads * head_dim, -1)
        elif name.endswith(".ff.net.0.proj.weight"):
            value, gate = tensor.chunk(2, dim=0)
            yield native_name(name).replace(".ff.net.0.proj.weight", ".mlp.fc1.weight"), torch.cat([gate, value], dim=0)
        else:
            yield native_name(name), tensor
    if pending:
        raise RuntimeError(f"The attention {sorted(pending)[0]} lacks some of its query, key and value weights.")


def _quantized(module: str) -> bool:
    """Return whether the linear *module* (MiniMax name) is quantized by the shipped recipe, as the fork's ``QuantConfig`` decides it: the core linears of the blocks and the refiner, and the block AdaLN projections."""
    return module.endswith(_CORE_LINEARS) or _BLOCK_ADALN.fullmatch(module) is not None


def _stored(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Return an unquantized tensor in the dtype the release stores it in: float32 for its float32 modules, else bfloat16."""
    import torch

    return tensor.to(torch.float32 if name.startswith(_FP32_PREFIXES) else torch.bfloat16).contiguous()


def _merged(tensors: Iterable[tuple[str, torch.Tensor]], merger: LoraMerger) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield *tensors* with each weight the turbo LoRA targets merged by *merger*."""
    for name, tensor in tensors:
        module, _, leaf = name.rpartition(".")
        yield name, merger.apply(module, tensor) if leaf == "weight" else tensor


def _mlx_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize *weight* in bfloat16 with ``mx.quantize`` (8 bits, groups of 32); return the packed weight, scales and biases as torch tensors."""
    import mlx.core as mx
    import numpy as np
    import torch

    source = mx.array(weight.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()).view(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=_GROUP_SIZE, bits=_BITS)

    def _bf16(array: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(array.astype(mx.bfloat16).view(mx.uint16))).view(torch.bfloat16)

    return torch.from_numpy(np.array(packed)), _bf16(scales), _bf16(biases)
