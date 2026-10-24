"""Low-memory, one-block-at-a-time loading for quantized or BF16 MiniMax-H3 DiTs."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import (
    DENSE_DEQUANT_PROFILE_OFF,
    MiniMaxH3DiT,
    TransformerBlock,
    apply_dense_dequant_profile_to_block,
    normalize_dense_dequant_profile,
)
from .quantize import (
    QuantConfig,
    _class_predicate,
    apply_quantization_structure,
    apply_quantized_slots,
)


# The rejected exact-key safetensors provider is intentionally not an active
# streamed-block mode. Keep only the reference MLX shard loader on generation paths.
BLOCK_LOAD_MODES = ("mlx",)


def _normalize_block_load_mode(mode: str) -> str:
    if mode not in BLOCK_LOAD_MODES:
        allowed = ", ".join(BLOCK_LOAD_MODES)
        raise ValueError(f"unknown block load mode {mode!r}; expected one of: {allowed}")
    return mode


class _BlockSlot(nn.Module):
    """One reusable block under the same `blocks.0` path as a checkpoint."""

    def __init__(
        self,
        config: DiTConfig,
        quantization: QuantConfig | None,
        source_block_index: int = 0,
    ) -> None:
        super().__init__()
        self.blocks = [TransformerBlock(config)]
        if quantization is not None:
            apply_quantization_structure(
                self,
                quantization.for_block_slot(source_block_index),
            )


class QuantizedBlockProvider:
    """Load quantized or full-precision transformer blocks lazily from indexed safetensors.

    By default the provider keeps the historical one reusable block slot.  The
    opt-in ``stream_block_group_size`` creates a small group of reusable slots and
    loads adjacent blocks from the same safetensors shard pass before serving them
    one by one.  This changes residency/IO granularity only; block order and math
    are unchanged.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        block_load_mode: str = "mlx",
        stream_block_group_size: int = 1,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.block_load_mode = _normalize_block_load_mode(block_load_mode)
        self.stream_block_group_size = int(stream_block_group_size)
        if self.stream_block_group_size <= 0:
            raise ValueError(f"stream_block_group_size must be positive, got {stream_block_group_size}")
        self.config = DiTConfig.from_json(self.model_dir / "config.json")
        quant_path = self.model_dir / "quant_config.json"
        if quant_path.is_file():
            with quant_path.open() as handle:
                raw_quant = json.load(handle)
            self.quantization: QuantConfig | None = QuantConfig(
                bits=int(raw_quant["bits"]),
                group_size=int(raw_quant["group_size"]),
                mode=str(raw_quant.get("mode", "affine")),
                quantize_adaln=bool(raw_quant.get("quantize_adaln", False)),
                adaln_bits=int(raw_quant.get("adaln_bits") or 8),
                overrides={path: None for path in raw_quant.get("bf16_layers", [])},
            )
        else:
            self.quantization = None
        with (self.model_dir / "model.safetensors.index.json").open() as handle:
            self.weight_map = json.load(handle)["weight_map"]
        initial_indices = [
            min(index, self.config.num_layers - 1)
            for index in range(self.stream_block_group_size)
        ]
        self.slots = [
            _BlockSlot(self.config, self.quantization, source_block_index=index)
            for index in initial_indices
        ]
        self._slot_overrides = [
            self.quantization.for_block_slot(index).overrides
            if self.quantization is not None
            else None
            for index in initial_indices
        ]
        self.slot = self.slots[0]
        self.expected = {key for key, _ in tree_flatten(self.slot.parameters())}
        self.current_index: int | None = None
        self.current_group_start: int | None = None
        self.current_group_end: int | None = None
        self.current_group_load_kind: tuple[bool, bool] | None = None
        self.current_lora = None
        self.turbo_lora = None
        self.refiner_loras = None
        self.final_lora = None
        self.logical_bytes_loaded = 0
        self.group_load_count = 0
        self.shard_load_count = 0
        self.group_cache_hit_count = 0
        self.dense_dequant_profile = DENSE_DEQUANT_PROFILE_OFF
        self.dense_dequant_attention_qkv_tile_size = 2048
        self.dense_dequant_ffn_fc2_tile_size = 1024
        self.dense_dequant_attention_out_tile_size = 2048

    def set_block_load_mode(self, mode: str) -> None:
        """Validate legacy callers while keeping only the reference mx.load path active."""

        self.block_load_mode = _normalize_block_load_mode(mode)

    def set_dense_dequant_profile(
        self,
        profile: str | None,
        *,
        attention_qkv_tile_size: int = 2048,
        ffn_fc2_tile_size: int = 1024,
        attention_out_tile_size: int = 2048,
    ) -> None:
        """Configure the opt-in/provenance dense-dequant profile for streamed blocks."""

        self.dense_dequant_profile = normalize_dense_dequant_profile(profile)
        self.dense_dequant_attention_qkv_tile_size = int(attention_qkv_tile_size)
        self.dense_dequant_ffn_fc2_tile_size = int(ffn_fc2_tile_size)
        self.dense_dequant_attention_out_tile_size = int(attention_out_tile_size)

    def set_turbo_lora(
        self,
        path: str | Path,
        *,
        alpha: float | None = None,
        scale: float = 1.0,
    ) -> None:
        path = Path(path)
        if path.is_dir():
            from .native_turbo_lora import NativeTurboLoRAProvider

            if alpha is not None:
                raise ValueError(
                    "native Turbo records alpha=rank per target; do not pass a fixed alpha"
                )
            self.turbo_lora = NativeTurboLoRAProvider(
                path,
                num_blocks=self.config.num_layers,
                num_refiner_blocks=self.config.token_refiner_num_layers,
                scale=scale,
            )
            self.final_lora = self.turbo_lora.load_final()
        else:
            from .turbo_lora import TurboLoRAProvider

            self.turbo_lora = TurboLoRAProvider(
                path,
                num_blocks=self.config.num_layers,
                num_refiner_blocks=self.config.token_refiner_num_layers,
                hidden_size=self.config.hidden_size,
                inner_dim=self.config.inner_dim,
                ffn_hidden_size=self.config.ffn_hidden_size,
                alpha=alpha,
                scale=scale,
            )
            self.final_lora = None
        self.refiner_loras = self.turbo_lora.load_refiners()

    @property
    def block_count(self) -> int:
        return self.config.num_layers

    def _ensure_slot_structure(self, slot_index: int, block_index: int) -> None:
        if self.quantization is None:
            return
        mapped = self.quantization.for_block_slot(block_index)
        if self._slot_overrides[slot_index] != mapped.overrides:
            self.slots[slot_index] = _BlockSlot(
                self.config,
                self.quantization,
                source_block_index=block_index,
            )
            self._slot_overrides[slot_index] = mapped.overrides
        if slot_index == 0:
            self.slot = self.slots[0]
            self.expected = {key for key, _ in tree_flatten(self.slot.parameters())}

    @staticmethod
    def _expected_for_load(
        slot: _BlockSlot,
        *,
        include_adaln: bool,
        adaln_only: bool,
    ) -> set[str]:
        expected = {key for key, _ in tree_flatten(slot.parameters())}
        if adaln_only:
            return {key for key in expected if ".adaln_proj." in key}
        if include_adaln:
            return expected
        return {key for key in expected if ".adaln_proj." not in key}

    def _slot_for_loaded_index(self, index: int) -> _BlockSlot:
        if self.current_group_start is None or self.current_group_end is None:
            raise RuntimeError("no streamed block group is loaded")
        return self.slots[index - self.current_group_start]

    def _prepare_block_slot(self, block: TransformerBlock) -> None:
        clear_fc1_dense_cache = getattr(block.mlp, "clear_fc1_dense_dequant_cache", None)
        if clear_fc1_dense_cache is not None:
            clear_fc1_dense_cache()
        clear_fc2_dense_cache = getattr(block.mlp, "clear_fc2_dense_dequant_cache", None)
        if clear_fc2_dense_cache is not None:
            clear_fc2_dense_cache()
        clear_out_dense_cache = getattr(block.attn, "clear_out_dense_dequant_cache", None)
        if clear_out_dense_cache is not None:
            clear_out_dense_cache()
        apply_dense_dequant_profile_to_block(
            block,
            self.dense_dequant_profile,
            attention_qkv_tile_size=self.dense_dequant_attention_qkv_tile_size,
            ffn_fc2_tile_size=self.dense_dequant_ffn_fc2_tile_size,
            attention_out_tile_size=self.dense_dequant_attention_out_tile_size,
        )

    def _set_current_lora(self, index: int, *, adaln_only: bool) -> None:
        del adaln_only
        self.current_lora = (
            self.turbo_lora.load_block(index)
            if self.turbo_lora is not None
            else None
        )

    def load_block(
        self,
        index: int,
        *,
        include_adaln: bool = True,
        adaln_only: bool = False,
    ) -> TransformerBlock:
        if not 0 <= index < self.block_count:
            raise IndexError(f"block index {index} outside 0..{self.block_count - 1}")
        load_kind = (bool(include_adaln), bool(adaln_only))
        if (
            self.current_group_start is not None
            and self.current_group_end is not None
            and self.current_group_load_kind == load_kind
            and self.current_group_start <= index < self.current_group_end
        ):
            self.group_cache_hit_count += 1
            self.current_index = index
            self._set_current_lora(index, adaln_only=adaln_only)
            return self._slot_for_loaded_index(index).blocks[0]

        group_start = (index // self.stream_block_group_size) * self.stream_block_group_size
        group_end = min(group_start + self.stream_block_group_size, self.block_count)
        for offset, block_index in enumerate(range(group_start, group_end)):
            self._ensure_slot_structure(offset, block_index)
        expected_by_slot = [
            self._expected_for_load(
                self.slots[offset],
                include_adaln=include_adaln,
                adaln_only=adaln_only,
            )
            for offset in range(group_end - group_start)
        ]
        updates_by_slot: list[list[tuple[str, mx.array]]] = [[] for _ in range(group_end - group_start)]
        found_by_slot: list[set[str]] = [set() for _ in range(group_end - group_start)]
        # The previous block has already been synchronized by the caller. Drop its arrays before
        # opening the next multi-gigabyte BF16 shard so old and new blocks do not overlap.
        for offset in range(group_end - group_start):
            slot = self.slots[offset]
            all_keys = {key for key, _ in tree_flatten(slot.parameters())}
            slot.update(tree_unflatten([(key, mx.array(0, dtype=mx.uint32)) for key in all_keys]))
        mx.synchronize()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        key_routes: dict[str, tuple[int, str]] = {}
        by_shard: dict[str, list[str]] = {}
        for block_index in range(group_start, group_end):
            slot_index = block_index - group_start
            source_prefix = f"blocks.{block_index}."
            target_prefix = "blocks.0."
            source_keys = [key for key in self.weight_map if key.startswith(source_prefix)]
            if adaln_only:
                source_keys = [key for key in source_keys if ".adaln_proj." in key]
            elif not include_adaln:
                source_keys = [key for key in source_keys if ".adaln_proj." not in key]
            for source_key in source_keys:
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key not in expected_by_slot[slot_index]:
                    raise KeyError(f"streaming slot has no parameter {target_key!r}")
                key_routes[source_key] = (slot_index, target_key)
                by_shard.setdefault(self.weight_map[source_key], []).append(source_key)

        for shard_name, keys in by_shard.items():
            arrays = mx.load(str(self.model_dir / shard_name))
            self.shard_load_count += 1
            for source_key in keys:
                slot_index, target_key = key_routes[source_key]
                tensor = arrays[source_key]
                updates_by_slot[slot_index].append((target_key, tensor))
                found_by_slot[slot_index].add(target_key)
                self.logical_bytes_loaded += tensor.nbytes

        for offset, block_index in enumerate(range(group_start, group_end)):
            missing = sorted(expected_by_slot[offset] - found_by_slot[offset])
            if missing:
                raise KeyError(f"block {block_index} is missing {len(missing)} tensors, e.g. {missing[:4]}")
            slot = self.slots[offset]
            slot.update(tree_unflatten(updates_by_slot[offset]))
            self._prepare_block_slot(slot.blocks[0])

        self.current_group_start = group_start
        self.current_group_end = group_end
        self.current_group_load_kind = load_kind
        self.group_load_count += 1
        self.current_index = index
        self._set_current_lora(index, adaln_only=adaln_only)
        return self._slot_for_loaded_index(index).blocks[0]


def load_streaming_dit(
    model_dir: str | Path,
    *,
    turbo_lora_path: str | Path | None = None,
    turbo_lora_alpha: float | None = None,
    turbo_lora_scale: float = 1.0,
    block_load_mode: str = "mlx",
    stream_block_group_size: int = 1,
    verbose: bool = False,
) -> tuple[MiniMaxH3DiT, QuantizedBlockProvider]:
    """Load static DiT weights and leave the 50 main blocks file-backed."""
    model_dir = Path(model_dir)
    provider = QuantizedBlockProvider(
        model_dir,
        block_load_mode=block_load_mode,
        stream_block_group_size=stream_block_group_size,
    )
    model = MiniMaxH3DiT(provider.config, build_blocks=False)

    # Quantized checkpoints need matching packed slots. A source BF16 checkpoint keeps the normal
    # dense MLX module structure and loads the serialized arrays without conversion.
    if provider.quantization is not None:
        predicate = _class_predicate(provider.quantization)

        def static_predicate(path: str, module: nn.Module):
            if path.startswith("blocks."):
                return False
            return predicate(path, module)

        apply_quantized_slots(model, static_predicate)

    expected = {
        key
        for key, _ in tree_flatten(model.parameters())
        if not key.startswith("blocks.")
    }
    source_keys = [key for key in provider.weight_map if not key.startswith("blocks.")]
    by_shard: dict[str, list[str]] = {}
    for key in source_keys:
        by_shard.setdefault(provider.weight_map[key], []).append(key)

    updates: list[tuple[str, mx.array]] = []
    for shard_name, keys in by_shard.items():
        arrays = mx.load(str(model_dir / shard_name))
        for key in keys:
            if key in expected:
                updates.append((key, arrays[key]))

    found = {key for key, _ in updates}
    missing = sorted(expected - found)
    if missing:
        raise KeyError(f"streaming DiT static weights missing {len(missing)} tensors, e.g. {missing[:4]}")
    model.update(tree_unflatten(updates))
    mx.eval(*(tensor for _, tensor in updates))
    if turbo_lora_path is not None:
        provider.set_turbo_lora(
            turbo_lora_path,
            alpha=turbo_lora_alpha,
            scale=turbo_lora_scale,
        )
    if verbose:
        size = sum(tensor.nbytes for _, tensor in updates) / 1e9
        precision = "quantized" if provider.quantization is not None else "BF16/full-precision"
        print(
            f"loaded {len(updates)} static tensors ({size:.2f} GB, {precision}); "
            f"main blocks stream lazily in groups of {provider.stream_block_group_size}"
        )
    return model, provider


__all__ = ["BLOCK_LOAD_MODES", "QuantizedBlockProvider", "load_streaming_dit"]