"""MLX-native, file-streamed Larry MiniMax-H3 Turbo LoRA provider."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx

from .turbo_lora import LoRAWeights

_MAIN_RE = re.compile(
    r"^blocks\.(\d+)\."
    r"(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)\."
    r"lora_([AB])\.weight$"
)
_REFINER_RE = re.compile(
    r"^token_refiner\.blocks\.(\d+)\."
    r"(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\."
    r"lora_([AB])\.weight$"
)
_FINAL_RE = re.compile(r"^final_layer\.adaln_proj\.linear\.lora_([AB])\.weight$")

MAIN_TARGETS = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
    "adaln_proj.linear",
)
REFINER_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
FINAL_TARGET = "final_layer.adaln_proj.linear"


class NativeTurboLoRAProvider:
    """Strictly validate and stream a canonical native H3 Turbo adapter directory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        num_blocks: int,
        num_refiner_blocks: int,
        scale: float = 1.0,
    ) -> None:
        self.directory = Path(directory)
        if scale < 0:
            raise ValueError("Turbo LoRA scale must be non-negative")
        self.multiplier = float(scale)
        manifest_path = self.directory / "conversion_manifest.json"
        index_path = self.directory / "adapter.safetensors.index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"native Turbo directory requires conversion_manifest.json and adapter index: {self.directory}"
            )
        self.manifest = json.loads(manifest_path.read_text())
        index = json.loads(index_path.read_text())
        if self.manifest.get("format") != "minimax-h3-mlx-native-turbo-lora-v1":
            raise ValueError(f"unsupported native Turbo format: {self.manifest.get('format')!r}")
        contract = self.manifest.get("contract", {})
        if contract.get("alpha_policy") != "alpha=rank" or float(
            contract.get("runtime_multiplier", -1)
        ) != 1.0:
            raise ValueError("native Turbo manifest must record alpha=rank and multiplier=1")
        if not contract.get("all_source_tensors_consumed") or not contract.get(
            "post_write_tensor_values_exact"
        ):
            raise ValueError("native Turbo conversion is not complete/exact")
        self.weight_map = dict(index.get("weight_map") or {})
        self.num_blocks = int(num_blocks)
        self.num_refiner_blocks = int(num_refiner_blocks)
        expected_tensors = (self.num_blocks * len(MAIN_TARGETS) + self.num_refiner_blocks * len(REFINER_TARGETS) + 1) * 2
        if len(self.weight_map) != expected_tensors:
            raise ValueError(
                f"native Turbo index has {len(self.weight_map)} tensors, expected {expected_tensors}"
            )
        self.key_map: dict[tuple[str, int, str, str], str] = {}
        self.ranks: dict[tuple[str, int, str], int] = {}
        self._parse_and_validate()

    def _parse_and_validate(self) -> None:
        shapes = {
            str(row["mlx_key"]): tuple(int(v) for v in row["shape"])
            for row in self.manifest.get("mapping", [])
        }
        dtypes = {
            str(row["mlx_key"]): str(row["dtype"])
            for row in self.manifest.get("mapping", [])
        }
        if set(shapes) != set(self.weight_map):
            raise ValueError("native Turbo manifest/index tensor sets differ")
        for key in self.weight_map:
            match = _MAIN_RE.fullmatch(key)
            if match:
                raw_index, target, side = match.groups()
                family, index = "block", int(raw_index)
            else:
                match = _REFINER_RE.fullmatch(key)
                if match:
                    raw_index, target, side = match.groups()
                    family, index = "refiner", int(raw_index)
                else:
                    match = _FINAL_RE.fullmatch(key)
                    if not match:
                        raise ValueError(f"unsupported native Turbo key: {key}")
                    (side,) = match.groups()
                    family, index, target = "final", 0, FINAL_TARGET
            identity = (family, index, target, side)
            if identity in self.key_map:
                raise ValueError(f"duplicate native Turbo tensor: {identity}")
            if dtypes[key] != "BF16" or len(shapes[key]) != 2:
                raise TypeError(f"native Turbo tensor must be a BF16 matrix: {key}")
            self.key_map[identity] = key

        required: list[tuple[str, int, str]] = []
        required.extend(
            ("block", index, target)
            for index in range(self.num_blocks)
            for target in MAIN_TARGETS
        )
        required.extend(
            ("refiner", index, target)
            for index in range(self.num_refiner_blocks)
            for target in REFINER_TARGETS
        )
        required.append(("final", 0, FINAL_TARGET))
        for family, index, target in required:
            a_key = self.key_map.get((family, index, target, "A"))
            b_key = self.key_map.get((family, index, target, "B"))
            if a_key is None or b_key is None:
                raise KeyError(f"native Turbo pair missing: {(family, index, target)}")
            rank = shapes[a_key][0]
            if shapes[b_key][-1] != rank:
                raise ValueError(f"native Turbo rank mismatch: {(family, index, target)}")
            self.ranks[(family, index, target)] = rank
        if len(self.key_map) != len(required) * 2:
            raise ValueError(
                f"native Turbo has {len(self.key_map)} mapped tensors, expected {len(required) * 2}"
            )

    def _load(self, family: str, index: int, targets: tuple[str, ...]) -> LoRAWeights:
        keys = [
            self.key_map[(family, index, target, side)]
            for target in targets
            for side in ("A", "B")
        ]
        shard_names = {self.weight_map[key] for key in keys}
        if len(shard_names) != 1:
            raise ValueError(f"native Turbo component spans multiple shards: {(family, index)}")
        shard_path = self.directory / next(iter(shard_names))
        arrays = mx.load(str(shard_path))
        if set(arrays) != set(keys):
            raise KeyError(
                f"native Turbo shard tensor set mismatch for {(family, index)}: "
                f"{sorted(set(keys) - set(arrays))[:4]} missing"
            )
        tensors = {}
        for target in targets:
            a = arrays[self.key_map[(family, index, target, "A")]]
            b = arrays[self.key_map[(family, index, target, "B")]]
            mx.eval(a, b)
            tensors[target] = (a, b)
        return LoRAWeights(tensors=tensors, multiplier=self.multiplier)

    def load_block(self, index: int) -> LoRAWeights:
        if not 0 <= index < self.num_blocks:
            raise IndexError(index)
        return self._load("block", index, MAIN_TARGETS)

    def load_refiners(self) -> list[LoRAWeights]:
        return [self._load("refiner", index, REFINER_TARGETS) for index in range(self.num_refiner_blocks)]

    def load_final(self) -> LoRAWeights:
        return self._load("final", 0, (FINAL_TARGET,))


__all__ = [
    "FINAL_TARGET",
    "MAIN_TARGETS",
    "NativeTurboLoRAProvider",
    "REFINER_TARGETS",
]
