"""Streaming PEFT LoRA support for the MiniMax-H3 Turbo adapter."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

DEFAULT_TURBO_LORA_ALPHA = 8.0

TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_KEY = re.compile(
    r"(?:^|.*\.)(transformer_blocks|token_refiner\.refiner_blocks)\."
    r"(\d+)\.(attn\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))\."
    r"lora_([AB])\.default\.weight$"
)


@dataclass(frozen=True)
class LoRAWeights:
    """One block's logical Diffusers projections, held outside the base module tree."""

    tensors: dict[str, tuple[mx.array, mx.array]]
    multiplier: float

    def has(self, target: str) -> bool:
        return target in self.tensors

    def apply(self, target: str, inputs: mx.array) -> mx.array:
        lora_a, lora_b = self.tensors[target]
        hidden = inputs.astype(lora_a.dtype) @ lora_a.T
        output = hidden @ lora_b.T
        return (output * self.multiplier).astype(inputs.dtype)


def _metadata_alpha(metadata: dict[str, str] | None) -> float | None:
    """Return a LoRA alpha encoded in safetensors metadata, if present."""
    if not metadata:
        return None
    for key in ("alpha", "lora_alpha", "peft_lora_alpha"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Turbo LoRA metadata {key} must be numeric, got {value!r}") from exc
    return None


class TurboLoRAProvider:
    """Validate one PEFT adapter and load only the requested H3 block's tensors."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_blocks: int,
        num_refiner_blocks: int,
        hidden_size: int,
        inner_dim: int,
        ffn_hidden_size: int,
        alpha: float | None = None,
        scale: float = 1.0,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Turbo LoRA not found: {self.path}")
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value < 0:
            raise ValueError("Turbo LoRA scale must be finite and non-negative")
        if alpha is not None:
            alpha_value = float(alpha)
            if not math.isfinite(alpha_value) or alpha_value <= 0:
                raise ValueError("Turbo LoRA alpha must be finite and positive")

        self.metadata: dict[str, str] = {}
        self.key_map: dict[tuple[str, int, str, str], str] = {}
        self.shapes: dict[tuple[str, int, str, str], tuple[int, ...]] = {}
        ranks: set[int] = set()
        unsupported: list[str] = []
        with safe_open(self.path, framework="np") as checkpoint:
            self.metadata = dict(checkpoint.metadata() or {})
            for key in checkpoint.keys():
                match = _KEY.fullmatch(key)
                if match is None:
                    unsupported.append(key)
                    continue
                family, raw_index, target, side = match.groups()
                kind = "block" if family == "transformer_blocks" else "refiner"
                index = int(raw_index)
                identity = (kind, index, target, side)
                if identity in self.key_map:
                    raise ValueError(f"duplicate Turbo LoRA tensor for {identity}: {key}")
                shape = tuple(checkpoint.get_slice(key).get_shape())
                if len(shape) != 2:
                    raise ValueError(f"Turbo LoRA tensor {key} must be a matrix, got {shape}")
                self.key_map[identity] = key
                self.shapes[identity] = shape
                if side == "A":
                    ranks.add(int(shape[0]))

        if unsupported:
            raise ValueError(
                f"Turbo LoRA contains {len(unsupported)} unsupported keys, e.g. {unsupported[:3]}"
            )
        if len(ranks) != 1:
            raise ValueError(f"Turbo LoRA must use one rank, found {sorted(ranks)}")
        self.rank = ranks.pop()
        resolved_alpha = float(alpha) if alpha is not None else (_metadata_alpha(self.metadata) or DEFAULT_TURBO_LORA_ALPHA)
        if not math.isfinite(resolved_alpha) or resolved_alpha <= 0:
            raise ValueError(f"Turbo LoRA alpha must be finite and positive, got {resolved_alpha}")
        self.alpha = resolved_alpha
        self.scale = scale_value
        self.multiplier = self.scale * self.alpha / self.rank
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.target_shapes = {
            "attn.to_q": (inner_dim, hidden_size),
            "attn.to_k": (inner_dim, hidden_size),
            "attn.to_v": (inner_dim, hidden_size),
            "attn.to_out.0": (hidden_size, inner_dim),
            "ff.net.0.proj": (2 * ffn_hidden_size, hidden_size),
            "ff.net.2": (hidden_size, ffn_hidden_size),
        }
        self._validate_groups()

    def _validate_groups(self) -> None:
        missing: list[tuple[str, int, str, str]] = []
        for kind, count in (("block", self.num_blocks), ("refiner", self.num_refiner_blocks)):
            for index in range(count):
                for target in TARGETS:
                    for side in ("A", "B"):
                        identity = (kind, index, target, side)
                        if identity not in self.key_map:
                            missing.append(identity)
                    if missing and missing[-1][:3] == (kind, index, target):
                        continue
                    shape_a = self.shapes[(kind, index, target, "A")]
                    shape_b = self.shapes[(kind, index, target, "B")]
                    output_dims, input_dims = self.target_shapes[target]
                    expected_a = (self.rank, input_dims)
                    expected_b = (output_dims, self.rank)
                    if shape_a != expected_a or shape_b != expected_b:
                        raise ValueError(
                            f"Turbo LoRA shape mismatch for {kind} {index} {target}: "
                            f"A{shape_a}, B{shape_b}; expected A{expected_a}, B{expected_b}"
                        )
        if missing:
            raise KeyError(f"Turbo LoRA is missing {len(missing)} tensors, e.g. {missing[:3]}")

        expected_groups = self.num_blocks + self.num_refiner_blocks
        expected_tensors = expected_groups * len(TARGETS) * 2
        if len(self.key_map) != expected_tensors:
            raise ValueError(
                f"Turbo LoRA has {len(self.key_map)} supported tensors, expected {expected_tensors}"
            )

    def load(self, kind: str, index: int) -> LoRAWeights:
        arrays = mx.load(str(self.path))
        tensors = {}
        for target in TARGETS:
            lora_a = arrays[self.key_map[(kind, index, target, "A")]]
            lora_b = arrays[self.key_map[(kind, index, target, "B")]]
            mx.eval(lora_a, lora_b)
            tensors[target] = (lora_a, lora_b)
        return LoRAWeights(tensors, self.multiplier)

    def load_block(self, index: int) -> LoRAWeights:
        return self.load("block", index)

    def load_refiners(self) -> list[LoRAWeights]:
        return [self.load("refiner", index) for index in range(self.num_refiner_blocks)]


__all__ = ["DEFAULT_TURBO_LORA_ALPHA", "LoRAWeights", "TARGETS", "TurboLoRAProvider"]
