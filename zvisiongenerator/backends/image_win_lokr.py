"""LyCORIS LoKr adapters for the diffusers image backend, applied the way mflux applies them on macOS.

diffusers loads only standard LoRA (``lora_A``/``lora_B``). A LoKr adapter adds ``scale * kron(w1, w2)`` to a
linear layer's weight. It is applied as an extra term on the layer's output, computed without building the
Kronecker product (``w1 @ X @ w2ᵀ``), so it works the same on bfloat16, FP8-stored (q8) and NF4 (q4) layers and
never changes their weights. The scale follows mflux and LyCORIS: ``alpha / rank`` when a factor is stored
decomposed, else 1.

Original layer names are mapped onto the diffusers model with diffusers' own LoRA key converters, run on stand-in
tensors whose rows carry their own index. That also covers fused layers the converters split (FLUX.2's ``qkv``
becomes ``to_q``/``to_k``/``to_v``): each target computes the small factorized product and keeps its own rows.

Only files with LoKr tensors are split; the rest of such a file goes to diffusers. Like mflux, layers that do
not map onto the model are skipped with a warning.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["LoraParts", "apply_lokr", "has_lokr_tensors", "split_lora_tensors"]

_LOKR_TENSORS = frozenset({"lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2"})
_LORA_TENSORS = frozenset({"lora_A.weight", "lora_B.weight", "lora_down.weight", "lora_up.weight", "lora_A.bias", "lora_B.bias"})
# Tensors that belong to whichever adapter shares their layer.
_SHARED_TENSORS = frozenset({"alpha", "dora_scale"})
_COMPONENTS = ("transformer", "text_encoder")
# Layer-name prefixes of kohya-format files (e.g. lora_unet_double_blocks_0_img_attn_qkv).
_KOHYA_PREFIXES = ("lora_unet_", "lora_te")


@dataclass(frozen=True)
class LoraParts:
    """A LoRA file's tensors, split by how they are applied."""

    rest: dict[str, torch.Tensor] = field(default_factory=dict)  # every non-LoKr tensor, for diffusers
    lora: dict[str, torch.Tensor] = field(default_factory=dict)  # the standard LoRA tensors among *rest*
    lokr: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)  # LoKr tensors by original layer name

    @property
    def unsupported(self) -> tuple[str, ...]:
        """Return the keys in *rest* that are not standard LoRA tensors (e.g. LyCORIS ``diff``)."""
        return tuple(sorted(self.rest.keys() - self.lora.keys()))


def has_lokr_tensors(keys: Iterable[str]) -> bool:
    """Return whether any of a LoRA file's tensor *keys* belongs to a LoKr adapter."""
    return any(_split_key(key)[1] in _LOKR_TENSORS for key in keys)


def split_lora_tensors(state: dict[str, torch.Tensor]) -> LoraParts:
    """Split a LoRA file's tensors into LoKr layers and the rest, noting which of the rest are standard LoRA."""
    layers: dict[str, dict[str, str]] = {}
    for key in state:
        layer, tensor = _split_key(key)
        layers.setdefault(layer, {})[tensor] = key
    rest: dict[str, torch.Tensor] = {}
    lora: dict[str, torch.Tensor] = {}
    lokr: dict[str, dict[str, torch.Tensor]] = {}
    for layer, tensors in layers.items():
        is_lokr = bool(_LOKR_TENSORS & tensors.keys())
        is_lora = bool(_LORA_TENSORS & tensors.keys())
        for tensor, key in tensors.items():
            if is_lokr and (tensor in _LOKR_TENSORS or tensor in _SHARED_TENSORS):
                lokr.setdefault(layer, {})[tensor] = state[key]
                continue
            rest[key] = state[key]
            if is_lora and (tensor in _LORA_TENSORS or tensor in _SHARED_TENSORS):
                lora[key] = state[key]
    return LoraParts(rest, lora, lokr)


def apply_lokr(pipeline: Any, layers: dict[str, dict[str, torch.Tensor]], strength: float, *, pin_memory: bool) -> tuple[str, ...]:
    """Add the LoKr adapters in *layers* (by original layer name) to *pipeline*'s layers at *strength*.

    The adapter tensors stay in system memory as bfloat16, pinned when *pin_memory* is set (for fast copies to
    a CUDA GPU), and are copied to the GPU as each layer runs. Like mflux, layers that do not map onto the model
    (layers diffusers cannot convert for this model, such as text-encoder layers of some families) are skipped;
    every layer is checked before any is applied.

    Returns:
        The original names of the skipped layers (all of them when the LoRA does not match the model at all).

    Raises:
        ValueError: When a layer's factors do not fit the layer they map to (a LoRA made for a different model).
    """
    factors = {layer: _lokr_factors(tensors) for layer, tensors in layers.items()}
    shapes = {layer: (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]) for layer, (w1, w2, _scale) in factors.items()}
    resolved: dict[str, list[tuple[str, range, torch.nn.Module]]] = {}
    for layer, targets in _map_layers(type(pipeline), shapes).items():
        try:
            resolved[layer] = [(target, rows, _resolve(pipeline, target)) for target, rows in targets]
        except AttributeError:
            continue  # the converter passed through a name that is no layer of this model
    for layer, targets in resolved.items():
        for target, rows, linear in targets:
            _check_fits(layer, target, linear, len(rows), shapes[layer][1])
    for layer, targets in resolved.items():
        w1, w2, scale = factors[layer]
        # Targets split from one fused layer share its factors; each keeps its own rows of the product.
        host_w1, host_w2 = _host(w1, pin_memory), _host(w2, pin_memory)
        for target, rows, linear in targets:
            kept = None if rows == range(shapes[layer][0]) else rows
            _add_term(pipeline, target, linear, _LoKrTerm(host_w1, host_w2, scale * strength, kept))
    return tuple(sorted(shapes.keys() - resolved.keys()))


def _check_fits(layer: str, target: str, linear: torch.nn.Module, out_features: int, in_features: int) -> None:
    """Raise ``ValueError`` when an ``out_features × in_features`` delta does not fit the layer *linear*."""
    if not hasattr(linear, "out_features") or not hasattr(linear, "in_features"):
        raise ValueError(f"LoKr layer {layer} maps to {target}, which is not a linear layer; the LoRA was probably made for a different model.")
    expected = (linear.out_features, linear.in_features)
    if expected != (out_features, in_features):
        raise ValueError(f"LoKr layer {layer} ({out_features}×{in_features}) does not fit {target} ({expected[0]}×{expected[1]}); the LoRA was probably made for a different model.")


def _host(tensor: torch.Tensor, pin_memory: bool) -> torch.Tensor:
    """Return *tensor* as contiguous bfloat16 in system memory, pinned when *pin_memory* is set."""
    host = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return host.pin_memory() if pin_memory else host


class _LoKrTerm(torch.nn.Module):
    """``scale * kron(w1, w2) @ x``, computed as ``w1 @ X @ w2ᵀ`` without building the product.

    For a target split from a fused layer, only its *rows* of the product are kept. The factors are plain
    attributes in system memory, copied to the input's device when the layer runs: model offloading neither
    tracks nor moves them, and they never hold GPU memory between layers.
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, scale: float, rows: range | None = None) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.scale = scale
        self.rows = rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.to(x.device, non_blocking=True)
        w2 = self.w2.to(x.device, non_blocking=True)
        groups = x.reshape(*x.shape[:-1], w1.shape[1], w2.shape[1]).to(w1.dtype)
        out = torch.matmul(torch.matmul(w1, groups), w2.transpose(0, 1)).reshape(*x.shape[:-1], -1)
        if self.rows is not None:
            out = out[..., self.rows.start : self.rows.stop]
        return (out * self.scale).to(x.dtype)


class _AdaptedLinear(torch.nn.Module):
    """A linear layer plus adapter terms added to its output; the layer itself is left unchanged."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        self.terms = torch.nn.ModuleList()

    @property
    def weight(self) -> torch.Tensor:
        """Return the base layer's weight, for code that inspects it."""
        return self.base.weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.base(x, *args, **kwargs)
        for term in self.terms:
            out = out + term(x)
        return out


def _split_key(key: str) -> tuple[str, str]:
    """Split a tensor key into its layer name and tensor name, e.g. ``a.b.lora_A.weight`` -> ``(a.b, lora_A.weight)``."""
    parts = key.rsplit(".", 2)
    if len(parts) == 3 and f"{parts[1]}.{parts[2]}" in _LORA_TENSORS:
        return parts[0], f"{parts[1]}.{parts[2]}"
    layer, _, tensor = key.rpartition(".")
    return layer, tensor


def _lokr_factors(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Rebuild a layer's LoKr factors and scale, as mflux's ``rebuild_lokr_factors`` does.

    Raises:
        ValueError: For factors that are missing, or LoKr variants without a factorized form (Tucker, DoRA).
    """
    if "lokr_t2" in tensors or "dora_scale" in tensors:
        raise ValueError("LoKr layers with a Tucker core or DoRA scale are not supported on this platform.")
    rank: int | None = None
    w1, w2 = tensors.get("lokr_w1"), tensors.get("lokr_w2")
    if w1 is None:
        if "lokr_w1_a" not in tensors or "lokr_w1_b" not in tensors:
            raise ValueError("A LoKr layer needs lokr_w1 or both lokr_w1_a and lokr_w1_b.")
        rank = tensors["lokr_w1_b"].shape[0]
        w1 = tensors["lokr_w1_a"].float() @ tensors["lokr_w1_b"].float()
    if w2 is None:
        if "lokr_w2_a" not in tensors or "lokr_w2_b" not in tensors:
            raise ValueError("A LoKr layer needs lokr_w2 or both lokr_w2_a and lokr_w2_b.")
        rank = tensors["lokr_w2_b"].shape[0]
        w2 = tensors["lokr_w2_a"].float() @ tensors["lokr_w2_b"].float()
    scale = float(tensors["alpha"].item()) / rank if "alpha" in tensors and rank is not None else 1.0
    return w1, w2, scale


def _map_layers(pipeline_class: type, shapes: dict[str, tuple[int, int]]) -> dict[str, list[tuple[str, range]]]:
    """Map original layer names to ``(component.layer, output rows)`` targets; layers that do not map are left out.

    Converters reject a whole file when one layer name is unknown to them (for example a text-encoder layer),
    so a rejected set is mapped again one layer at a time.
    """
    try:
        return _convert_stand_ins(pipeline_class, shapes)
    except ValueError, KeyError:
        targets: dict[str, list[tuple[str, range]]] = {}
        for layer, shape in shapes.items():
            try:
                targets.update(_convert_stand_ins(pipeline_class, {layer: shape}))
            except ValueError, KeyError:
                continue
        return targets


def _convert_stand_ins(pipeline_class: type, shapes: dict[str, tuple[int, int]]) -> dict[str, list[tuple[str, range]]]:
    """Run diffusers' LoRA converter for *pipeline_class* on stand-in LoRAs and read back each layer's targets.

    Each layer gets a stand-in LoRA whose ``lora_B`` holds a unique row number per output row, so whatever the
    converter renames or splits, the rows that arrive at each target name the source layer and its slice.

    Raises:
        ValueError: When the converter rejects the layers or a target receives rows it cannot map back.
    """
    starts: list[int] = []
    layers: list[str] = []
    stand_in: dict[str, torch.Tensor] = {}
    offset = 0
    for layer, (out_features, in_features) in shapes.items():
        starts.append(offset)
        layers.append(layer)
        # diffusers runs its kohya converters only on lora_down/lora_up keys, so kohya-named layers get those.
        down, up = ("lora_down", "lora_up") if layer.startswith(_KOHYA_PREFIXES) else ("lora_A", "lora_B")
        stand_in[f"{layer}.{down}.weight"] = torch.zeros(1, in_features)
        stand_in[f"{layer}.{up}.weight"] = torch.arange(offset, offset + out_features, dtype=torch.float64).unsqueeze(1)
        offset += out_features
    converted = pipeline_class.lora_state_dict(stand_in)
    if isinstance(converted, tuple):
        converted = converted[0]
    targets: dict[str, list[tuple[str, range]]] = {}
    for key, value in converted.items():
        if not key.endswith("lora_B.weight"):
            continue
        rows = value[:, 0].round().long().tolist()
        index = bisect.bisect_right(starts, rows[0]) - 1
        start = rows[0] - starts[index]
        if rows != list(range(rows[0], rows[0] + len(rows))) or start + len(rows) > shapes[layers[index]][0]:
            raise ValueError(f"Could not map LoKr layer {layers[index]} onto the model.")
        target = key.removesuffix(".lora_B.weight")
        targets.setdefault(layers[index], []).append((target, range(start, start + len(rows))))
    return targets


def _resolve(pipeline: Any, target: str) -> torch.nn.Module:
    """Return the layer *target* names (``transformer.``-prefixed or bare) on *pipeline*."""
    component, layer = _component_and_layer(target)
    module = getattr(pipeline, component).get_submodule(layer)
    return module.base if isinstance(module, _AdaptedLinear) else module


def _add_term(pipeline: Any, target: str, linear: torch.nn.Module, term: torch.nn.Module) -> None:
    """Add *term* to the layer at *target*, wrapping it in an :class:`_AdaptedLinear` the first time."""
    component, layer = _component_and_layer(target)
    root = getattr(pipeline, component)
    existing = root.get_submodule(layer)
    if isinstance(existing, _AdaptedLinear):
        existing.terms.append(term)
        return
    wrapper = _AdaptedLinear(linear)
    wrapper.terms.append(term)
    parent_name, _, child_name = layer.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child_name, wrapper)


def _component_and_layer(target: str) -> tuple[str, str]:
    """Split a converted target name into the pipeline component and the layer inside it (transformer by default)."""
    component, _, rest = target.partition(".")
    return (component, rest) if component in _COMPONENTS else ("transformer", target)
