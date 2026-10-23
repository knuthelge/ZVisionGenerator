"""Detect which model family a LoRA file was trained for, from its safetensors header alone.

The checks are pure functions of the parsed header and metadata, so no tensor data is read and nothing imports torch.
Training metadata decides first. Without a usable label, key names and the shapes of the adapted weights fingerprint
the family, and a hidden-size guard rejects a fingerprint whose weights cannot belong to that model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

__all__ = ["LORA_FAMILY_VERSION", "LoraFamily", "detect_lora_family"]

# Bump when the rules change so cached detections are recomputed.
LORA_FAMILY_VERSION = 3

_METADATA_KEYS = ("ss_base_model_version", "release_base_model", "modelspec.architecture")
_KEY_PREFIXES = ("diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model.")
# Suffix of the first weight of each adapter layout; the rest of the key names the adapted module.
_BASE_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lokr_w1", ".lokr_w2")
_LORA_PAIRS = ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"))
# The single-block ``linear1`` output width of each double/single-block family.
_LINEAR1_OUT = {36864: ("flux2_klein", "9b"), 27648: ("flux2_klein", "4b"), 21504: ("flux1", None)}
# The single-block ``proj_mlp`` output width of the FLUX.1 diffusers layout (4 x the hidden size).
_PROJ_MLP_OUT = {12288: ("flux1", None)}
# Kohya names a block ``single_blocks_0_linear1``; the dotted form is ``single_blocks.0.linear1``.
_KOHYA_BLOCK = re.compile(r"^lora_unet_((?:single|double)_blocks)_(\d+)_")
# The model hidden size that must appear among the adapted weights' input or output widths.
_HIDDEN_SIZE = {
    ("zimage", None): 3840,
    ("ideogram4", None): 4608,
    ("krea2", None): 6144,
    ("ltx", None): 4096,
    ("flux2_klein", "9b"): 4096,
    ("flux2_klein", "4b"): 3072,
    ("flux1", None): 3072,
}


@dataclass(frozen=True)
class LoraFamily:
    """The detected model family of a LoRA.

    Attributes:
        family: Family id (``zimage``, ``flux2_klein``, ...), or None when it could not be told.
        size_variant: Size within the family (``9b``, ``4b``), or None when the family has no sizes or it is unknown.
        source: ``"metadata"``, ``"fingerprint"`` or ``"none"``.
    """

    family: str | None
    size_variant: str | None
    source: str


def detect_lora_family(header: Mapping[str, Any], metadata: Mapping[str, str]) -> LoraFamily:
    """Return the family of the LoRA with this safetensors *header* and *metadata*; metadata wins over the fingerprint."""
    labelled = _family_from_metadata(metadata)
    if labelled is not None:
        return LoraFamily(labelled[0], labelled[1], "metadata")
    fingerprint = _family_from_fingerprint(header)
    if fingerprint is not None:
        return LoraFamily(fingerprint[0], fingerprint[1], "fingerprint")
    return LoraFamily(None, None, "none")


def _family_from_metadata(metadata: Mapping[str, str]) -> tuple[str, str | None] | None:
    """Return ``(family, size_variant)`` named by the training metadata, or None when it names no known family."""
    value = " ".join(str(metadata.get(key, "")) for key in _METADATA_KEYS).lower()
    if "ideogram4" in value:
        return ("ideogram4", None)
    if "klein_9b" in value or "klein-9b" in value:
        return ("flux2_klein", "9b")
    if "klein_4b" in value or "klein-4b" in value:
        return ("flux2_klein", "4b")
    if "krea2" in value:
        return ("krea2", None)
    if "zimage" in value or "z-image" in value:
        return ("zimage", None)
    if "ltx2" in value:
        return ("ltx", None)
    if "flux1" in value or "flux.1" in value or "flux-1" in value:
        return ("flux1", None)
    return None


def _strip_prefix(name: str) -> str:
    """Return *name* without one leading trainer prefix, with a Kohya block name written in its dotted form."""
    kohya = _KOHYA_BLOCK.match(name)
    if kohya is not None:
        return f"{kohya.group(1)}.{kohya.group(2)}.{name[kohya.end() :]}"
    for prefix in _KEY_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _shape(header: Mapping[str, Any], key: str) -> list[int] | None:
    """Return the two-dimensional shape of tensor *key*, or None when it is missing or not a matrix."""
    entry = header.get(key)
    shape = entry.get("shape") if isinstance(entry, Mapping) else None
    if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(size, int) for size in shape):
        return shape
    return None


def _module_dims(header: Mapping[str, Any], base: str) -> tuple[int, int] | None:
    """Return ``(in, out)`` features of the module at raw key *base* for LoRA or LoKr, or None when unknown."""
    for down, up in _LORA_PAIRS:
        a, b = _shape(header, base + down), _shape(header, base + up)
        if a is not None and b is not None:
            return a[1], b[0]
    w1, w2 = _shape(header, base + ".lokr_w1"), _shape(header, base + ".lokr_w2")
    if w1 is not None and w2 is not None:
        return w1[1] * w2[1], w1[0] * w2[0]
    return None


def _adapted_dims(header: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Return ``(in, out)`` features per adapted module, keyed by its prefix-stripped name."""
    bases = {key[: -len(suffix)] for key in header for suffix in _BASE_SUFFIXES if key.endswith(suffix)}
    dims: dict[str, tuple[int, int]] = {}
    for base in bases:
        found = _module_dims(header, base)
        if found is not None:
            dims[_strip_prefix(base)] = found
    return dims


def _family_from_fingerprint(header: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """Return the fingerprinted ``(family, size_variant)`` that also passes the hidden-size guard, or None."""
    keys = {_strip_prefix(key) for key in header}
    dims = _adapted_dims(header)
    found = _match_fingerprint(keys, dims)
    if found is None:
        return None
    hidden = _HIDDEN_SIZE[found]
    if not any(hidden in pair for pair in dims.values()):
        return None
    return found


def _by_single_block_width(dims: Mapping[str, tuple[int, int]], prefix: str, suffix: str, families: Mapping[int, tuple[str, str | None]]) -> tuple[str, str | None] | None:
    """Return the family whose output width in *families* is that of a single-block module named ``<prefix>*<suffix>``, or None."""
    outs = {out for base, (_, out) in dims.items() if base.startswith(prefix) and base.endswith(suffix)}
    return next((families[out] for out in families if out in outs), None)


def _by_layer_width(dims: Mapping[str, tuple[int, int]]) -> tuple[str, str | None] | None:
    """Return the family whose hidden size is an input or output width of a ``layers.N`` module, or None.

    Z-Image and Ideogram 4 share the ``layers.N.attention`` layout, so a LoRA without a marker key is told apart by width.
    """
    widths = {width for base, pair in dims.items() if base.startswith("layers.") for width in pair}
    return next((family for family in (("zimage", None), ("ideogram4", None)) if _HIDDEN_SIZE[family] in widths), None)


def _match_fingerprint(keys: set[str], dims: Mapping[str, tuple[int, int]]) -> tuple[str, str | None] | None:
    """Return the first family whose key pattern matches *keys*, before the hidden-size guard."""

    def has(*parts: str) -> bool:
        return any(part in key for key in keys for part in parts)

    def starts(prefix: str, part: str = "") -> bool:
        return any(key.startswith(prefix) and part in key for key in keys)

    if starts("single_blocks.") and starts("double_blocks."):
        return _by_single_block_width(dims, "single_blocks.", "linear1", _LINEAR1_OUT)
    if has(".attn.gate", ".attn.to_gate") and has("txtfusion", "text_fusion") and not has("attn1."):
        return ("krea2", None)
    if starts("single_transformer_blocks.") and starts("transformer_blocks."):
        return _by_single_block_width(dims, "single_transformer_blocks.", "attn.to_qkv_mlp_proj", _LINEAR1_OUT) or _by_single_block_width(dims, "single_transformer_blocks.", "proj_mlp", _PROJ_MLP_OUT)
    if has("transformer_blocks.") and has(".attn1.", ".attn2."):
        return ("ltx", None)
    if has("layers.") and has("noise_refiner", "context_refiner", "cap_embedder"):
        return ("zimage", None)
    if starts("layers.", "adaLN_modulation"):
        return ("zimage", None)
    if starts("layers.", "adaln_modulation"):
        return ("ideogram4", None)
    if starts("layers."):
        return _by_layer_width(dims)
    return None
