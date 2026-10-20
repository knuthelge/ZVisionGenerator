"""Read a single-file LTX-2.3 checkpoint for conversion, one tensor at a time.

Accepts Lightricks' full checkpoint layout (``model.diffusion_model.*`` plus the bundled VAEs and vocoder) and
transformer-only exports (``diffusion_model.*`` or unprefixed native names); transformer names are normalized to
``model.diffusion_model.*``. Scaled-FP8 weights (ComfyUI's ``weight_scale`` / ``comfy_quant``) are dequantized as they
stream, and Lightricks' distilled LoRA can be merged in on the way.

Two checks need no network: whether the checkpoint is plain dev or carries Lightricks' step distillation, and whether
its text connectors equal stock's. Both compare SHA-256 hashes of tensors (as bfloat16) with the ones pinned in
``ltx23_hashes.json``, taken from Lightricks' own checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from zvisiongenerator.converters.convert_checkpoint import FP8_AUX_SUFFIXES, FP8_WEIGHT_SCALE_SUFFIXES
from zvisiongenerator.utils.safetensors_stream import TensorFile, iter_tensors, read_header

if TYPE_CHECKING:
    import torch

__all__ = [
    "LtxCheckpoint",
    "check_lora",
    "connectors_status",
    "distillation",
    "inspect_checkpoint",
    "transformer_tensors",
]

CHECKPOINT_PREFIX = "model.diffusion_model."
_EXPORT_PREFIX = "diffusion_model."
_PROJECTION_PREFIX = "text_embedding_projection."
_CONNECTOR_PREFIXES = (f"{CHECKPOINT_PREFIX}video_embeddings_connector.", f"{CHECKPOINT_PREFIX}audio_embeddings_connector.", _PROJECTION_PREFIX)
# The native transformer's top-level modules: an unprefixed export is recognised by them.
_NATIVE_MODULES = frozenset(
    {
        "adaln_single",
        "audio_adaln_single",
        "audio_embeddings_connector",
        "audio_patchify_proj",
        "audio_proj_out",
        "audio_prompt_adaln_single",
        "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "patchify_proj",
        "proj_out",
        "prompt_adaln_single",
        "scale_shift_table",
        "transformer_blocks",
        "video_embeddings_connector",
    }
)
# Where diffusers' layout differs from the native one at the top level (native: patchify_proj., adaln_single.).
_DIFFUSERS_MODULES = ("proj_in.", "time_embed.")
# What tells LTX-2.3 apart, checked from the header.
_BLOCK_COUNT = 48
_SHAPES = {f"{CHECKPOINT_PREFIX}patchify_proj.weight": [4096, 128], f"{CHECKPOINT_PREFIX}audio_patchify_proj.weight": [2048, 128]}
_GATE_LOGITS = f"{CHECKPOINT_PREFIX}transformer_blocks.0.attn1.to_gate_logits.weight"
_PROMPT_ADALN = f"{CHECKPOINT_PREFIX}prompt_adaln_single."
_FF_BIAS = f"{CHECKPOINT_PREFIX}transformer_blocks.0.ff.net.0.proj.bias"
_BLOCK_RE = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}transformer_blocks\.(\d+)\.")
_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
_FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")
_LORA_A, _LORA_B, _LORA_ALPHA = ".lora_A.weight", ".lora_B.weight", ".alpha"
_HASHES_FILE = "ltx23_hashes.json"

Distillation = Literal["dev", "distilled", "unknown"]
ConnectorsStatus = Literal["absent", "stock", "changed"]


@dataclass(frozen=True)
class LtxCheckpoint:
    """A checked LTX-2.3 checkpoint: where its tensors are and what they are called once normalized.

    Attributes:
        path: The checkpoint file.
        transformer: Normalized transformer name → name in the file (without the connectors).
        connectors: Normalized connector and ``text_embedding_projection`` name → name in the file.
        shapes: Normalized name → shape, for every name above.
        scales: Normalized weight name → name of its scalar FP8 scale in the file.
        fp8_weights: How many transformer weights are stored in FP8.
        bundles_components: Whether the file also holds the VAEs or vocoder (Lightricks' full layout).
    """

    path: Path
    transformer: dict[str, str]
    connectors: dict[str, str]
    shapes: dict[str, list[int]]
    scales: dict[str, str]
    fp8_weights: int
    bundles_components: bool


def inspect_checkpoint(path: Path) -> LtxCheckpoint:
    """Read *path*'s header and check that it holds an LTX-2.3 transformer in a layout the converter reads.

    Raises:
        ValueError: If the file is not safetensors, is in diffusers' layout, is not LTX-2.3, or holds a quantization
            other than per-tensor scaled FP8.
    """
    _data_start, header = read_header(path)
    names = {normalized: name for name in header if (normalized := _normalized_name(name)) is not None}
    if not names:
        raise ValueError(f"{path.name} holds no LTX transformer weights.")
    _check_layout(path, names.keys())
    tensors = {normalized: name for normalized, name in names.items() if not normalized.endswith(FP8_WEIGHT_SCALE_SUFFIXES + FP8_AUX_SUFFIXES)}
    transformer = {normalized: name for normalized, name in tensors.items() if not normalized.startswith(_CONNECTOR_PREFIXES)}
    connectors = {normalized: name for normalized, name in tensors.items() if normalized.startswith(_CONNECTOR_PREFIXES)}
    _check_version(path, transformer.keys(), {normalized: header[name]["shape"] for normalized, name in transformer.items()})
    scales = _fp8_scales(names)
    _check_fp8_formats(path, [names[normalized] for normalized in names if normalized.endswith(".comfy_quant")])
    for normalized, scale_name in scales.items():
        if header[scale_name]["shape"] not in ([], [1]):
            raise ValueError(f"{normalized} uses block-wise FP8 scales; only per-tensor scaled FP8 is supported.")
    return LtxCheckpoint(
        path=path,
        transformer=transformer,
        connectors=connectors,
        shapes={normalized: header[name]["shape"] for normalized, name in {**transformer, **connectors}.items()},
        scales=scales,
        fp8_weights=sum(header[name]["dtype"] in _FP8_DTYPES for name in transformer.values()),
        bundles_components=any(name.startswith(("vae.", "audio_vae.", "vocoder.")) for name in header),
    )


def _normalized_name(name: str) -> str | None:
    """Return *name* as a ``model.diffusion_model.*`` (or ``text_embedding_projection.*``) name, or ``None`` when it is no transformer tensor."""
    if name.startswith((CHECKPOINT_PREFIX, _PROJECTION_PREFIX)):
        return name
    if name.startswith(_EXPORT_PREFIX):
        return f"model.{name}"
    head = name.split(".", 1)[0]
    if head in _NATIVE_MODULES or f"{head}." in _DIFFUSERS_MODULES:
        return CHECKPOINT_PREFIX + name
    return None


def _check_layout(path: Path, names: Any) -> None:
    """Refuse a checkpoint in diffusers' transformer layout, which this converter does not map."""
    stripped = [name.removeprefix(CHECKPOINT_PREFIX) for name in names]
    if any(name.startswith(_DIFFUSERS_MODULES) for name in stripped) and not any(name.startswith("patchify_proj.") for name in stripped):
        raise ValueError(f"{path.name} is in diffusers' layout; convert a single-file checkpoint in Lightricks' or ComfyUI's layout instead.")


def _check_version(path: Path, names: Any, shapes: dict[str, list[int]]) -> None:
    """Refuse a checkpoint that is not LTX-2.3: 2.0 lacks the gated attention and prompt modulation, 2.5 the feed-forward bias."""
    names = set(names)
    if _GATE_LOGITS not in names or not any(name.startswith(_PROMPT_ADALN) for name in names):
        raise ValueError(f"{path.name} is not an LTX-2.3 checkpoint (it looks like LTX-2.0).")
    if _FF_BIAS not in names:
        raise ValueError(f"{path.name} is not an LTX-2.3 checkpoint (its feed-forward layers have no bias, as in LTX-2.5).")
    blocks = {int(match.group(1)) for name in names if (match := _BLOCK_RE.match(name))}
    if len(blocks) != _BLOCK_COUNT or any(shapes.get(name) != shape for name, shape in _SHAPES.items()):
        raise ValueError(f"{path.name} is not an LTX-2.3 22B checkpoint ({len(blocks)} blocks, where LTX-2.3 has {_BLOCK_COUNT}).")


def _fp8_scales(names: dict[str, str]) -> dict[str, str]:
    """Return each scaled-FP8 weight's normalized name → the name of its scale in the file."""
    return {normalized.rsplit(".", 1)[0] + ".weight": name for normalized, name in names.items() if normalized.endswith(FP8_WEIGHT_SCALE_SUFFIXES)}


def _check_fp8_formats(path: Path, quant_names: list[str]) -> None:
    """Refuse ``comfy_quant`` entries that describe anything but per-tensor FP8."""
    if not quant_names:
        return
    wanted = set(quant_names)
    for name, tensor in iter_tensors(path, wanted.__contains__):
        quant_format = json.loads(bytes(tensor.tolist()).decode()).get("format")
        if quant_format not in _FP8_FORMATS:
            raise ValueError(f"{name} uses the quantization format {quant_format!r}; only scaled FP8 checkpoints are supported.")


def distillation(checkpoint: LtxCheckpoint) -> Distillation:
    """Return whether *checkpoint* is plain dev, carries one of Lightricks' step distillations, or can't be told.

    Reads the four small tensors Lightricks' distilled LoRA changes (a few MB) and compares their hashes with the
    pinned ones. Tensors stored in FP8 never match (FP8 is lossy), so such files come out ``unknown``.
    """
    pinned = _pinned()["distillation"]
    names = next(iter(pinned.values())).keys()
    if any(name not in checkpoint.transformer for name in names):
        return "unknown"
    found = _hashes(checkpoint, {checkpoint.transformer[name]: name for name in names})
    matches = [label for label, hashes in pinned.items() if hashes == found]
    if matches == ["dev"]:
        return "dev"
    return "distilled" if matches else "unknown"


def connectors_status(checkpoint: LtxCheckpoint) -> ConnectorsStatus:
    """Return whether *checkpoint*'s text connectors are absent, equal to stock's, or changed.

    The video connector, the audio connector and ``text_embedding_projection`` are checked as three groups: each
    may be missing (transformer-only exports often lack the projection), but one that is there must have all of
    stock's tensors. Equal means every tensor present has the pinned hash as bfloat16, whatever dtype it is
    stored in; scaled-FP8 copies never match, since FP8 rounds. This reads the connector tensors (about 6 GB in
    bfloat16).
    """
    if not checkpoint.connectors:
        return "absent"
    pinned = _pinned()["connectors"]
    present = checkpoint.connectors.keys()
    if not present <= pinned.keys():
        return "changed"
    for prefix in _CONNECTOR_PREFIXES:
        group = {name for name in pinned if name.startswith(prefix)}
        if group & present and not group <= present:
            return "changed"
    found = _hashes(checkpoint, {name: normalized for normalized, name in checkpoint.connectors.items()})
    return "stock" if all(found[name] == pinned[name] for name in present) else "changed"


def _hashes(checkpoint: LtxCheckpoint, wanted: dict[str, str]) -> dict[str, str]:
    """Return the SHA-256 of each tensor in *wanted* (file name → normalized name) as bfloat16 bytes, by normalized name."""
    import torch

    return {
        wanted[name]: hashlib.sha256(_dequantized(tensor, None).to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, tensor in iter_tensors(checkpoint.path, wanted.__contains__)
    }


@cache
def _pinned() -> dict[str, Any]:
    """Return the pinned hashes of ``ltx23_hashes.json``."""
    return json.loads(resources.files(__package__).joinpath(_HASHES_FILE).read_text(encoding="utf-8"))


def check_lora(checkpoint: LtxCheckpoint, lora: Path) -> None:
    """Check that every weight the LoRA at *lora* changes is in *checkpoint*, with a matching shape.

    Raises:
        ValueError: When the LoRA targets a weight the checkpoint lacks, or its factors don't fit it.
    """
    header = read_header(lora)[1]
    for target, (down, up) in _lora_pairs(header).items():
        shape = checkpoint.shapes.get(target)
        if shape is None:
            raise ValueError(f"The LoRA {lora.name} changes {target}, which {checkpoint.path.name} does not have.")
        if [header[up]["shape"][0], header[down]["shape"][1]] != shape:
            raise ValueError(f"The LoRA {lora.name} does not fit {target} of {checkpoint.path.name}.")


def _lora_pairs(header: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Return each target weight's normalized name → the LoRA's ``(lora_A, lora_B)`` tensor names."""
    pairs = {}
    for name in header:
        if name.endswith(_LORA_A):
            base = name.removesuffix(_LORA_A)
            target = _normalized_name(f"{base}.weight")
            if target is not None and f"{base}{_LORA_B}" in header:
                pairs[target] = (name, f"{base}{_LORA_B}")
    return pairs


def transformer_tensors(
    checkpoint: LtxCheckpoint,
    *,
    lora: Path | None = None,
    merge_device: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the transformer's tensors under their normalized names, in file order, one at a time.

    Scaled-FP8 weights are multiplied by their scale (read before the weights, so no weight waits for one), other
    FP8 tensors are cast to bfloat16, and with *lora* each weight it changes gets ``lora_B @ lora_A`` (times
    ``alpha / rank`` when the LoRA sets an alpha) added before it is yielded.

    Args:
        checkpoint: The checked checkpoint.
        lora: A LoRA file to merge in at strength 1.0, read one factor pair at a time.
        merge_device: Where the merge's products are computed (a CUDA device is much faster); the CPU by default.
        progress: Called with ``(tensors done, total)`` after each tensor.
    """
    files = {name: normalized for normalized, name in checkpoint.transformer.items()}
    scales = {checkpoint.scales[normalized]: normalized for normalized in checkpoint.scales if normalized in checkpoint.transformer}
    scale_values = {scales[name]: tensor for name, tensor in iter_tensors(checkpoint.path, scales.__contains__)}
    lora_file = TensorFile(lora) if lora is not None else None
    pairs = _lora_pairs(lora_file.header) if lora_file is not None else {}
    try:
        for done, (name, tensor) in enumerate(iter_tensors(checkpoint.path, files.__contains__), start=1):
            normalized = files[name]
            tensor = _dequantized(tensor, scale_values.get(normalized))
            if normalized in pairs and lora_file is not None:
                tensor = _merged(tensor, lora_file, *pairs[normalized], merge_device)
            yield normalized, tensor
            if progress is not None:
                progress(done, len(files))
    finally:
        if lora_file is not None:
            lora_file.close()


def _dequantized(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Return *tensor* times its FP8 *scale* in bfloat16, or an unscaled FP8 tensor cast to bfloat16; other dtypes as they are."""
    import torch

    if scale is not None:
        return (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.to(torch.bfloat16)
    return tensor


def _merged(weight: torch.Tensor, lora: TensorFile, down_name: str, up_name: str, device: Any) -> torch.Tensor:
    """Return *weight* with the LoRA's ``up @ down`` added, in *weight*'s dtype."""
    import torch

    down = lora.read(down_name).to(device=device, dtype=torch.float32)
    up = lora.read(up_name).to(device=device, dtype=torch.float32)
    alpha_name = down_name.removesuffix(_LORA_A) + _LORA_ALPHA
    scale = float(lora.read(alpha_name).item()) / down.shape[0] if alpha_name in lora.header else 1.0
    merged = weight.to(device=device, dtype=torch.float32).addmm_(up, down, alpha=scale)
    return merged.to(weight.dtype).cpu()
