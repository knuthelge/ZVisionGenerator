"""Read a single-file LTX-2.3 or LTX-2.5 checkpoint for conversion, one tensor at a time.

Accepts Lightricks' full checkpoint layout (``model.diffusion_model.*`` plus the bundled VAEs and vocoder) and
transformer-only exports (``diffusion_model.*`` or unprefixed native names); transformer names are normalized to
``model.diffusion_model.*``. Quantized weights (scaled FP8, INT8 and ComfyUI's ConvRot formats, see
:mod:`zvisiongenerator.utils.quantized_weights`) are decoded as they stream, and Lightricks' distilled LoRA can be
merged in on the way. The version is read from the header: LTX-2.5 drops the video feed-forward's bias and keeps the
audio one.

Two checks need no network: whether the checkpoint is plain dev or carries Lightricks' step distillation, and whether
its text connectors equal stock's. Both compare SHA-256 hashes of tensors (as bfloat16) with the ones pinned for the
checkpoint's version (``ltx23_hashes.json``, ``ltx25_hashes.json``), taken from Lightricks' own checkpoints.
"""

from __future__ import annotations

from collections import Counter

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from zvisiongenerator.utils.ltx_converted import LTX_MODEL_TYPES
from zvisiongenerator.utils.quantized_weights import AUX_SUFFIXES, decoded_shape, decoded_tensors, quantization_formats
from zvisiongenerator.utils.safetensors_stream import TensorFile, iter_tensors, read_header, read_metadata

if TYPE_CHECKING:
    import torch

__all__ = [
    "CHECKPOINT_PREFIX",
    "CONNECTOR_PREFIXES",
    "LtxCheckpoint",
    "check_lora",
    "checkpoint_config",
    "connector_tensors",
    "connectors_status",
    "distillation",
    "file_order",
    "inspect_checkpoint",
    "require_version",
    "transformer_tensors",
]

CHECKPOINT_PREFIX = "model.diffusion_model."
_EXPORT_PREFIX = "diffusion_model."
_PROJECTION_PREFIX = "text_embedding_projection."
# The text connectors inside the transformer's prefix; diffusers' single-file conversion drops them from the transformer.
CONNECTOR_PREFIXES = (f"{CHECKPOINT_PREFIX}video_embeddings_connector.", f"{CHECKPOINT_PREFIX}audio_embeddings_connector.")
_CONNECTOR_PREFIXES = (*CONNECTOR_PREFIXES, _PROJECTION_PREFIX)
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
        "keyframes_abs_pos_embedding",
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
# What tells the versions apart, checked from the header.
_BLOCK_COUNT = 48
_SHAPES = {f"{CHECKPOINT_PREFIX}patchify_proj.weight": [4096, 128], f"{CHECKPOINT_PREFIX}audio_patchify_proj.weight": [2048, 128]}
_GATE_LOGITS = f"{CHECKPOINT_PREFIX}transformer_blocks.0.attn1.to_gate_logits.weight"
_PROMPT_ADALN = f"{CHECKPOINT_PREFIX}prompt_adaln_single."
_FF_BIAS = f"{CHECKPOINT_PREFIX}transformer_blocks.0.ff.net.0.proj.bias"
_AUDIO_FF_BIAS = f"{CHECKPOINT_PREFIX}transformer_blocks.0.audio_ff.net.0.proj.bias"
_BLOCK_RE = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}transformer_blocks\.(\d+)\.")
# torchao's Float8Tensor (a quantized repack, not a fine-tune) stores its weights under these suffixes.
_TORCHAO_SUFFIXES = ("._weight_qdata", "._weight_scale")
_LORA_A, _LORA_B, _LORA_ALPHA = ".lora_A.weight", ".lora_B.weight", ".alpha"
_HASHES_FILES = {"2.3": "ltx23_hashes.json", "2.5": "ltx25_hashes.json"}

Distillation = Literal["dev", "distilled", "unknown"]
ConnectorsStatus = Literal["absent", "stock", "changed", "unknown"]


# Dtypes that hold stock's bfloat16 connectors exactly; anything narrower rounds them.
_EXACT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})


@dataclass(frozen=True)
class LtxCheckpoint:
    """A checked LTX checkpoint: its version, where its tensors are and what they are called once normalized.

    Attributes:
        path: The checkpoint file.
        version: ``2.3`` or ``2.5``.
        transformer: Normalized transformer name → name in the file (without the connectors).
        connectors: Normalized connector and ``text_embedding_projection`` name → name in the file.
        shapes: Normalized name → shape, for every name above.
        quantized: Normalized weight name → its quantization format, for every quantized weight above.
        bundles_components: Whether the file also holds the VAEs or vocoder (Lightricks' full layout).
    """

    path: Path
    version: str
    transformer: dict[str, str]
    connectors: dict[str, str]
    shapes: dict[str, list[int]]
    quantized: dict[str, str]
    bundles_components: bool

    @property
    def precision(self) -> dict[str, int]:
        """Return how many transformer weights use each quantization format (empty when none is quantized)."""
        return dict(Counter(found for name, found in self.quantized.items() if name in self.transformer))


def inspect_checkpoint(path: Path) -> LtxCheckpoint:
    """Read *path*'s header and check that it holds an LTX-2.3 or LTX-2.5 transformer in a layout the converter reads.

    Raises:
        ValueError: If the file is not safetensors, is a LoRA, is in diffusers' layout, is neither LTX-2.3 nor LTX-2.5,
            or holds a quantization the converter can't decode.
    """
    _data_start, header = read_header(path)
    if any(name.endswith(_LORA_A) or ".lora_down." in name for name in header):
        raise ValueError(f"{path.name} is a LoRA, not a checkpoint; import it on the LoRAs tab instead.")
    if any(name.endswith(_TORCHAO_SUFFIXES) for name in header):
        raise ValueError(f"{path.name} is a torchao-quantized repack (Float8Tensor), not a fine-tune; it can't be converted.")
    names = {normalized: name for name in header if (normalized := _normalized_name(name)) is not None}
    if not names:
        raise ValueError(f"{path.name} holds no LTX transformer weights.")
    _check_layout(path, names.keys())
    tensors = {normalized: name for normalized, name in names.items() if not normalized.endswith(AUX_SUFFIXES)}
    transformer = {normalized: name for normalized, name in tensors.items() if not normalized.startswith(_CONNECTOR_PREFIXES)}
    connectors = {normalized: name for normalized, name in tensors.items() if normalized.startswith(_CONNECTOR_PREFIXES)}
    version = _version(path, transformer.keys(), {normalized: header[name]["shape"] for normalized, name in transformer.items()})
    formats = quantization_formats([path])
    return LtxCheckpoint(
        path=path,
        version=version,
        transformer=transformer,
        connectors=connectors,
        shapes={normalized: decoded_shape(header, name, formats.get(name)) for normalized, name in tensors.items()},
        quantized={normalized: str(formats[name]["format"]) for normalized, name in tensors.items() if name in formats},
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


def _version(path: Path, names: Any, shapes: dict[str, list[int]]) -> str:
    """Return ``2.3`` or ``2.5`` for the transformer *names*: both have gated attention and prompt modulation (2.0 has neither); 2.5's video feed-forward has no bias while its audio one keeps it.

    Raises:
        ValueError: If it is LTX-2.0, neither version, or not the 22B model.
    """
    names = set(names)
    if _GATE_LOGITS not in names or not any(name.startswith(_PROMPT_ADALN) for name in names):
        raise ValueError(f"{path.name} is not an LTX-2.3 or LTX-2.5 checkpoint (it looks like LTX-2.0).")
    if _FF_BIAS in names:
        version = "2.3"
    elif _AUDIO_FF_BIAS in names:
        version = "2.5"
    else:
        raise ValueError(f"{path.name} is not an LTX-2.3 or LTX-2.5 checkpoint (none of its feed-forward layers has a bias).")
    blocks = {int(match.group(1)) for name in names if (match := _BLOCK_RE.match(name))}
    if len(blocks) != _BLOCK_COUNT or any(shapes.get(name) != shape for name, shape in _SHAPES.items()):
        raise ValueError(f"{path.name} is not an LTX-{version} 22B checkpoint ({len(blocks)} blocks, where it has {_BLOCK_COUNT}).")
    return version


def require_version(checkpoint: LtxCheckpoint, model_type: str) -> None:
    """Refuse *checkpoint* when it is not the version of *model_type* (``ltx-2.3`` or ``ltx-2.5``), naming the type to pick.

    Raises:
        ValueError: When the versions differ.
    """
    wanted = LTX_MODEL_TYPES[model_type]
    if checkpoint.version != wanted:
        raise ValueError(f"{checkpoint.path.name} is an LTX-{checkpoint.version} checkpoint, not LTX-{wanted}; convert it as ltx-{checkpoint.version}.")


def distillation(checkpoint: LtxCheckpoint) -> Distillation:
    """Return whether *checkpoint* is plain dev, carries one of Lightricks' step distillations, or can't be told.

    Reads the small tensors Lightricks' distilled LoRA changes (a few MB) and compares their hashes with the pinned
    ones of the checkpoint's version. Quantized tensors never match (quantizing is lossy), so such files come out
    ``unknown``, and so does a version with no pins.
    """
    pinned = _pinned(checkpoint.version).get("distillation")
    if not pinned:
        return "unknown"
    names = next(iter(pinned.values())).keys()
    if any(name not in checkpoint.transformer or name in checkpoint.quantized for name in names):
        return "unknown"
    found = _hashes(checkpoint, {checkpoint.transformer[name]: name for name in names})
    matches = [label for label, hashes in pinned.items() if hashes == found]
    if matches == ["dev"]:
        return "dev"
    return "distilled" if matches else "unknown"


def connectors_status(checkpoint: LtxCheckpoint) -> ConnectorsStatus:
    """Return whether *checkpoint*'s text connectors are absent, equal to stock's, changed, or can't be told.

    The video connector, the audio connector and ``text_embedding_projection`` are checked as three groups: each
    may be missing (transformer-only exports often lack the projection), but one that is there must have all of
    stock's tensors. Equal means every tensor present has the pinned hash as bfloat16, whatever dtype it is
    stored in. Quantized connectors, or ones stored in a narrower dtype such as FP8, can't match (rounding
    changes them), so with stock's names they come out ``unknown``: most fine-tunes don't train the connectors, and the caller decides. This reads the connector
    tensors (about 4 to 6 GB in bfloat16).
    """
    if not checkpoint.connectors:
        return "absent"
    pinned = _pinned(checkpoint.version).get("connectors")
    if not pinned:
        return "unknown"
    present = checkpoint.connectors.keys()
    if not present <= pinned.keys():
        return "changed"
    for prefix in _CONNECTOR_PREFIXES:
        group = {name for name in pinned if name.startswith(prefix)}
        if group & present and not group <= present:
            return "changed"
    header = read_header(checkpoint.path)[1]
    if any(name in checkpoint.quantized or header[checkpoint.connectors[name]]["dtype"] not in _EXACT_DTYPES for name in present):
        return "unknown"
    found = _hashes(checkpoint, {name: normalized for normalized, name in checkpoint.connectors.items()})
    return "stock" if all(found[name] == pinned[name] for name in present) else "changed"


def _hashes(checkpoint: LtxCheckpoint, wanted: dict[str, str]) -> dict[str, str]:
    """Return the SHA-256 of each tensor in *wanted* (file name → normalized name) as bfloat16 bytes, by normalized name."""
    import torch

    return {wanted[name]: hashlib.sha256(tensor.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name, tensor in iter_tensors(checkpoint.path, wanted.__contains__)}


@cache
def _pinned(version: str) -> dict[str, Any]:
    """Return the pinned hashes of LTX-*version*, empty when it has none."""
    resource = resources.files(__package__).joinpath(_HASHES_FILES[version])
    return json.loads(resource.read_text(encoding="utf-8")) if resource.is_file() else {}


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

    Quantized weights are decoded to bfloat16, unscaled FP8 tensors are cast to it, and with *lora* each weight it
    changes gets ``lora_B @ lora_A`` (times ``alpha / rank`` when the LoRA sets an alpha) added before it is yielded.

    Args:
        checkpoint: The checked checkpoint.
        lora: A LoRA file to merge in at strength 1.0, read one factor pair at a time.
        merge_device: Where quantized weights are decoded and the merge's products computed (a GPU is much faster); the CPU by default.
        progress: Called with ``(tensors done, total)`` after each tensor.
    """
    lora_file = TensorFile(lora) if lora is not None else None
    pairs = _lora_pairs(lora_file.header) if lora_file is not None else {}
    try:
        for done, (normalized, tensor) in enumerate(_tensors(checkpoint, checkpoint.transformer, merge_device), start=1):
            if normalized in pairs and lora_file is not None:
                tensor = _merged(tensor, lora_file, *pairs[normalized], merge_device)
            yield normalized, tensor
            if progress is not None:
                progress(done, len(checkpoint.transformer))
    finally:
        if lora_file is not None:
            lora_file.close()


def connector_tensors(checkpoint: LtxCheckpoint) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the text connectors' and ``text_embedding_projection``'s tensors under their normalized names, in file order, quantized weights decoded."""
    yield from _tensors(checkpoint, checkpoint.connectors)


def file_order(checkpoint: LtxCheckpoint, names: Iterable[str]) -> list[str]:
    """Return the normalized *names* in the order the checkpoint stores their tensors, which is the order the tensor iterators yield them in."""
    header = read_header(checkpoint.path)[1]
    files = {**checkpoint.transformer, **checkpoint.connectors}
    return sorted(names, key=lambda name: header[files[name]]["data_offsets"][0])


def checkpoint_config(checkpoint: LtxCheckpoint) -> dict[str, Any] | None:
    """Return the model config Lightricks' layout carries in the file's metadata (``config``), or ``None`` when there is none with a transformer section."""
    raw = read_metadata(checkpoint.path).get("config")
    try:
        config = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return config if isinstance(config, dict) and isinstance(config.get("transformer"), dict) else None


def _tensors(checkpoint: LtxCheckpoint, wanted: dict[str, str], device: Any = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the tensors of *wanted* (normalized name → name in the file) under their normalized names, in file order, quantized weights decoded (on *device*)."""
    files = {name: normalized for normalized, name in wanted.items()}
    for name, tensor in decoded_tensors([checkpoint.path], files.__contains__, device=device):
        yield files[name], tensor


def _merged(weight: torch.Tensor, lora: TensorFile, down_name: str, up_name: str, device: Any) -> torch.Tensor:
    """Return *weight* with the LoRA's ``up @ down`` added, in *weight*'s dtype."""
    import torch

    down = lora.read(down_name).to(device=device, dtype=torch.float32)
    up = lora.read(up_name).to(device=device, dtype=torch.float32)
    alpha_name = down_name.removesuffix(_LORA_A) + _LORA_ALPHA
    scale = float(lora.read(alpha_name).item()) / down.shape[0] if alpha_name in lora.header else 1.0
    merged = weight.to(device=device, dtype=torch.float32).addmm_(up, down, alpha=scale)
    return merged.to(weight.dtype).cpu()
