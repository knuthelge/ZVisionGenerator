"""LTX-2.3 weights for the CUDA video backend: the transformer from a single-file checkpoint and the Gemma text encoder.

The distilled transformer comes from Lightricks' single-file checkpoint (e.g. ``ltx-2.3-22b-distilled-1.1``), renamed
to diffusers' layout with diffusers' own key conversion. Every other component comes from the diffusers repo.

The weights are prepared once per quantize level and kept as a *prepared copy* (sharded safetensors plus a
manifest) under the cache folder: the transformer in NF4 (q4, the linear layers of its blocks) or FP8 (q8), and
the text encoder in FP8 at either level. Both are written one tensor at a time, so a 38 GB transformer never
sits in memory at full precision.

NF4 weights are packed by bitsandbytes on the GPU and saved with their quantization state, in the layout
``Linear4bit`` saves itself in; they load back into ``Linear4bit`` layers on the CPU.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.image_win_quant import cast_to_fp8, empty_diffusers_model, fill_weights, kept_in_compute_dtype, link_or_copy
from zvisiongenerator.backends.safetensors_stream import ShardWriter, iter_tensors, read_header, shard_files
from zvisiongenerator.utils.stored_quant import MANIFEST_NAME, discard_partial, manifest_matches, partial_dir, promote_partial, sweep_stale_partials, write_manifest

if TYPE_CHECKING:
    import torch

__all__ = [
    "PREPARED_BITS",
    "component_elements",
    "load_connectors",
    "ensure_prepared",
    "load_text_encoder",
    "load_transformer",
    "text_encoder_bytes",
    "text_encoder_sources",
    "transformer_elements",
    "transformer_name_map",
    "transformer_sources",
    "write_text_encoder",
    "write_transformer",
]

PREPARED_BITS = (4, 8)
_LAYOUT_VERSION = 1
_INDEX_NAME = "model.safetensors.index.json"
_SHARD_PREFIX = "model"
_CONFIG_NAME = "config.json"
_DIFFUSERS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_TRANSFORMERS_INDEX_NAME = "model.safetensors.index.json"
_CHECKPOINT_PREFIX = "model.diffusion_model."
_BLOCKS_PREFIX = "transformer_blocks."
_NF4_BLOCKSIZE = 64
# What one saved NF4 weight consists of, by name suffix: the packed data and its double-quantized state.
_NF4_PARTS = frozenset({"", "absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state.bitsandbytes__nf4"})
# LTX-2.3's prompt modulation layers, which diffusers 0.41's single-file conversion leaves under their checkpoint names.
_LTX23_RENAMES = (("prompt_adaln_single.", "prompt_adaln."), ("audio_prompt_adaln_single.", "audio_prompt_adaln."))
# The diffusers text-encoder checkpoint uses transformers' older Gemma 3 names.
_LEGACY_TEXT_ENCODER_PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("vision_tower.vision_model.", "model.vision_tower."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
)


def transformer_name_map(names: Iterable[str]) -> dict[str, str]:
    """Map the transformer tensors among a single-file LTX-2 checkpoint's *names* to their diffusers names.

    Other components in the checkpoint (VAEs, vocoder, text connectors) are left out.
    """
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    # The conversion renames keys and drops the connectors; each value carries its original name through.
    originals = {name: name for name in names if name.startswith(_CHECKPOINT_PREFIX)}
    return {original: _rename_ltx23(converted) for converted, original in convert_ltx2_transformer_to_diffusers(originals).items()}


def _rename_ltx23(name: str) -> str:
    """Apply the LTX-2.3 renames diffusers' single-file conversion (written for LTX-2.0) does not know."""
    for old, new in _LTX23_RENAMES:
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def transformer_sources(source: Path) -> list[Path]:
    """Return the files the transformer at *source* is read from: a single-file checkpoint, or a diffusers folder's shards."""
    return [source] if source.is_file() else _folder_files(source, _DIFFUSERS_INDEX_NAME)


def text_encoder_sources(repo_dir: Path) -> list[Path]:
    """Return the weight files of the text encoder of the diffusers repo at *repo_dir*."""
    return _folder_files(repo_dir / "text_encoder", _TRANSFORMERS_INDEX_NAME)


def transformer_elements(source: Path) -> tuple[int, int]:
    """Count the transformer's elements at *source* from the file headers: ``(2-D weights in its blocks, the rest)``."""
    blocks = rest = 0
    for name, shape in _transformer_shapes(source):
        count = _numel(shape)
        if name.startswith(_BLOCKS_PREFIX) and len(shape) == 2:
            blocks += count
        else:
            rest += count
    return blocks, rest


def component_elements(component_dir: Path) -> int:
    """Count the elements of the component in the diffusers folder *component_dir* from its file headers."""
    files = [path for index in (_DIFFUSERS_INDEX_NAME, _TRANSFORMERS_INDEX_NAME) if (component_dir / index).is_file() for path in shard_files(component_dir, index)]
    files = files or sorted(component_dir.glob("*.safetensors"))
    return sum(_numel(info["shape"]) for path in files for info in read_header(path)[1].values())


def text_encoder_bytes(component_dir: Path) -> int:
    """Return the bytes the FP8 text encoder in *component_dir* holds: one per element, two where it stays bfloat16."""
    files = [path for index in (_DIFFUSERS_INDEX_NAME, _TRANSFORMERS_INDEX_NAME) if (component_dir / index).is_file() for path in shard_files(component_dir, index)]
    files = files or sorted(component_dir.glob("*.safetensors"))
    return sum(_numel(info["shape"]) * (2 if kept_in_compute_dtype(name) else 1) for path in files for name, info in read_header(path)[1].items())


def _transformer_shapes(source: Path) -> Iterator[tuple[str, list[int]]]:
    """Yield ``(diffusers name, shape)`` for each transformer tensor at *source*, read from the headers only."""
    if source.is_file():
        header = read_header(source)[1]
        for original, name in transformer_name_map(header).items():
            yield name, header[original]["shape"]
        return
    for path in transformer_sources(source):
        for name, info in read_header(path)[1].items():
            yield name, info["shape"]


def _transformer_tensors(source: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the transformer's tensors at *source* under their diffusers names (a single-file checkpoint is renamed)."""
    if source.is_file():
        names = transformer_name_map(read_header(source)[1])
        for name, tensor in iter_tensors(source, names.__contains__):
            yield names[name], tensor
        return
    for path in transformer_sources(source):
        yield from iter_tensors(path)


def _folder_files(directory: Path, index_name: str) -> list[Path]:
    """Return the shards *directory*'s index lists, or its one safetensors file when it is not sharded."""
    if (directory / index_name).is_file():
        return shard_files(directory, index_name)
    return sorted(directory.glob("*.safetensors"))


def _numel(shape: Iterable[int]) -> int:
    """Return the number of elements of a tensor of *shape*."""
    count = 1
    for size in shape:
        count *= size
    return count


def ensure_prepared(target: Path, sources: Iterable[Path], bits: int, write: Callable[[Path], None], reuse: Iterable[Path] = ()) -> Path:
    """Return *target* once it holds a current prepared copy of *sources* at *bits*, writing it first if needed.

    The copy is written into a hidden partial folder and renamed into place once complete, so an interrupted
    write never leaves a copy that looks finished. A copy made from other source files, or in an older layout, is replaced.

    Args:
        target: Where the prepared copy lives.
        sources: The files it is made from; their identity (resolved name and size) is recorded.
        bits: The quantize level.
        write: Writes the copy's files into the folder it is given.
        reuse: Other places the same copy may already be; a current one is hard-linked (copied across file
            systems) instead of written again, so the files are stored once.
    """
    manifest = _manifest(sources, bits)
    if manifest_matches(target, manifest):
        return target
    existing = next((path for path in reuse if path != target and manifest_matches(path, manifest)), None)
    target.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_partials(target)
    partial = partial_dir(target)
    try:
        if existing is not None:
            # The manifest is written anew, never through a link to the other copy's.
            shutil.copytree(existing, partial, copy_function=link_or_copy, ignore=shutil.ignore_patterns(MANIFEST_NAME))
        else:
            partial.mkdir()
            write(partial)
        write_manifest(partial, manifest)
        promote_partial(partial, target)
    except BaseException:
        discard_partial(partial)
        raise
    return target


def _manifest(sources: Iterable[Path], bits: int) -> dict[str, Any]:
    """Return the manifest of a prepared copy of *sources* at *bits* in the current layout."""
    files = [Path(os.path.realpath(path)) for path in sources]
    return {
        "layout": _LAYOUT_VERSION,
        "bits": bits,
        "format": _storage_format(bits),
        "sources": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
    }


def _storage_format(bits: int) -> str:
    """Return the storage format of *bits*: NF4 copies depend on bitsandbytes' serialization."""
    if bits == 4:
        major_minor = ".".join(importlib.metadata.version("bitsandbytes").split(".")[:2])
        return f"nf4-bitsandbytes-{major_minor}"
    return "fp8-e4m3"


def write_transformer(source: Path, config: dict[str, Any], target: Path, bits: int, device: Any) -> None:
    """Write the transformer in the checkpoint *source* to *target* at *bits*, one tensor at a time.

    Args:
        source: A single-file LTX-2 checkpoint, or a diffusers transformer folder.
        config: The diffusers transformer config (``transformer/config.json`` of the diffusers repo).
        target: The new folder to write.
        bits: 4 (NF4 block linears, the rest in bfloat16) or 8 (FP8 layer weights).
        device: The CUDA device bitsandbytes packs NF4 weights on.

    Raises:
        RuntimeError: When the checkpoint does not cover the transformer.
    """
    import torch

    model = _empty_transformer(config, bits, torch.bfloat16)
    storage = {key: tensor.dtype for key, tensor in model.state_dict().items()}
    nf4_weights = {f"{name}.weight" for name in _block_linear_names(model)} if bits == 4 else set()
    writer = ShardWriter(target, _SHARD_PREFIX, _INDEX_NAME)
    written: set[str] = set()
    for key, tensor in _transformer_tensors(source):
        if key not in storage:
            continue
        writer.add(_pack_nf4(key, tensor, device) if key in nf4_weights else {key: tensor.to(storage[key])})
        written.add(key)
    _require_coverage(storage.keys() - written, str(source))
    writer.close()
    (target / _CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_transformer(config: dict[str, Any], *, prepared: Path, bits: int, compute_dtype: torch.dtype) -> Any:
    """Load the LTX-2 transformer on the CPU from its prepared copy at *bits*.

    Args:
        config: The diffusers transformer config.
        prepared: The prepared copy at *bits*.
        bits: 4 (NF4 block linears) or 8 (FP8 layer weights).
        compute_dtype: The dtype layers compute in.
    """
    model = _empty_transformer(config, bits, compute_dtype)
    linears = _swap_in_linear4bit(model, compute_dtype) if bits == 4 else {}
    tensors = (item for path in shard_files(prepared, _INDEX_NAME) for item in iter_tensors(path))
    fill_weights(model, _place_nf4(tensors, linears), str(prepared), skipped=linears.keys())
    return model.eval()


def _empty_transformer(config: dict[str, Any], bits: int, compute_dtype: torch.dtype) -> Any:
    """Return the transformer without weights, its layer weights set to FP8 storage at q8."""
    model = empty_diffusers_model(config, compute_dtype)
    return cast_to_fp8(model, compute_dtype) if bits == 8 else model


def _block_linear_names(model: Any) -> list[str]:
    """Return the names of the linear layers inside the transformer blocks: the layers NF4 packs."""
    import torch

    return [name for name, module in model.named_modules() if name.startswith(_BLOCKS_PREFIX) and isinstance(module, torch.nn.Linear)]


def _pack_nf4(key: str, weight: torch.Tensor, device: Any) -> dict[str, torch.Tensor]:
    """Pack *weight* to NF4 on *device* and return it with its quantization state, named as ``Linear4bit`` saves them."""
    import torch
    from bitsandbytes.functional import quantize_4bit

    packed, state = quantize_4bit(weight.to(device=device, dtype=torch.bfloat16), blocksize=_NF4_BLOCKSIZE, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8)
    return {key: packed.cpu(), **{f"{key}.{name}": value.cpu() for name, value in state.as_dict(packed=True).items()}}


def _swap_in_linear4bit(model: Any, compute_dtype: torch.dtype) -> dict[str, Any]:
    """Replace the block linears of the empty *model* with empty ``Linear4bit`` layers; return them by weight name."""
    import torch
    from bitsandbytes.nn import Linear4bit

    layers: dict[str, Any] = {}
    for name in _block_linear_names(model):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        linear = getattr(parent, child)
        layer = Linear4bit(
            linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8, device="meta"
        )
        setattr(parent, child, layer)
        layers[f"{name}.weight"] = layer
    return layers


def _place_nf4(tensors: Iterable[tuple[str, torch.Tensor]], layers: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
    """Put each NF4 weight of *tensors* into its ``Linear4bit`` layer in *layers*, and yield every other tensor.

    A weight's parts are saved in one shard but not next to each other, so they are gathered until complete.

    Raises:
        RuntimeError: When a weight's parts are incomplete at the end.
    """
    from bitsandbytes.nn import Params4bit

    parts: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        owner = _nf4_owner(name, layers)
        if owner is None:
            yield name, tensor
            continue
        found = parts.setdefault(owner, {})
        found[name[len(owner) :].lstrip(".")] = tensor
        if found.keys() == _NF4_PARTS:
            data = found.pop("")
            layer = layers[owner]
            layer.weight = Params4bit.from_prequantized(data=data, quantized_stats=found, requires_grad=False, device="cpu", module=layer)
            del parts[owner]
    if parts:
        raise RuntimeError(f"The NF4 weight {sorted(parts)[0]} is incomplete in the prepared copy.")


def _nf4_owner(name: str, layers: dict[str, Any]) -> str | None:
    """Return the NF4 weight *name* belongs to (the weight itself or part of its quantization state), else ``None``."""
    if name in layers:
        return name
    head, separator, _ = name.partition(".weight.")
    owner = f"{head}.weight"
    return owner if separator and owner in layers else None


def load_connectors(component_dir: Path, *, compute_dtype: torch.dtype) -> Any:
    """Load the text connectors from the diffusers folder *component_dir* on the CPU, with FP8 weight storage.

    The model is built empty and filled one tensor at a time, so the full-precision weights are never held beside
    the FP8 ones.
    """
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    config = json.loads((component_dir / _CONFIG_NAME).read_text(encoding="utf-8"))
    model = cast_to_fp8(empty_diffusers_model(config, compute_dtype, LTX2TextConnectors), compute_dtype)
    files = _folder_files(component_dir, _DIFFUSERS_INDEX_NAME)
    fill_weights(model, (item for path in files for item in iter_tensors(path)), str(component_dir))
    return model.eval()


def write_text_encoder(source_dir: Path, target: Path) -> None:
    """Write the FP8 copy of the text encoder of the diffusers repo at *source_dir* to *target*, one tensor at a time.

    Raises:
        RuntimeError: When the checkpoint does not cover the text encoder.
    """
    import torch

    model = cast_to_fp8(_empty_text_encoder(source_dir, torch.bfloat16), torch.bfloat16)
    storage = {key: tensor.dtype for key, tensor in model.state_dict().items()}
    writer = ShardWriter(target, _SHARD_PREFIX, _INDEX_NAME)
    written: set[str] = set()
    for name, tensor in _source_text_encoder_tensors(source_dir, storage.keys()):
        if name in storage:
            writer.add({name: tensor.to(storage[name])})
            written.add(name)
    _require_coverage(storage.keys() - written - _tied_names(model), str(source_dir / "text_encoder"))
    writer.close()


def load_text_encoder(source_dir: Path, *, prepared: Path, compute_dtype: torch.dtype) -> Any:
    """Load the Gemma text encoder on the CPU from its *prepared* FP8 copy; *source_dir* (the repo) gives its config."""
    model = cast_to_fp8(_empty_text_encoder(source_dir, compute_dtype), compute_dtype)
    tensors = (item for path in shard_files(prepared, _INDEX_NAME) for item in iter_tensors(path))
    fill_weights(model, tensors, str(prepared), skipped=_tied_names(model))
    model.tie_weights()
    return model.eval()


def _empty_text_encoder(source_dir: Path, compute_dtype: torch.dtype) -> Any:
    """Return the text encoder the repo at *source_dir* names, without weights, its parameters in *compute_dtype*.

    Buffers keep their own dtype (the rotary frequencies stay float32).
    """
    import transformers
    from accelerate import init_empty_weights

    class_name = json.loads((source_dir / "model_index.json").read_text(encoding="utf-8"))["text_encoder"][1]
    config = transformers.AutoConfig.from_pretrained(str(source_dir / "text_encoder"))
    with init_empty_weights():
        model = getattr(transformers, class_name)(config)
    for param in model.parameters():
        param.data = param.data.to(compute_dtype)
    return model


def _source_text_encoder_tensors(source_dir: Path, model_names: Iterable[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the text encoder's tensors from *source_dir* under the model's names (see :data:`_LEGACY_TEXT_ENCODER_PREFIXES`)."""
    known = set(model_names)
    for path in text_encoder_sources(source_dir):
        for name, tensor in iter_tensors(path):
            yield _text_encoder_name(name, known), tensor


def _text_encoder_name(name: str, known: set[str]) -> str:
    """Return the model's name for the checkpoint tensor *name*, renaming an older Gemma 3 name."""
    if name in known:
        return name
    for old, new in _LEGACY_TEXT_ENCODER_PREFIXES:
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def _tied_names(model: Any) -> set[str]:
    """Return the names of *model*'s tied weights, which the checkpoint does not store."""
    tied = getattr(model, "_tied_weights_keys", None) or {}
    return set(tied)


def _require_coverage(missing: set[str], source: str) -> None:
    """Raise when *missing* is not empty: the source does not provide every tensor of the model."""
    if missing:
        raise RuntimeError(f"The weights in {source} do not cover the model ({len(missing)} tensors missing, e.g. {sorted(missing)[0]}).")
