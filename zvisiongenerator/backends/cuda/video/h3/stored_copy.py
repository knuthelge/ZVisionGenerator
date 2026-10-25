"""The stored 4-bit copy of the MiniMax H3 transformer: written once, loaded per clip.

The copy folds three things into one set of files, so the stock checkpoint is never needed at run time:

* The turbo LoRA is merged into the bf16 weights at the config scale (which *replaces* ``alpha / rank``), before anything else.
* Every block's AdaLN projection becomes a table per timestep key (:mod:`~zvisiongenerator.backends.cuda.video.h3.adaln`), written
  for the whole key union so any step count, row set and mode loads a subset of one file.
* The linears of the transformer blocks are packed to NF4 with bitsandbytes. The token refiner and everything outside the
  blocks stay in bfloat16 (the refiner is merged with user LoRAs in place, which a packed weight cannot take).

A converted fine-tune (:func:`write_folded`) is written the same way from any tensor source under diffusers' names.

Loading builds the model empty, installs the folded tables for the keys a clip needs, fills it from the copy, applies the user
LoRAs by target (refiner merge, AdaLN table delta, PEFT on the NF4 layers) and sets up block streaming.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.block_streaming import move_outside, stream_blocks
from zvisiongenerator.backends.cuda.lora_peft import is_unmatched_adapter_error, warn_unmatched_lora
from zvisiongenerator.backends.cuda.quant import empty_diffusers_model, fill_weights, nf4_linear_names, nf4_storage_format, pack_nf4, place_nf4, swap_in_linear4bit
from zvisiongenerator.backends.cuda.video.h3 import lora as h3_lora
from zvisiongenerator.utils.safetensors_stream import ShardWriter, iter_tensors, shard_files
from zvisiongenerator.utils.h3_preset import is_folder_name
from zvisiongenerator.utils.stored_quant import ensure_stored_copy, file_identities, stored_quant_name
from zvisiongenerator.utils.quantized_weights import decoded_tensors

if TYPE_CHECKING:
    import torch

    from zvisiongenerator.backends.cuda.video.h3.adaln import Key

__all__ = [
    "Tensors",
    "copy_manifest",
    "copy_name",
    "copy_settings",
    "ensure_copy",
    "load_copy",
    "write_copy",
    "write_folded",
]

# A source of transformer tensors under diffusers' names: called with a name filter (``None`` for all), yields ``(name, tensor)``.
Tensors = Callable[[Callable[[str], bool] | None], Iterator[tuple[str, "torch.Tensor"]]]

_LAYOUT_VERSION = 1
_BITS = 4
_FAMILY = "h3"
_BLOCKS_PREFIX = "transformer_blocks."
_NF4_EXCLUDE = {"adaln_proj"}  # a name part: the folded projection has no weight to pack
_TABLES_NAME = "adaln_tables.safetensors"
_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_SHARD_PREFIX = "model"
_TIME_EMBEDDER = "time_embedder."
_ADALN_TENSOR = re.compile(r"transformer_blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)")
_ADAPTER_PREFIX = "transformer"


def copy_manifest(shards: Iterable[Path], lora_file: Path, scale: float, keys: Iterable[Key]) -> dict[str, Any]:
    """Return the manifest of the stored copy of *shards* with the turbo LoRA at *scale* and AdaLN tables for *keys*.

    Every value is JSON-native, so the manifest compares equal after a round trip through its file. A copy with any other
    source file, scale, key union or NF4 serialization is not current.
    """
    return {**copy_settings(scale, keys), "sources": file_identities([*shards, lora_file])}


def copy_settings(scale: float | None, keys: Iterable[Key]) -> dict[str, Any]:
    """Return what makes a folded copy current besides its sources: its layout, the NF4 serialization, the turbo merge scale (``None`` without one) and the fold keys.

    A converted folder records these too; when they differ from what this machine would write, it must be converted again.
    """
    return {
        "layout": _LAYOUT_VERSION,
        "family": _FAMILY,
        "bits": _BITS,
        "format": nf4_storage_format(),
        "lora_scale": None if scale is None else float(scale),
        "fold_keys": [list(key) for key in keys],
    }


def copy_name(model_name: str | None, model_ref: str) -> str:
    """Return the name a model's stored copy goes by (the copy folder is ``<name>@q4``), so each source model keeps a copy of its own.

    The name the model was picked by is used when it is a plain folder name (``h3``, an installed fine-tune), which is how the
    Models page counts and deletes the copy with the model. Any other pick is named after its reference: a Hub repository
    (``owner--repo``, any revision dropped) or a local folder (its name and a digest of its full path, so equally named folders
    stay apart).

    Args:
        model_name: The name the model was picked by, if the caller has one.
        model_ref: The resolved model: a Hub reference or a local H3 folder.
    """
    if model_name and is_folder_name(model_name):
        return model_name
    ref = model_ref.strip()
    folder = Path(ref).expanduser()
    if folder.is_dir():
        return f"{folder.resolve().name}-{hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:8]}"
    return ref.partition("@")[0].replace("/", "--")


def ensure_copy(models_dir: Path, name: str, shards: Sequence[Path], config: dict[str, Any], lora_file: Path, scale: float, keys: Sequence[Key], device: Any) -> Path:
    """Return the stored copy under *models_dir*, writing it first when it is missing or not current.

    Args:
        models_dir: The models folder; the copy lives at ``<models>/<name>@q4``.
        name: The source model's name (see :func:`copy_name`); the stock ``h3`` keeps ``h3@q4``.
        shards: The transformer's safetensors files, under the model's own (diffusers) tensor names.
        config: The diffusers transformer config.
        lora_file: The turbo LoRA.
        scale: The turbo merge factor, which replaces the file's ``alpha / rank``.
        keys: The AdaLN fold keys to build tables for.
        device: The CUDA device bitsandbytes packs weights on.
    """
    manifest = copy_manifest(shards, lora_file, scale, keys)
    return ensure_stored_copy(models_dir / stored_quant_name(name, _BITS), manifest, lambda target: write_copy(shards, config, lora_file, scale, keys, target, device), flush=True)


def write_copy(shards: Sequence[Path], config: dict[str, Any], lora_file: Path, scale: float, keys: Sequence[Key], target: Path, device: Any) -> None:
    """Write the folded 4-bit copy of the transformer in *shards* to the folder *target*, one tensor at a time.

    Each weight the turbo LoRA targets is merged at *scale* first. The AdaLN projections are not stored: their tables for
    every key of *keys* are built from the merged weights and written to one file. The linears under ``transformer_blocks.``
    are packed to NF4 on *device*; every other tensor is stored in the dtype the model keeps it in.

    Raises:
        RuntimeError: When the shards do not cover the transformer, or the turbo LoRA targets a module they lack.
    """
    label = ", ".join(path.name for path in shards)
    write_folded(lambda wanted: _source_tensors(shards, wanted), config, (h3_lora.load_pairs(lora_file), scale), keys, target, device, label=label)


def write_folded(
    tensors: Tensors,
    config: dict[str, Any],
    turbo: tuple[dict[str, h3_lora.Pair], float] | None,
    keys: Sequence[Key],
    target: Path,
    device: Any,
    *,
    label: str,
) -> None:
    """Write the folded 4-bit copy of the transformer *tensors* yields (diffusers names) to the folder *target*, one tensor at a time.

    With *turbo* (the LoRA's factors and the scale that replaces ``alpha / rank``), each weight it targets is merged first.
    The AdaLN projections are not stored: their tables for every key of *keys* are built from the merged weights and written
    to one file. The linears under ``transformer_blocks.`` are packed to NF4 on *device*; every other tensor is stored in the
    dtype the model keeps it in.

    Args:
        tensors: The source; it is read twice, once for the time embedder and once for everything.
        config: The diffusers transformer config.
        turbo: The turbo LoRA to merge, or ``None`` for a source that is already step-distilled.
        keys: The AdaLN fold keys to build tables for.
        target: The folder to write.
        device: The CUDA device bitsandbytes packs weights on.
        label: Names the source in error messages.

    Raises:
        RuntimeError: When the source does not cover the transformer, or the turbo LoRA targets a module it lacks.
    """
    import torch

    from zvisiongenerator.backends.cuda.video.h3 import adaln

    model = _empty_model(config, torch.bfloat16)
    storage = {name: tensor.dtype for name, tensor in model.state_dict().items()}
    nf4_weights = {f"{name}.weight" for name in nf4_linear_names(model, prefix=_BLOCKS_PREFIX, exclude=_NF4_EXCLUDE)}
    expected_adaln = _adaln_tensor_names(storage)
    temb = _timestep_embeddings(adaln, tensors, config, keys, device, label)
    pairs, scale = turbo if turbo is not None else ({}, 0.0)
    merger = h3_lora.LoraMerger(pairs, scale, device)
    writer = ShardWriter(target, _SHARD_PREFIX, _INDEX_NAME)
    adaln_parts: dict[int, dict[str, torch.Tensor]] = {}
    tables: dict[int, dict[Key, torch.Tensor]] = {}
    written: set[str] = set()
    for name, tensor in tensors(None):
        if name not in storage:
            continue
        module, _, leaf = name.rpartition(".")
        if leaf == "weight":
            tensor = merger.apply(module, tensor)
        if name in expected_adaln:
            block = _adaln_block(name)
            adaln_parts.setdefault(block, {})[leaf] = tensor.to(storage[name])
            if adaln_parts[block].keys() == _adaln_leaves(expected_adaln, block):
                tables[block] = _block_tables(adaln, adaln_parts.pop(block), temb, device)
        else:
            writer.add(pack_nf4(name, tensor, device) if name in nf4_weights else {name: tensor.to(storage[name])})
        written.add(name)
    _require_coverage((storage.keys() - written) | {name for name in expected_adaln if _adaln_block(name) in adaln_parts}, label)
    if merger.unmerged():
        raise RuntimeError(f"The turbo LoRA targets {len(merger.unmerged())} modules the transformer does not have, e.g. {merger.unmerged()[0]}.")
    writer.close()
    adaln.write_tables(target / _TABLES_NAME, [tables[block] for block in sorted(tables)])
    (target / _CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_copy(
    copy_dir: Path,
    keys: Sequence[Key],
    device: Any,
    *,
    loras: Sequence[tuple[Path | str, float]] = (),
    lora_names: Sequence[str] = (),
) -> Any:
    """Load the transformer from the stored copy *copy_dir* on the CPU, ready to stream blocks to *device*.

    The model is built empty and filled from the copy; only the AdaLN tables of *keys* are read. Each user LoRA is applied
    by target before streaming starts: the token refiner is merged into its bf16 weights, the AdaLN part is added to the
    loaded tables, and the rest goes to PEFT on the NF4 layers.

    Args:
        copy_dir: A copy written by :func:`write_copy`.
        keys: The AdaLN fold keys the clip's forwards use.
        device: The GPU the blocks stream to and the layers outside the blocks live on.
        loras: ``(path, weight)`` of each user LoRA.
        lora_names: The PEFT adapter name of each LoRA, in order; ``lora_<index>`` by default.

    Raises:
        ValueError: If *lora_names* does not match *loras*, or a LoRA does not fit the model.
        KeyError: If a key was not built into the copy.
    """
    import torch
    from diffusers import MiniMaxH3Transformer3DModel

    from zvisiongenerator.backends.cuda.video.h3 import adaln

    names = list(lora_names) or [f"lora_{index}" for index in range(len(loras))]
    if len(names) != len(loras):
        raise ValueError(f"Got {len(names)} LoRA names for {len(loras)} LoRAs.")
    config = json.loads((copy_dir / _CONFIG_NAME).read_text(encoding="utf-8"))
    model = empty_diffusers_model(config, torch.bfloat16, MiniMaxH3Transformer3DModel)
    blocks = model.transformer_blocks
    tables = adaln.read_tables(copy_dir / _TABLES_NAME, list(keys), len(blocks))
    adaln.install_fold(model, tables)
    layers = swap_in_linear4bit(model, nf4_linear_names(model, prefix=_BLOCKS_PREFIX, exclude=_NF4_EXCLUDE), torch.bfloat16)
    stored = (item for path in shard_files(copy_dir, _INDEX_NAME) for item in iter_tensors(path))
    fill_weights(model, place_nf4(stored, layers), str(copy_dir), skipped=layers.keys())
    active = _apply_user_loras(adaln, model, tables, list(keys), loras, names)
    if active:
        model.set_adapters([name for name, _ in active], [weight for _, weight in active])
    model.eval()
    stream_blocks(blocks, device)
    move_outside(model, blocks, device)
    return model


# --- writing ---------------------------------------------------------------------------------------------------------


def _empty_model(config: dict[str, Any], dtype: torch.dtype) -> Any:
    """Return the H3 transformer *config* describes without weights."""
    from diffusers import MiniMaxH3Transformer3DModel

    return empty_diffusers_model(config, dtype, MiniMaxH3Transformer3DModel)


def _source_tensors(shards: Iterable[Path], wanted: Callable[[str], bool] | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` of every shard in turn, scaled FP8 weights multiplied out into bfloat16."""
    yield from decoded_tensors(shards, wanted)


def _adaln_tensor_names(storage: dict[str, Any]) -> set[str]:
    """Return the names of the AdaLN projection tensors of the blocks, which become tables instead of being stored.

    Raises:
        RuntimeError: If the model has an ``adaln_proj`` tensor other than a block's ``linear`` weight and bias.
    """
    inside = {name for name in storage if ".adaln_proj." in name}
    odd = sorted(name for name in inside if _ADALN_TENSOR.fullmatch(name) is None)
    if odd:
        raise RuntimeError(f"The transformer has AdaLN tensors the fold does not cover, e.g. {odd[0]}.")
    return inside


def _adaln_block(name: str) -> int:
    """Return the block index of the AdaLN projection tensor *name*."""
    match = _ADALN_TENSOR.fullmatch(name)
    if match is None:
        raise ValueError(f"{name} is not an AdaLN projection tensor.")
    return int(match.group(1))


def _adaln_leaves(names: set[str], block: int) -> set[str]:
    """Return which of ``weight`` and ``bias`` the AdaLN projection of *block* has."""
    return {name.rpartition(".")[2] for name in names if name.startswith(f"{_BLOCKS_PREFIX}{block}.")}


def _timestep_embeddings(adaln: Any, tensors: Tensors, config: dict[str, Any], keys: Iterable[Key], device: Any, label: str) -> dict[Key, torch.Tensor]:
    """Return ``{key: timestep embedding}`` from the source's (float32) time embedder."""
    state = {name.removeprefix(_TIME_EMBEDDER): tensor for name, tensor in tensors(lambda name: name.startswith(_TIME_EMBEDDER))}
    if not state:
        raise RuntimeError(f"{label} holds no time embedder.")
    time_proj, embedder = adaln.load_time_embedder(config, state, device)
    return {key: adaln.compute_temb(time_proj, embedder, key, device) for key in keys}


def _block_tables(adaln: Any, parts: dict[str, torch.Tensor], temb: dict[Key, torch.Tensor], device: Any) -> dict[Key, torch.Tensor]:
    """Return the tables of one block's AdaLN projection from its merged *parts* (``weight`` and maybe ``bias``)."""
    import torch

    weight = parts["weight"]
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in parts, dtype=weight.dtype, device="meta")
    linear.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
    if "bias" in parts:
        linear.bias = torch.nn.Parameter(parts["bias"].to(device), requires_grad=False)
    holder = torch.nn.Module()
    holder.linear = linear
    return {key: table.cpu() for key, table in adaln.build_tables(holder, temb).items()}


def _require_coverage(missing: set[str], label: str) -> None:
    """Raise when *missing* is not empty: the source *label* does not provide every tensor of the model."""
    if missing:
        raise RuntimeError(f"The weights in {label} do not cover the model ({len(missing)} tensors missing, e.g. {sorted(missing)[0]}).")


# --- loading ---------------------------------------------------------------------------------------------------------


def _apply_user_loras(
    adaln: Any,
    model: Any,
    tables: Sequence[dict[Key, torch.Tensor]],
    keys: Sequence[Key],
    loras: Sequence[tuple[Path | str, float]],
    names: Sequence[str],
) -> list[tuple[str, float]]:
    """Apply each user LoRA to the loaded *model* and return ``(adapter name, weight)`` of the PEFT adapters it added.

    Per LoRA the token refiner is merged first, then the AdaLN tables get their delta, then PEFT loads the rest.
    """
    active: list[tuple[str, float]] = []
    temb: dict[Key, torch.Tensor] | None = None
    for (path, weight), name in zip(loras, names, strict=True):
        loaded = h3_lora.load_lora(path)
        parts = h3_lora.split_user_lora(loaded.state, weight, scales=loaded.scales)
        h3_lora.merge_refiner(model, parts.refiner)
        if parts.adaln:
            temb = temb or {key: adaln.compute_temb(model.time_proj, model.time_embedder, key, "cpu") for key in keys}
            _add_table_deltas(adaln, model, tables, temb, parts.adaln)
        if parts.peft.state and _load_peft(model, parts.peft, name, str(path)):
            active.append((name, parts.peft.weight))
    return active


def _add_table_deltas(adaln: Any, model: Any, tables: Sequence[dict[Key, torch.Tensor]], temb: dict[Key, torch.Tensor], deltas: dict[int, h3_lora.Scaled]) -> None:
    """Add each block's AdaLN LoRA to its loaded tables, in place (the folded projection's buffers and *tables*)."""
    for block, (down, up, scale) in deltas.items():
        if not 0 <= block < len(tables):
            raise ValueError(f"LoRA target transformer_blocks.{block}.adaln_proj.linear is not a block of this model ({len(tables)} blocks).")
        updated = adaln.lora_table_delta(tables[block], temb, down, up, scale)
        folded = model.transformer_blocks[block].adaln_proj
        for position, key in enumerate(tables[block]):
            getattr(folded, f"table{position}").copy_(updated[key])
            tables[block][key] = updated[key]


def _load_peft(model: Any, part: h3_lora.PeftPart, adapter_name: str, path: str) -> bool:
    """Load the PEFT part of a user LoRA as adapter *adapter_name* and return whether it matched; warn when it did not."""
    try:
        model.load_lora_adapter(h3_lora.peft_state(part), prefix=_ADAPTER_PREFIX, adapter_name=adapter_name)
    except Exception as exc:
        if is_unmatched_adapter_error(exc):
            warn_unmatched_lora(path)
            return False
        raise RuntimeError(f"Failed to load LoRA '{path}': {exc}") from exc
    return True
