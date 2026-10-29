"""Read a MiniMax H3 fine-tune for conversion, one tensor at a time, under diffusers' names.

Three layouts are read, from one file or a sharded set (picking any shard reads the set its index lists):

- **ComfyUI** (one file): MiniMax's own names (``blocks.N.attn.qkv_proj`` ...), bare or under ``model.diffusion_model.``
  or ``diffusion_model.``, with the fused QKV stored as ``[q; k; v]``.
- **MiniMax native** (a sharded set with an index, as ``FL2VA/transformer/``): the same names, with the fused QKV
  interleaved per head (``[q0 k0 v0, q1 k1 v1, ...]``), the way MiniMax's reference loader reads it.
- **Diffusers** (``transformer_blocks.N.attn.to_q`` ...), as diffusers saves the transformer.

The mapping from MiniMax's names is a rename, plus two reorders: the fused QKV is split into ``to_q``/``to_k``/``to_v``
and the gated feed-forward's ``[gate; value]`` halves are swapped to diffusers' ``[value; gate]``. No tensor is
transposed. Quantized weights (see :mod:`zvisiongenerator.utils.quantized_weights`) are decoded as they stream.

A **pruned** checkpoint has no time embedder: its AdaLN projections read an ``r``-dimensional curve of the timestep
(``adaln_t_table``, one row per point of a grid over ``[0, 1]``, interpolated linearly, as ComfyUI reads it) instead of
the 2688-wide embedding, and skip the SiLU. That curve is an affine projection of stock's ``silu(time embedding)``
(fitted on its grid to about 1e-12), so the reader un-prunes it: with the projection ``c(t) = P s(t) + q`` fitted once,
each projection ``W c(t) + b`` becomes the stock-shaped ``(W P) s(t) + (b + W q)``, and stock's time embedder is
yielded in place of the missing one. The model then runs, takes LoRAs and folds exactly as stock does.

Everything here reads headers only, except the tensor iterator.
"""

from __future__ import annotations

from collections import Counter

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from zvisiongenerator.utils.quantized_weights import AUX_SUFFIXES, decoded_shape, decoded_tensors, quantization_formats
from zvisiongenerator.utils.safetensors_stream import read_header_and_metadata, read_stream_tensors

if TYPE_CHECKING:
    import torch

__all__ = [
    "H3Checkpoint",
    "STOCK_CONFIG",
    "curve_projection",
    "diffusers_tensors",
    "inspect_h3",
    "native_name",
    "stock_time_embedder",
]

# The transformer config of stock MiniMax H3 (the diffusers ``transformer/config.json``); every checkpoint must match it.
STOCK_CONFIG: dict[str, Any] = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
}
# The pruned checkpoint's timestep curve.
CURVE_TABLE = "adaln_t_table"
_TIME_EMBEDDER = "time_embedder."
_TIME_EMBEDDER_TENSORS = ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")
_DIFFUSERS_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
# How closely a pruned checkpoint's curve must match the projection of stock's embedding, relative to its size.
_CURVE_FIT_TOLERANCE = 1e-3
# Directions of stock's embedding over the grid weaker than this (relative to the strongest) are left out of the fit: they
# carry rounding noise, and fitting them makes the widened weights large enough to lose the result in bfloat16 rounding.
_CURVE_FIT_RCOND = 1e-5
# The projections that read the curve in a pruned checkpoint (diffusers names, without ``.weight``/``.bias``).
_CURVE_READERS = re.compile(r"(transformer_blocks\.\d+\.adaln_proj\.linear|norm_out\.linear)\.(weight|bias)")
_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# Tensors a checkpoint may carry that the model recomputes or doesn't hold: the rotary frequencies, and the marker of the
# older scaled-FP8 layout (its ``scale_weight`` tensors are read as scales).
_DROPPED = ("rope.inv_freq", "scaled_fp8")
_NATIVE_MARK = "blocks.0.attn.qkv_proj.weight"
_DIFFUSERS_MARK = "transformer_blocks.0.attn.to_q.weight"
# The diffusers layout's folder for the reference-to-video transformer.
_REF_FOLDER = "transformer_ref"
_LORA_MARKS = (".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.", "lora_unet_")
# What a file's name or metadata may say about a step-distilled (turbo) checkpoint. Hints only: nothing is decided from them.
_TURBO_WORDS = re.compile(r"turbo|lightning|lightx2v|distill|fused|(?<![0-9])[48][-_ ]?steps?", re.IGNORECASE)
_SHIFT_KEY = re.compile(r"shift", re.IGNORECASE)
_RENAMES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
)
_INNER_RENAMES = ((".attn.q_norm.", ".attn.norm_q."), (".attn.k_norm.", ".attn.norm_k."), (".attn.out_proj.", ".attn.to_out.0."), (".mlp.fc2.", ".ff.net.2."))


@dataclass(frozen=True)
class H3Checkpoint:
    """A checked MiniMax H3 checkpoint: its files, layout and what the converter needs to know before reading weights.

    Attributes:
        path: The file the user picked.
        files: Every file of the checkpoint (one, or the shards its index lists), in index order.
        layout: ``comfyui``, ``native`` or ``diffusers``.
        names: Tensor name in the files → its diffusers name (several for a fused QKV; none for a dropped tensor).
        shapes: Diffusers name → shape, once decoded, mapped and (for a pruned checkpoint) un-pruned: stock's.
        quantized: Tensor name in the files → its quantization format, for every quantized weight.
        curve_rank: The width of a pruned checkpoint's AdaLN input (its curve's rank); ``None`` for a full checkpoint.
        turbo_hints: What the file's name or metadata says that hints at a step-distilled checkpoint.
    """

    path: Path
    files: tuple[Path, ...]
    layout: str
    names: dict[str, tuple[str, ...]]
    shapes: dict[str, list[int]]
    quantized: dict[str, str]
    curve_rank: int | None
    turbo_hints: tuple[str, ...]

    @property
    def precision(self) -> dict[str, int]:
        """Return how many weights use each quantization format (empty when none is quantized)."""
        return dict(Counter(self.quantized.values()))


def inspect_h3(path: Path) -> H3Checkpoint:
    """Read the headers of the H3 checkpoint at *path* (and its sibling shards) and check that the converter reads it.

    Raises:
        ValueError: If it is not a safetensors file, is a LoRA, is not MiniMax H3 (another model, a PDD or Ref2VA
            variant, other sizes) or holds a quantization that can't be decoded; the message says which.
    """
    if path.suffix != ".safetensors":
        raise ValueError(f"{path.name} is not a .safetensors file; GGUF and other formats can't be converted.")
    files = _checkpoint_files(path)
    headers: dict[str, dict[str, Any]] = {}
    metadata: dict[str, str] = {}
    for file in files:
        header, found = read_header_and_metadata(file)
        headers.update(header)
        metadata = metadata or found
    if any(mark in name for name in headers for mark in _LORA_MARKS):
        raise ValueError(f"{path.name} is a LoRA, not a checkpoint; import it on the LoRAs tab instead.")
    stripped = {_strip_prefix(name): name for name in headers if not name.endswith(AUX_SUFFIXES)}
    if _DIFFUSERS_MARK in stripped:
        layout = "diffusers"
    elif _NATIVE_MARK in stripped:
        layout = "native" if len(files) > 1 else "comfyui"
    else:
        raise ValueError(f"{path.name} is not a MiniMax H3 transformer.")
    _check_variant(path, metadata)
    _check_partition(path)
    formats = quantization_formats(files)
    shapes_in_file = {short: decoded_shape(headers, name, formats.get(name)) for short, name in stripped.items()}
    curve_rank = _curve_rank(path, shapes_in_file)
    expected = _expected_shapes(curve_rank)
    names: dict[str, tuple[str, ...]] = {}
    shapes: dict[str, list[int]] = {}
    for short, name in stripped.items():
        targets = _diffusers_names(short) if layout != "diffusers" else ((short,) if short not in _DROPPED else ())
        names[name] = targets
        for target, shape in zip(targets, _target_shapes(short, shapes_in_file[short], len(targets)), strict=True):
            shapes[target] = shape
    _check_shapes(path, shapes, expected)
    if curve_rank is not None:
        shapes = _expected_shapes(None)  # what un-pruning yields
    return H3Checkpoint(
        path=path,
        files=files,
        layout=layout,
        names=names,
        shapes=shapes,
        quantized={name: str(record["format"]) for name, record in formats.items() if name in names},
        curve_rank=curve_rank,
        turbo_hints=_turbo_hints(path, files, metadata),
    )


def _checkpoint_files(path: Path) -> tuple[Path, ...]:
    """Return the files of the checkpoint *path* belongs to: the shards a sibling index lists when it lists *path*, else *path* alone."""
    for index in sorted(path.parent.glob("*.safetensors.index.json")):
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        except OSError, ValueError:
            continue
        shards = sorted(set(weight_map.values()))
        if path.name in shards:
            missing = [name for name in shards if not (path.parent / name).is_file()]
            if missing:
                raise ValueError(f"{path.name} is one shard of a set whose index lists {missing[0]}, which is missing.")
            return tuple(path.parent / name for name in shards)
    return (path,)


def _strip_prefix(name: str) -> str:
    """Return *name* without a ComfyUI checkpoint prefix."""
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _check_variant(path: Path, metadata: dict[str, str]) -> None:
    """Refuse the H3 variants the app does not run, by what the file's config metadata declares."""
    try:
        transformer = json.loads(metadata.get("config", "{}")).get("transformer", {})
    except ValueError:
        return
    if isinstance(transformer, dict) and int(transformer.get("pdd_steps", 1) or 1) > 1:
        raise ValueError(f"{path.name} is a PDD checkpoint (stacked output heads), which the app doesn't run.")


def _check_partition(path: Path) -> None:
    """Refuse a checkpoint that its folder says is not the first-and-last-frame (``fl2va``) partition.

    MiniMax's native layout names the partition in ``model_index.json`` beside the transformer folder (``Ref2VA/`` holds a
    reference-to-video transformer with the same tensor names and shapes as ``FL2VA/``); the diffusers layout keeps it in
    ``transformer_ref/``. A file with no such index (a ComfyUI single file) is let through.

    Raises:
        ValueError: If the folder is the reference-to-video partition, or any partition other than ``fl2va``.
    """
    if path.parent.name == _REF_FOLDER:
        raise ValueError(f"{path.name} is a Ref2VA (reference-to-video) transformer, which the app doesn't convert; use the FL2VA transformer.")
    index = path.parent.parent / "model_index.json"
    try:
        partition = json.loads(index.read_text(encoding="utf-8")).get("_minimax_h3", {}).get("partition")
    except OSError, ValueError:
        return
    if isinstance(partition, str) and partition.lower() != "fl2va":
        raise ValueError(f"{path.name} is the {partition} partition of MiniMax H3 ({index.name} says so), which the app doesn't convert; use the FL2VA transformer.")


def _curve_rank(path: Path, shapes: dict[str, list[int]]) -> int | None:
    """Return the curve rank of a pruned checkpoint (the width of its AdaLN input), or ``None`` for a full one.

    Raises:
        ValueError: When the curve table and the projections disagree.
    """
    table = shapes.get(CURVE_TABLE)
    if table is None:
        return None
    projection = shapes.get("blocks.0.adaln_proj.linear.weight") or shapes.get("transformer_blocks.0.adaln_proj.linear.weight") or [0, 0]
    if len(table) != 2 or table[0] < 2 or table[1] != projection[1]:
        raise ValueError(f"{path.name}'s pruned AdaLN curve {table} does not fit its projections {projection}.")
    return int(table[1])


def _diffusers_names(native: str) -> tuple[str, ...]:
    """Return the diffusers name(s) of the MiniMax-named tensor *native* (three for a fused QKV, none for a dropped tensor)."""
    if native in _DROPPED:
        return ()
    if native == CURVE_TABLE:
        return (CURVE_TABLE,)
    name = native
    for old, new in _RENAMES:
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    for old, new in _INNER_RENAMES:
        name = name.replace(old, new)
    if name.endswith(".attn.qkv_proj.weight"):
        stem = name.removesuffix("qkv_proj.weight")
        return (f"{stem}to_q.weight", f"{stem}to_k.weight", f"{stem}to_v.weight")
    return (name.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"),)


def native_name(name: str) -> str:
    """Return MiniMax's name of the diffusers H3 tensor *name*: the reader's renames undone (a QKV or the gated feed-forward also needs its tensor regrouped)."""
    for new, old in ((new, old) for old, new in _RENAMES):
        if name.startswith(new):
            name = old + name[len(new) :]
            break
    for old, new in _INNER_RENAMES:
        name = name.replace(new, old)
    return name


def _target_shapes(native: str, shape: list[int], count: int) -> list[list[int]]:
    """Return the shapes of the *count* diffusers tensors *native* (of *shape*) maps to: a fused QKV splits its rows in three."""
    if count == 3:
        return [[shape[0] // 3, *shape[1:]]] * 3
    return [shape] * count


def _expected_shapes(curve_rank: int | None) -> dict[str, list[int]]:
    """Return every diffusers tensor of stock H3 and its shape; with *curve_rank*, the pruned layout (a curve, no time embedder)."""
    config = STOCK_CONFIG
    hidden, inner, ffn = config["hidden_size"], config["num_attention_heads"] * config["attention_head_dim"], config["ffn_dim"]
    head = config["attention_head_dim"]
    adaln_in = curve_rank or config["time_embed_dim"]
    patch = config["in_channels"] * config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]
    shapes: dict[str, list[int]] = {
        "proj_in.weight": [hidden, patch],
        "proj_in.bias": [hidden],
        "audio_proj_in.weight": [hidden, config["audio_in_channels"]],
        "audio_proj_in.bias": [hidden],
        "context_embedder.weight": [hidden, config["text_dim"]],
        "context_embedder.bias": [hidden],
        "token_refiner.final_norm.weight": [hidden],
        "norm_out.norm.weight": [hidden],
        "norm_out.linear.weight": [2 * hidden, adaln_in],
        "norm_out.linear.bias": [2 * hidden],
        "proj_out.weight": [patch, hidden],
        "proj_out.bias": [patch],
        "audio_proj_out.weight": [config["audio_in_channels"], hidden],
        "audio_proj_out.bias": [config["audio_in_channels"]],
    }
    if curve_rank is None:
        shapes |= {
            "time_embedder.linear_1.weight": [config["time_embed_hidden_dim"], config["freq_dim"]],
            "time_embedder.linear_1.bias": [config["time_embed_hidden_dim"]],
            "time_embedder.linear_2.weight": [config["time_embed_dim"], config["time_embed_hidden_dim"]],
            "time_embedder.linear_2.bias": [config["time_embed_dim"]],
        }
    for prefix, count, has_adaln in (("transformer_blocks", config["num_layers"], True), ("token_refiner.refiner_blocks", config["num_refiner_layers"], False)):
        for index in range(count):
            block = f"{prefix}.{index}"
            shapes |= {
                f"{block}.norm1.weight": [hidden],
                f"{block}.norm2.weight": [hidden],
                f"{block}.attn.to_q.weight": [inner, hidden],
                f"{block}.attn.to_k.weight": [inner, hidden],
                f"{block}.attn.to_v.weight": [inner, hidden],
                f"{block}.attn.norm_q.weight": [head],
                f"{block}.attn.norm_k.weight": [head],
                f"{block}.attn.to_out.0.weight": [hidden, inner],
                f"{block}.ff.net.0.proj.weight": [2 * ffn, hidden],
                f"{block}.ff.net.2.weight": [hidden, ffn],
            }
            if has_adaln:
                shapes |= {f"{block}.adaln_proj.linear.weight": [6 * 3 * hidden, adaln_in], f"{block}.adaln_proj.linear.bias": [6 * 3 * hidden]}
    return shapes


def _check_shapes(path: Path, shapes: dict[str, list[int]], expected: dict[str, list[int]]) -> None:
    """Raise unless *shapes* (the mapped checkpoint, plus a pruned one's curve) has exactly stock's tensors and shapes."""
    found = {name: shape for name, shape in shapes.items() if name != CURVE_TABLE}
    extra = sorted(found.keys() - expected.keys())
    missing = sorted(expected.keys() - found.keys())
    if extra or missing:
        what = f"has tensors stock H3 lacks, e.g. {extra[0]}" if extra else f"lacks {len(missing)} of stock H3's tensors, e.g. {missing[0]}"
        raise ValueError(f"{path.name} is not a MiniMax H3 FL2VA checkpoint the app runs: it {what}.")
    wrong = sorted(name for name, shape in found.items() if shape != expected[name])
    if wrong:
        raise ValueError(f"{path.name} is not a stock-sized MiniMax H3 checkpoint: {wrong[0]} is {found[wrong[0]]}, where stock has {expected[wrong[0]]}.")


def _turbo_hints(path: Path, files: tuple[Path, ...], metadata: dict[str, str]) -> tuple[str, ...]:
    """Return what *path*'s name (or, in the Hugging Face cache, its repository's) or metadata says that hints at a step-distilled checkpoint, as short phrases."""
    hints: list[str] = []
    repos = [parent.name.split("--", 2)[-1] for parent in path.parents if parent.name.startswith("models--")]
    for label in dict.fromkeys([path.stem, *(file.stem for file in files[:1]), *repos[:1]]):
        if match := _TURBO_WORDS.search(label):
            hints.append(f"the name says '{match.group(0)}'")
            break
    for key, value in metadata.items():
        if key == "license":
            continue
        if _SHIFT_KEY.search(key):
            hints.append(f"its metadata sets {key} = {value[:20]}")
        elif isinstance(value, str) and len(value) < 2000 and (match := _TURBO_WORDS.search(value)):
            hints.append(f"its metadata '{key}' says '{match.group(0)}'")
    return tuple(dict.fromkeys(hints))


def diffusers_tensors(
    checkpoint: H3Checkpoint,
    wanted: Callable[[str], bool] | None = None,
    *,
    device: Any = None,
    progress: Callable[[int, int], None] | None = None,
    time_embedder: dict[str, torch.Tensor] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the checkpoint's tensors under diffusers' names, in stock's shapes, in file order.

    Quantized weights are decoded to bfloat16; the fused QKV of a native set is de-interleaved per head first. A pruned
    checkpoint is un-pruned (see the module docstring): stock's *time_embedder* comes first, then each projection that
    read the curve in stock's shape (a bias once its weight has passed).

    Args:
        checkpoint: The checked checkpoint.
        wanted: Accepts the diffusers names to yield; every one when ``None``.
        device: Where quantized weights are decoded and pruned projections widened; the CPU by default.
        progress: Called with ``(tensors done, total)`` after each tensor of the files.
        time_embedder: Stock's time embedder (``linear_1``/``linear_2`` names), required for a pruned checkpoint.

    Raises:
        ValueError: For a pruned checkpoint without *time_embedder*, or whose curve is not a projection of its embedding.
    """
    import torch

    projection = None
    if checkpoint.curve_rank is not None:
        if time_embedder is None:
            raise ValueError(f"{checkpoint.path.name} is a pruned checkpoint: reading it needs stock's time embedder.")
        table = dict(decoded_tensors(checkpoint.files, lambda name: checkpoint.names.get(name) == (CURVE_TABLE,)))
        projection = curve_projection(next(iter(table.values())), time_embedder, checkpoint.path.name)
        for name, tensor in sorted(time_embedder.items()):
            if wanted is None or wanted(_TIME_EMBEDDER + name):
                yield _TIME_EMBEDDER + name, tensor
    unpruned = _Unpruner(projection, device) if projection is not None else None

    def _source_wanted(name: str) -> bool:
        targets = checkpoint.names.get(name, ())
        if targets == (CURVE_TABLE,):
            return False
        if unpruned is not None and any(_CURVE_READERS.fullmatch(target) for target in targets):
            return wanted is None or any(wanted(f"{target.rpartition('.')[0]}.{leaf}") for target in targets for leaf in ("weight", "bias"))
        return bool(targets) and (wanted is None or any(wanted(target) for target in targets))

    total = sum(bool(targets) and targets != (CURVE_TABLE,) for targets in checkpoint.names.values())
    heads, head_dim = STOCK_CONFIG["num_attention_heads"], STOCK_CONFIG["attention_head_dim"]
    for done, (name, tensor) in enumerate(decoded_tensors(checkpoint.files, _source_wanted, device=device), start=1):
        targets = checkpoint.names[name]
        if unpruned is not None and _CURVE_READERS.fullmatch(targets[0]):
            for target, widened in unpruned.add(targets[0], tensor):
                if wanted is None or wanted(target):
                    yield target, widened
        elif len(targets) == 3:
            if checkpoint.layout == "native":
                grouped = tensor.reshape(heads, 3 * head_dim, -1)
                parts = [part.reshape(heads * head_dim, -1) for part in grouped.split(head_dim, dim=1)]
            else:
                parts = list(tensor.split(heads * head_dim))
            for target, part in zip(targets, parts, strict=True):
                if wanted is None or wanted(target):
                    yield target, part.contiguous()
        elif checkpoint.layout != "diffusers" and targets[0].endswith(".ff.net.0.proj.weight"):
            gate, value = tensor.chunk(2, dim=0)
            yield targets[0], torch.cat([value, gate], dim=0)
        elif wanted is None or wanted(targets[0]):
            yield targets[0], tensor
        if progress is not None:
            progress(done, total)


class _Unpruner:
    """Widen a pruned checkpoint's curve-reading projections to stock's shape as their weights and biases stream past.

    A bias needs its weight (``b + W q``), so a bias that comes first waits for it.
    """

    def __init__(self, projection: tuple[torch.Tensor, torch.Tensor], device: Any) -> None:
        self.matrix, self.offset = projection  # P (rank, 2688) and q (rank,), float32
        self.device = device
        self.shifts: dict[str, torch.Tensor] = {}
        self.biases: dict[str, torch.Tensor] = {}

    def add(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        """Take the weight or bias *name* and yield what is ready: a widened weight, and a bias once its weight has passed."""
        import torch

        module, _, leaf = name.rpartition(".")
        if leaf == "weight":
            weight = tensor.to(device=self.device, dtype=torch.float32)
            self.shifts[module] = (weight @ self.offset.to(weight.device)).cpu()
            yield name, (weight @ self.matrix.to(weight.device)).cpu()
            if module in self.biases:
                yield from self._bias(module, self.biases.pop(module))
        elif module in self.shifts:
            yield from self._bias(module, tensor)
        else:
            self.biases[module] = tensor

    def _bias(self, module: str, bias: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        import torch

        yield f"{module}.bias", bias.to(torch.float32) + self.shifts.pop(module)


def curve_projection(table: torch.Tensor, time_embedder: dict[str, torch.Tensor], label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(P, q)``, the affine map with ``curve(t) = P silu(temb(t)) + q`` on the curve *table*'s grid, from stock's *time_embedder*.

    Fitted by least squares in float64 over every grid point, leaving out the embedding's directions that carry only
    rounding noise (so the widened weights stay as small as stock's), and returned in float32.

    Raises:
        ValueError: When the curve is not such a projection (a fine-tuned time embedder), naming *label*.
    """
    import torch

    grid = torch.linspace(0.0, 1.0, table.shape[0], dtype=torch.float64)
    embedded = torch.nn.functional.silu(_time_embedding(grid, time_embedder)).double()
    design = torch.cat([embedded, torch.ones(embedded.shape[0], 1, dtype=torch.float64)], dim=1)
    curve = table.double()
    solution = torch.linalg.lstsq(design, curve, rcond=_CURVE_FIT_RCOND, driver="gelsd").solution
    residual = ((design @ solution - curve).norm() / curve.norm()).item()
    if residual > _CURVE_FIT_TOLERANCE:
        raise ValueError(f"{label}'s pruned AdaLN curve is not a projection of stock's time embedding (fit error {residual:.1e}), so it can't be un-pruned.")
    return solution[:-1].T.contiguous().float(), solution[-1].float()


def _time_embedding(timesteps: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return stock H3's time embedding of *timesteps* in float64: the flipped sinusoid of diffusers' ``Timesteps`` through *state*'s two linears."""
    import math

    import torch

    half = STOCK_CONFIG["freq_dim"] // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    angles = timesteps.double()[:, None] * frequencies[None, :]
    sinusoid = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(sinusoid, state["linear_1.weight"].double(), state["linear_1.bias"].double()))
    return torch.nn.functional.linear(hidden, state["linear_2.weight"].double(), state["linear_2.bias"].double())


def stock_time_embedder(
    repo_id: str,
    revision: str,
    *,
    index: str = _DIFFUSERS_INDEX,
    native: bool = False,
    download_file: Callable[..., str] | None = None,
    open_remote: Callable[[str, str, str], tuple[IO[bytes], int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Return stock H3's time embedder (``linear_1``/``linear_2`` names) from the H3 transformer of *repo_id*: its shard index, then only the embedder's bytes of the shards that hold it, read by range.

    Args:
        repo_id: The repository.
        revision: Its revision.
        index: The shard index's path in the repository; the shards sit beside it. Stock's diffusers transformer by default.
        native: The transformer uses MiniMax's names (the MLX INT8 transformer keeps the time embedder unquantized).
        download_file: Called as ``hf_hub_download`` for the index; defaults to it.
        open_remote: Called with ``(repo_id, path, revision)``; returns a seekable stream of that file and its size. Opens it on the Hub by default.

    Raises:
        RuntimeError: If they can't be read or don't hold the whole time embedder.
    """
    if download_file is None:
        from huggingface_hub import hf_hub_download as download_file
    if open_remote is None:
        open_remote = _open_on_hub
    names = {(native_name(f"{_TIME_EMBEDDER}{name}") if native else f"{_TIME_EMBEDDER}{name}"): name for name in _TIME_EMBEDDER_TENSORS}
    folder = index.rpartition("/")[0]
    state: dict[str, torch.Tensor] = {}
    try:
        weight_map = json.loads(Path(download_file(repo_id=repo_id, filename=index, revision=revision)).read_text(encoding="utf-8"))["weight_map"]
        for shard in sorted({weight_map[name] for name in names if name in weight_map}):
            path = f"{folder}/{shard}" if folder else shard
            handle, size = open_remote(repo_id, path, revision)
            with handle:
                found = read_stream_tensors(handle, f"{repo_id}/{path}", size, names.__contains__)
            state.update({names[name]: tensor for name, tensor in found.items()})
    except Exception as exc:
        raise RuntimeError(f"Converting a pruned MiniMax H3 checkpoint needs stock's time embedder from {repo_id}, which couldn't be read: {exc}") from exc
    if state.keys() != set(_TIME_EMBEDDER_TENSORS):
        raise RuntimeError(f"The transformer of {repo_id} holds no complete time embedder.")
    return state


def _open_on_hub(repo_id: str, path: str, revision: str) -> tuple[IO[bytes], int]:
    """Open *path* of *repo_id* at *revision* on the Hub for range reads, with its size."""
    from huggingface_hub import HfFileSystem

    handle = HfFileSystem().open(f"{repo_id}/{path}", "rb", revision=revision)
    return handle, handle.size
