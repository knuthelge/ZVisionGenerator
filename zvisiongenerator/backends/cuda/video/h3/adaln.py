"""The exact AdaLN fold of the MiniMax H3 transformer: per-block modulation tables keyed by a forward's timesteps.

Every block's ``adaln_proj`` computes ``linear(silu(temb))`` from the timestep embedding alone, and ``temb`` is a
function of the forward's ``timestep`` argument only: the distinct row timesteps of that forward. So the output depends
on those values and nothing else. A table per (block, key) computed once with the stock projection and the same k rows is
bit-identical to computing it in the forward, and the projection's weights (the bulk of a block's parameters) never
need to be loaded at run time.

A key is the sorted tuple of a forward's distinct row timesteps. Which values occur depends on the conditioning rows of
the clip (:data:`~zvisiongenerator.backends.cuda.video.h3.schedule.ROW_SETS`) and on the schedule, so the tables cover
every key of every row set of the longest schedule; shorter schedules reuse a subset of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from zvisiongenerator.backends.cuda.video.h3.schedule import (
    AUDIO_COND_T,
    KEYFRAME_T,
    ROW_SETS,
    build_schedulers,
    row_timesteps,
)

if TYPE_CHECKING:
    from diffusers import MiniMaxH3Scheduler
    from torch.utils.hooks import RemovableHandle

    from zvisiongenerator.utils.h3_preset import H3Schedule

__all__ = [
    "FoldHandle",
    "FoldState",
    "FoldedAdaLN",
    "Key",
    "build_tables",
    "compute_temb",
    "fold_keys",
    "install_fold",
    "load_time_embedder",
    "lora_table_delta",
    "read_tables",
    "step_keys",
    "table_keys",
    "unique_keys",
    "write_tables",
]

Key = tuple[float, ...]  # the sorted distinct row timesteps of one forward

_TEXT_ROWS = 1  # text rows only inherit the video timestep, so one is enough to enumerate the keys
# Rows per modality in the dummy layouts of step_keys: (video rows, audio rows, conditioning video rows, conditioning audio rows).
_DUMMY_LAYOUTS = {
    "plain": (1, 1, 0, 0),
    "keyframe": (2, 1, 1, 0),
    "extend": (2, 3, 1, 2),
}


def step_keys(video: MiniMaxH3Scheduler, audio: MiniMaxH3Scheduler, row_set: str) -> list[Key]:
    """Return the key of every forward of the schedulers' schedule for a clip with the conditioning rows of *row_set*.

    The keys come from the same row-timestep function the sampler uses, run on a dummy layout with the conditioning
    rows first: ``plain`` has none, ``keyframe`` has one video row at the keyframe level, ``extend`` has one video row
    at the keyframe level and two audio rows at the fully clean level. The result does not depend on the canvas.

    Raises:
        ValueError: If *row_set* is not one of ``ROW_SETS`` or the two schedulers differ in length.
    """
    if row_set not in _DUMMY_LAYOUTS:
        raise ValueError(f"Unknown row set {row_set!r}; expected one of {ROW_SETS}.")
    if len(video.timesteps) != len(audio.timesteps):
        raise ValueError(f"The video and audio schedules differ in length ({len(video.timesteps)} and {len(audio.timesteps)}).")
    video_rows, audio_rows, condition_video, condition_audio = _DUMMY_LAYOUTS[row_set]
    video_idx = torch.arange(_TEXT_ROWS, _TEXT_ROWS + video_rows)
    audio_idx = torch.arange(_TEXT_ROWS + video_rows, _TEXT_ROWS + video_rows + audio_rows)
    keys: list[Key] = []
    for t, audio_t in zip(video.timesteps, audio.timesteps, strict=True):
        unique, _ = row_timesteps(
            video_idx,
            audio_idx,
            _TEXT_ROWS,
            float(t),
            float(audio_t),
            keyframe_t=KEYFRAME_T,
            audio_cond_t=AUDIO_COND_T,
            condition_video_rows=condition_video,
            condition_audio_rows=condition_audio,
        )
        keys.append(tuple(float(value) for value in unique))
    return keys


def unique_keys(keys: Iterable[Key]) -> list[Key]:
    """Return *keys* without repeats, in first-appearance order."""
    return list(dict.fromkeys(keys))


def _schedule_keys(schedule: H3Schedule, config_dir: Path | None) -> list[Key]:
    """Return the distinct keys of *schedule* over every row set, in row-set then forward order."""
    video, audio = build_schedulers(schedule, config_dir)
    return unique_keys(key for row_set in ROW_SETS for key in step_keys(video, audio, row_set))


def fold_keys(schedules: Mapping[int, H3Schedule], config_dir: Path | None = None) -> list[Key]:
    """Return every key the fold tables must cover for the given schedules (by step count).

    The longest schedule defines the keys; every other schedule's keys must already be among them, so the tables stay
    one set however many step counts the model offers.

    Raises:
        ValueError: If *schedules* is empty, or a schedule needs a key the longest one does not have (naming it).
    """
    if not schedules:
        raise ValueError("At least one schedule is required to derive the fold keys.")
    longest = max(schedules, key=lambda steps: schedules[steps].forwards)
    keys = _schedule_keys(schedules[longest], config_dir)
    covered = set(keys)
    for steps, schedule in schedules.items():
        missing = [key for key in _schedule_keys(schedule, config_dir) if key not in covered]
        if missing:
            raise ValueError(f"The {steps}-step schedule needs {len(missing)} fold keys the {longest}-step schedule does not have, e.g. {missing[0]}.")
    return keys


def load_time_embedder(config: Mapping[str, Any], state: Mapping[str, torch.Tensor], device: torch.device | str) -> tuple[nn.Module, nn.Module]:
    """Return ``(time_proj, time_embedder)`` built from the transformer *config* and the checkpoint's ``time_embedder.*`` *state*.

    *state* maps the names without the ``time_embedder.`` prefix to tensors; the embedder runs in float32 like the
    transformer's own.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    time_proj = Timesteps(num_channels=config["freq_dim"], flip_sin_to_cos=True, downscale_freq_shift=0)
    embedder = TimestepEmbedding(in_channels=config["freq_dim"], time_embed_dim=config["time_embed_hidden_dim"], out_dim=config["time_embed_dim"])
    embedder.load_state_dict({name: value.float() for name, value in state.items()})
    return time_proj.to(device), embedder.to(device).float().eval()


def compute_temb(time_proj: nn.Module, time_embedder: nn.Module, key: Key, device: torch.device | str) -> torch.Tensor:
    """Return the timestep embedding of *key* exactly as ``MiniMaxH3Transformer3DModel.forward`` computes it."""
    from diffusers.models.modeling_utils import get_parameter_dtype

    timestep = torch.tensor(key, dtype=torch.float32, device=device)
    with torch.no_grad():
        return time_embedder(time_proj(timestep).to(get_parameter_dtype(time_embedder)))


def build_tables(adaln: nn.Module, temb_by_key: Mapping[Key, torch.Tensor]) -> dict[Key, torch.Tensor]:
    """Return ``{key: (k, 6 * 3 * hidden) table}`` from the modulation module *adaln* (anything with a ``.linear``).

    The projection is applied as the stock module applies it, ``linear(silu(temb).to(parameter dtype))``, with the
    same k rows, so each table is bit-identical to the stock output for that key.
    """
    from diffusers.models.modeling_utils import get_parameter_dtype

    dtype = get_parameter_dtype(adaln.linear)
    with torch.no_grad():
        return {key: adaln.linear(nn.functional.silu(temb).to(dtype)) for key, temb in temb_by_key.items()}


def lora_table_delta(
    tables: Mapping[Key, torch.Tensor],
    temb_by_key: Mapping[Key, torch.Tensor],
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
) -> dict[Key, torch.Tensor]:
    """Return *tables* with a LoRA on the AdaLN projection added: ``table + scale * ((silu(temb) @ down.T) @ up.T)``.

    The delta is computed in float32 per key with the same k rows and added to the table in float32; the sum is cast
    back to the table's dtype. *down* is the LoRA's A matrix ``(rank, in)`` and *up* its B matrix ``(out, rank)``;
    *scale* already folds the weight and ``alpha / rank``. The inputs are not modified.
    """
    folded: dict[Key, torch.Tensor] = {}
    with torch.no_grad():
        for key, table in tables.items():
            temb = nn.functional.silu(temb_by_key[key].to(table.device, torch.float32))
            delta = (temb @ down.to(table.device, torch.float32).T) @ up.to(table.device, torch.float32).T
            folded[key] = (table.float() + scale * delta).to(table.dtype)
    return folded


class FoldState:
    """The key of the forward in flight, shared by every folded projection of one transformer and its pre-hook."""

    def __init__(self, keys: Sequence[Key]) -> None:
        self.keys = list(keys)
        self.index = {key: position for position, key in enumerate(self.keys)}
        self.current: int | None = None
        self.hits: list[int] = []

    def select(self, timestep: torch.Tensor) -> int:
        """Set the current key from a forward's ``timestep`` argument and return its position.

        Raises:
            KeyError: If no table was built for that key (there is no fallback to a stock projection).
        """
        key = tuple(float(value) for value in timestep.detach().float().cpu().tolist())
        if key not in self.index:
            raise KeyError(f"No AdaLN fold table for timestep key {key} ({len(self.keys)} keys loaded); rebuild the H3 copy for this schedule.")
        self.current = self.index[key]
        self.hits.append(self.current)
        return self.current


class FoldedAdaLN(nn.Module):
    """Stands in for ``adaln_proj``: returns the six modulation chunks of the current key from precomputed tables.

    The tables are non-persistent buffers, so moving the block moves them and they never enter a state dict.
    """

    def __init__(self, state: FoldState, tables: Sequence[torch.Tensor], hidden_size: int) -> None:
        super().__init__()
        self.state = state
        self.hidden_size = hidden_size
        self.count = len(tables)
        for position, table in enumerate(tables):
            self.register_buffer(f"table{position}", table, persistent=False)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        position = self.state.current
        if position is None:
            raise RuntimeError("FoldedAdaLN ran outside a transformer forward: no key was selected.")
        table: torch.Tensor = getattr(self, f"table{position}")
        if table.shape[0] != temb.shape[0]:
            raise ValueError(f"The fold table has {table.shape[0]} timestep rows, the forward has {temb.shape[0]}.")
        return table.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


@dataclass(frozen=True)
class FoldHandle:
    """What :func:`install_fold` returns: the shared state and the transformer's pre-hook."""

    state: FoldState
    hook: RemovableHandle

    def remove(self) -> None:
        """Remove the pre-hook. The folded projections stay installed; they cannot be reverted."""
        self.hook.remove()


def install_fold(transformer: nn.Module, tables: Sequence[Mapping[Key, torch.Tensor]]) -> FoldHandle:
    """Replace each block's ``adaln_proj`` with a folded projection over ``tables[block][key]``, selecting the key before each forward.

    Raises:
        ValueError: If *tables* does not hold one equally keyed mapping per block.
    """
    blocks = transformer.transformer_blocks
    if len(tables) != len(blocks):
        raise ValueError(f"Got tables for {len(tables)} blocks, the transformer has {len(blocks)}.")
    keys = list(tables[0])
    if any(list(block_tables) != keys for block_tables in tables):
        raise ValueError("Every block must have tables for the same keys in the same order.")
    state = FoldState(keys)
    hidden = int(transformer.config.hidden_size)
    for block, block_tables in zip(blocks, tables, strict=True):
        block.adaln_proj = FoldedAdaLN(state, [block_tables[key] for key in keys], hidden)

    def select(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        state.select(kwargs["timestep"] if "timestep" in kwargs else args[3])

    return FoldHandle(state, transformer.register_forward_pre_hook(select, with_kwargs=True))


# --- table files: ``k<id>.b<block>`` tensors, with the key list in the metadata ---------------------------------------


def write_tables(path: Path, per_block: Sequence[Mapping[Key, torch.Tensor]]) -> dict[str, int]:
    """Write the per-block tables to *path* (CPU copies, one tensor per key and block) and return the counts and byte size.

    The keys are stored sorted, as lists, in the file's metadata; the tensor ``k0003.b05`` is key 3 of block 5.

    Raises:
        ValueError: If the blocks do not all have the same keys, or there are none.
    """
    from safetensors.torch import save_file

    if not per_block:
        raise ValueError("At least one block of tables is required.")
    keys = sorted(per_block[0])
    if any(sorted(block) != keys for block in per_block):
        raise ValueError("Every block must have tables for the same keys.")
    tensors = {f"k{index:04d}.b{block:02d}": tables[key].detach().cpu().contiguous() for index, key in enumerate(keys) for block, tables in enumerate(per_block)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"keys": json.dumps([list(key) for key in keys]), "blocks": str(len(per_block))})
    return {"keys": len(keys), "blocks": len(per_block), "bytes": path.stat().st_size}


def table_keys(path: Path) -> list[Key]:
    """Return the keys stored in the table file *path*, in id order."""
    from zvisiongenerator.utils.safetensors_stream import read_metadata

    return [tuple(float(value) for value in key) for key in json.loads(read_metadata(path)["keys"])]


def read_tables(path: Path, wanted: Sequence[Key], blocks: int) -> list[dict[Key, torch.Tensor]]:
    """Read the tables of the *wanted* keys for each of *blocks* blocks from *path*, and nothing else.

    Raises:
        KeyError: If a wanted key was not built into the file (rebuild the H3 copy).
        ValueError: If the file holds a different number of blocks.
    """
    from zvisiongenerator.utils.safetensors_stream import TensorFile, read_metadata

    stored_blocks = int(read_metadata(path)["blocks"])
    if stored_blocks != blocks:
        raise ValueError(f"{path.name} holds tables for {stored_blocks} blocks, the transformer has {blocks}.")
    ids = {key: index for index, key in enumerate(table_keys(path))}
    missing = [key for key in wanted if key not in ids]
    if missing:
        raise KeyError(f"{path.name} has no table for {len(missing)} timestep keys, e.g. {missing[0]}; rebuild the H3 copy for this schedule.")
    per_block: list[dict[Key, torch.Tensor]] = [{} for _ in range(blocks)]
    with TensorFile(path) as handle:
        for block in range(blocks):
            for key in wanted:
                per_block[block][key] = handle.read(f"k{ids[key]:04d}.b{block:02d}")
    return per_block
