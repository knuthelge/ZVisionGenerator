"""Give a loader a LoRA file under the tensor names its model family reads.

:func:`normalise_lora_keys` decides the names. This module applies them for the two kinds of loader:

* A loader that takes tensors (diffusers' ``load_lora_weights``, the MOSS and MiniMax H3 loaders) gets the file's
  tensors under their new names from :func:`normalised_lora_state`, with ``alpha`` folded into the weights where the
  converter cannot take it.
* A loader that takes only a path (mflux, ``ltx-pipelines-mlx``) gets a copy of the file whose header carries the new
  names from :func:`readable_lora_files`. A rename moves no tensor data, so the copy is the same bytes under a new
  header and the file's metadata, dtypes and ``alpha`` tensors are kept. The copy lives in the data folder's
  ``cache/loras/``, in a folder of its own under the source's file name, only for as long as its caller needs it: it is removed when the caller's ``with`` block ends, never
  reused by a later job, and a copy a crash left behind is removed the next time a LoRA is loaded. Each copy is locked
  by the process that made it for as long as it exists (``flock``), which is how that sweep tells a copy in use from
  one left behind.

A file that already has the names its family reads is never touched: both functions leave it to the loader as it is.
This module imports torch only when it reads tensors.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib
import json
import os
from pathlib import Path
import shutil
import struct
from typing import IO, TYPE_CHECKING
import uuid

from zvisiongenerator.utils.lora_keys import folds_alpha, normalise_lora_keys
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.safetensors_stream import read_header, read_metadata

try:
    import fcntl
except ImportError:  # Windows has no flock; only macOS makes copies, and without flock nothing is swept
    fcntl = None

if TYPE_CHECKING:
    import torch

__all__ = ["normalised_lora_state", "readable_lora_files"]

_HEADER_ALIGNMENT = 8
_COPY_CHUNK_BYTES = 8 * 1024**2


def normalised_lora_state(path: str | Path, family: str, loader: str = "diffusers") -> dict[str, torch.Tensor] | None:
    """Return the tensors of the LoRA at *path* under the names *loader* reads for *family*, or None when the file needs no change.

    None also means *path* is not a readable ``.safetensors`` file, which the loader reports in its own way.

    Args:
        path: The LoRA file.
        family: The family of the model the LoRA is loaded on.
        loader: What takes the tensors: ``diffusers``, ``moss`` or ``h3`` (see :func:`~zvisiongenerator.utils.lora_keys.normalise_lora_keys`).

    Returns:
        The tensors, or None when the loader can read the file as it is. Where the loader cannot take an ``alpha``
        tensor, each is multiplied into its layer's ``lora_A`` (``alpha / rank``) and removed.
    """
    names = _tensor_names(path)
    if names is None:
        return None
    renames = normalise_lora_keys(names, family, loader=loader)
    fold = folds_alpha(family, loader=loader)
    if not renames and not (fold and any(name.endswith(".alpha") for name in names)):
        return None
    from safetensors.torch import load_file

    state = {renames.get(name, name): tensor for name, tensor in load_file(str(path)).items()}
    return _fold_alpha(state) if fold else state


@contextlib.contextmanager
def readable_lora_files(paths: Sequence[str | Path], family: str, loader: str, folder: Path | None = None) -> Iterator[list[str]]:
    """Yield the paths to hand *loader* for the LoRAs at *paths*: each file it cannot read as it is becomes a renamed copy, removed when the block ends.

    A copy keeps the file's metadata and tensors, ``alpha`` included: the loaders that take only a path apply
    ``alpha / rank`` themselves. A file the loader reads as it is is never copied. Every call makes its own copies, so a
    copy is never shared with another job; copies a crashed process left behind are removed first.

    Args:
        paths: The LoRA files.
        family: The family of the model the LoRAs are loaded on.
        loader: What reads the paths: ``mflux`` or ``mlx`` (see :func:`~zvisiongenerator.utils.lora_keys.normalise_lora_keys`).
        folder: Where copies are made (default: ``cache/loras`` in the data folder).

    Yields:
        One path per entry of *paths*, in order.
    """
    if not paths:
        yield []
        return
    folder = folder if folder is not None else get_ziv_data_dir() / "cache" / "loras"
    _sweep_abandoned_copies(folder)
    with contextlib.ExitStack() as copies:
        yield [_readable_path(path, family, loader, folder, copies) for path in paths]


def _readable_path(path: str | Path, family: str, loader: str, folder: Path, copies: contextlib.ExitStack) -> str:
    """Return *path*, or a renamed copy of it that *copies* removes when it closes, when *loader* cannot read the names in it."""
    names = _tensor_names(path)
    if names is None:
        return str(path)
    renames = normalise_lora_keys(names, family, loader=loader)
    if not renames:
        return str(path)
    return str(copies.enter_context(_renamed_copy(Path(path), folder, renames)))


@contextlib.contextmanager
def _renamed_copy(source: Path, folder: Path, renames: dict[str, str]) -> Iterator[Path]:
    """Yield a new file under *folder*: *source* with its tensors renamed by *renames*; it is locked while it exists and removed on exit.

    Each copy gets a folder of its own and keeps the source's file name, so a loader that logs the path names the LoRA.
    """
    own_folder = folder / uuid.uuid4().hex
    own_folder.mkdir(parents=True)
    target = own_folder / source.name
    temporary = own_folder / f".{source.name}.tmp"
    try:
        with open(temporary, "wb") as writer:
            _lock(writer)  # held until the file is closed, through the rename: the sweep never sees it unlocked
            _write_renamed(source, writer, renames)
            os.replace(temporary, target)
            yield target
    finally:
        target.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            own_folder.rmdir()


def _lock(handle: IO[bytes]) -> None:
    """Lock *handle*'s file for as long as it stays open; nothing where the platform or the file system has no ``flock``."""
    if fcntl is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _sweep_abandoned_copies(folder: Path) -> None:
    """Remove the copies under *folder* that no process holds the lock of, and their emptied folders: what a crashed or killed process left behind."""
    if fcntl is None or not folder.is_dir():
        return
    for own_folder in folder.iterdir():
        if not own_folder.is_dir():
            continue
        for path in own_folder.iterdir():
            try:
                with open(path, "rb") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises while the process that made it is alive
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        with contextlib.suppress(OSError):
            own_folder.rmdir()


def _tensor_names(path: str | Path) -> list[str] | None:
    """Return the tensor names in the safetensors header of *path*, or None when it is not a readable ``.safetensors`` file."""
    source = Path(path)
    if source.suffix.lower() != ".safetensors" or not source.is_file():
        return None
    try:
        return list(read_header(source)[1])
    except OSError, ValueError:
        return None


def _fold_alpha(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return *state* with each layer's ``alpha / rank`` multiplied into its ``lora_A`` and its ``alpha`` removed.

    An ``alpha`` with no ``lora_A`` and ``lora_B`` beside it (a LoKr layer's) stays.
    """
    for key in [name for name in state if name.endswith(".alpha")]:
        module = key.removesuffix(".alpha")
        down_key = f"{module}.lora_A.weight"
        if down_key in state and f"{module}.lora_B.weight" in state:
            down = state[down_key]
            state[down_key] = down * (float(state.pop(key)) / down.shape[0])
    return state


def _write_renamed(source: Path, writer: IO[bytes], renames: dict[str, str]) -> None:
    """Write *source* to *writer* with its tensors renamed by *renames*: a new header, and the data bytes as they are."""
    data_start, header = read_header(source)
    entries: dict[str, object] = {}
    metadata = read_metadata(source)
    if metadata:
        entries["__metadata__"] = metadata
    entries.update({renames.get(name, name): entry for name, entry in header.items()})
    blob = json.dumps(entries, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % _HEADER_ALIGNMENT)
    writer.write(struct.pack("<Q", len(blob)) + blob)
    with open(source, "rb") as reader:
        reader.seek(data_start)
        shutil.copyfileobj(reader, writer, _COPY_CHUNK_BYTES)
    writer.flush()
