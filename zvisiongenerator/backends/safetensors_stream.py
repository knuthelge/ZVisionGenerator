"""Read and write safetensors files one tensor at a time, keeping them out of the page cache.

``safetensors.safe_open`` maps the whole file, and the kernel keeps every page it touched cached until the handle
closes. A single checkpoint can be larger than system memory, and reclaiming that cache under load is the memory
pressure that gets processes killed on Linux. :func:`iter_tensors` reads each tensor with plain file reads instead and
drops its byte range from the cache right after; :class:`ShardWriter` writes shards of bounded size and flushes
each one out of the cache once written.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.backends.memory_cuda import drop_cached_file

if TYPE_CHECKING:
    import torch

__all__ = ["ShardWriter", "iter_tensors", "read_header", "shard_files"]

_HEADER_LENGTH_BYTES = 8
_DEFAULT_SHARD_BYTES = 2 * 1024**3


def read_header(path: Path) -> tuple[int, dict[str, dict]]:
    """Return the byte offset where *path*'s tensor data starts, and its header without the metadata entry.

    Raises:
        ValueError: If *path* is not a safetensors file.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) != _HEADER_LENGTH_BYTES:
            raise ValueError(f"{path} is not a safetensors file.")
        (length,) = struct.unpack("<Q", prefix)
        try:
            header = json.loads(handle.read(length))
        except ValueError as exc:
            raise ValueError(f"{path} is not a safetensors file.") from exc
    header.pop("__metadata__", None)
    return _HEADER_LENGTH_BYTES + length, header


def iter_tensors(path: Path, wanted: Callable[[str], bool] | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for each tensor in *path* that *wanted* accepts, in file order.

    Each tensor is read into memory of its own (never a view of the file), and its bytes leave the page cache
    as soon as it is read.

    Raises:
        ValueError: If the file holds a dtype this reader does not know.
    """
    import torch

    data_start, header = read_header(path)
    entries = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
    with open(path, "rb") as handle:
        fd = handle.fileno()
        for name, info in entries:
            if wanted is not None and not wanted(name):
                continue
            begin, end = info["data_offsets"]
            dtype = _dtype(info["dtype"], path)
            buffer = bytearray(end - begin)
            handle.seek(data_start + begin)
            if handle.readinto(buffer) != len(buffer):
                raise ValueError(f"{path} ended inside tensor {name}.")
            _drop_range(fd, data_start + begin, end - begin)
            tensor = torch.frombuffer(buffer, dtype=dtype) if buffer else torch.empty(0, dtype=dtype)
            yield name, tensor.reshape(info["shape"])


def shard_files(directory: Path, index_name: str) -> list[Path]:
    """Return the shard files *directory*'s index (named *index_name*) lists, in name order.

    Raises:
        FileNotFoundError: If the index is missing.
    """
    index = json.loads((directory / index_name).read_text(encoding="utf-8"))
    return [directory / name for name in sorted(set(index["weight_map"].values()))]


def _dtype(name: str, path: Path) -> torch.dtype:
    """Return the torch dtype of safetensors dtype *name*."""
    import torch

    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BOOL": torch.bool,
    }
    if name not in dtypes:
        raise ValueError(f"{path} holds tensors of dtype {name}, which cannot be read here.")
    return dtypes[name]


def _drop_range(fd: int, offset: int, length: int) -> None:
    """Drop a byte range of an open file from the page cache, where the platform can."""
    if hasattr(os, "posix_fadvise"):  # Linux only
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass  # the cache is only a memory aid


class ShardWriter:
    """Write tensors into numbered safetensors shards of bounded size, with a diffusers-style index.

    Tensors are held only until their shard is full. Each shard is flushed to disk and dropped from the page
    cache once written. Call :meth:`close` to write the last shard and the index.
    """

    def __init__(self, directory: Path, prefix: str, index_name: str, max_shard_bytes: int = _DEFAULT_SHARD_BYTES) -> None:
        self._directory = directory
        self._prefix = prefix
        self._index_name = index_name
        self._max_shard_bytes = max_shard_bytes
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._shard_names: list[str] = []
        self._weight_map: dict[str, int] = {}
        self._total_bytes = 0
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, tensors: dict[str, torch.Tensor]) -> None:
        """Queue *tensors* together in one shard, writing the current shard first when they would grow it past the limit.

        Tensors that must be read back together (a quantized weight and its quantization state) are added in one call.
        """
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if self._pending and self._pending_bytes + size > self._max_shard_bytes:
            self._flush()
        self._pending.update({name: tensor.contiguous() for name, tensor in tensors.items()})
        self._pending_bytes += size

    def close(self) -> None:
        """Write the last shard and the index that maps every tensor to its shard file."""
        if self._pending or not self._shard_names:
            self._flush()
        count = len(self._shard_names)
        files = [f"{self._prefix}-{number + 1:05d}-of-{count:05d}.safetensors" for number in range(count)]
        for temporary, final in zip(self._shard_names, files, strict=True):
            (self._directory / temporary).rename(self._directory / final)
        index = {"metadata": {"total_size": self._total_bytes}, "weight_map": {name: files[shard] for name, shard in sorted(self._weight_map.items())}}
        (self._directory / self._index_name).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    def _flush(self) -> None:
        """Write the pending tensors as the next shard and release them."""
        from safetensors.torch import save_file

        name = f"{self._prefix}-part-{len(self._shard_names):05d}.safetensors"
        path = self._directory / name
        save_file(self._pending, str(path), metadata={"format": "pt"})
        drop_cached_file(path)
        for tensor_name in self._pending:
            self._weight_map[tensor_name] = len(self._shard_names)
        self._shard_names.append(name)
        self._total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0
