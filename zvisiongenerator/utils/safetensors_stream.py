"""Read and write safetensors files one tensor at a time, keeping them out of the page cache.

``safetensors.safe_open`` maps the whole file, and the kernel keeps every page it touched cached until the handle
closes. A single checkpoint can be larger than system memory, and reclaiming that cache under load is the memory
pressure that gets processes killed on Linux. :func:`iter_tensors` reads each tensor with plain file reads instead and
drops its byte range from the cache right after; :class:`TensorFile` does the same for tensors read by name;
:class:`ShardWriter` writes shards of bounded size and flushes each one out of the cache once written;
:class:`SingleFileWriter` writes one file of any size, its header first and then each tensor as it comes.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

from zvisiongenerator.utils.page_cache import drop_cached_file

if TYPE_CHECKING:
    import torch

__all__ = ["ShardWriter", "SingleFileWriter", "TensorFile", "iter_tensors", "read_header", "read_metadata", "shard_files"]

_HEADER_LENGTH_BYTES = 8
_DEFAULT_SHARD_BYTES = 2 * 1024**3
# Bytes per element of each safetensors dtype the writer accepts.
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "I16": 2, "U32": 4, "I32": 4, "I64": 8}
# The data that follows the header starts on a multiple of this.
_HEADER_ALIGNMENT = 8


def read_header(path: Path) -> tuple[int, dict[str, dict]]:
    """Return the byte offset where *path*'s tensor data starts, and its header without the metadata entry.

    Raises:
        ValueError: If *path* is not a safetensors file.
    """
    data_start, header = _raw_header(path)
    header.pop("__metadata__", None)
    return data_start, header


def read_metadata(path: Path) -> dict[str, str]:
    """Return the ``__metadata__`` entry of *path*'s header (string keys and values), empty when it has none.

    Raises:
        ValueError: If *path* is not a safetensors file.
    """
    metadata = _raw_header(path)[1].get("__metadata__")
    return metadata if isinstance(metadata, dict) else {}


def _raw_header(path: Path) -> tuple[int, dict]:
    """Return the byte offset where *path*'s tensor data starts, and its whole header."""
    with open(path, "rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) != _HEADER_LENGTH_BYTES:
            raise ValueError(f"{path} is not a safetensors file.")
        (length,) = struct.unpack("<Q", prefix)
        try:
            header = json.loads(handle.read(length))
        except ValueError as exc:
            raise ValueError(f"{path} is not a safetensors file.") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} is not a safetensors file.")
    return _HEADER_LENGTH_BYTES + length, header


def iter_tensors(path: Path, wanted: Callable[[str], bool] | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for each tensor in *path* that *wanted* accepts, in file order.

    Each tensor is read into memory of its own (never a view of the file), and its bytes leave the page cache
    as soon as it is read.

    Raises:
        ValueError: If the file holds a dtype this reader does not know.
    """
    data_start, header = read_header(path)
    entries = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
    with open(path, "rb") as handle:
        for name, info in entries:
            if wanted is not None and not wanted(name):
                continue
            yield name, _read_tensor(handle, path, data_start, name, info)


class TensorFile:
    """Read single tensors of one safetensors file by name, in any order, keeping them out of the page cache.

    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data_start, self.header = read_header(path)
        self._handle = open(path, "rb")

    def read(self, name: str) -> torch.Tensor:
        """Return the tensor *name* in memory of its own.

        Raises:
            KeyError: If the file has no tensor *name*.
        """
        return _read_tensor(self._handle, self._path, self._data_start, name, self.header[name])

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def __enter__(self) -> TensorFile:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _read_tensor(handle: IO[bytes], path: Path, data_start: int, name: str, info: dict) -> torch.Tensor:
    """Read the tensor *name* described by its header entry *info* from the open file *handle*, then drop its bytes from the page cache."""
    import torch

    begin, end = info["data_offsets"]
    dtype = _dtype(info["dtype"], path)
    buffer = bytearray(end - begin)
    handle.seek(data_start + begin)
    if handle.readinto(buffer) != len(buffer):
        raise ValueError(f"{path} ended inside tensor {name}.")
    _drop_range(handle.fileno(), data_start + begin, end - begin)
    tensor = torch.frombuffer(buffer, dtype=dtype) if buffer else torch.empty(0, dtype=dtype)
    return tensor.reshape(info["shape"])


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


class SingleFileWriter:
    """Write one safetensors file whose tensors are all known up front, holding only the tensor being written.

    The header, with every tensor's offsets, is written first; then each tensor's bytes are passed to
    :meth:`write` in the declared order. Call :meth:`close` (or leave the ``with`` block) once all are written. A
    file left incomplete by an error has a header promising data it lacks, so write it where it is discarded on
    failure.
    """

    def __init__(self, path: Path, entries: Sequence[tuple[str, str, list[int]]]) -> None:
        """Create *path* and write its header for *entries*: ``(name, safetensors dtype, shape)``, in the order they will be written.

        Raises:
            ValueError: If a dtype is not one the writer knows, or a name repeats.
        """
        self._path = path
        self._sizes: list[tuple[str, int]] = []
        header: dict[str, dict] = {}
        offset = 0
        for name, dtype, shape in entries:
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"Cannot write {name} as {dtype}.")
            if name in header:
                raise ValueError(f"{name} is listed twice.")
            size = _numel(shape) * _DTYPE_BYTES[dtype]
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
            self._sizes.append((name, size))
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-(_HEADER_LENGTH_BYTES + len(encoded)) % _HEADER_ALIGNMENT)
        self._handle = open(path, "wb")
        self._handle.write(struct.pack("<Q", len(encoded)) + encoded)
        self._next = 0

    def write(self, name: str, data: memoryview | bytes | bytearray | object) -> None:
        """Write the bytes of tensor *name* (anything exposing the buffer protocol, e.g. a NumPy array), the next one declared.

        Raises:
            ValueError: If *name* is not the next tensor declared, or *data* is not its size.
        """
        if self._next >= len(self._sizes) or self._sizes[self._next][0] != name:
            expected = self._sizes[self._next][0] if self._next < len(self._sizes) else "nothing"
            raise ValueError(f"Wrote {name} where {expected} comes next.")
        view = memoryview(data).cast("B")  # type: ignore[arg-type]
        if view.nbytes != self._sizes[self._next][1]:
            raise ValueError(f"{name} is {view.nbytes} bytes; its header says {self._sizes[self._next][1]}.")
        self._handle.write(view)
        self._next += 1

    def close(self) -> None:
        """Close the file, then flush it out of the page cache.

        Raises:
            ValueError: If a declared tensor was never written.
        """
        if self._handle.closed:
            return
        self._handle.close()
        if self._next != len(self._sizes):
            raise ValueError(f"{self._path.name} is missing {self._sizes[self._next][0]} and the tensors after it.")
        drop_cached_file(self._path)

    def __enter__(self) -> SingleFileWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_exc: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._handle.close()


def _numel(shape: Sequence[int]) -> int:
    """Return the element count of *shape* (1 for a scalar)."""
    count = 1
    for size in shape:
        count *= size
    return count
