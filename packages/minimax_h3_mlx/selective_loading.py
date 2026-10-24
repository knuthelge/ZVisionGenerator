"""Lossless, key-selective safetensors loading helpers.

The low-memory H3 path streams one DiT block at a time.  The current MLX loader
uses ``mx.load(shard)`` and then filters the returned dictionary to the keys for
that block.  That is numerically safe, but it leaves an open question for 24GB
Macs: does opening a multi-gigabyte shard induce avoidable page-in and file-cache
pressure when the caller only needs a few tensors?

This module is a tiny lossless prototype for the alternative: use the
safetensors index to open only shards that contain requested keys, then read only
those exact tensors by key.  It does not dequantize, rewrite, reshard, download,
or approximate weights.  Returned arrays contain the exact serialized tensor
values for the requested keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from safetensors import safe_open

_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    """Header metadata for one tensor selected from a safetensors shard."""

    key: str
    shard: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class ShardSelection:
    """The subset of keys selected from one physical shard."""

    shard: str
    path: str
    shard_size_bytes: int
    selected_tensor_count: int
    selected_tensor_bytes: int
    keys: tuple[str, ...]


def index_path(model_dir: str | Path) -> Path:
    """Return the standard safetensors index path for a model directory."""

    return Path(model_dir) / "model.safetensors.index.json"


def load_weight_map(model_dir: str | Path) -> dict[str, str]:
    """Load ``weight_map`` from ``model.safetensors.index.json``."""

    path = index_path(model_dir)
    with path.open() as handle:
        raw = json.load(handle)
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path} does not contain a non-empty weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def _dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_NBYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc


def _numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _tensor_header(path: Path, key: str) -> tuple[tuple[int, ...], str, int]:
    with safe_open(str(path), framework="np") as handle:
        if key not in handle.keys():
            raise KeyError(f"tensor {key!r} is not present in {path}")
        view = handle.get_slice(key)
        shape = tuple(int(dim) for dim in view.get_shape())
        dtype = str(view.get_dtype())
    return shape, dtype, _numel(shape) * _dtype_nbytes(dtype)


def _read_safetensors_payload(path: Path, key: str) -> tuple[dict[str, Any], bytes]:
    """Read one tensor's raw serialized payload from a safetensors file."""

    with path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"{path} is too small to contain a safetensors header")
        header_len = int.from_bytes(header_len_raw, "little")
        header = json.loads(handle.read(header_len))
        record = header.get(key)
        if not isinstance(record, dict):
            raise KeyError(f"tensor {key!r} is not present in {path}")
        offsets = record.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"tensor {key!r} in {path} has invalid data_offsets")
        start, end = int(offsets[0]), int(offsets[1])
        if end < start:
            raise ValueError(f"tensor {key!r} in {path} has negative payload length")
        handle.seek(8 + header_len + start)
        payload = handle.read(end - start)
    if len(payload) != end - start:
        raise ValueError(f"tensor {key!r} in {path} ended before its declared payload")
    return record, payload


def _load_bf16_mlx_tensor(path: Path, key: str) -> Any:
    """Load one BF16 tensor as MLX by preserving its raw uint16 bit pattern."""

    import mlx.core as mx

    record, payload = _read_safetensors_payload(path, key)
    shape = tuple(int(dim) for dim in record.get("shape", ()))
    dtype = str(record.get("dtype"))
    if dtype != "BF16":
        raise ValueError(f"tensor {key!r} in {path} is {dtype}, not BF16")
    expected = _numel(shape) * _dtype_nbytes(dtype)
    if len(payload) != expected:
        raise ValueError(f"tensor {key!r} in {path} has {len(payload)} bytes, expected {expected}")
    bits = np.frombuffer(payload, dtype="<u2").reshape(shape)
    return mx.array(bits, dtype=mx.uint16).view(mx.bfloat16)


def build_tensor_plan(
    model_dir: str | Path,
    keys: Iterable[str],
    *,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable plan for exact-key loading.

    Args:
        model_dir: Directory with ``model.safetensors.index.json`` and shards.
        keys: Tensor keys to load.  Order is preserved and duplicates are
            rejected so byte accounting is unambiguous.
        cwd: Optional base for rendering relative paths in reports.

    Returns:
        A JSON-serializable dictionary containing per-tensor headers,
        per-shard selections, requested bytes, and an upper-bound I/O
        amplification ratio for loaders that effectively touch whole shards.
    """

    root = Path(model_dir)
    render_base = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    requested = [str(key) for key in keys]
    if not requested:
        raise ValueError("at least one tensor key is required")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate tensor keys would make byte accounting ambiguous")

    weight_map = load_weight_map(root)
    missing = [key for key in requested if key not in weight_map]
    if missing:
        raise KeyError(f"unknown tensor key(s): {missing[:5]}")

    specs: list[TensorSpec] = []
    by_shard: dict[str, list[TensorSpec]] = {}
    for key in requested:
        shard = weight_map[key]
        shard_path = root / shard
        shape, dtype, nbytes = _tensor_header(shard_path, key)
        spec = TensorSpec(key=key, shard=shard, shape=shape, dtype=dtype, nbytes=nbytes)
        specs.append(spec)
        by_shard.setdefault(shard, []).append(spec)

    shard_records: list[ShardSelection] = []
    opened_shard_bytes = 0
    for shard, shard_specs in sorted(by_shard.items()):
        shard_path = root / shard
        shard_size = shard_path.stat().st_size
        opened_shard_bytes += shard_size
        try:
            rendered = str(shard_path.resolve().relative_to(render_base))
        except ValueError:
            rendered = str(shard_path)
        shard_records.append(
            ShardSelection(
                shard=shard,
                path=rendered,
                shard_size_bytes=shard_size,
                selected_tensor_count=len(shard_specs),
                selected_tensor_bytes=sum(spec.nbytes for spec in shard_specs),
                keys=tuple(spec.key for spec in shard_specs),
            )
        )

    requested_bytes = sum(spec.nbytes for spec in specs)
    return {
        "schema_version": 1,
        "model_dir": str(root),
        "index_path": str(index_path(root)),
        "tensor_count": len(specs),
        "selected_shard_count": len(shard_records),
        "requested_tensor_bytes": requested_bytes,
        "opened_shard_bytes": opened_shard_bytes,
        "whole_shard_io_amplification_upper_bound": (
            opened_shard_bytes / requested_bytes if requested_bytes else None
        ),
        "tensors": [asdict(spec) for spec in specs],
        "shards": [asdict(record) for record in shard_records],
        "lossless_boundary": {
            "exact_values": True,
            "dequantizes": False,
            "rewrites_or_reshards": False,
            "loads_unrequested_tensors": False,
        },
    }


def _keys_by_shard_from_plan(plan: Mapping[str, Any]) -> dict[str, list[str]]:
    by_shard: dict[str, list[str]] = {}
    for tensor in plan["tensors"]:
        by_shard.setdefault(str(tensor["shard"]), []).append(str(tensor["key"]))
    return by_shard


def load_selected_tensors(model_dir: str | Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Load exactly the requested tensors from indexed safetensors shards.

    The returned dictionary has the same keys as requested and NumPy arrays with
    the serialized values.  No tensor outside ``keys`` is materialized by this
    function.
    """

    root = Path(model_dir)
    requested = [str(key) for key in keys]
    plan = build_tensor_plan(root, requested)
    by_shard = _keys_by_shard_from_plan(plan)

    loaded: dict[str, np.ndarray] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(root / shard), framework="np") as handle:
            for key in shard_keys:
                loaded[key] = np.asarray(handle.get_tensor(key))
    return {key: loaded[key] for key in requested}


def load_selected_mlx_tensors(model_dir: str | Path, keys: Iterable[str]) -> dict[str, Any]:
    """Load exactly the requested tensors as MLX arrays.

    This is the integration path for ``QuantizedBlockProvider``.  It keeps the
    same explicit-key plan and ordering as :func:`load_selected_tensors`.  It
    asks safetensors for MLX arrays directly where supported and preserves BF16
    tensors with a raw uint16 bit-view fallback for real H3 block weights.
    """

    # Importing mlx.core first initializes the top-level ``mlx.core`` attribute
    # that safetensors' MLX framework adapter expects.
    import mlx.core as _mx  # noqa: F401

    root = Path(model_dir)
    requested = [str(key) for key in keys]
    plan = build_tensor_plan(root, requested)
    by_shard = _keys_by_shard_from_plan(plan)

    dtype_by_key = {str(tensor["key"]): str(tensor["dtype"]) for tensor in plan["tensors"]}
    loaded: dict[str, Any] = {}
    for shard, shard_keys in by_shard.items():
        shard_path = root / shard
        non_bf16 = [key for key in shard_keys if dtype_by_key[key] != "BF16"]
        for key in shard_keys:
            if dtype_by_key[key] == "BF16":
                loaded[key] = _load_bf16_mlx_tensor(shard_path, key)
        if non_bf16:
            with safe_open(str(shard_path), framework="mlx") as handle:
                for key in non_bf16:
                    loaded[key] = handle.get_tensor(key)
    return {key: loaded[key] for key in requested}


def fingerprint_array(array: np.ndarray) -> dict[str, Any]:
    """Return a stable byte fingerprint for a NumPy array."""

    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(dim) for dim in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "nbytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def fingerprint_tensors(tensors: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Fingerprint a tensor dictionary returned by a loader."""

    return {key: fingerprint_array(value) for key, value in tensors.items()}


__all__ = [
    "TensorSpec",
    "ShardSelection",
    "build_tensor_plan",
    "fingerprint_array",
    "fingerprint_tensors",
    "index_path",
    "load_selected_mlx_tensors",
    "load_selected_tensors",
    "load_weight_map",
]
