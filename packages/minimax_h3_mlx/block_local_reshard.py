"""Lossless block-local safetensors layout builder for streamed DiT blocks.

The streamed MiniMax-H3 DiT provider already loads by checkpoint key through
``mx.load``.  This module keeps that math path unchanged and changes only the
physical safetensors layout: selected ``blocks.N.*`` tensors are copied byte for
byte into small block-local shards, while the generated index preserves the
original tensor keys and uses only relative shard paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .selective_loading import load_weight_map

_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.")
REQUIRED_TRANSFORMER_FILES = ("config.json", "quant_config.json")


@dataclass(frozen=True)
class CopiedTensorRecord:
    """One tensor payload copied into the derived block-local layout."""

    key: str
    source_shard: str
    target_shard: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    payload_sha256: str


@dataclass(frozen=True)
class WrittenShardRecord:
    """One safetensors file written by the resharder."""

    shard: str
    tensor_count: int
    payload_bytes: int
    file_size_bytes: int
    keys: tuple[str, ...]


@dataclass(frozen=True)
class BlockLocalLayoutResult:
    """Machine-readable summary of a block-local resharding operation."""

    schema_version: int
    source_dir: str
    output_dir: str
    block_indices: tuple[int, ...]
    include_static: bool
    split_adaln: bool
    copied_config_files: tuple[str, ...]
    tensor_count: int
    payload_bytes: int
    index_path: str
    manifest_path: str
    shards: tuple[WrittenShardRecord, ...]
    tensors: tuple[CopiedTensorRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "block_indices": list(self.block_indices),
            "include_static": self.include_static,
            "split_adaln": self.split_adaln,
            "copied_config_files": list(self.copied_config_files),
            "tensor_count": self.tensor_count,
            "payload_bytes": self.payload_bytes,
            "index_path": self.index_path,
            "manifest_path": self.manifest_path,
            "shards": [asdict(record) for record in self.shards],
            "tensors": [asdict(record) for record in self.tensors],
            "path_policy": {
                "index_weight_map_values_are_relative": True,
                "no_symlinks_created": True,
            },
            "lossless_boundary": {
                "preserves_tensor_keys": True,
                "copies_serialized_tensor_payload_bytes": True,
                "dequantizes": False,
                "changes_model_math": False,
            },
        }


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too small to contain a safetensors header")
        header_len = int.from_bytes(raw_len, "little")
        payload_limit = path.stat().st_size - 8
        if header_len < 0 or header_len > payload_limit:
            raise ValueError(f"{path} has an invalid safetensors header length {header_len}")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path} ended before its declared safetensors header")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not a JSON object")
    return header


def _read_tensor_record_and_payload(path: Path, key: str) -> tuple[dict[str, Any], bytes]:
    header = _read_safetensors_header(path)
    record = header.get(key)
    if not isinstance(record, dict):
        raise KeyError(f"tensor {key!r} is not present in {path}")
    offsets = record.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"tensor {key!r} in {path} has invalid data_offsets")
    start, end = int(offsets[0]), int(offsets[1])
    if start < 0 or end < start:
        raise ValueError(f"tensor {key!r} in {path} has invalid byte range {offsets!r}")
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        header_len = int.from_bytes(raw_len, "little")
        handle.seek(8 + header_len + start)
        payload = handle.read(end - start)
    if len(payload) != end - start:
        raise ValueError(f"tensor {key!r} in {path} ended before its declared payload")
    copied = {name: value for name, value in record.items() if name != "data_offsets"}
    return copied, payload


def _write_raw_safetensors(path: Path, entries: Sequence[tuple[str, dict[str, Any], bytes]]) -> WrittenShardRecord:
    if not entries:
        raise ValueError(f"refusing to write empty safetensors shard {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    offset = 0
    keys: list[str] = []
    payload_bytes = 0
    for key, source_record, payload in entries:
        if key in header:
            raise ValueError(f"duplicate tensor key {key!r} for {path}")
        record = dict(source_record)
        end = offset + len(payload)
        record["data_offsets"] = [offset, end]
        header[key] = record
        offset = end
        payload_bytes += len(payload)
        keys.append(key)

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(len(header_bytes).to_bytes(8, "little"))
            handle.write(header_bytes)
            for _, _, payload in entries:
                handle.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return WrittenShardRecord(
        shard=path.as_posix(),
        tensor_count=len(entries),
        payload_bytes=payload_bytes,
        file_size_bytes=path.stat().st_size,
        keys=tuple(keys),
    )


def infer_block_indices(weight_map: Mapping[str, str]) -> tuple[int, ...]:
    """Return sorted block indices present in a transformer weight map."""

    indices = {int(match.group(1)) for key in weight_map for match in [_BLOCK_KEY_RE.match(key)] if match}
    return tuple(sorted(indices))


def block_tensor_keys(
    weight_map: Mapping[str, str],
    block_index: int,
    *,
    include_adaln: bool = True,
    adaln_only: bool = False,
) -> tuple[str, ...]:
    """Return indexed source keys for one block using provider-compatible filters."""

    prefix = f"blocks.{block_index}."
    keys = [key for key in weight_map if key.startswith(prefix)]
    if adaln_only:
        keys = [key for key in keys if ".adaln_proj." in key]
    elif not include_adaln:
        keys = [key for key in keys if ".adaln_proj." not in key]
    return tuple(keys)


def _group_name_for_key(key: str, *, split_adaln: bool) -> str:
    if split_adaln and ".adaln_proj." in key:
        return "adaln"
    return "core"


def _target_block_shard(block_index: int, group_name: str) -> str:
    return f"blocks/block-{block_index:03d}-{group_name}.safetensors"


def _target_static_shard(source_shard: str) -> str:
    return f"static/{Path(source_shard).name}"


def _copy_required_transformer_files(source_dir: Path, output_dir: Path) -> tuple[str, ...]:
    copied: list[str] = []
    for filename in REQUIRED_TRANSFORMER_FILES:
        src = source_dir / filename
        if not src.is_file():
            raise FileNotFoundError(f"required transformer file is missing: {src}")
        dst = output_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(filename)
    return tuple(copied)


def _copy_payload_group(
    source_dir: Path,
    output_dir: Path,
    weight_map: Mapping[str, str],
    target_shard: str,
    keys: Sequence[str],
) -> tuple[WrittenShardRecord, tuple[CopiedTensorRecord, ...]]:
    entries: list[tuple[str, dict[str, Any], bytes]] = []
    records: list[CopiedTensorRecord] = []
    for key in keys:
        source_shard = weight_map.get(key)
        if source_shard is None:
            raise KeyError(f"source index has no entry for tensor {key!r}")
        if Path(source_shard).is_absolute():
            raise ValueError(f"source index shard path for {key!r} is absolute: {source_shard}")
        source_path = source_dir / source_shard
        source_record, payload = _read_tensor_record_and_payload(source_path, key)
        dtype = str(source_record.get("dtype"))
        shape = tuple(int(dim) for dim in source_record.get("shape", ()))
        entries.append((key, source_record, payload))
        records.append(
            CopiedTensorRecord(
                key=key,
                source_shard=source_shard,
                target_shard=target_shard,
                dtype=dtype,
                shape=shape,
                nbytes=len(payload),
                payload_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    shard_record = _write_raw_safetensors(output_dir / target_shard, entries)
    return (
        WrittenShardRecord(
            shard=target_shard,
            tensor_count=shard_record.tensor_count,
            payload_bytes=shard_record.payload_bytes,
            file_size_bytes=shard_record.file_size_bytes,
            keys=shard_record.keys,
        ),
        tuple(records),
    )


def create_block_local_layout(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    block_indices: Iterable[int] | None = None,
    include_static: bool = False,
    split_adaln: bool = True,
) -> BlockLocalLayoutResult:
    """Create a project-local block-local transformer layout.

    Args:
        source_dir: Existing quantized transformer directory.
        output_dir: New, empty directory to receive config files, shards, index, and manifest.
        block_indices: Block indices to copy. ``None`` means every indexed block.
        include_static: Also copy non-``blocks.*`` tensors into static-only shards, making a
            complete transformer layout when all blocks are selected.
        split_adaln: Put each block's AdaLN tensors in a separate local shard so
            ``adaln_only`` cache construction and steady-state ``include_adaln=False`` loads do
            not force each other through ``mx.load``.
    """

    src = Path(source_dir)
    dst = Path(output_dir)
    if not (src / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"source transformer index is missing: {src / 'model.safetensors.index.json'}")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"output directory already exists and is not empty: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    weight_map = load_weight_map(src)
    selected_blocks = tuple(int(i) for i in (infer_block_indices(weight_map) if block_indices is None else block_indices))
    if not selected_blocks:
        raise ValueError("at least one block index is required")

    copied_files = _copy_required_transformer_files(src, dst)
    new_weight_map: dict[str, str] = {}
    shard_records: list[WrittenShardRecord] = []
    tensor_records: list[CopiedTensorRecord] = []

    for block_index in selected_blocks:
        keys = block_tensor_keys(weight_map, block_index)
        if not keys:
            raise KeyError(f"source index contains no tensors for block {block_index}")
        grouped: dict[str, list[str]] = {}
        for key in keys:
            grouped.setdefault(_group_name_for_key(key, split_adaln=split_adaln), []).append(key)
        for group_name, group_keys in sorted(grouped.items()):
            target_shard = _target_block_shard(block_index, group_name)
            shard_record, copied = _copy_payload_group(src, dst, weight_map, target_shard, group_keys)
            shard_records.append(shard_record)
            tensor_records.extend(copied)
            for key in group_keys:
                new_weight_map[key] = target_shard

    if include_static:
        by_source_shard: dict[str, list[str]] = {}
        for key, source_shard in weight_map.items():
            if not key.startswith("blocks."):
                by_source_shard.setdefault(source_shard, []).append(key)
        for source_shard, keys in sorted(by_source_shard.items()):
            target_shard = _target_static_shard(source_shard)
            shard_record, copied = _copy_payload_group(src, dst, weight_map, target_shard, keys)
            shard_records.append(shard_record)
            tensor_records.extend(copied)
            for key in keys:
                new_weight_map[key] = target_shard

    for key, shard in new_weight_map.items():
        if Path(shard).is_absolute():
            raise ValueError(f"internal error: generated absolute shard path for {key!r}: {shard}")

    index_payload = {
        "metadata": {"total_size": sum(record.nbytes for record in tensor_records)},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    index_path = dst / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

    result = BlockLocalLayoutResult(
        schema_version=1,
        source_dir=str(src),
        output_dir=str(dst),
        block_indices=selected_blocks,
        include_static=include_static,
        split_adaln=split_adaln,
        copied_config_files=copied_files,
        tensor_count=len(tensor_records),
        payload_bytes=sum(record.nbytes for record in tensor_records),
        index_path="model.safetensors.index.json",
        manifest_path="block_local_reshard_manifest.json",
        shards=tuple(shard_records),
        tensors=tuple(tensor_records),
    )
    (dst / result.manifest_path).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return result


__all__ = [
    "BlockLocalLayoutResult",
    "CopiedTensorRecord",
    "WrittenShardRecord",
    "block_tensor_keys",
    "create_block_local_layout",
    "infer_block_indices",
]
