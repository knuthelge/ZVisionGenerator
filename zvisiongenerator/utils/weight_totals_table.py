"""Fill the ``weight_totals`` table of ``model_metadata.yaml`` from the Hugging Face headers (a developer command).

Run it after changing a pin of an image or LTX model, or adding one::

    uv run python -m zvisiongenerator.utils.weight_totals --regenerate [--only TEXT]

It reads each repo the memory verdict needs (see :func:`memory_sources`) at its pinned revision, through the safetensors
headers only, and rewrites the table in place. The developer's Hugging Face token is used, so a gated repo is read when the
account has accepted its licence. A repo that cannot be read is reported and keeps its old row.

Besides what :mod:`weight_totals` documents for a row, the LTX rows hold what the LTX estimates count:

- ``ltx_cuda``: ``connectors`` (the elements of the connectors folder) and ``text_encoder_bytes`` (the FP8 text encoder's bytes), of a diffusers repo.
- ``transformers``: for each transformer, a single file (by name) or the repo's own ``transformer`` folder, ``block_matrices`` (the elements of the 2-D weights inside its blocks) and ``transformer_rest``.
- ``ltx_mlx``: ``connector``, ``transformer``, ``decoders`` and ``upsampler`` (bfloat16 bytes) of a macOS pack, with ``gemma`` when the pack bundles its text encoder; ``gemma`` alone for the Gemma 3 repo.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from zvisiongenerator.downloads.declaration import model_targets
from zvisiongenerator.downloads.target import DownloadTarget, is_literal
from zvisiongenerator.utils.model_files import ltx_mlx_bundled_text_encoder, weight_file_names
from zvisiongenerator.utils.model_memory import ltx_mlx_weights, totals_from_header
from zvisiongenerator.utils.model_metadata import METADATA_FILE, PLATFORM_KEYS, load_model_metadata, model_entry, model_ids
from zvisiongenerator.utils.weight_totals import TABLE_KEY, HubReader, build_row, row_key

__all__ = [
    "IMAGE",
    "LTX_CUDA_REPO",
    "LTX_MLX_GEMMA",
    "LTX_MLX_PACK",
    "LTX_TRANSFORMER_FILE",
    "RowSource",
    "main",
    "memory_sources",
    "regenerate",
    "render_table",
    "write_table",
]

_IMAGE_ROLES = frozenset({"model", "components", "sft"})
_IMAGE_FILE_ROLES = frozenset({"transformer", "turbo_lora"})
_TRANSFORMER = "transformer"
_LTX_CUDA_FOLDERS = frozenset({"connectors", "text_encoder", _TRANSFORMER})
_PLATFORMS = sorted(PLATFORM_KEYS)
_SIMPLE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")

IMAGE = "image"
LTX_CUDA_REPO = "ltx_cuda_repo"
LTX_TRANSFORMER_FILE = "ltx_transformer_file"
LTX_MLX_PACK = "ltx_mlx_pack"
LTX_MLX_GEMMA = "ltx_mlx_gemma"


@dataclass(frozen=True)
class RowSource:
    """One download of a model whose weights feed its memory verdict, and what to read from it.

    Attributes:
        target: The download (repo, revision and file patterns), as the metadata declares it.
        kind: What the row holds: :data:`IMAGE` (the repo's diffusers components, or just the *files* it is a source of),
            :data:`LTX_CUDA_REPO`, :data:`LTX_TRANSFORMER_FILE`, :data:`LTX_MLX_PACK` or :data:`LTX_MLX_GEMMA`.
        files: For :data:`LTX_TRANSFORMER_FILE` and an :data:`IMAGE` source of single files (a transformer, a LoRA), the safetensors files whose totals the row holds.
    """

    target: DownloadTarget
    kind: str
    files: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """The row's key in the table."""
        return row_key(self.target.repo_id, self.target.revision)


def memory_sources(model_id: str) -> tuple[RowSource, ...]:
    """Return the downloads of the pinned image or LTX model *model_id* whose weights its memory verdict reads, on every platform, without repeats.

    Raises:
        ValueError: If *model_id* is neither an image model nor an LTX video model.
    """
    entry = model_entry(model_id)
    ltx = entry is not None and entry["kind"] == "video" and entry.get("family") == "ltx"
    if entry is None or not (ltx or entry["kind"] == IMAGE):
        raise ValueError(f"{model_id} is neither an image model nor an LTX video model.")
    sources: dict[tuple[str, str, tuple[str, ...]], RowSource] = {}
    for platform_key in _PLATFORMS:
        for role, target in model_targets(model_id, platform_key) or ():
            kind = _ltx_kind(role, platform_key) if ltx else _image_kind(role)
            if kind is None:
                continue
            named = tuple(pattern for pattern in target.allow_patterns or () if is_literal(pattern) and pattern.endswith(".safetensors"))
            source = RowSource(target, kind, named if role in _IMAGE_FILE_ROLES or kind == LTX_TRANSFORMER_FILE else ())
            sources[(source.key, kind, source.files)] = source
    return tuple(sources.values())


def _image_kind(role: str) -> str | None:
    return IMAGE if role in _IMAGE_ROLES | _IMAGE_FILE_ROLES else None


def _ltx_kind(role: str, platform_key: str) -> str | None:
    mac = platform_key == "darwin"
    return {"model": LTX_MLX_PACK if mac else LTX_CUDA_REPO, "text_encoder": LTX_MLX_GEMMA if mac else None, "transformer": None if mac else LTX_TRANSFORMER_FILE}.get(role)


def regenerate(reader: HubReader, old: Mapping[str, Mapping[str, Any]], model_filter: Callable[[str], bool] = lambda key: True) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Read the table's rows from the Hub.

    Args:
        reader: Reads the Hub, with the developer's token.
        old: The current table; a repo that cannot be read keeps its row.
        model_filter: Takes a row key; a repo it rejects keeps its old row without being read.

    Returns:
        The new table (a row for every repo any pinned image or LTX model needs), and for each repo that could not be read the reason.
    """
    wanted: dict[str, list[RowSource]] = {}
    for model_id in (*model_ids("image"), *(model_id for model_id in model_ids("video") if model_entry(model_id)["family"] == "ltx")):
        for source in memory_sources(model_id):
            wanted.setdefault(source.key, []).append(source)
    table: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for key in sorted(wanted):
        if not model_filter(key):
            if key in old:
                table[key] = dict(old[key])
            continue
        try:
            table[key] = _read_row(reader, wanted[key])
        except Exception as exc:  # noqa: BLE001 - a repo that cannot be read (gated, gone, offline) is reported, never fatal
            failures[key] = _reason(exc, wanted[key][0].target.repo_id)
            if key in old:
                table[key] = dict(old[key])
    return table, failures


def _reason(exc: BaseException, repo_id: str) -> str:
    from zvisiongenerator.downloads.hub_access import describe_refusal

    return describe_refusal(exc, repo_id) or f"{type(exc).__name__}: {exc}"


def _read_row(reader: HubReader, sources: list[RowSource]) -> dict[str, Any]:
    """Return the merged row of the repo that *sources* (all of one key) read."""
    target = sources[0].target
    row: dict[str, Any] = {}
    if any(source.kind == IMAGE and not source.files for source in sources):
        row = build_row(reader, target.repo_id, target.revision)
    image_files = tuple(sorted({name for source in sources if source.kind == IMAGE for name in source.files}))
    if image_files:
        _merge(row, build_row(reader, target.repo_id, target.revision, image_files))
    builders = {LTX_CUDA_REPO: _ltx_cuda_repo, LTX_TRANSFORMER_FILE: _ltx_transformer_file, LTX_MLX_PACK: _ltx_mlx_pack, LTX_MLX_GEMMA: _ltx_mlx_gemma}
    for source in sources:
        if source.kind in builders:
            _merge(row, builders[source.kind](reader, source))
    return row


def _merge(row: dict[str, Any], extra: Mapping[str, Any]) -> None:
    """Add *extra*'s entries to *row*, merging nested mappings."""
    for name, value in extra.items():
        if isinstance(value, Mapping) and isinstance(row.get(name), dict):
            row[name] |= value
        else:
            row[name] = value


def _listing(reader: HubReader, source: RowSource) -> tuple[dict[str, Any], list[str]]:
    """Return the row's start (the commit of an unpinned repo) and the safetensors files of the repo that the download's patterns select."""
    from huggingface_hub.utils import filter_repo_objects

    target = source.target
    sha, names = reader.listing(target.repo_id, target.revision)
    selected = filter_repo_objects(
        names, allow_patterns=None if target.allow_patterns is None else list(target.allow_patterns), ignore_patterns=None if target.ignore_patterns is None else list(target.ignore_patterns)
    )
    return ({} if target.revision else {"revision": sha}), [name for name in selected if name.endswith(".safetensors")]


def _elements(header: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(math.prod(spec["shape"]) for spec in header.values())


def _ltx_cuda_repo(reader: HubReader, source: RowSource) -> dict[str, Any]:
    """Return the connectors' elements, the FP8 text encoder's bytes and, when the repo carries its own, the transformer's element counts."""
    from zvisiongenerator.backends.cuda.quant import kept_in_compute_dtype
    from zvisiongenerator.backends.cuda.video.weights import count_transformer_elements

    row, names = _listing(reader, source)
    target = source.target
    names = [name for name in names if name.rpartition("/")[0] in _LTX_CUDA_FOLDERS]
    headers = reader.headers(target.repo_id, target.revision, names)

    def folder(component: str) -> list[dict[str, dict[str, Any]]]:
        return [headers[name] for name in names if name.rpartition("/")[0] == component]

    row["ltx_cuda"] = {
        "connectors": sum(_elements(header) for header in folder("connectors")),
        "text_encoder_bytes": sum(math.prod(spec["shape"]) * (2 if kept_in_compute_dtype(tensor) else 1) for header in folder("text_encoder") for tensor, spec in header.items()),
    }
    transformer = folder(_TRANSFORMER)
    if transformer:
        blocks, rest = count_transformer_elements((tensor, spec["shape"]) for header in transformer for tensor, spec in header.items())
        row["transformers"] = {_TRANSFORMER: {"block_matrices": blocks, "transformer_rest": rest}}
    return row


def _ltx_transformer_file(reader: HubReader, source: RowSource) -> dict[str, Any]:
    """Return the element counts of the transformer in each single-file checkpoint the source names."""
    from zvisiongenerator.backends.cuda.video.weights import count_transformer_elements
    from zvisiongenerator.converters.ltx_checkpoint import CHECKPOINT_PREFIX, CONNECTOR_PREFIXES

    row, _ = _listing(reader, source)
    target = source.target
    headers = reader.headers(target.repo_id, target.revision, list(source.files))
    counts = {}
    for name, header in headers.items():
        shapes = ((tensor.removeprefix(CHECKPOINT_PREFIX), spec["shape"]) for tensor, spec in header.items() if tensor.startswith(CHECKPOINT_PREFIX) and not tensor.startswith(CONNECTOR_PREFIXES))
        blocks, rest = count_transformer_elements(shapes)
        counts[name] = {"block_matrices": blocks, "transformer_rest": rest}
    return row | {"transformers": counts}


def _ltx_mlx_pack(reader: HubReader, source: RowSource) -> dict[str, Any]:
    """Return the bfloat16 bytes of the components of the pack, picked the way the pipeline picks its files."""
    row, names = _listing(reader, source)
    target = source.target
    missing = [name for name in target.allow_patterns or () if is_literal(name) and name.endswith(".safetensors") and name not in names]
    if missing:
        raise FileNotFoundError(f"{target.repo_id} has no file {missing[0]}.")
    headers = reader.headers(target.repo_id, target.revision, names)
    sizes = {name: totals_from_header(header).bfloat16_bytes for name, header in headers.items()}
    with TemporaryDirectory() as pack_dir, TemporaryDirectory() as stand_in_dir:
        pack, stand_in = Path(pack_dir), Path(stand_in_dir)
        for name in target.allow_patterns or ():
            (pack / name).touch()
        (stand_in / "model.safetensors").touch()  # a pack without its own text encoder reads the Gemma 3 repo, which has its own row
        weights = ltx_mlx_weights(pack, stand_in, file_bytes=lambda path: sizes.get(path.name, 0))
        bundled = ltx_mlx_bundled_text_encoder(pack) is not None
    if weights is None:
        raise ValueError(f"{target.repo_id} does not hold an LTX pack: its files are not the ones the pipeline reads.")
    figures = {"connector": weights.connector, "transformer": weights.transformer, "decoders": weights.decoders}
    if weights.upsampler is not None:
        figures["upsampler"] = weights.upsampler
    if bundled:
        figures["gemma"] = weights.gemma
    return row | {"ltx_mlx": figures}


def _ltx_mlx_gemma(reader: HubReader, source: RowSource) -> dict[str, Any]:
    """Return the bfloat16 bytes of the Gemma 3 text encoder repo."""
    row, names = _listing(reader, source)
    target = source.target
    weights = weight_file_names(names, None)
    headers = reader.headers(target.repo_id, target.revision, weights)
    return row | {"ltx_mlx": {"gemma": sum(totals_from_header(header).bfloat16_bytes for header in headers.values())}}


# --- the file --------------------------------------------------------------------------------------------------------


def render_table(table: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the ``weight_totals:`` section of the metadata file for *table*: a block mapping, with each mapping of plain values on one line."""
    if not table:
        return f"{TABLE_KEY}: {{}}\n"
    return "\n".join([f"{TABLE_KEY}:", *_block(table, 1)]) + "\n"


def _block(mapping: Mapping[str, Any], depth: int) -> list[str]:
    lines: list[str] = []
    pad = "  " * depth
    for name, value in mapping.items():
        if not isinstance(value, Mapping):
            lines.append(f"{pad}{_scalar(name)}: {_scalar(value)}")
        elif not any(isinstance(item, Mapping) for item in value.values()):
            lines.append(f"{pad}{_scalar(name)}: {_flow(value)}")
        else:
            lines += [f"{pad}{_scalar(name)}:", *_block(value, depth + 1)]
    return lines


def _flow(mapping: Mapping[str, Any]) -> str:
    return "{ " + ", ".join(f"{_scalar(name)}: {_scalar(value)}" for name, value in mapping.items()) + " }" if mapping else "{}"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return text if text[:1].isalpha() and set(text) <= _SIMPLE and text not in {"true", "false", "null"} else '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_table(path: Path, table: Mapping[str, Mapping[str, Any]]) -> None:
    """Replace the ``weight_totals:`` section at the end of the metadata file *path* with *table*.

    Raises:
        ValueError: If the file has no ``weight_totals:`` section to replace.
    """
    text = path.read_text(encoding="utf-8")
    marker = f"\n{TABLE_KEY}:"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{path} has no {TABLE_KEY}: section.")
    path.write_text(text[: start + 1] + render_table(table), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the developer command: ``--regenerate`` reads the Hub and rewrites the table; ``--only TEXT`` limits it to the repos whose key contains TEXT."""
    parser = argparse.ArgumentParser(prog="python -m zvisiongenerator.utils.weight_totals", description="Fill the weight_totals table of model_metadata.yaml from the Hugging Face headers.")
    parser.add_argument("--regenerate", action="store_true", required=True, help="Read the pinned repos and rewrite the table.")
    parser.add_argument("--only", metavar="TEXT", help="Read only the repos whose key contains TEXT; the other rows stay as they are.")
    args = parser.parse_args(argv)
    old = load_model_metadata().get(TABLE_KEY) or {}
    table, failures = regenerate(HubReader(), old, (lambda key: args.only in key) if args.only else (lambda key: True))
    write_table(METADATA_FILE, table)
    print(f"Wrote {len(table)} rows to {METADATA_FILE.name}.")
    for key, reason in failures.items():
        print(f"{'kept the old row of' if key in table else 'no row for'} {key}: {reason}", file=sys.stderr)
    return 1 if any(key not in table for key in failures) else 0
