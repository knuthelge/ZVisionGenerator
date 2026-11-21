"""Resolve the weights a model's memory verdict reads: from its files when downloaded, else from a shipped table, else from the Hugging Face header.

:func:`resolve_image_weights` and :func:`resolve_ltx_weights` return a :class:`ResolvedWeights`: the
:class:`~zvisiongenerator.utils.memory_verdict.MemoryWeights` for ``judge`` and the family and size the job needs. Each
tries three sources in order:

1. The model's local safetensors headers (a model that is downloaded).
2. The ``weight_totals`` table of ``model_metadata.yaml``, keyed by the ``ref@sha`` of each pinned download.
3. The Hugging Face header of the repo, only when the caller allows it (``hub=True``): it is a network call of up to
   :data:`HUB_TIMEOUT_SECONDS`, so a caller on a request path that must stay fast never allows it.

A table row, and the row the Hub path builds, describe one repo at one revision:

- ``revision``: the commit the row was read at, for a repo the metadata does not pin.
- ``family`` and ``size``: the image family of a diffusers repo, and Klein's size.
- ``components``: each weighted folder's totals (the fields of :class:`WeightTotals`; zero and false ones left out).
- ``text_encoder_text_only``: MOSS-Image's text encoder without its vision tower, which a unified-memory backend loads.
- ``files``: the totals and ``bytes`` of single files the model names (a transformer file, a LoRA).
- ``ltx_cuda``, ``transformers``: LTX on CUDA, the elements its estimate counts (see :mod:`weight_totals_table`).
- ``ltx_mlx``: LTX on macOS, the bfloat16 bytes of the components of a pack or of its Gemma repo.

The developer command that fills the table is ``uv run python -m zvisiongenerator.utils.weight_totals --regenerate``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.declaration import model_targets, role_targets
from zvisiongenerator.utils.ideogram4_instant import instant_source
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.ltx_facts import ltx_companion_file, ltx_diffusers_quantize, ltx_gemma_repo_id
from zvisiongenerator.utils.memory_verdict import MemoryWeights
from zvisiongenerator.utils.model_files import find_local_model_dir, is_krea2_model, is_ltx_mlx_layout, is_moss_model, ltx_mlx_bundled_text_encoder, weight_file_names, weighted_components
from zvisiongenerator.utils.model_memory import (
    LtxCudaMemoryEstimate,
    LtxMlxWeights,
    WeightTotals,
    estimate_ltx_cuda_system_memory,
    image_component_totals,
    ltx_cuda_system_memory,
    ltx_mlx_weights,
    read_image_components,
    totals_from_header,
    turbo_lora_bytes,
)
from zvisiongenerator.utils.model_metadata import load_model_metadata, model_entry
from zvisiongenerator.utils.moss_build import TURBO_DIR_NAME, moss_source
from zvisiongenerator.utils.paths import join_huggingface_revision, parse_huggingface_repo_reference, split_huggingface_revision
from zvisiongenerator.utils.stored_quant import stored_quant_bits

__all__ = [
    "HUB_TIMEOUT_SECONDS",
    "HubReader",
    "TABLE_KEY",
    "ResolvedWeights",
    "build_row",
    "fetch_hub_row",
    "resolve_image_weights",
    "resolve_ltx_weights",
    "row_key",
    "totals_from_row",
    "totals_to_row",
    "weight_table",
]

TABLE_KEY = "weight_totals"
"""The top-level section of the model metadata that holds the shipped rows."""

HUB_TIMEOUT_SECONDS = 10.0
"""How long the Hub path may take in all, and how long one of its requests may."""

_MAX_HEADER_REQUESTS = 8
_UNIFIED_PLATFORM = "darwin"
_INSTANT_BUILD = "ideogram4_instant"
_MOSS_BUILD = "moss"
_TRANSFORMER = "transformer"
_TEXT_ENCODER = "text_encoder"
_TOTALS_DEFAULTS = WeightTotals()

_log = logging.getLogger(__name__)

FetchRow = Callable[[str, str | None, tuple[str, ...]], dict[str, Any] | None]


@dataclass(frozen=True)
class ResolvedWeights:
    """A model's weights as the memory verdict reads them.

    Attributes:
        weights: What ``judge`` takes for the model's media and platform.
        family: The image family (``"flux2_klein"``, ``"ideogram4"``, ``"moss"`` ...) or ``"ltx"``; ``None`` when it could not be detected.
        model_size: Klein's ``"4b"`` or ``"9b"``, otherwise ``None``.
    """

    weights: MemoryWeights
    family: str | None = None
    model_size: str | None = None

    @property
    def stored_bits(self) -> int | None:
        """The level the weights are stored at, which wins over the requested one; ``None`` for a model that is not a stored copy."""
        return self.weights.stored_bits


def row_key(repo_id: str, revision: str | None) -> str:
    """Return the key of the table row for *repo_id* at *revision*: ``owner/repo@sha``, or the bare repo id of a repo the metadata does not pin."""
    return join_huggingface_revision(repo_id, revision)


def weight_table() -> Mapping[str, Mapping[str, Any]]:
    """Return the shipped table, keyed by :func:`row_key`; part of the shared metadata parse, so read-only."""
    return load_model_metadata().get(TABLE_KEY) or {}


def totals_to_row(totals: WeightTotals) -> dict[str, Any]:
    """Return *totals* as a table entry: its fields, those at their default (zero, false) left out."""
    return {key: value for key, value in asdict(totals).items() if value != getattr(_TOTALS_DEFAULTS, key)}


def totals_from_row(entry: Mapping[str, Any]) -> WeightTotals:
    """Return the :class:`WeightTotals` a table entry (see :func:`totals_to_row`) describes."""
    return WeightTotals(**{field.name: entry[field.name] for field in fields(WeightTotals) if field.name in entry})


def resolve_image_weights(
    name: str | None,
    resolved_path: str,
    *,
    platform_key: str,
    hub: bool = False,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> ResolvedWeights | None:
    """Return the weights of an image model, or ``None`` when no source has them.

    Args:
        name: The model as picked (``"klein9b"``), which selects its pinned downloads; ``None`` for a repo or folder that is no built-in model.
        resolved_path: What the name resolves to: a model folder, ``owner/repo[@revision]``, or the folder a built model lives in.
        platform_key: ``sys.platform`` of the machine; macOS reads MOSS-Image's text encoder without its vision tower, and Ideogram 4's FP8 transformer.
        hub: Whether the Hugging Face header may be read when the files and the table have nothing. It is a network call, so a request
            that must stay fast leaves it off.
        find_local_dir: Locates a downloaded model directory.
    """
    unified = platform_key == _UNIFIED_PLATFORM
    model_dir = find_local_dir(resolved_path)
    local = None if model_dir is None else _local_image(model_dir, unified=unified)
    if local is not None:
        return local
    fetch = fetch_hub_row if hub else None
    return _image_from_rows(name, resolved_path, platform_key=platform_key, fetch=fetch)


def resolve_ltx_weights(
    name: str,
    resolved_path: str,
    *,
    platform_key: str,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> ResolvedWeights | None:
    """Return the weights of the built-in LTX model *name* on *platform_key*, or ``None`` when its files and the table have nothing.

    Local files give a converted folder or a downloaded model; the table gives a built-in one before it is downloaded. The
    Hugging Face header is not read: an LTX model is judged from its built-in entry or its files.

    Args:
        name: The model as picked (``"ltx-4"``).
        resolved_path: What the name resolves to: a model folder or ``owner/repo[@revision]``.
        platform_key: ``sys.platform`` of the machine.
        find_local_dir: Locates a downloaded model directory.
    """
    weights = _local_ltx(name, resolved_path, platform_key, find_local_dir)
    if weights is not None:
        return ResolvedWeights(weights, family="ltx")
    weights = _table_ltx(name, platform_key)
    return None if weights is None else ResolvedWeights(weights, family="ltx")


def fetch_hub_row(repo_id: str, revision: str | None, files: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Return the row for *repo_id* at *revision* read from the Hugging Face header, or ``None`` when it cannot be read in time.

    The call is bounded by :data:`HUB_TIMEOUT_SECONDS` in all and follows the download rules for authentication: the token
    from the environment or the saved login is sent, and a repo it cannot open gives ``None``.

    Args:
        repo_id: The Hugging Face repo.
        revision: Branch, tag or commit; ``None`` reads the default branch.
        files: Single files of the repo whose totals the row must hold (see :func:`build_row`).
    """
    deadline = time.monotonic() + HUB_TIMEOUT_SECONDS
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        row = pool.submit(_build_hub_row, repo_id, revision, files, deadline).result(timeout=HUB_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - whatever the network or the Hub does, the answer is "no verdict", never an error
        _log.debug("Could not read the weights of %s from Hugging Face: %s", row_key(repo_id, revision), _describe(exc, repo_id))
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return row


def _build_hub_row(repo_id: str, revision: str | None, files: tuple[str, ...], deadline: float) -> dict[str, Any]:
    return build_row(HubReader(deadline=deadline), repo_id, revision, files)


def _describe(exc: BaseException, repo_id: str) -> str:
    """Return one line for a failed Hub read: the refusal's advice when the Hub refused, else the error itself."""
    from zvisiongenerator.downloads.hub_access import describe_refusal

    return describe_refusal(exc, repo_id) or f"{type(exc).__name__}: {exc}"


class HubReader:
    """Reads repo listings and safetensors headers from the Hugging Face Hub within a deadline.

    Args:
        api: A ``HfApi`` (created when ``None``); it sends the token the download rules use.
        deadline: A ``time.monotonic`` time every request must finish by; ``None`` bounds each request by itself.
    """

    def __init__(self, api: Any = None, *, deadline: float | None = None) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self._api = api
        self._deadline = deadline

    def _timeout(self) -> float:
        if self._deadline is None:
            return 3 * HUB_TIMEOUT_SECONDS
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Reading the Hugging Face header took too long.")
        return min(remaining, HUB_TIMEOUT_SECONDS)

    def listing(self, repo_id: str, revision: str | None) -> tuple[str, list[str]]:
        """Return the commit a revision names and the files of the repo at it."""
        info = self._api.model_info(repo_id, revision=revision, timeout=self._timeout())
        return info.sha, [sibling.rfilename for sibling in info.siblings]

    def headers(self, repo_id: str, revision: str | None, names: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the safetensors header (tensor name to dtype, shape and data offsets) of each file in *names*."""
        if not names:
            return {}
        with ThreadPoolExecutor(max_workers=min(_MAX_HEADER_REQUESTS, len(names))) as pool:
            return dict(zip(names, pool.map(lambda name: self._header(repo_id, revision, name), names), strict=True))

    def _header(self, repo_id: str, revision: str | None, name: str) -> dict[str, dict[str, Any]]:
        parsed = self._api.parse_safetensors_file_metadata(repo_id, name, revision=revision, timeout=self._timeout())
        return {tensor: {"dtype": info.dtype, "shape": list(info.shape), "data_offsets": list(info.data_offsets)} for tensor, info in parsed.tensors.items()}

    def model_index(self, repo_id: str, revision: str | None) -> Path:
        """Return the local path of the repo's ``model_index.json``, downloaded if it is not in the cache."""
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id, "model_index.json", revision=revision, etag_timeout=self._timeout()))


def build_row(reader: HubReader, repo_id: str, revision: str | None, files: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the row (see the module docstring) of *repo_id* at *revision*, read through *reader*.

    Without *files* the row describes the repo as a diffusers image model: its family, size and the totals of each weighted
    component, as the local files would give them. With *files* it holds only the totals and ``bytes`` (the size of the
    tensor data) of those single files, which is what a repo that serves as a source of one file needs.

    Args:
        reader: Reads the Hub.
        repo_id: The Hugging Face repo.
        revision: Branch, tag or commit; ``None`` reads the default branch, and the row records the commit it read.
        files: Single safetensors files of the repo to describe instead of its components.

    Raises:
        FileNotFoundError: If the repo has no ``model_index.json`` (without *files*) or lacks one of *files*.
        ValueError: If the repo is an image model the app does not support, or a tensor has a dtype the estimate does not know.
    """
    sha, names = reader.listing(repo_id, revision)
    row: dict[str, Any] = {} if revision else {"revision": sha}
    if not files:
        if "model_index.json" not in names:
            raise FileNotFoundError(f"{repo_id} has no model_index.json: it is not a diffusers image model.")
        return row | _layout_row(reader, repo_id, revision, names)
    missing = [name for name in files if name not in names]
    if missing:
        raise FileNotFoundError(f"{repo_id} has no file {missing[0]}.")
    headers = reader.headers(repo_id, revision, list(files))
    return row | {"files": {name: totals_to_row(totals_from_header(headers[name])) | {"bytes": _data_bytes(headers[name])} for name in files}}


def _data_bytes(header: Mapping[str, Mapping[str, Any]]) -> int:
    """Return the size of the tensor data a header describes (the header itself, a few hundred KB at most, is not counted)."""
    return max((int(spec["data_offsets"][1]) for spec in header.values()), default=0)


def _layout_row(reader: HubReader, repo_id: str, revision: str | None, names: list[str]) -> dict[str, Any]:
    """Return the family, size and component totals of the diffusers repo whose listing is *names*."""
    snapshot = reader.model_index(repo_id, revision).parent
    info = detect_image_model(repo_id)
    selected = weight_file_names(names, weighted_components(snapshot), krea2=is_krea2_model(snapshot))
    headers = reader.headers(repo_id, revision, selected)
    components: dict[str, WeightTotals] = {}
    text_only = WeightTotals()
    for name in selected:
        component = name.partition("/")[0] if "/" in name else ""
        components[component] = components.get(component, WeightTotals()) + image_component_totals(component, headers[name], unified=False)
        if component == _TEXT_ENCODER:
            text_only += image_component_totals(component, headers[name], unified=info.family == "moss")
    row: dict[str, Any] = {"family": info.family}
    if info.size is not None:
        row["size"] = info.size
    row["components"] = {component: totals_to_row(totals) for component, totals in components.items()}
    if _TEXT_ENCODER in components and text_only != components[_TEXT_ENCODER]:
        row["text_encoder_text_only"] = totals_to_row(text_only)
    return row


# --- images: from the files ----------------------------------------------------------------------------------------


def _local_image(model_dir: Path, *, unified: bool) -> ResolvedWeights | None:
    """Return the weights of the image model in *model_dir* from its safetensors headers, or ``None`` when they are unreadable."""
    components = read_image_components(model_dir, unified=unified)
    if components is None:
        return None
    try:
        info = detect_image_model(str(model_dir))
        family, size = info.family, info.size
    except ValueError, FileNotFoundError:
        family = size = None
    lora = turbo_lora_bytes(model_dir) if is_moss_model(model_dir) else 0
    weights = MemoryWeights(components=components, stored_bits=stored_quant_bits(model_dir), turbo_lora_bytes=lora)
    return ResolvedWeights(weights, family=family, model_size=size)


# --- images: from the table, or the Hub ----------------------------------------------------------------------------


def _lookup_row(fetch: FetchRow | None, repo_id: str, revision: str | None, files: tuple[str, ...] = ()) -> Mapping[str, Any] | None:
    """Return the row of *repo_id* at *revision*: the table's when it holds it (and every one of *files*), else the Hub's when *fetch* is given."""
    row = weight_table().get(row_key(repo_id, revision))
    if row is not None and all(name in (row.get("files") or {}) for name in files):
        return row
    return None if fetch is None else fetch(repo_id, revision, files)


def _image_from_rows(name: str | None, resolved_path: str, *, platform_key: str, fetch: FetchRow | None) -> ResolvedWeights | None:
    """Compose an image model's weights from the rows of the repos it is made from."""
    entry = model_entry(name) if name else None
    if entry is None or entry["kind"] != "image":
        reference = parse_huggingface_repo_reference(resolved_path)
        return None if reference is None else _plain_image(fetch, reference.repo_id, reference.revision)
    build = entry.get("build")
    if build == _INSTANT_BUILD:
        return _instant_image(fetch, platform_key)
    if build == _MOSS_BUILD:
        return _moss_image(fetch, name, platform_key)
    target = next(iter(role_targets(model_targets(name, platform_key) or (), "model")), None)
    return None if target is None else _plain_image(fetch, target.repo_id, target.revision)


def _plain_image(fetch: FetchRow | None, repo_id: str, revision: str | None) -> ResolvedWeights | None:
    """Return the weights of a diffusers repo as it is downloaded, or ``None`` when no row holds its components."""
    row = _lookup_row(fetch, repo_id, revision)
    if row is None or not row.get("components"):
        return None
    components = {name: totals_from_row(entry) for name, entry in row["components"].items()}
    return ResolvedWeights(MemoryWeights(components=components), family=row.get("family"), model_size=row.get("size"))


def _instant_image(fetch: FetchRow | None, platform_key: str) -> ResolvedWeights | None:
    """Compose Ideogram 4 Instant: the base repo's linked components, and a transformer in the format the platform builds it in."""
    from zvisiongenerator.utils.ideogram4_instant import LINKED_COMPONENTS

    source = instant_source(platform_key)
    base = _lookup_row(fetch, source.components_repo, source.components_revision)
    if base is None:
        return None
    entries = base.get("components") or {}
    if source.transformer_format == "fp8":
        transformer = entries.get(_TRANSFORMER)  # the base FP8 repo's own transformer: the build keeps its names and storage
    else:
        row = _lookup_row(fetch, source.transformer_repo, source.transformer_revision, (source.transformer_file,))
        transformer = None if row is None else row["files"][source.transformer_file]
    if transformer is None:
        return None
    components = {name: totals_from_row(entry) for name, entry in entries.items() if name in LINKED_COMPONENTS}
    components[_TRANSFORMER] = totals_from_row(transformer)
    return ResolvedWeights(MemoryWeights(components=components), family="ideogram4")


def _moss_image(fetch: FetchRow | None, name: str, platform_key: str) -> ResolvedWeights | None:
    """Compose MOSS-Image: the SFT repo's linked components, and for Turbo the LoRA's size; a unified-memory backend drops the vision tower."""
    from zvisiongenerator.utils.moss_build import LINKED_COMPONENTS

    source = moss_source(platform_key)
    base = _lookup_row(fetch, source.sft_repo, source.sft_revision)
    if base is None:
        return None
    entries = dict(base.get("components") or {})
    if platform_key == _UNIFIED_PLATFORM and "text_encoder_text_only" in base:
        entries[_TEXT_ENCODER] = base["text_encoder_text_only"]
    components = {component: totals_from_row(entry) for component, entry in entries.items() if component in LINKED_COMPONENTS}
    lora = 0
    if name == TURBO_DIR_NAME:
        turbo = _lookup_row(fetch, source.turbo_repo, source.turbo_revision, (source.turbo_file,))
        if turbo is None:
            return None
        lora = int(turbo["files"][source.turbo_file]["bytes"])
    return ResolvedWeights(MemoryWeights(components=components, turbo_lora_bytes=lora), family="moss")


# --- LTX -----------------------------------------------------------------------------------------------------------


def _local_ltx(name: str, resolved_path: str, platform_key: str, find_local_dir: Callable[[str], Path | None]) -> MemoryWeights | None:
    """Return the weights of an LTX model from its files on disk, or ``None`` when some are missing."""
    if platform_key != _UNIFIED_PLATFORM:
        estimate = ltx_cuda_system_memory(resolved_path, name, find_local_dir)
        return None if estimate is None else MemoryWeights(ltx_cuda=estimate)
    model_dir = find_local_dir(resolved_path)
    if model_dir is None:
        return None
    text_encoder_dir = find_local_dir(ltx_gemma_repo_id()) if is_ltx_mlx_layout(model_dir) and ltx_mlx_bundled_text_encoder(model_dir) is None else None
    mlx = ltx_mlx_weights(model_dir, text_encoder_dir)
    return None if mlx is None else MemoryWeights(ltx_mlx=mlx)


def _table_ltx(name: str, platform_key: str) -> MemoryWeights | None:
    """Return the weights of the built-in LTX model *name* from the table, or ``None`` when it has no complete rows."""
    if model_targets(name, platform_key) is None:
        return None
    if platform_key == _UNIFIED_PLATFORM:
        mlx = _table_ltx_mlx(name)
        return None if mlx is None else MemoryWeights(ltx_mlx=mlx)
    cuda = _table_ltx_cuda(name, platform_key)
    return None if cuda is None else MemoryWeights(ltx_cuda=cuda)


def _table_ltx_cuda(name: str, platform_key: str) -> LtxCudaMemoryEstimate | None:
    """Return the system memory of LTX on CUDA from the rows of its diffusers repo and its single-file transformer."""
    quantize = ltx_diffusers_quantize(name)
    target = next(iter(role_targets(model_targets(name, platform_key) or (), "model")), None)
    if quantize is None or target is None:
        return None
    row = _lookup_row(None, target.repo_id, target.revision)
    figures = None if row is None else row.get("ltx_cuda")
    source = ltx_companion_file(name, platform_key, _TRANSFORMER)
    if source is None:
        counts = None if row is None else (row.get("transformers") or {}).get(_TRANSFORMER)  # the repo's own transformer folder
    else:
        repo_id, revision = split_huggingface_revision(source[0])
        companion = _lookup_row(None, repo_id, revision)
        counts = None if companion is None else (companion.get("transformers") or {}).get(source[1])
    if figures is None or counts is None:
        return None
    return estimate_ltx_cuda_system_memory(
        block_matrices=counts["block_matrices"],
        transformer_rest=counts["transformer_rest"],
        connectors=figures["connectors"],
        text_encoder=figures["text_encoder_bytes"],
        quantize=quantize,
    )


def _table_ltx_mlx(name: str) -> LtxMlxWeights | None:
    """Return the component bytes of an LTX MLX pack from the rows of the pack and, for a pack without its own, of the Gemma 3 repo."""
    pairs = model_targets(name, _UNIFIED_PLATFORM) or ()
    pack_target = next(iter(role_targets(pairs, "model")), None)
    pack = None if pack_target is None else _lookup_row(None, pack_target.repo_id, pack_target.revision)
    figures = None if pack is None else pack.get("ltx_mlx")
    if figures is None:
        return None
    gemma = figures.get("gemma")
    if gemma is None:
        encoder_target = next(iter(role_targets(pairs, "text_encoder")), None)
        encoder = None if encoder_target is None else _lookup_row(None, encoder_target.repo_id, encoder_target.revision)
        gemma = None if encoder is None else (encoder.get("ltx_mlx") or {}).get("gemma")
    if gemma is None:
        return None
    return LtxMlxWeights(
        gemma=gemma,
        connector=figures["connector"],
        transformer=figures["transformer"],
        decoders=figures["decoders"],
        upsampler=figures.get("upsampler"),
        bundled_encoder=figures.get("gemma") is not None,
    )


if __name__ == "__main__":
    from zvisiongenerator.utils.weight_totals_table import main

    raise SystemExit(main())
