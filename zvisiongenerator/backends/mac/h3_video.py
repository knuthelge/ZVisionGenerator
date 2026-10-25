"""macOS MiniMax H3 video backend (the vendored Argus MLX port, ``minimax_h3_mlx``).

``minimax_h3_mlx`` and ``mlx`` are imported only when a clip is made, so this module imports anywhere.

What this module adds around the fork's pipeline:

* File resolution (:func:`resolve_mlx_files`) and a one-time 4-bit text-encoder copy that also carries the vision tower
  (:func:`ensure_text_encoder_copy`).
* LoRAs. The fork takes one adapter of one rank, so user LoRAs are combined with the turbo LoRA into one rank-concatenated
  adapter (:func:`combine_loras`); with none, the turbo file is passed straight through.
* The image modes (image-to-video, first and last frame, Extend) run through :mod:`~zvisiongenerator.backends.mac.h3_staged`,
  which holds one component in memory at a time; text-to-video runs the fork's low-memory pipeline.
* Extend as image-to-video from the source's last frame.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.mac.h3_staged import generate_conditioned
from zvisiongenerator.backends.mac.memory import release_memory
from zvisiongenerator.core.video_family import FrameRule
from zvisiongenerator.core.video_types import ExtendSegment
from zvisiongenerator.utils.alignment import align_frames
from zvisiongenerator.utils.audio_resample import resample_audio
from zvisiongenerator.utils.h3_memory import H3_FPS, H3_FRAMES_PER_CHUNK, H3_ROW_PIXELS
from zvisiongenerator.utils.h3_preset import H3MlxFiles, H3MlxPreset, H3Schedule, HubFiles
from zvisiongenerator.utils.hf_snapshot import snapshot_download as hub_snapshot_download
from zvisiongenerator.utils.hf_snapshot import snapshot_with
from zvisiongenerator.utils.paths import get_ziv_data_dir, parse_huggingface_repo_reference
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE, EXTEND_FPS_TOLERANCE
from zvisiongenerator.utils.stored_quant import ensure_stored_copy, file_identities, text_encoder_copy_name
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_splice import build_segment_command, mux_audio, write_pcm16_wav, write_segment_frames
from zvisiongenerator.utils.video_tail import read_source_tail

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

    from zvisiongenerator.core.video_types import ExtendSource

__all__ = ["H3_MLX_COMMIT", "H3MlxVideoBackend", "combine_loras", "ensure_text_encoder_copy", "local_int8_transformer", "local_transformer_dirs", "resolve_mlx_files"]

# The commit of the vendored fork; equals ``H3_MLX_COMMIT`` in the Makefile (a test compares them).
H3_MLX_COMMIT = "3fcc9f8c557cf353400b1acf737b985e524a509f"

MEMORY_LIMIT_GB = 16.0

# The frame rate, the 17-frame group of an extension and the size alignment are the model's, shared with the CUDA backend.
_FPS_TEXT = f"{H3_FPS}/1"
_MODEL_INFO = VideoModelInfo(family="h3", backend="h3", supports_i2v=True, default_fps=H3_FPS, frame_alignment=H3_FRAMES_PER_CHUNK, resolution_alignment=H3_ROW_PIXELS)

_MODES = ("t2v", "i2v", "extend")
_LOAD_KEYS = frozenset({"mode", "low_memory", "loras", "upscale", "model_name", "preset", "lora", "lora_scale", "verbose", "models_dir"})
_PROGRESS_PHASE = "video_generate"

# The text-encoder copy: its folder is ``<models>/<name>@mlx-te<bits>``; the vision tower goes into its own shard.
_COPY_LAYOUT = 1
_VISION_PREFIX = "model.visual."
_VISION_SHARD = "model-visual.safetensors"
_INDEX_NAME = "model.safetensors.index.json"

# The tensors the fork's ``TurboLoRAProvider`` accepts (tests/test_h3_mlx_contract.py pins them): one adapter of PEFT
# diffusers keys, for the transformer blocks and the token refiner, six projections each.
_FAMILIES = ("transformer_blocks", "token_refiner.refiner_blocks")
_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_SLOT = rf"({'|'.join(re.escape(family) for family in _FAMILIES)})\.(\d+)\.({'|'.join(re.escape(target) for target in _TARGETS)})"
_MODULE_PATTERN = re.compile(_SLOT)
_ADAPTER_KEY_PATTERN = re.compile(rf"(?:^|.*\.){_SLOT}\.lora_([AB])\.default\.weight$")
_UNSUPPORTED_LORA = "MiniMax H3 on macOS takes LoRAs for transformer blocks and the token refiner only."

# A LoRA slot: the block family, its index and the projection.
_Slot = tuple[str, int, str]


@dataclass(frozen=True)
class _H3MlxModel:
    """The handle ``load_model`` returns: the resolved files and the settings every clip is made with.

    The pipeline (text-to-video) or the staged components (image modes) are built per clip.
    """

    files: H3MlxFiles
    text_encoder_dir: Path
    turbo_lora_path: Path | None
    turbo_lora_alpha: float | None
    turbo_lora_scale: float
    mode: str
    verbose: bool
    frame_rule: FrameRule
    schedules: dict[int, H3Schedule]


@dataclass(frozen=True)
class _Clip:
    """A generated clip: RGB frames ``(frames, height, width, 3)`` and stereo audio ``(2, samples)`` at ``sample_rate``."""

    video: np.ndarray
    audio: np.ndarray
    sample_rate: int


# ---- File resolution ------------------------------------------------------------------------------------------------


# Where a local H3 folder may keep a transformer: the fork's default (beside the other components), then the release's top level.
_LOCAL_TRANSFORMER_DIRS = ("FL2VA/transformer", "transformer")
_INT8_CONFIG_FILES = ("config.json", "quant_config.json")


def _holds_weights(folder: Path) -> bool:
    """Whether *folder* holds transformer weights: safetensors shards or a shard index."""
    return folder.is_dir() and ((folder / _INDEX_NAME).is_file() or any(folder.glob("*.safetensors")))


def local_transformer_dirs(local: Path) -> list[Path]:
    """Return the folders of the local H3 folder *local* that hold transformer weights, in the order they are tried."""
    return [folder for folder in (local / name for name in _LOCAL_TRANSFORMER_DIRS) if _holds_weights(folder)]


def _int8_missing(folder: Path) -> list[str]:
    """Return what the fork's ``load_dit`` reads from *folder* and the folder lacks (empty for a complete INT8 transformer)."""
    missing = [name for name in (*_INT8_CONFIG_FILES, _INDEX_NAME) if not (folder / name).is_file()]
    if missing:
        return missing
    try:
        weight_map = json.loads((folder / _INDEX_NAME).read_text(encoding="utf-8"))["weight_map"]
        return [name for name in sorted(set(weight_map.values())) if not (folder / name).is_file()]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{folder / _INDEX_NAME} is not a valid safetensors index: {error}") from error


def _is_int8_layout(folder: Path) -> bool:
    """Whether *folder* is an INT8 transformer in the fork's layout (a ``quant_config.json`` beside its config and index)."""
    return (folder / "quant_config.json").is_file() and (folder / "config.json").is_file()


def local_int8_transformer(local: Path) -> Path | None:
    """Return the MLX INT8 transformer folder of the local H3 folder *local*, or ``None`` when it holds no transformer.

    The fork loads ``config.json``, ``quant_config.json`` and ``model.safetensors.index.json`` with its shards from one folder.

    Raises:
        ValueError: If *local* holds a transformer that is not in that layout (the release's bf16 ``transformer/`` or
            ``FL2VA/transformer``, or a fine-tune of them): it must be converted first, and the stock transformer must
            not stand in for it.
        FileNotFoundError: If an INT8 transformer lacks its index or a shard.
    """
    candidates = local_transformer_dirs(local)
    if not candidates:
        return None
    for folder in candidates:
        if _is_int8_layout(folder):
            if missing := _int8_missing(folder):
                raise FileNotFoundError(f"The INT8 transformer in {folder} is missing: {', '.join(missing)}.")
            return folder
    raise ValueError(
        f"{local} holds a transformer ({', '.join(str(folder.relative_to(local)) for folder in candidates)}) that is not in the MLX INT8 layout. "
        "MiniMax H3 on macOS needs the fork's INT8 layout (config.json, quant_config.json, model.safetensors.index.json and its shards). "
        "Convert the fine-tune on the Models page (MiniMax H3) or with `ziv model model --model-type h3`, or remove the transformer from the folder to use the stock INT8 transformer."
    )


def resolve_mlx_files(model_ref: str, preset: H3MlxPreset, *, lora: HubFiles | None, snapshot_download: Callable[..., str] | None = None) -> H3MlxFiles:
    """Return the on-disk files of the macOS H3 model, downloading what is missing.

    Args:
        model_ref: A local folder holding ``FL2VA/`` or a Hub reference; a Hub reference takes the preset's component files.
            A local folder with an MLX INT8 transformer (see :func:`local_int8_transformer`) uses it; one without any
            transformer takes the preset's INT8 transformer.
        preset: The macOS preset: the INT8 transformer and the ``FL2VA`` components.
        lora: The turbo LoRA's repository and file (one file), or ``None`` for a converted folder, which holds its turbo merge.
        snapshot_download: Called as ``snapshot_download(repo_id=, revision=, allow_patterns=)``; defaults to the Hub's.

    Raises:
        FileNotFoundError: If a local folder holds no ``FL2VA/model_index.json``, or the LoRA file is missing.
        ValueError: If *lora* does not name exactly one file, or a local folder holds a transformer the Mac cannot load.
    """
    download = hub_snapshot_download if snapshot_download is None else snapshot_download
    if lora is not None and len(lora.files) != 1:
        raise ValueError(f"The turbo LoRA must be one file, got {len(lora.files)}.")
    local = Path(model_ref).expanduser()
    transformer_dir: Path | None = None
    if local.is_dir() or parse_huggingface_repo_reference(model_ref) is None:
        fl2va_dir = local / "FL2VA"
        if not (fl2va_dir / "model_index.json").is_file():
            raise FileNotFoundError(f"{local} holds no FL2VA/model_index.json; point at a MiniMax H3 folder or repository.")
        transformer_dir = local_int8_transformer(local)
    else:
        fl2va_dir = snapshot_with(preset.components, download) / "FL2VA"
    lora_file = snapshot_with(lora, download) / lora.files[0] if lora is not None else None
    if lora_file is not None and not lora_file.is_file():
        raise FileNotFoundError(f"Turbo LoRA not found: {lora_file}")
    if transformer_dir is None:
        transformer_dir = snapshot_with(preset.transformer, download)
    return H3MlxFiles(fl2va_dir=fl2va_dir, transformer_dir=transformer_dir, text_encoder_source=fl2va_dir / "text_encoder", lora_file=lora_file)


# ---- Text-encoder copy ----------------------------------------------------------------------------------------------


def ensure_text_encoder_copy(models_dir: Path, name: str, files: H3MlxFiles, preset: H3MlxPreset) -> Path:
    """Return the 4-bit text-encoder folder ``<models_dir>/<name>@mlx-te<bits>``, converting it from the release on first use.

    The fork's converter keeps the language tower only, so the vision tensors (``model.visual.*``) are copied into a shard of
    their own and merged into the index; image modes load the vision tower from the same folder.

    Args:
        models_dir: The models folder.
        name: The model's name; the copy is its sibling.
        files: The resolved files; ``text_encoder_source`` is the release's text encoder.
        preset: The quantization to convert to.

    Raises:
        FileNotFoundError: If the release's text encoder has no index.
    """
    source = files.text_encoder_source
    index = source / _INDEX_NAME
    if not index.is_file():
        raise FileNotFoundError(f"MiniMax H3 text encoder index not found: {index}")
    target = models_dir / text_encoder_copy_name(name, preset.text_encoder_bits)
    manifest = {
        "layout": _COPY_LAYOUT,
        "family": "h3",
        "kind": "mlx-text-encoder",
        "fork": H3_MLX_COMMIT,
        "bits": preset.text_encoder_bits,
        "group_size": preset.text_encoder_group_size,
        "layers": preset.text_encoder_layers,
        "sources": file_identities([index, *sorted(source.glob("model-*.safetensors"))]),
    }
    return ensure_stored_copy(target, manifest, lambda folder: _write_text_encoder_copy(source, folder, preset), flush=True)


def _write_text_encoder_copy(source: Path, folder: Path, preset: H3MlxPreset) -> None:
    """Convert *source*'s language tower into *folder* and add its vision tensors."""
    from minimax_h3_mlx.quantize_text_encoder import convert

    convert(source, folder, bits=preset.text_encoder_bits, group_size=preset.text_encoder_group_size, num_layers=preset.text_encoder_layers)
    _copy_vision_tensors(source, folder)


def _copy_vision_tensors(source: Path, folder: Path) -> None:
    """Copy the ``model.visual.*`` tensors of *source* into a shard of *folder* and add them to its index.

    Raises:
        ValueError: If *source* holds no vision tensors.
        FileNotFoundError: If a shard holding them is missing.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_map = json.loads((source / _INDEX_NAME).read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(_VISION_PREFIX):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise ValueError(f"The text encoder at {source} has no {_VISION_PREFIX}* tensors, so image modes could not load the vision tower.")
    missing = sorted(shard for shard in by_shard if not (source / shard).is_file())
    if missing:
        raise FileNotFoundError(f"The shard(s) with the vision tensors are missing from {source}: {', '.join(missing)}")
    tensors = {}
    for shard, keys in by_shard.items():
        with safe_open(str(source / shard), framework="pt") as reader:
            tensors.update({key: reader.get_tensor(key) for key in keys})
    save_file(tensors, str(folder / _VISION_SHARD), metadata={"format": "pt"})
    index_path = folder / _INDEX_NAME
    merged = json.loads(index_path.read_text(encoding="utf-8"))
    merged["weight_map"].update(dict.fromkeys(tensors, _VISION_SHARD))
    merged.setdefault("metadata", {})["total_size"] = int(merged.get("metadata", {}).get("total_size", 0)) + sum(tensor.nbytes for tensor in tensors.values())
    index_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---- LoRAs ----------------------------------------------------------------------------------------------------------


def _turbo_rank(path: Path) -> int:
    """Return the rank of the turbo LoRA file (its ``lora_A`` height, one rank for the whole file).

    Raises:
        ValueError: If the file holds no adapter tensors or more than one rank.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as reader:
        ranks = {reader.get_slice(key).get_shape()[0] for key in reader.keys() if _ADAPTER_KEY_PATTERN.fullmatch(key) and key.endswith(".lora_A.default.weight")}
    if len(ranks) != 1:
        raise ValueError(f"The turbo LoRA {path.name} must use one rank, found {sorted(ranks) or 'no adapter tensors'}.")
    return ranks.pop()


def _slot_of(module: str) -> _Slot:
    """Return the (family, index, projection) of a converted module name.

    Raises:
        ValueError: If the module is not a transformer-block or token-refiner projection the fork's adapter format takes.
    """
    match = _MODULE_PATTERN.fullmatch(module)
    if match is None:
        raise ValueError(_UNSUPPORTED_LORA)
    return match.group(1), int(match.group(2)), match.group(3)


def _read_user_lora(path: Path, weight: float) -> dict[_Slot, tuple[torch.Tensor, torch.Tensor]]:
    """Return a user LoRA's ``{slot: (A, B)}`` in bf16 with ``weight * alpha / rank`` baked into B.

    Raises:
        ValueError: If the file targets anything but block or refiner projections.
    """
    import torch

    from zvisiongenerator.utils.h3_lora import load_lora, pairs_from_state

    lora = load_lora(path)
    factors = {}
    for module, (down, up) in pairs_from_state(lora.state).items():
        factor = weight * lora.scales.get(module, 1.0)
        factors[_slot_of(module)] = (down.to(torch.bfloat16), (up.float() * factor).to(torch.bfloat16))
    return factors


def _read_turbo_lora(path: Path, scale: float) -> dict[_Slot, tuple[torch.Tensor, torch.Tensor]]:
    """Return the turbo LoRA's ``{slot: (A, B)}`` in bf16 with *scale* (and no alpha / rank) baked into B.

    Raises:
        ValueError: If the file holds a key the fork's adapter format does not take, or half of a pair.
    """
    import torch
    from safetensors.torch import load_file

    halves: dict[_Slot, dict[str, torch.Tensor]] = {}
    for key, tensor in load_file(str(path)).items():
        match = _ADAPTER_KEY_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"The turbo LoRA {path.name} holds a key the MiniMax H3 adapter format does not take: {key}")
        slot = (match.group(1), int(match.group(2)), match.group(3))
        halves.setdefault(slot, {})[match.group(4)] = tensor if match.group(4) == "A" else (tensor.float() * scale).to(torch.bfloat16)
    incomplete = [slot for slot, pair in halves.items() if len(pair) != 2]
    if incomplete:
        raise ValueError(f"The turbo LoRA {path.name} has only one of lora_A and lora_B for {incomplete[0]}.")
    return {slot: (pair["A"].to(torch.bfloat16), pair["B"]) for slot, pair in halves.items()}


def combine_loras(turbo: Path | None, turbo_scale: float, user: Sequence[tuple[Path, float]], target: Path) -> Path:
    """Write one adapter that holds the turbo LoRA (when given) and every user LoRA, in the format the fork's turbo provider reads.

    Each file's up-projection is scaled by its strength (the turbo's is *turbo_scale* alone; a user file's is ``weight * alpha /
    rank``). The files' ranks are concatenated per projection, with the slots a file does not cover left at zero, so the sum
    of the files' deltas is exact. The fork needs every projection of every block, so the adapter covers them all. The adapter
    is written in bf16 with its metadata alpha equal to the total rank; pass it with ``turbo_lora_scale=1.0`` and no alpha
    override.

    Args:
        turbo: The turbo LoRA file (PEFT diffusers keys), or ``None`` for a converted model, which holds its turbo merge.
        turbo_scale: The turbo's strength.
        user: ``(file, strength)`` of each user LoRA, in any format diffusers' H3 loader converts.
        target: Where to write the adapter.

    Returns:
        *target*.

    Raises:
        ValueError: If a user LoRA targets anything but transformer-block and token-refiner projections, or a projection
            the model does not have, or with factors of the wrong shape.
    """
    import torch
    from safetensors.torch import save_file

    contributions = [_read_user_lora(Path(path), float(weight)) for path, weight in user]
    if turbo is not None:
        contributions.insert(0, _read_turbo_lora(Path(turbo), float(turbo_scale)))
    # The turbo covers every projection; without it, the model's config lists them.
    shapes = {slot: (down.shape[1], up.shape[0]) for slot, (down, up) in contributions[0].items()} if turbo is not None else _slot_shapes()
    ranks = [max(down.shape[0] for down, _ in contribution.values()) for contribution in contributions]
    offsets = [sum(ranks[:index]) for index in range(len(ranks))]
    total = sum(ranks)

    stray = sorted({slot for contribution in contributions for slot in contribution} - set(shapes))
    if stray:
        family, index, projection = stray[0]
        raise ValueError(f"A LoRA targets {family}.{index}.{projection}, which MiniMax H3 does not have.")

    tensors: dict[str, torch.Tensor] = {}
    for slot in sorted(shapes):
        family, index, projection = slot
        inputs, outputs = shapes[slot]
        down = torch.zeros(total, inputs, dtype=torch.bfloat16)
        up = torch.zeros(outputs, total, dtype=torch.bfloat16)
        for contribution, offset in zip(contributions, offsets, strict=True):
            if slot not in contribution:
                continue
            lora_a, lora_b = contribution[slot]
            if lora_a.shape[1] != inputs or lora_b.shape[0] != outputs or lora_a.shape[0] != lora_b.shape[1]:
                raise ValueError(f"A LoRA's factors for {family}.{index}.{projection} (A {tuple(lora_a.shape)}, B {tuple(lora_b.shape)}) do not fit the model's ({inputs} in, {outputs} out).")
            rank = lora_a.shape[0]
            down[offset : offset + rank] = lora_a
            up[:, offset : offset + rank] = lora_b
        prefix = f"{family}.{index}.{projection}"
        tensors[f"{prefix}.lora_A.default.weight"] = down
        tensors[f"{prefix}.lora_B.default.weight"] = up

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    try:
        save_file(tensors, str(partial), metadata={"alpha": str(float(total))})
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _slot_shapes() -> dict[_Slot, tuple[int, int]]:
    """Return every projection the fork's adapter covers, with its ``(inputs, outputs)``, from stock H3's config."""
    from zvisiongenerator.converters.h3_checkpoint import STOCK_CONFIG

    hidden, inner, ffn = STOCK_CONFIG["hidden_size"], STOCK_CONFIG["num_attention_heads"] * STOCK_CONFIG["attention_head_dim"], STOCK_CONFIG["ffn_dim"]
    projections = {
        "attn.to_q": (hidden, inner),
        "attn.to_k": (hidden, inner),
        "attn.to_v": (hidden, inner),
        "attn.to_out.0": (inner, hidden),
        "ff.net.0.proj": (hidden, 2 * ffn),
        "ff.net.2": (ffn, hidden),
    }
    counts = {"transformer_blocks": STOCK_CONFIG["num_layers"], "token_refiner.refiner_blocks": STOCK_CONFIG["num_refiner_layers"]}
    return {(family, index, projection): shape for family, count in counts.items() for index in range(count) for projection, shape in projections.items()}


# ---- Pipeline runs --------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _process_limits(mx: Any) -> Iterator[None]:
    """Put the MLX cache and memory limits the fork's low-memory mode sets in place, and restore the previous ones after.

    The fork sets them process-wide and never restores them, which would leave a web server's later image jobs with no buffer
    cache and a 16 GB ceiling. The staged image modes run under the same limits.
    """
    previous_cache = mx.set_cache_limit(0)
    previous_memory = mx.set_memory_limit(int(MEMORY_LIMIT_GB * 1e9))
    try:
        yield
    finally:
        mx.set_cache_limit(previous_cache)
        mx.set_memory_limit(previous_memory)


def _converted(folder: Path) -> bool:
    """Return whether *folder* is a converted H3 folder.

    Raises:
        RuntimeError: When it was converted on Windows or Linux.
    """
    from zvisiongenerator.backends.mac.h3_converted import STORAGE
    from zvisiongenerator.utils.h3_converted import read_converted_h3

    manifest = read_converted_h3(folder) if folder.is_dir() else None
    if manifest is not None and manifest.get("storage") != STORAGE:
        raise RuntimeError(f"{folder.name} was converted on Windows or Linux; convert the checkpoint again on this Mac.")
    return manifest is not None


def _load_pipeline(model: _H3MlxModel, schedule: H3Schedule) -> Any:
    """Build the fork's low-memory text-to-video pipeline for one clip, with the sigma shifts of *schedule*."""
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline.from_pretrained(
        checkpoint_dir=model.files.fl2va_dir,
        transformer_dir=model.files.transformer_dir,
        text_encoder_dir=model.text_encoder_dir,
        turbo_lora_path=model.turbo_lora_path,
        turbo_lora_alpha=model.turbo_lora_alpha,
        turbo_lora_scale=model.turbo_lora_scale,
        sigma_shift_video=schedule.video_shift,
        sigma_shift_audio=schedule.audio_shift,
        stream_blocks=True,
        low_memory=True,
        load_vision=False,
        memory_limit_gb=MEMORY_LIMIT_GB,
        verbose=model.verbose,
    )


def _schedule(model: _H3MlxModel, steps: int) -> H3Schedule:
    """Return the preset's sigma schedule for *steps* transformer forwards.

    Raises:
        ValueError: If the preset has no schedule for *steps*, naming the supported counts.
    """
    if steps not in model.schedules:
        raise ValueError(f"MiniMax H3 runs {' or '.join(str(count) for count in sorted(model.schedules))} steps, got {steps}.")
    return model.schedules[steps]


def _progress(callback: Callable[[dict[str, Any]], None] | None, current: int, total: int) -> None:
    """Report *current* of *total* steps."""
    if callback is not None:
        callback({"phase": _PROGRESS_PHASE, "current_step": current, "total_steps": total})


def _generate(
    model: _H3MlxModel,
    prompt: str,
    *,
    width: int,
    height: int,
    frames: int,
    seed: int,
    steps: int,
    images: Sequence[Image.Image] = (),
    anchors: Sequence[str] = (),
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> _Clip:
    """Make one clip and release everything it held.

    Text-to-video runs the fork's low-memory pipeline, which has no step hook, so it reports its start and end. With images the
    staged path reports every denoise step.

    Raises:
        ValueError: If images are given to a model loaded for text-to-video, or the preset has no schedule for *steps*.
    """
    schedule = _schedule(model, steps)

    import mlx.core as mx
    import numpy as np

    if images and model.mode == "t2v":
        raise ValueError("This MiniMax H3 model was loaded for text-to-video; reload it for image-to-video to use images.")
    pipe = None
    try:
        with _process_limits(mx):
            if images:
                _progress(step_callback, 0, schedule.forwards)
                video, audio, sample_rate = generate_conditioned(
                    fl2va_dir=model.files.fl2va_dir,
                    transformer_dir=model.files.transformer_dir,
                    text_encoder_dir=model.text_encoder_dir,
                    turbo_lora_path=model.turbo_lora_path,
                    turbo_lora_alpha=model.turbo_lora_alpha,
                    turbo_lora_scale=model.turbo_lora_scale,
                    schedule=schedule,
                    prompt=prompt,
                    width=width,
                    height=height,
                    frames=frames,
                    seed=seed,
                    images=images,
                    anchors=anchors,
                    step_callback=lambda current, total: _progress(step_callback, current, total),
                    verbose=model.verbose,
                )
                return _Clip(video=video, audio=audio, sample_rate=sample_rate)
            _progress(step_callback, 0, 1)
            pipe = _load_pipeline(model, schedule)
            result = pipe(
                prompt,
                duration_seconds=frames / H3_FPS,
                num_inference_steps=schedule.num_inference_steps,
                seed=seed,
                height=height,
                width=width,
                verbose=model.verbose,
            )
            clip = _Clip(video=np.asarray(result.video), audio=np.asarray(result.audio), sample_rate=int(result.sample_rate))
            del result
    finally:
        pipe = None
        release_memory()
    _progress(step_callback, 1, 1)
    return clip


# ---- Writing clips --------------------------------------------------------------------------------------------------


def _fit_image(path: str | Path, width: int, height: int) -> Image.Image:
    """Open an image as RGB, scaled to cover ``width x height`` and centre-cropped to it (as LTX does)."""
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        return ImageOps.fit(opened.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    """Return *audio* cut, or padded with silence, to exactly *samples* samples."""
    import numpy as np

    if audio.shape[1] >= samples:
        return audio[:, :samples]
    return np.pad(audio, ((0, 0), (0, samples - audio.shape[1])))


def _write_clip(clip: _Clip, output_path: Path) -> Path:
    """Encode a clip's frames and mux its audio, writing *output_path* atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=output_path.parent, prefix=".h3-"))
    try:
        _, height, width, _ = clip.video.shape
        video = work / "video.mp4"
        write_segment_frames(clip.video, build_segment_command(width, height, _FPS_TEXT, video))
        wav = work / "audio.wav"
        # The audio never outlasts the picture: the container's length is the longer track's.
        write_pcm16_wav(clip.audio[:, : round(len(clip.video) / H3_FPS * clip.sample_rate)], wav, clip.sample_rate)
        muxed = work / "muxed.mp4"
        mux_audio(video, wav, muxed)
        os.replace(muxed, output_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return output_path


def _reject_upscale(kwargs: dict[str, Any]) -> None:
    """Refuse the upscale options: MiniMax H3 on macOS has no upscale.

    Raises:
        ValueError: If ``stage1_steps`` or ``review`` is set.
    """
    if kwargs.get("stage1_steps") is not None or kwargs.get("review") is not None:
        raise ValueError("MiniMax H3 on macOS does not support upscale.")


class H3MlxVideoBackend:
    """Video backend for MiniMax H3 on macOS (the vendored MLX port)."""

    name: str = "h3"

    def load_model(self, model_path: str, **kwargs: Any) -> tuple[Any, VideoModelInfo]:
        """Resolve the model's files, make the text-encoder copy, and prepare the LoRA; the pipeline is built per clip.

        Args:
            model_path: A local folder holding ``FL2VA/`` or the Hub reference of the MiniMax H3 repository. A local folder
                may carry an MLX INT8 transformer, which replaces the preset's (see :func:`local_int8_transformer`).
            **kwargs:
                mode: ``"t2v"`` (default), ``"i2v"`` or ``"extend"``. Text-to-video streams the text encoder and loads no
                    vision tower; the image modes encode the keyframes and the prompt, denoise and decode in stages, each
                    loading one component and dropping it before the next.
                preset: The ``H3MlxPreset`` (required): its files, its frame rule and its sigma schedules, which also set the
                    allowed ``steps``.
                lora: The turbo LoRA's ``HubFiles`` (required).
                lora_scale: The turbo LoRA's strength (required; the config's ``lora.scale``).
                loras: ``(path, weight)`` of each user LoRA.
                model_name: The name the text-encoder copy is stored under; defaults to ``"h3"``.
                models_dir: The models folder; defaults to the app's.
                verbose: Whether the fork prints its progress; defaults to True (the web runner passes False).
                low_memory: Accepted and ignored. upscale: Must be falsy.

        Returns:
            Tuple of (backend-private model handle, VideoModelInfo).

        Raises:
            ValueError: For an unknown option, a missing preset or LoRA, an upscale request, a user LoRA this port cannot take,
                or a local folder whose transformer is not in the MLX INT8 layout.
        """
        unknown = sorted(set(kwargs) - _LOAD_KEYS)
        if unknown:
            raise ValueError(f"MiniMax H3 on macOS does not take the option(s): {', '.join(unknown)}.")
        if kwargs.get("upscale"):
            raise ValueError("MiniMax H3 on macOS does not support upscale.")
        mode = kwargs.get("mode", "t2v")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {', '.join(_MODES)}, got {mode!r}.")
        preset, lora, lora_scale = kwargs.get("preset"), kwargs.get("lora"), kwargs.get("lora_scale")
        if not isinstance(preset, H3MlxPreset) or not isinstance(lora, HubFiles) or lora_scale is None:
            raise ValueError("MiniMax H3 on macOS needs its preset and turbo LoRA to load (preset=, lora=, lora_scale=).")
        scale = float(lora_scale)
        user_loras = [(Path(path), float(weight)) for path, weight in kwargs.get("loras") or ()]

        # A converted folder holds its turbo merge (or needs none), so it gets no turbo LoRA.
        converted = _converted(Path(model_path).expanduser())
        files = resolve_mlx_files(model_path, preset, lora=None if converted else lora)
        if converted:
            # Stock's own text encoder files, so stock and every converted model share one copy and none rebuilds it.
            files = replace(files, text_encoder_source=snapshot_with(preset.components, hub_snapshot_download) / "FL2VA" / "text_encoder")
        directory: str | None = None
        lora_path: Path | None
        if user_loras:
            directory = tempfile.mkdtemp(prefix="ziv-h3-lora-")
            try:
                adapter = combine_loras(None if converted else files.lora_file, scale, user_loras, Path(directory) / "adapter.safetensors")
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            lora_path, alpha, adapter_scale = adapter, None, 1.0
        elif converted:
            lora_path, alpha, adapter_scale = None, None, 1.0
        else:
            lora_path, alpha, adapter_scale = files.lora_file, scale * _turbo_rank(files.lora_file), 1.0
        # A converted model shares stock's text encoder, so its copy is stock's.
        copy_name = "h3" if converted else kwargs.get("model_name") or "h3"
        try:
            text_encoder_dir = ensure_text_encoder_copy(Path(kwargs.get("models_dir") or get_ziv_data_dir() / "models"), copy_name, files, preset)
        except BaseException:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
            raise
        model = _H3MlxModel(files, text_encoder_dir, lora_path, alpha, adapter_scale, mode, bool(kwargs.get("verbose", True)), preset.frame_rule, preset.schedules)
        if directory is not None:
            weakref.finalize(model, shutil.rmtree, directory, True)
        return model, _MODEL_INFO

    def text_to_video(
        self,
        model: Any,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate a video, with sound, from a text prompt.

        Args:
            model: Handle returned by ``load_model``.
            prompt: Text prompt.
            width: Video width (a multiple of 32).
            height: Video height (a multiple of 32).
            num_frames: Frames at 24 fps; the fork rounds up to 17 * n + 5.
            seed: Random seed.
            steps: Transformer forwards; a step count of the preset's schedules (4 or 8).
            output_path: Path to save the video.
            step_callback: Receives a progress event when the clip starts and when it ends.
            **kwargs: ``stage1_steps`` and ``review`` are refused.

        Returns:
            Path to the generated video.

        Raises:
            ValueError: If an upscale option is set, or the preset has no schedule for *steps*.
        """
        _reject_upscale(kwargs)
        clip = _generate(model, prompt, width=width, height=height, frames=num_frames, seed=seed, steps=steps, step_callback=step_callback)
        return _write_clip(clip, Path(output_path))

    def image_to_video(
        self,
        model: Any,
        image_path: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate a video, with sound, from a first-frame image (and optionally a last-frame image) and a prompt.

        Args:
            model: Handle returned by ``load_model`` with ``mode="i2v"``.
            image_path: The first frame, scaled to cover the canvas and centre-cropped.
            prompt: Text prompt.
            width: Video width.
            height: Video height.
            num_frames: Frames at 24 fps.
            seed: Random seed.
            steps: Transformer forwards.
            output_path: Path to save the video.
            step_callback: Receives a progress event when the clip starts and after every denoise step.
            **kwargs: ``end_image``, the last frame's path; ``stage1_steps`` and ``review`` are refused.

        Returns:
            Path to the generated video.

        Raises:
            ValueError: If an upscale option is set, or the model was loaded for text-to-video.
        """
        _reject_upscale(kwargs)
        images = [_fit_image(image_path, width, height)]
        anchors = ["first"]
        if kwargs.get("end_image"):
            images.append(_fit_image(kwargs["end_image"], width, height))
            anchors.append("last")
        clip = _generate(model, prompt, width=width, height=height, frames=num_frames, seed=seed, steps=steps, images=images, anchors=anchors, step_callback=step_callback)
        return _write_clip(clip, Path(output_path))

    def extend_video(
        self,
        model: Any,
        source: ExtendSource,
        prompt: str,
        *,
        extend_frames: int,
        seed: int,
        steps: int,
        segment_dir: str | Path,
        audio_output: bool,
        step_callback: Any | None = None,
    ) -> ExtendSegment | None:
        """Generate the continuation of a clip as image-to-video from its last frame.

        The last source frame is the first-frame anchor of a clip of ``extend_frames + 1`` frames moved to the frame rule. Its first frame
        repeats the anchor, so the segment is the next ``extend_frames`` frames, with the audio cut to the same span.

        Args:
            model: Handle returned by ``load_model`` with ``mode="extend"`` (or ``"i2v"``).
            source: The checked clip to continue (24 fps, a size that is a multiple of 32).
            prompt: Text prompt for what happens next.
            extend_frames: New frames; a positive multiple of 17.
            seed: Random seed.
            steps: Transformer forwards; a step count of the preset's schedules.
            segment_dir: Existing directory that receives ``segment.mp4`` and, with audio, ``segment.wav``.
            audio_output: Whether to write the new part's sound (only when the clip has any).
            step_callback: Receives a progress event when the clip starts and after every denoise step.

        Returns:
            The generated segment: its video and, with audio, a 48 kHz WAV that starts at the new part.

        Raises:
            ValueError: If ``extend_frames`` is not a positive multiple of 17, the source is not 24 fps or not a multiple of
                32 in size, the model was loaded for text-to-video, or the preset has no schedule for *steps*.
            RuntimeError: If the clip's last frame cannot be read, or the pipeline returns fewer frames than asked.
        """
        if extend_frames < H3_FRAMES_PER_CHUNK or extend_frames % H3_FRAMES_PER_CHUNK:
            raise ValueError(f"extend_frames must be a positive multiple of {H3_FRAMES_PER_CHUNK}, got {extend_frames}.")
        if abs(source.fps - H3_FPS) > EXTEND_FPS_TOLERANCE:
            raise ValueError(f"MiniMax H3 extends 24 fps clips only; the source runs at {source.fps:g} fps.")
        if source.width % H3_ROW_PIXELS or source.height % H3_ROW_PIXELS:
            raise ValueError(f"MiniMax H3 extends clips whose size is a multiple of {H3_ROW_PIXELS}; the source is {source.width}x{source.height}.")
        if model.mode == "t2v":
            raise ValueError("This MiniMax H3 model was loaded for text-to-video; reload it with mode='extend' to extend a clip.")
        _schedule(model, steps)

        from PIL import Image

        last_frame = Image.fromarray(read_source_tail(source)[-1])
        clip = _generate(
            model,
            prompt,
            width=source.width,
            height=source.height,
            frames=align_frames(extend_frames + 1, model.frame_rule),
            seed=seed,
            steps=steps,
            images=[last_frame],
            anchors=["first"],
            step_callback=step_callback,
        )
        if len(clip.video) < 1 + extend_frames:
            raise RuntimeError(f"MiniMax H3 returned {len(clip.video)} frames, expected at least {1 + extend_frames}.")

        directory = Path(segment_dir)
        video_path = directory / "segment.mp4"
        command = build_segment_command(source.width, source.height, source.fps_text, video_path, color_space=source.color_space, color_primaries=source.color_primaries, color_trc=source.color_trc)
        write_segment_frames(clip.video[1 : 1 + extend_frames], command)

        audio_path: Path | None = None
        if audio_output and source.has_audio:
            rate = clip.sample_rate
            span = clip.audio[:, round(rate / H3_FPS) : round((1 + extend_frames) * rate / H3_FPS)]
            audio_path = directory / "segment.wav"
            write_pcm16_wav(_fit_length(resample_audio(span, rate, AUDIO_SAMPLE_RATE), round(extend_frames / H3_FPS * AUDIO_SAMPLE_RATE)), audio_path)
        return ExtendSegment(video_path=video_path, audio_path=audio_path, lead_samples=0)
