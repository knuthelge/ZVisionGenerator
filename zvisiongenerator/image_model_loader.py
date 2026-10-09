"""Load an image model, reusing or creating its stored quant when a quantize level is selected.

Quantization stays opt-in. When a job selects a level, the first load saves the quantized base weights in
the models directory as ``<name>@q<bits>`` and later loads read that copy instead of quantizing again.
``<name>`` is the installed model's folder, or the alias the user picked for a Hugging Face model (whose
downloaded files are the source). LoRAs are never baked into a stored quant: they are applied at load time
on top of it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.stored_quant import (
    build_manifest,
    copy_detection_files,
    discard_partial,
    flush_to_disk,
    is_current,
    parse_stored_quant_name,
    partial_dir,
    promote_partial,
    stored_quant_dir,
    stored_quant_name,
    sweep_stale_partials,
    write_manifest,
)

__all__ = ["LOADING_PHASE", "SAVING_QUANT_PHASE", "LoadPlan", "create_stored_quant", "load_image_model", "plan_image_model_load", "save_stored_quant"]

SAVING_QUANT_PHASE = "saving_quant"
# Reported when the model loads again after its stored quant was saved.
LOADING_PHASE = "loading"
_PRECISION = "bfloat16"


@dataclass(frozen=True)
class LoadPlan:
    """How to load a model: from *path* at *quantize*, optionally then saving a stored quant at *create*.

    *source* is the local folder the stored quant is made from, or ``None`` while a Hugging Face model is not
    downloaded yet (it is resolved again after the load downloads it).
    """

    path: str
    quantize: int | None
    create: Path | None = None
    source: Path | None = None


def plan_image_model_load(
    model_path: str,
    quantize: int | None,
    *,
    models_dir: Path,
    backend_format: str | None,
    model_name: str | None = None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> LoadPlan:
    """Decide whether to load the source, an existing stored quant, or create one after loading.

    Args:
        model_path: Resolved model path (a local directory, alias target, or repo id).
        quantize: Selected quantize level, or ``None``.
        models_dir: The installed-models directory, where stored quants live.
        backend_format: The backend's :meth:`ImageBackend.stored_quant_format`, ``None`` when unsupported.
        model_name: The name the user picked (an installed model or alias). Models outside *models_dir* get
            a stored quant only under a plain picked name; a raw repo id or path is quantized at load.
        find_local_dir: Resolves a model reference to its fully downloaded local folder.
    """
    path = Path(model_path).expanduser()
    if parse_stored_quant_name(path.name) is not None:
        # A stored quant selected directly is already quantized.
        return LoadPlan(model_path, None)
    if quantize is None or backend_format is None:
        return LoadPlan(model_path, quantize)
    if _is_installed(path, models_dir):
        source: Path | None = path
        target = stored_quant_dir(path, quantize)
    elif model_name is not None and model_name != model_path and _is_plain_name(model_name):
        # An alias: resolution mapped the picked name to a repo id or path. An unmapped name (e.g. a folder
        # in the current directory) is a raw path and gets no copy.
        source = find_local_dir(model_path)
        target = models_dir / stored_quant_name(model_name, quantize)
    else:
        return LoadPlan(model_path, quantize)
    if source is not None and is_current(target, source, quantize, backend_format):
        return LoadPlan(str(target), None)
    return LoadPlan(model_path, quantize, create=target, source=source)


def save_stored_quant(
    backend: ImageBackend,
    model: Any,
    *,
    source: Path,
    target: Path,
    bits: int,
    backend_format: str,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Save *model* as *source*'s stored quant at *target*; return whether the copy is now in place.

    The write runs on the calling thread, so a stop request takes effect once it finishes: when *cancelled*
    is then true the copy is discarded. See :func:`_store` for how partial and failed saves are handled.
    """
    return _store(lambda partial: backend.save_quantized(model, str(partial)), source=source, target=target, bits=bits, backend_format=backend_format, cancelled=cancelled)


def _write_stored_quant(
    backend: ImageBackend,
    *,
    source: Path,
    target: Path,
    bits: int,
    backend_format: str,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Write *source*'s stored quant at *target* from its files, without loading the model; return whether it is in place.

    The backend checks *cancelled* while it writes, so a stop takes effect after its current step.
    """
    return _store(
        lambda partial: backend.write_quantized_files(str(source), str(partial), bits, cancelled),
        source=source,
        target=target,
        bits=bits,
        backend_format=backend_format,
        cancelled=cancelled,
    )


def create_stored_quant(backend: ImageBackend, source: Path, target: Path, bits: int) -> bool:
    """Create *source*'s stored quant at *target* the way the backend stores *bits*; return whether it is in place.

    Raises:
        RuntimeError: When the backend cannot store quantized weights.
    """
    backend_format = backend.stored_quant_format(bits)
    if backend_format is None:
        raise RuntimeError(f"The {backend.name} backend cannot store quantized weights.")
    if backend.quantizes_from_files(bits):
        return _write_stored_quant(backend, source=source, target=target, bits=bits, backend_format=backend_format)
    model, _info = backend.load_model(str(source), quantize=bits, precision=_PRECISION)
    return save_stored_quant(backend, model, source=source, target=target, bits=bits, backend_format=backend_format)


def _store(
    write: Callable[[Path], None],
    *,
    source: Path,
    target: Path,
    bits: int,
    backend_format: str,
    cancelled: Callable[[], bool] | None,
) -> bool:
    """Run *write* into a hidden partial folder, complete it and rename it to *target*; return whether it is in place.

    Leftover partial folders of *target* from an interrupted earlier save are removed first. A failure warns
    and leaves no folder behind; the job carries on without the copy. When *cancelled* is true once the write
    returns, the copy is discarded. Interrupts (Ctrl-C, exit) also discard the partial folder.
    """
    sweep_stale_partials(target)
    partial = partial_dir(target)
    try:
        write(partial)
        if cancelled is not None and cancelled():
            discard_partial(partial)
            return False
        copy_detection_files(source, partial)
        flush_to_disk(partial)
        write_manifest(partial, build_manifest(source, bits, backend_format))
    except Exception as exc:  # noqa: BLE001 - a failed save must never fail the job
        discard_partial(partial)
        warnings.warn(f"Could not save the q{bits} copy of {source.name} ({exc}); it is quantized at load instead.", stacklevel=3)
        return False
    except BaseException:
        discard_partial(partial)
        raise
    try:
        promote_partial(partial, target)
    except OSError as exc:
        discard_partial(partial)
        warnings.warn(f"Could not store the q{bits} copy of {source.name} ({exc}); it is quantized at load instead.", stacklevel=3)
        return False
    return True


def load_image_model(
    backend: ImageBackend,
    model_path: str,
    *,
    quantize: int | None,
    models_dir: Path,
    model_name: str | None = None,
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    on_phase: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    release_memory: Callable[[], None] | None = None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> tuple[Any, ImageModelInfo]:
    """Load *model_path*, reusing its stored quant at *quantize* or creating it on first use.

    Args:
        backend: The image backend.
        model_path: Resolved model path.
        quantize: Selected quantize level, or ``None`` (never creates a stored quant).
        models_dir: The installed-models directory, where stored quants live.
        model_name: The name the user picked; names the stored quant of a Hugging Face (alias) model.
        lora_paths: LoRAs applied at load time (never baked into the stored quant).
        lora_weights: Scale per LoRA.
        on_phase: Receives :data:`SAVING_QUANT_PHASE` when the stored quant starts saving (after the model has
            loaded, or before it loads when the backend writes the level from the source files), then
            :data:`LOADING_PHASE` when the model loads again after the save.
        cancelled: Checked when the save finishes (and during a write from files); when true the copy is
            discarded (the job's own stop handling follows).
        release_memory: Frees accelerator memory between the LoRA-free load used for saving and the LoRA load.
        find_local_dir: Resolves a model reference to its fully downloaded local folder.
    """
    backend_format = backend.stored_quant_format(quantize) if quantize is not None else None
    plan = plan_image_model_load(model_path, quantize, models_dir=models_dir, backend_format=backend_format, model_name=model_name, find_local_dir=find_local_dir)
    if plan.create is None or plan.quantize is None or backend_format is None:
        return backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)
    if backend.quantizes_from_files(plan.quantize):
        return _write_then_load(backend, plan, plan.create, plan.quantize, backend_format, lora_paths=lora_paths, lora_weights=lora_weights, on_phase=on_phase, cancelled=cancelled)

    # Saved weights must be LoRA-free: mflux bakes LoRAs into what it saves.
    model, info = backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION)
    # A Hugging Face model that was not downloaded before is now, unless the load read it some other way.
    source = plan.source or find_local_dir(plan.path)
    saved = False
    if source is not None:
        if on_phase is not None:
            on_phase(SAVING_QUANT_PHASE)
        saved = save_stored_quant(backend, model, source=source, target=plan.create, bits=plan.quantize, backend_format=backend_format, cancelled=cancelled)
    if not lora_paths or (not saved and cancelled is not None and cancelled()):
        # A stopped job quits before its first generation, so it never needs the LoRA load.
        return model, info
    del model
    if release_memory is not None:
        release_memory()
    if on_phase is not None and source is not None:
        on_phase(LOADING_PHASE)
    if saved:
        return backend.load_model(str(plan.create), quantize=None, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)
    return backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)


def _write_then_load(
    backend: ImageBackend,
    plan: LoadPlan,
    target: Path,
    bits: int,
    backend_format: str,
    *,
    lora_paths: list[str] | None,
    lora_weights: list[float] | None,
    on_phase: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
) -> tuple[Any, ImageModelInfo]:
    """Write *plan*'s stored quant at *bits* to *target* from the source files, then load it with the LoRAs.

    A model that is not downloaded yet is loaded (and downloaded) at its quantize level instead, and the next
    job writes the copy. A failed write also loads at the quantize level. A write stopped by the user returns
    ``(None, info)`` without loading anything: the runners quit before their first image and never use the model.
    """
    if plan.source is not None:
        if on_phase is not None:
            on_phase(SAVING_QUANT_PHASE)
        if _write_stored_quant(backend, source=plan.source, target=target, bits=bits, backend_format=backend_format, cancelled=cancelled):
            if on_phase is not None:
                on_phase(LOADING_PHASE)
            return backend.load_model(str(target), quantize=None, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)
        if cancelled is not None and cancelled():
            return None, detect_image_model(str(plan.source))  # the local copy: no Hub request on the way out
    return backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)


def _is_plain_name(name: str) -> bool:
    """Return whether *name* can name a folder in the models directory (an alias, not a repo id or path)."""
    return bool(name) and name not in {".", ".."} and not any(char in name for char in "/\\:~") and parse_stored_quant_name(name) is None


def _is_installed(path: Path, models_dir: Path) -> bool:
    """Return whether *path* is a model folder directly inside *models_dir*."""
    try:
        return path.is_dir() and path.parent.resolve() == models_dir.resolve()
    except OSError:
        return False
