"""Load an image model, reusing or creating its stored quant when a quantize level is selected.

Quantization stays opt-in. When a job selects a level, the first load saves the quantized base weights in
the models directory as ``<name>@q<bits>`` and later loads read that copy instead of quantizing again.
``<name>`` is the installed model's folder, or the alias the user picked for a Hugging Face model (whose
downloaded files are the source). LoRAs are never baked into a stored quant: they are applied at load time
on top of it.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.stored_quant import (
    build_manifest,
    copy_detection_files,
    discard_partial,
    is_current,
    parse_stored_quant_name,
    partial_dir,
    promote_partial,
    stored_quant_dir,
    stored_quant_name,
    write_manifest,
)

__all__ = ["SAVING_QUANT_PHASE", "LoadPlan", "load_image_model", "plan_image_model_load", "save_stored_quant"]

SAVING_QUANT_PHASE = "saving_quant"
_POLL_SECONDS = 0.2
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
    elif model_name is not None and _is_plain_name(model_name):
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
    poll_seconds: float = _POLL_SECONDS,
) -> bool:
    """Save *model* as *source*'s stored quant at *target*; return whether the copy is now in place.

    The copy is written to a hidden partial folder and renamed when complete. A failure warns and leaves no
    folder behind; the job carries on with the in-memory model. When *cancelled* turns true the save is
    abandoned at once: the write finishes in the background and its partial folder is removed.
    """
    partial = partial_dir(target)
    done = threading.Event()
    abandoned = threading.Event()
    outcome: dict[str, BaseException] = {}

    def _write() -> None:
        try:
            backend.save_quantized(model, str(partial))
            copy_detection_files(source, partial)
            write_manifest(partial, build_manifest(source, bits, backend_format))
        except BaseException as exc:  # noqa: BLE001 - reported to the waiting thread
            outcome["error"] = exc
        finally:
            done.set()
            if abandoned.is_set() or "error" in outcome:
                discard_partial(partial)

    threading.Thread(target=_write, name="ziv-save-quant", daemon=True).start()
    while not done.wait(poll_seconds):
        if cancelled is not None and cancelled():
            abandoned.set()
            if done.is_set():
                discard_partial(partial)
            return False
    if "error" in outcome:
        warnings.warn(f"Could not save the q{bits} copy of {source.name} ({outcome['error']}); it is quantized at load instead.", stacklevel=2)
        return False
    try:
        promote_partial(partial, target)
    except OSError as exc:
        discard_partial(partial)
        warnings.warn(f"Could not store the q{bits} copy of {source.name} ({exc}); it is quantized at load instead.", stacklevel=2)
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
        on_phase: Receives :data:`SAVING_QUANT_PHASE` when the model has loaded and its stored quant is being saved.
        cancelled: Polled while saving; when true the save is abandoned (the job's own stop handling follows).
        release_memory: Frees accelerator memory between the LoRA-free load used for saving and the LoRA load.
        find_local_dir: Resolves a model reference to its fully downloaded local folder.
    """
    plan = plan_image_model_load(model_path, quantize, models_dir=models_dir, backend_format=backend.stored_quant_format(), model_name=model_name, find_local_dir=find_local_dir)
    if plan.create is None:
        return backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)

    # Saved weights must be LoRA-free: mflux bakes LoRAs into what it saves.
    model, info = backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION)
    # A Hugging Face model that was not downloaded before is now, unless the load read it some other way.
    source = plan.source or find_local_dir(plan.path)
    saved = False
    if source is not None:
        if on_phase is not None:
            on_phase(SAVING_QUANT_PHASE)
        saved = save_stored_quant(backend, model, source=source, target=plan.create, bits=plan.quantize, backend_format=backend.stored_quant_format(), cancelled=cancelled)
    if not lora_paths or (not saved and cancelled is not None and cancelled()):
        # A stopped job quits before its first generation, so it never needs the LoRA load.
        return model, info
    del model
    if release_memory is not None:
        release_memory()
    if saved:
        return backend.load_model(str(plan.create), quantize=None, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)
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
