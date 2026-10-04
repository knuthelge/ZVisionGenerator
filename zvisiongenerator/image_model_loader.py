"""Load an image model, reusing or creating its stored quant when a quantize level is selected.

Quantization stays opt-in. When a job selects a level for a model installed in the models directory, the
first load saves the quantized base weights next to the source (``<name>@q<bits>``) and later loads read
that copy instead of quantizing again. LoRAs are never baked into a stored quant: they are applied at load
time on top of it.
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
from zvisiongenerator.utils.stored_quant import (
    build_manifest,
    copy_detection_files,
    discard_partial,
    is_current,
    parse_stored_quant_name,
    partial_dir,
    promote_partial,
    stored_quant_dir,
    write_manifest,
)

__all__ = ["SAVING_QUANT_PHASE", "LoadPlan", "load_image_model", "plan_image_model_load", "save_stored_quant"]

SAVING_QUANT_PHASE = "saving_quant"
_POLL_SECONDS = 0.2
_PRECISION = "bfloat16"


@dataclass(frozen=True)
class LoadPlan:
    """How to load a model: from *path* at *quantize*, optionally first saving a stored quant at *create*."""

    path: str
    quantize: int | None
    create: Path | None = None


def plan_image_model_load(model_path: str, quantize: int | None, *, models_dir: Path, backend_format: str | None) -> LoadPlan:
    """Decide whether to load the source, an existing stored quant, or create one first.

    Args:
        model_path: Resolved model path (a local directory, alias target, or repo id).
        quantize: Selected quantize level, or ``None``.
        models_dir: The installed-models directory; only models directly inside it get stored quants.
        backend_format: The backend's :meth:`ImageBackend.stored_quant_format`, ``None`` when unsupported.
    """
    path = Path(model_path).expanduser()
    if parse_stored_quant_name(path.name) is not None:
        # A stored quant selected directly is already quantized.
        return LoadPlan(model_path, None)
    if quantize is None or backend_format is None or not _is_installed(path, models_dir):
        return LoadPlan(model_path, quantize)
    target = stored_quant_dir(path, quantize)
    if is_current(target, path, quantize, backend_format):
        return LoadPlan(str(target), None)
    return LoadPlan(model_path, quantize, create=target)


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
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    on_phase: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    release_memory: Callable[[], None] | None = None,
) -> tuple[Any, ImageModelInfo]:
    """Load *model_path*, reusing its stored quant at *quantize* or creating it on first use.

    Args:
        backend: The image backend.
        model_path: Resolved model path.
        quantize: Selected quantize level, or ``None`` (never creates a stored quant).
        models_dir: The installed-models directory.
        lora_paths: LoRAs applied at load time (never baked into the stored quant).
        lora_weights: Scale per LoRA.
        on_phase: Receives :data:`SAVING_QUANT_PHASE` before a stored quant is created.
        cancelled: Polled while saving; when true the save is abandoned (the job's own stop handling follows).
        release_memory: Frees accelerator memory between the LoRA-free load used for saving and the LoRA load.
    """
    plan = plan_image_model_load(model_path, quantize, models_dir=models_dir, backend_format=backend.stored_quant_format())
    if plan.create is None:
        return backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION, lora_paths=lora_paths, lora_weights=lora_weights)

    if on_phase is not None:
        on_phase(SAVING_QUANT_PHASE)
    source = Path(plan.path).expanduser()
    # Saved weights must be LoRA-free: mflux bakes LoRAs into what it saves.
    model, info = backend.load_model(plan.path, quantize=plan.quantize, precision=_PRECISION)
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


def _is_installed(path: Path, models_dir: Path) -> bool:
    """Return whether *path* is a model folder directly inside *models_dir*."""
    try:
        return path.is_dir() and path.parent.resolve() == models_dir.resolve()
    except OSError:
        return False
