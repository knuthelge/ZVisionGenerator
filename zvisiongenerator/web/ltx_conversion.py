"""LTX checkpoint conversion from the Models page: inspect a checkpoint, then convert it as a background job with progress.

An LTX conversion takes minutes, so it runs the converter CLI in a subprocess on its own thread and reads its
``Converting: <stage> (done/total)`` lines. The caller's exclusive slot (no generation job while it runs, since the
conversion holds about 21 GB and the GPU) is held until the subprocess ends.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.ltx_convert import PROGRESS_PREFIX, distill_advice, inspect_ltx
from zvisiongenerator.utils.subprocess_env import utf8_child_env

__all__ = ["conversion_status", "inspect_checkpoint_for_form", "recommended_ltx_quantize", "start_conversion"]

_PROGRESS_RE = re.compile(rf"^{re.escape(PROGRESS_PREFIX)}(?P<stage>.+?)(?: \((?P<done>\d+)/(?P<total>\d+)\))?$")
# The q8 stock model's estimate decides the level a machine converts at.
_REFERENCE_ALIAS = "ltx-8"
_SYSTEM_FITS_RATIO = 0.8
# Unified memory of the smallest Mac that runs ltx-8; a Mac sold as "32 GB" reports exactly 32 GiB.
_UNIFIED_Q8_BYTES = 32 * 1024**3

_lock = threading.Lock()
_status: dict[str, Any] = {"state": "idle"}


def inspect_checkpoint_for_form(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return what the convert form shows for the LTX checkpoint at *path*: its layout, distillation and the level to preselect.

    Raises:
        ValueError: If it is not an LTX-2.3 checkpoint the converter reads.
    """
    inspection = inspect_ltx(path)
    return {
        "layout": inspection.layout,
        "fp8_weights": inspection.checkpoint.fp8_weights,
        "distillation": inspection.distillation,
        "advice": distill_advice(inspection.distillation, "Distill on"),
        "recommend_distill": inspection.distillation == "dev",
        "recommended_quantize": recommended_ltx_quantize(config),
    }


def recommended_ltx_quantize(config: dict[str, Any]) -> int:
    """Return 8 when this machine holds LTX at q8 comfortably, else 4 (also when that can't be estimated).

    With unified memory (Apple Silicon) that is 32 GB or more, the smallest Mac that runs ``ltx-8``; with a
    discrete GPU, the q8 stock model's system-memory estimate within 80% of the system memory.
    """
    from zvisiongenerator.backends import accelerator_memory_budget_or_none
    from zvisiongenerator.utils.model_memory import ltx_cuda_system_memory
    from zvisiongenerator.utils.paths import resolve_model_path

    budget = accelerator_memory_budget_or_none()
    if budget is not None and budget.unified_bytes:
        return 8 if budget.unified_bytes >= _UNIFIED_Q8_BYTES else 4
    if budget is None or not budget.system_bytes:
        return 4
    try:
        resolved = resolve_model_path(_REFERENCE_ALIAS, aliases=config.get("model_aliases", {}), platform_key=sys.platform)
        system = ltx_cuda_system_memory(resolved, _REFERENCE_ALIAS, config)
    except Exception:  # noqa: BLE001 - an estimate that fails only means no recommendation beyond q4
        return 4
    return 8 if system is not None and system.taking_turns <= budget.system_bytes * _SYSTEM_FITS_RATIO else 4


def start_conversion(args: list[str], name: str, slot: AbstractContextManager[None]) -> None:
    """Run the converter CLI with *args* in the background, holding the entered *slot* until it ends.

    Args:
        args: The converter's arguments (``model --input ... --model-type ltx-2.3 ...``).
        name: The model's name, for the status.
        slot: The exclusive slot, already entered; it is exited when the conversion ends.
    """
    with _lock:
        _status.clear()
        _status.update({"state": "running", "name": name, "stage": "Starting", "done": 0, "total": 0, "message": None})
    threading.Thread(target=_run, args=(args, slot), name="ziv-ltx-convert", daemon=True).start()


def conversion_status() -> dict[str, Any]:
    """Return the current or last LTX conversion's state (``idle``, ``running``, ``done`` or ``failed``) and progress."""
    with _lock:
        return dict(_status)


def _run(args: list[str], slot: AbstractContextManager[None]) -> None:
    """Run the converter and record its progress and outcome, then release *slot*."""
    lines: list[str] = []
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "zvisiongenerator.converters.convert_checkpoint", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=utf8_child_env(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            lines.append(line)
            _record_progress(line)
        code = process.wait()
        with _lock:
            if code == 0:
                _status.update({"state": "done", "message": f"Converted the checkpoint into '{_status.get('name')}'."})
            else:
                _status.update({"state": "failed", "message": _failure_message(lines, code)})
    except Exception as exc:  # noqa: BLE001 - reported as the job's failure
        with _lock:
            _status.update({"state": "failed", "message": str(exc)})
    finally:
        slot.__exit__(None, None, None)


def _failure_message(lines: list[str], code: int) -> str:
    """Return why the converter that printed *lines* exited with *code*: its last error, else its last line unless that is progress, else how it ended."""
    errors = [line.removeprefix("Error: ") for line in lines if line.startswith("Error: ")]
    if errors:
        return errors[-1]
    if lines and lines[-1] and _PROGRESS_RE.match(lines[-1]) is None:
        return lines[-1]
    if code < 0:
        return f"The conversion was stopped by signal {-code}, usually because the system ran out of memory."
    return f"The conversion stopped with exit code {code}."


def _record_progress(line: str) -> None:
    """Record the stage and counts of a progress *line*; other lines are ignored."""
    match = _PROGRESS_RE.match(line)
    if match is None:
        return
    with _lock:
        _status.update({"stage": match["stage"], "done": int(match["done"] or 0), "total": int(match["total"] or 0)})
