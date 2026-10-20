"""Convert a single-file LTX-2.3 checkpoint into a models-folder entry at one quantize level (``--model-type ltx-2.3``).

The folder is written into a hidden partial folder and moved into place once complete, so an interrupted
conversion never leaves a model that is listed and fails to load. Its layout depends on the platform (see
:func:`zvisiongenerator.backends.convert_ltx_checkpoint`).

Progress goes to standard output as ``Converting: <stage>`` lines, with ``(done/total)`` where the stage counts
tensors; the Web UI reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint, check_lora, distillation, inspect_checkpoint
from zvisiongenerator.utils.config import ltx_distilled_lora_source
from zvisiongenerator.utils.stored_quant import discard_partial, partial_dir, promote_partial, sweep_stale_partials

__all__ = ["LtxInspection", "convert_ltx", "distill_advice", "inspect_ltx", "progress_line"]

PROGRESS_PREFIX = "Converting: "
_STAGES = {
    "download": "Getting the base model files",
    "connectors": "Checking the text connectors",
    "transformer": "Writing the transformer",
    "text_encoder": "Writing the text encoder",
    "components": "Linking the other components",
}
# Print a counted stage at most this many times.
_PROGRESS_STEPS = 50
# {distill} is how the reader turns the merge on: the CLI flag, or the Web UI's checkbox.
_ADVICE = {
    "dev": "This is plain LTX-2.3 dev, which renders broken at the 8 steps the app runs LTX at. Convert it with {distill} to merge Lightricks' distilled LoRA.",
    "distilled": "This checkpoint carries Lightricks' step distillation, so it runs at 8 steps as it is.",
    "unknown": "Couldn't tell whether this checkpoint is step-distilled. If its model page says to use it with Lightricks' distilled LoRA, convert it with {distill}.",
}


@dataclass(frozen=True)
class LtxInspection:
    """What the converter found in a checkpoint, before converting it.

    Attributes:
        checkpoint: The checked checkpoint.
        distillation: ``dev``, ``distilled`` or ``unknown``.
        advice: What that means for converting it.
    """

    checkpoint: LtxCheckpoint
    distillation: str
    advice: str

    @property
    def layout(self) -> str:
        """``full`` for Lightricks' full checkpoint layout, ``transformer`` for a transformer-only export."""
        return "full" if self.checkpoint.bundles_components else "transformer"


def inspect_ltx(path: Path) -> LtxInspection:
    """Check the checkpoint at *path* and tell whether it is plain dev or step-distilled; nothing is downloaded.

    Raises:
        ValueError: If it is not an LTX-2.3 checkpoint the converter reads.
    """
    checkpoint = inspect_checkpoint(path)
    found = distillation(checkpoint)
    return LtxInspection(checkpoint=checkpoint, distillation=found, advice=distill_advice(found, "--distill"))


def distill_advice(found: str, distill: str) -> str:
    """Return what the distillation *found* (``dev``, ``distilled`` or ``unknown``) means for converting, naming the merge's switch as *distill*."""
    return _ADVICE[found].format(distill=distill)


def convert_ltx(input_path: Path, models_dir: Path, name: str, bits: int, *, distill: bool, force: bool, config: dict[str, Any]) -> Path:
    """Convert the LTX-2.3 checkpoint *input_path* into ``<models_dir>/<name>`` at *bits*; return the folder.

    Args:
        input_path: The single-file checkpoint.
        models_dir: The models folder.
        name: The new model's folder name; neither an existing folder (unless *force*) nor a configured alias.
        bits: 4 or 8, the only level stored.
        distill: Merge Lightricks' distilled LoRA in (for plain-dev checkpoints).
        force: Replace an existing folder of that name, once the new one is complete.
        config: The loaded config.

    Raises:
        ValueError: If the name is taken by an alias, or the checkpoint isn't one the converter reads.
        FileExistsError: If the folder exists and *force* is not set.
        RuntimeError: If the distilled LoRA can't be downloaded, or the conversion fails.
    """
    if bits not in (4, 8):
        raise ValueError("LTX checkpoints need --quantize 4 or 8: that level is the only one stored.")
    if name in (config.get("model_aliases") or {}):
        raise ValueError(f"'{name}' is a model alias; choose another name for the converted model.")
    target = models_dir / name
    if target.exists() and not force:
        raise FileExistsError(f"A model named '{name}' already exists. Choose another name, or pass --force to replace it.")
    inspection = inspect_ltx(input_path)
    checkpoint = inspection.checkpoint
    print(
        f"Checkpoint: {input_path.name} ({'full checkpoint' if inspection.layout == 'full' else 'transformer only'}, {len(checkpoint.transformer)} transformer tensors, {checkpoint.fp8_weights} in FP8)"
    )
    print(inspection.advice if not (distill and inspection.distillation == "dev") else "This is plain LTX-2.3 dev: merging Lightricks' distilled LoRA.")
    if distill and inspection.distillation == "distilled":
        print("Warning: --distill merges Lightricks' distilled LoRA into a checkpoint that already carries its distillation.")
    lora = _distilled_lora(config) if distill else None
    if lora is not None:
        check_lora(checkpoint, lora)
    from zvisiongenerator.backends import convert_ltx_checkpoint

    models_dir.mkdir(parents=True, exist_ok=True)
    sweep_stale_partials(target)
    partial = partial_dir(target)
    try:
        partial.mkdir()
        convert_ltx_checkpoint(checkpoint, partial, bits, config=config, models_dir=models_dir, lora=lora, report=_Progress())
        promote_partial(partial, target)
    except BaseException:
        discard_partial(partial)
        raise
    return target


def _distilled_lora(config: dict[str, Any]) -> Path:
    """Return the local path of Lightricks' distilled LoRA, downloading it once into the Hugging Face cache.

    Raises:
        RuntimeError: When it isn't configured or can't be downloaded.
    """
    source = ltx_distilled_lora_source(config)
    if source is None:
        raise RuntimeError("--distill needs video_model_presets.ltx.distilled_lora (repo and filename) in the config.")
    from huggingface_hub import hf_hub_download

    repo, filename = source
    print(f"Getting Lightricks' distilled LoRA ({filename}, 7.6 GB the first time)...", flush=True)
    try:
        return Path(hf_hub_download(repo_id=repo, filename=filename))
    except Exception as exc:  # noqa: BLE001 - any download failure stops before converting
        raise RuntimeError(f"--distill needs Lightricks' distilled LoRA {filename} from {repo}, which couldn't be downloaded: {exc}") from exc


def progress_line(stage: str, done: int, total: int) -> str:
    """Return the progress line of *stage* (a key of the stages), with ``(done/total)`` when the stage counts tensors."""
    label = _STAGES.get(stage, stage)
    return f"{PROGRESS_PREFIX}{label} ({done}/{total})" if total > 1 else f"{PROGRESS_PREFIX}{label}"


class _Progress:
    """Print a conversion's progress lines: each stage once, and a counted stage about every 2%."""

    def __init__(self) -> None:
        self._stage: str | None = None
        self._printed = 0

    def __call__(self, stage: str, done: int, total: int) -> None:
        step = max(total // _PROGRESS_STEPS, 1)
        if stage != self._stage or done == total or done - self._printed >= step:
            self._stage, self._printed = stage, done
            print(progress_line(stage, done, total), flush=True)
