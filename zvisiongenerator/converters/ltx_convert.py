"""Convert a single-file LTX-2.3 or LTX-2.5 checkpoint into a models-folder entry at one quantize level (``--model-type ltx-2.3`` / ``ltx-2.5``).

The folder is written into a hidden partial folder and moved into place once complete (see
:mod:`zvisiongenerator.converters.convert_target`, which also prints the progress lines the Web UI reads). Its layout
depends on the platform (see :func:`zvisiongenerator.backends.convert_ltx_checkpoint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint, check_lora, distillation, inspect_checkpoint, require_version
from zvisiongenerator.utils.config import ltx_distilled_lora_source, ltx_version_for_type
from zvisiongenerator.converters.convert_target import Progress, checked_target, write_into
from zvisiongenerator.utils.paths import split_huggingface_revision
from zvisiongenerator.utils.quantized_weights import precision_label

__all__ = ["LtxInspection", "convert_ltx", "distill_advice", "inspect_ltx"]
# {distill} is how the reader turns the merge on: the CLI flag, or the Web UI's checkbox.
_ADVICE = {
    "dev": "This is plain LTX-{version} dev, which renders broken at the 8 steps the app runs LTX at. Convert it with {distill} to merge Lightricks' distilled LoRA.",
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


def inspect_ltx(path: Path, model_type: str = "ltx-2.3") -> LtxInspection:
    """Check the checkpoint at *path* against *model_type* and tell whether it is plain dev or step-distilled; nothing is downloaded.

    Raises:
        ValueError: If it is not an LTX checkpoint the converter reads, or not the version of *model_type*.
    """
    checkpoint = inspect_checkpoint(path)
    require_version(checkpoint, model_type)
    found = distillation(checkpoint)
    return LtxInspection(checkpoint=checkpoint, distillation=found, advice=distill_advice(found, "--distill", checkpoint.version))


def distill_advice(found: str, distill: str, version: str) -> str:
    """Return what the distillation *found* (``dev``, ``distilled`` or ``unknown``) of an LTX-*version* checkpoint means for converting, naming the merge's switch as *distill*."""
    return _ADVICE[found].format(distill=distill, version=version)


def convert_ltx(input_path: Path, models_dir: Path, name: str, bits: int, *, distill: bool, force: bool, config: dict[str, Any], model_type: str = "ltx-2.3") -> Path:
    """Convert the LTX checkpoint *input_path* of *model_type* into ``<models_dir>/<name>`` at *bits*; return the folder.

    Args:
        input_path: The single-file checkpoint.
        models_dir: The models folder.
        name: The new model's folder name; neither an existing folder (unless *force*) nor a configured alias.
        bits: 4 or 8, the only level stored.
        distill: Merge Lightricks' distilled LoRA in (for plain-dev checkpoints).
        force: Replace an existing folder of that name, once the new one is complete.
        config: The loaded config.
        model_type: ``ltx-2.3`` or ``ltx-2.5``; the checkpoint must be that version.

    Raises:
        ValueError: If the name is taken by an alias, or the checkpoint isn't one the converter reads (or another version).
        FileExistsError: If the folder exists and *force* is not set.
        RuntimeError: If the distilled LoRA can't be downloaded, or the conversion fails.
    """
    if bits not in (4, 8):
        raise ValueError("LTX checkpoints need --quantize 4 or 8: that level is the only one stored.")
    target = checked_target(models_dir, name, force=force, config=config)
    inspection = inspect_ltx(input_path, model_type)
    checkpoint = inspection.checkpoint
    print(
        f"Checkpoint: {input_path.name} (LTX-{checkpoint.version}, {'full checkpoint' if inspection.layout == 'full' else 'transformer only'}, "
        f"{len(checkpoint.transformer)} transformer tensors, {precision_label(checkpoint.precision)})"
    )
    print(inspection.advice if not (distill and inspection.distillation == "dev") else f"This is plain LTX-{checkpoint.version} dev: merging Lightricks' distilled LoRA.")
    if distill and inspection.distillation == "distilled":
        print("Warning: --distill merges Lightricks' distilled LoRA into a checkpoint that already carries its distillation.")
    version = ltx_version_for_type(config, model_type)
    lora = _distilled_lora(config, version, checkpoint.version) if distill else None
    if lora is not None:
        check_lora(checkpoint, lora)
    from zvisiongenerator.backends import convert_ltx_checkpoint

    return write_into(target, lambda partial: convert_ltx_checkpoint(checkpoint, partial, bits, config=config, version=version, models_dir=models_dir, lora=lora, report=Progress()))


def _distilled_lora(config: dict[str, Any], version: str | None, label: str) -> Path:
    """Return the local path of Lightricks' distilled LoRA of LTX version block *version* (LTX-*label*), downloading it once into the Hugging Face cache.

    Raises:
        RuntimeError: When it isn't configured or can't be downloaded.
    """
    source = ltx_distilled_lora_source(config, version)
    if source is None:
        raise RuntimeError(f"--distill needs Lightricks' LTX-{label} distilled LoRA (distilled_lora: repo and filename) in the config.")
    from huggingface_hub import hf_hub_download

    reference, filename = source
    repo, revision = split_huggingface_revision(reference)
    print(f"Getting Lightricks' LTX-{label} distilled LoRA ({filename}, 8 to 9 GB the first time)...", flush=True)
    try:
        return Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision))
    except Exception as exc:  # noqa: BLE001 - any download failure stops before converting
        raise RuntimeError(f"--distill needs Lightricks' distilled LoRA {filename} from {repo}, which couldn't be downloaded: {exc}") from exc
