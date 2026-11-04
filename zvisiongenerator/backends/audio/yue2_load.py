"""Load YuE2's weights without the kit's memory-heavy paths: stream them in, and check file sizes instead of hashing.

The kit loads the model with transformers' memory-mapped ``from_pretrained``, which keeps the whole 7 GB file in the page
cache while the weights are copied (or pinned) elsewhere. :func:`load_model` reads each tensor with plain file reads
instead and builds the model from the tensors it was given. The kit's pipeline also sha256-hashes every weight file each
time it is built; :func:`size_checked_weights` swaps that for a size check against the files' manifest while one is built.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.core.audio_backend import raise_if_cancelled
from zvisiongenerator.utils.safetensors_stream import iter_tensors

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck

__all__ = ["load_model", "size_checked_weights"]

_MANIFEST = "weights_manifest.json"


def load_model(model_dir: Path | str, *, pin: bool, cancelled: CancelCheck | None = None) -> Any:
    """Build ``YuE2ForCausalLM`` on the CPU in bf16 from *model_dir*'s safetensors files, reading one tensor at a time.

    The model adopts the tensors as read, so nothing is copied and the file is never mapped.

    Args:
        model_dir: A model folder: ``config.json`` and the ``*.safetensors`` files.
        pin: Pin each tensor in page-locked memory as it is read, so a later copy to the GPU is fast and pins nothing again.
        cancelled: Called before each tensor is read and while the model is built from them; ``True`` stops the load.

    Returns:
        The model in eval mode.

    Raises:
        AudioCancelled: If *cancelled* returned ``True``; the tensors read so far are dropped.
    """
    import torch

    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    directory = Path(model_dir)
    state = {}
    for file in sorted(directory.glob("*.safetensors")):
        for name, tensor in iter_tensors(file):
            raise_if_cancelled(cancelled)
            state[name] = tensor.pin_memory() if pin else tensor
    raise_if_cancelled(cancelled)
    config = YuE2Config.from_pretrained(directory)
    with stop_while_building(cancelled):
        model = YuE2ForCausalLM.from_pretrained(None, config=config, state_dict=state, torch_dtype=torch.bfloat16).eval()
    raise_if_cancelled(cancelled)
    return model


@contextmanager
def size_checked_weights() -> Iterator[None]:
    """Make the kit's pipeline check weight file sizes instead of hashing them while the block runs.

    Raises:
        AttributeError: On entry, if the kit's pipeline module no longer has the ``model_identity`` function this replaces.
    """
    import yue2.pipeline as kit_pipeline

    original = kit_pipeline.model_identity
    kit_pipeline.model_identity = _size_checked_identity
    try:
        yield
    finally:
        kit_pipeline.model_identity = original


def _size_checked_identity(path: Path | str, verify: bool = True) -> dict[str, Any]:
    """Return the kit's ``model_identity`` record for *path*, taking sizes from the files and hashes from the manifest.

    Raises:
        FileNotFoundError: If the folder holds no safetensors file.
        ValueError: If a file's size differs from the manifest's, or the files differ from the manifest's list.
    """
    directory = Path(path)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    manifest = directory / _MANIFEST
    expected: dict[str, Any] | None = json.loads(manifest.read_text(encoding="utf-8")).get("files", {}) if manifest.is_file() else None
    entries = {}
    for file in files:
        size = file.stat().st_size
        declared = (expected or {}).get(file.name, {})
        if verify and expected is not None and declared.get("bytes") != size:
            raise ValueError(f"Weight file {file.name} is {size} bytes; the manifest says {declared.get('bytes', 'nothing')}.")
        entries[file.name] = {"sha256": declared.get("sha256"), "bytes": size}
    if expected is not None and set(entries) != set(expected):
        raise ValueError("Weight manifest has missing or unexpected shards")
    config_sha256 = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    return {"files": entries, "config_sha256": config_sha256}
