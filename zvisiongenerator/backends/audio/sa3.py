"""Stable Audio 3 Medium backend: ambience and sound effects of 1 to 380 seconds, 44.1 kHz stereo.

The model is the vendored ``packages/stable_audio_3``. It is built through the same factory the package's own loader
uses, but under an fp16 default dtype, and then filled tensor by tensor from the checkpoint. The fp32 model (9.2 GB) and
the fp32 state dict of the package's loader are never resident, and the checkpoint is never mapped into the page cache.
Attention is PyTorch's SDPA: ``flash_attn`` is optional and not needed.

The library clamps its output to -1..1 inside ``generate``. The backend keeps the decoder's output from before the
clamp by wrapping the instance's ``pretransform.decode`` for the length of one take, and returns it with
``AudioClip.preclamp`` set so the save step can scale a take that peaks above 1.0 instead of clipping it.
"""

from __future__ import annotations

import gc
import json
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.utils.audio_request import MIN_SOUND_SECONDS
from zvisiongenerator.utils.model_files import cached_snapshot_dirs
from zvisiongenerator.utils.safetensors_stream import iter_tensors
from zvisiongenerator.utils.torch_flags import TorchFlags, capture_torch_flags, restore_torch_flags

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioFiles, AudioPlatform, AudioPreset

__all__ = ["FAMILY", "MIN_SECONDS", "StableAudio3Backend"]

FAMILY = "stable_audio_3"
"""The model family this backend serves: the name ``backends.get_audio_backend`` resolves."""

MIN_SECONDS = MIN_SOUND_SECONDS
"""The shortest sound the model makes, in seconds (the one length rule the CLI and the Web UI also check)."""

PHASE = "Sampling"
"""The progress phase of a take: the diffusion steps."""

_CONFIG_FILE = "model_config.json"
_TEXT_ENCODER_DIR = "t5gemma-b-b-ul2"
_SETTINGS = ("steps", "cfg_scale", "sample_size")


@dataclass(frozen=True)
class _SoundFiles:
    """Where the model's three inputs are on disk.

    Attributes:
        config: The model config JSON.
        checkpoint: The fp32 safetensors checkpoint.
        text_encoder: The folder of the T5Gemma text encoder.
    """

    config: Path
    checkpoint: Path
    text_encoder: Path


def _snapshot(files: AudioFiles) -> Path:
    """Return the cache snapshot of *files*' repository at its pinned commit.

    Raises:
        FileNotFoundError: If that revision is not downloaded.
    """
    snapshots = cached_snapshot_dirs(files.repo_id, files.revision)
    if not snapshots:
        raise FileNotFoundError(f"Stable Audio 3 needs {files.repo_id} at {files.revision}, which is not downloaded.")
    return snapshots[0]


def _locate_files(platform_entry: AudioPlatform) -> _SoundFiles:
    """Return the config, checkpoint and text encoder folder named by the platform entry's file lists.

    The config is the entry that lists ``model_config.json``, the checkpoint its one literal ``.safetensors`` path and
    the text encoder the entry that lists a ``t5gemma-b-b-ul2/`` folder.

    Raises:
        ValueError: If the entry lists none of the three.
        FileNotFoundError: If a revision or a file is not downloaded.
    """
    found: dict[str, Path] = {}
    for files in platform_entry.files:
        for pattern in files.patterns:
            if pattern == _CONFIG_FILE:
                found.setdefault("config", _snapshot(files) / pattern)
            elif pattern.startswith(f"{_TEXT_ENCODER_DIR}/"):
                found.setdefault("text_encoder", _snapshot(files) / _TEXT_ENCODER_DIR)
            elif pattern.endswith(".safetensors") and "*" not in pattern:
                found.setdefault("checkpoint", _snapshot(files) / pattern)
    missing = [name for name in ("config", "checkpoint", "text_encoder") if name not in found]
    if missing:
        raise ValueError(f"The Stable Audio 3 platform entry lists no file for: {', '.join(missing)}.")
    for path in (found["config"], found["checkpoint"]):
        if not path.is_file():
            raise FileNotFoundError(f"Stable Audio 3 file {path} is not downloaded.")
    if not found["text_encoder"].is_dir():
        raise FileNotFoundError(f"Stable Audio 3 text encoder folder {found['text_encoder']} is not downloaded.")
    return _SoundFiles(found["config"], found["checkpoint"], found["text_encoder"])


def _model_config(config_path: Path, text_encoder: Path) -> dict[str, Any]:
    """Return the model config with its T5Gemma conditioner pointed at the local *text_encoder* folder.

    The Hub reference (``repo_id`` and ``subfolder``) is dropped, so the conditioner reads the folder and never the Hub.

    Raises:
        ValueError: If the config has no T5Gemma conditioner.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditioners = [entry["config"] for entry in config["model"]["conditioning"]["configs"] if entry.get("type") == "t5gemma"]
    if not conditioners:
        raise ValueError(f"{config_path} has no t5gemma conditioner.")
    for conditioner in conditioners:
        conditioner.pop("repo_id", None)
        conditioner.pop("subfolder", None)
        conditioner["model_path"] = str(text_encoder)
    return config


def _fill_from_checkpoint(model: Any, checkpoint: Path) -> None:
    """Copy the checkpoint's tensors into *model*'s parameters and buffers one at a time, converting each to the model's dtype.

    Names are matched the way the package's loader matches them (``remap_state_dict_keys``); a tensor the model lacks, or
    whose shape differs, is skipped with a warning, as the package's loader skips it.
    """
    from stable_audio_3.loading_utils import remap_state_dict_keys

    destinations = model.state_dict()  # views of the live tensors: copying into them fills the model
    for name, tensor in iter_tensors(checkpoint):
        key = next(iter(remap_state_dict_keys({name: None}, destinations)))
        destination = destinations.get(key)
        if destination is None or destination.shape != tensor.shape:
            warnings.warn(f"Stable Audio 3 checkpoint tensor {name} has no matching model tensor of the same shape; skipped.", stacklevel=2)
            continue
        destination.copy_(tensor)


@contextmanager
def _fp16_default() -> Iterator[None]:
    """Make new floating-point parameters fp16 for the length of the block, then restore the previous default dtype.

    The package's factory builds fp32 models, which takes 9.2 GB of memory for Stable Audio 3 Medium. Built under this
    default its parameters are fp16 from the start, about half that. Nothing the init computes survives: the
    checkpoint holds every parameter and buffer of the model, and the fill overwrites each. The T5Gemma text encoder
    loads at the dtype of its own files, whatever the default is.
    """
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _build_fp16(config: dict[str, Any], checkpoint: Path, device: str) -> Any:
    """Return the diffusion model built from *config* in fp16, filled from *checkpoint* and moved to *device*, ready to sample."""
    from stable_audio_3.factory import create_diffusion_cond_from_config

    with _fp16_default():
        model = create_diffusion_cond_from_config(config)
    model.half()  # the buffers the factory made in fp32
    _fill_from_checkpoint(model, checkpoint)
    model.to(device).eval().requires_grad_(False)
    model.use_lora = False
    model.lora_names = []
    return model


@contextmanager
def _keep_decoded(pretransform: Any) -> Iterator[list[Any]]:
    """Wrap ``pretransform.decode`` so each waveform it returns is appended to the yielded list; restore it on exit.

    The library calls the decoder from ``sample_diffusion`` and clamps what comes out. The wrapper sits on the instance
    only, for the length of the block, so the package is not edited and no other use of the model sees it.
    """
    original = pretransform.decode
    had_own = "decode" in vars(pretransform)
    kept: list[Any] = []

    def decode(*args: Any, **kwargs: Any) -> Any:
        waveform = original(*args, **kwargs)
        kept.append(waveform.detach())
        return waveform

    pretransform.decode = decode
    try:
        yield kept
    finally:
        if had_own:
            pretransform.decode = original
        else:
            del pretransform.decode


def _check_seconds(seconds: float | None, max_seconds: float) -> float:
    """Return the requested length, or raise when it is missing or outside ``MIN_SECONDS`` to *max_seconds*.

    Raises:
        ValueError: If *seconds* is ``None`` or outside the range the model makes.
    """
    if seconds is None:
        raise ValueError(f"Stable Audio 3 needs a length of {MIN_SECONDS:g} to {max_seconds:g} seconds.")
    if not MIN_SECONDS <= seconds <= max_seconds:
        raise ValueError(f"Stable Audio 3 makes sounds of {MIN_SECONDS:g} to {max_seconds:g} seconds; {seconds:g} seconds is outside that.")
    return float(seconds)


def _empty_cache(device: str) -> None:
    """Return the cached accelerator memory of *device* to the system."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class StableAudio3Backend:
    """Make a sound from a text description with Stable Audio 3 Medium on a torch device (``cuda`` or ``mps``).

    Implements :class:`zvisiongenerator.core.audio_backend.AudioBackend`.
    """

    def __init__(self, empty_cache: Callable[[str], None] = _empty_cache) -> None:
        self._empty_cache = empty_cache
        self._model: Any = None
        self._device = ""
        self._max_seconds = 0.0
        self._settings: dict[str, Any] = {}
        self._flags: TorchFlags | None = None

    def load(self, preset: AudioPreset, platform_entry: AudioPlatform, device: str) -> None:
        """Load the model of *preset* onto *device*, filling it tensor by tensor in fp16.

        Args:
            preset: The ``sa3`` preset; its settings must give ``steps``, ``cfg_scale`` and ``sample_size``.
            platform_entry: The preset's entry for this platform: the config, the checkpoint and the text encoder.
            device: ``cuda`` or ``mps``.

        Raises:
            ValueError: If a setting or a file entry is missing.
            FileNotFoundError: If a file is not downloaded.
        """
        from stable_audio_3.model import StableAudioModel

        self.close()
        settings = preset.settings_for(device)
        missing = [key for key in _SETTINGS if key not in settings]
        if missing:
            raise ValueError(f"Audio model '{preset.id}' has no setting for: {', '.join(missing)}.")
        if preset.limits.max_seconds is None:
            raise ValueError(f"Audio model '{preset.id}' has no max_seconds limit.")
        files = _locate_files(platform_entry)
        config = _model_config(files.config, files.text_encoder)
        self._device = device  # set first, so close() frees the device's cache after a failed load too
        self._flags = capture_torch_flags()  # StableAudioModel sets global numerics flags; close() puts them back
        try:
            self._model = StableAudioModel(_build_fp16(config, files.checkpoint, device), config, device, True)
            self._max_seconds = float(preset.limits.max_seconds)
            self._settings = {key: settings[key] for key in _SETTINGS}
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Make one sound for *request* and return its waveform from before the library's clamp.

        The library has a step callback, so progress is reported after each diffusion step and a stop request is
        honoured between steps, not only before the take.

        Args:
            request: The sound to make; ``seconds`` is required.
            on_progress: Called as ``on_progress("Sampling", step, steps, 1, 1)``.
            cancelled: Called between steps; ``True`` stops the take.

        Returns:
            A stereo 44.1 kHz clip with ``preclamp`` set: it may peak above 1.0.

        Raises:
            RuntimeError: If no model is loaded.
            ValueError: If the length is missing or outside the range the model makes.
            AudioCancelled: If ``cancelled`` returned ``True``.
        """
        if self._model is None:
            raise RuntimeError("Stable Audio 3 is not loaded.")
        seconds = _check_seconds(request.seconds, self._max_seconds)
        steps = int(self._settings["steps"])
        if cancelled():
            raise AudioCancelled
        on_progress(PHASE, 0, steps, 1, 1)

        def on_step(info: dict[str, Any]) -> None:
            on_progress(PHASE, int(info["i"]) + 1, steps, 1, 1)
            if cancelled():
                raise AudioCancelled

        sound = self._model
        with _keep_decoded(sound.model.pretransform) as decoded:
            sound.generate(
                prompt=request.prompt,
                duration=seconds,
                steps=steps,
                cfg_scale=float(self._settings["cfg_scale"]),
                seed=request.seed,
                sample_size=int(self._settings["sample_size"]),
                callback=on_step,
                disable_tqdm=True,
            )
        if not decoded:
            raise RuntimeError("Stable Audio 3 did not decode through the pretransform, so there is no waveform to return.")
        frames = int(seconds * sound.model.sample_rate)
        samples = decoded[-1][0, :, :frames].float().cpu().contiguous().numpy()
        return AudioClip(samples=samples, sample_rate=int(sound.model.sample_rate), preclamp=True)

    def close(self) -> None:
        """Free the model, return its accelerator memory and restore the torch flags the library changed.

        Safe to call twice, after a failed load and after a failed take.
        """
        device, flags = self._device, self._flags
        self._model = None
        self._device = ""
        self._flags = None
        try:
            gc.collect()
            if device:
                self._empty_cache(device)
        finally:
            if flags is not None:
                restore_torch_flags(flags)
