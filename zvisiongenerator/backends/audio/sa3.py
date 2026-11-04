"""Stable Audio 3 Medium backend: ambience and sound effects of 1 to 380 seconds, 44.1 kHz stereo.

The model is the vendored ``packages/stable_audio_3``. It is built through the same factory the package's own loader
uses, but under an fp16 default dtype, and then filled tensor by tensor from the checkpoint. The fp32 model (9.2 GB) and
the fp32 state dict of the package's loader are never resident, and the checkpoint is never mapped into the page cache.
Attention is PyTorch's SDPA: ``flash_attn`` is optional and not needed.

The library clamps its output to -1..1 inside ``generate``. The backend keeps the decoder's output from before the
clamp by wrapping the instance's ``pretransform.decode`` for the length of one take, and returns it so the runner can
scale a take that peaks above 1.0 instead of clipping it.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.audio_backend import DECODE_PHASE, AudioCancelled, raise_if_cancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.utils.library_noise import quiet_flash_attn_probe, silence_autotune_report
from zvisiongenerator.utils.safetensors_stream import iter_tensors
from zvisiongenerator.utils.torch_flags import default_dtype

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioPreset

__all__ = ["StableAudio3Backend"]

PHASE = "Sampling"
"""The progress phase of a take: the diffusion steps."""

_TEXT_ENCODER_DIR = "t5gemma-b-b-ul2"


def _model_config(config_path: Path, text_encoder: Path) -> dict[str, Any]:
    """Return the model config with its T5Gemma conditioner pointed at the local *text_encoder* folder.

    The Hub reference (``repo_id`` and ``subfolder``) is dropped, so the conditioner reads the folder and never the Hub.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for conditioner in (entry["config"] for entry in config["model"]["conditioning"]["configs"] if entry.get("type") == "t5gemma"):
        conditioner.pop("repo_id", None)
        conditioner.pop("subfolder", None)
        conditioner["model_path"] = str(text_encoder)
    return config


def _fill_from_checkpoint(model: Any, checkpoint: Path, cancelled: CancelCheck | None = None) -> None:
    """Copy the checkpoint's tensors into *model*'s parameters and buffers one at a time, converting each to the model's dtype.

    Names are matched the way the package's loader matches them (``remap_state_dict_keys``); a tensor the model lacks, or
    whose shape differs, is skipped with a warning, as the package's loader skips it. *cancelled* is called before each tensor;
    ``True`` raises :class:`AudioCancelled`.
    """
    from stable_audio_3.loading_utils import remap_state_dict_keys

    destinations = model.state_dict()  # views of the live tensors: copying into them fills the model
    for name, tensor in iter_tensors(checkpoint):
        raise_if_cancelled(cancelled)
        key = next(iter(remap_state_dict_keys({name: None}, destinations)))
        destination = destinations.get(key)
        if destination is None or destination.shape != tensor.shape:
            warnings.warn(f"Stable Audio 3 checkpoint tensor {name} has no matching model tensor of the same shape; skipped.", stacklevel=2)
            continue
        destination.copy_(tensor)


def _build_fp16(config: dict[str, Any], checkpoint: Path, device: str, cancelled: CancelCheck | None = None) -> Any:
    """Return the diffusion model built from *config* with an fp16 DiT and decoder, filled from *checkpoint* and moved to *device*, ready to sample."""
    import torch
    from stable_audio_3.factory import create_diffusion_cond_from_config

    # The package's factory builds fp32 models, 9.2 GB for Stable Audio 3 Medium. Built under an fp16 default its parameters
    # are fp16 from the start, about half that. Nothing the init computes survives: the checkpoint holds every parameter and
    # buffer of the model, and the fill overwrites each.
    raise_if_cancelled(cancelled)
    # The factory builds the DiT, the decoder and the T5Gemma conditioner in one call; the hook sees each layer it creates.
    with default_dtype(torch.float16), stop_while_building(cancelled):
        model = create_diffusion_cond_from_config(config)
    raise_if_cancelled(cancelled)
    # Only the DiT and the decoder run in fp16 (the buffers the factory made in fp32 are converted here). The conditioner's own
    # layers (the projection, the padding embedding, the number embedder) are small and stay fp32. The T5Gemma text encoder is
    # not a registered submodule of the conditioner, so no conversion of the model reaches it: it loads at the dtype of its own
    # files (bf16-trained, and it overflows in fp16) and is never cast to fp16 here. The package casts the conditioning it
    # produces to the DiT's dtype before sampling.
    model.model.half()
    model.pretransform.half()
    model.conditioner.float()
    raise_if_cancelled(cancelled)
    _fill_from_checkpoint(model, checkpoint, cancelled)
    model.to(device).eval().requires_grad_(False)
    raise_if_cancelled(cancelled)
    model.use_lora = False
    model.lora_names = []
    return model


@contextmanager
def _keep_decoded(pretransform: Any, on_decode: Callable[[int], None] | None = None) -> Iterator[list[Any]]:
    """Wrap ``pretransform.decode`` so each waveform it returns is appended to the yielded list; restore it on exit.

    The library calls the decoder from ``sample_diffusion`` and clamps what comes out. The wrapper sits on the instance
    only, for the length of the block, so the package is not edited and no other use of the model sees it.

    Args:
        pretransform: The model's pretransform, whose ``decode`` is wrapped.
        on_decode: Called with 0 when a decode starts and with 1 when it ends, which is how the Decode phase is reported.
    """
    original = pretransform.decode
    had_own = "decode" in vars(pretransform)
    kept: list[Any] = []

    def decode(*args: Any, **kwargs: Any) -> Any:
        if on_decode is not None:
            on_decode(0)
        waveform = original(*args, **kwargs)
        kept.append(waveform.detach())
        if on_decode is not None:
            on_decode(1)
        return waveform

    pretransform.decode = decode
    try:
        yield kept
    finally:
        if had_own:
            pretransform.decode = original
        else:
            del pretransform.decode


class StableAudio3Backend:
    """Make a sound from a text description with Stable Audio 3 Medium on a torch device (``cuda`` or ``mps``).

    Implements :class:`zvisiongenerator.core.audio_backend.AudioBackend`.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._settings: dict[str, Any] = {}

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the model of *preset* onto *device*, filling it tensor by tensor in fp16.

        Args:
            preset: The ``sa3`` preset: the checkpoint, the config and the text encoder files, in that order, and the
                settings ``steps``, ``cfg_scale`` and ``sample_size``.
            device: ``cuda`` or ``mps``.
            cancelled: Called while the model is built (before and after it, and for each layer created), before each
                checkpoint tensor is copied and after the model is on the device; ``True`` stops the load.

        Raises:
            FileNotFoundError: If a file is not downloaded.
            AudioCancelled: If *cancelled* returned ``True``; the half-filled model is freed.
        """
        with quiet_flash_attn_probe():  # the package prints that the optional flash_attn is missing, which it does not need
            from stable_audio_3.model import StableAudioModel
        silence_autotune_report()  # the first take compiles flex attention, which prints its kernel tuning tables
        raise_if_cancelled(cancelled)  # the import above takes seconds

        self.close()
        checkpoint_files, config_files, encoder_files = preset.files
        checkpoint = snapshot_dir(checkpoint_files) / checkpoint_files.patterns[0]
        config = _model_config(snapshot_dir(config_files) / config_files.patterns[0], snapshot_dir(encoder_files) / _TEXT_ENCODER_DIR)
        try:
            self._model = StableAudioModel(_build_fp16(config, checkpoint, device, cancelled), config, device, True)
            self._settings = preset.settings_for(device)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Make one sound for *request* and return its waveform from before the library's clamp.

        The library has a step callback, so progress is reported after each diffusion step and a stop request is
        honoured between steps, not only before the take.

        Args:
            request: The sound to make; ``seconds`` is required.
            on_progress: Called as ``on_progress("Sampling", step, steps, 1, 1)``, then ``on_progress("Decode", 0, 1, 1, 1)`` and
                ``on_progress("Decode", 1, 1, 1, 1)`` around the decoder that follows the last step.
            cancelled: Called between steps; ``True`` stops the take.

        Returns:
            A stereo 44.1 kHz clip from before the library's clamp, so it may peak above 1.0; the runner levels it.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
        """
        seconds = float(request.seconds)
        steps = int(self._settings["steps"])
        if cancelled():
            raise AudioCancelled
        on_progress(PHASE, 0, steps, 1, 1)

        def on_step(info: dict[str, Any]) -> None:
            on_progress(PHASE, int(info["i"]) + 1, steps, 1, 1)
            if cancelled():
                raise AudioCancelled

        def on_decode(done: int) -> None:
            on_progress(DECODE_PHASE, done, 1, 1, 1)

        sound = self._model
        with _keep_decoded(sound.model.pretransform, on_decode) as decoded:
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
        frames = int(seconds * sound.model.sample_rate)
        samples = decoded[-1][0, :, :frames].float().cpu().contiguous().numpy()
        return AudioClip(samples=samples, sample_rate=int(sound.model.sample_rate))

    def close(self) -> None:
        """Drop the model; the runner returns the accelerator memory and the torch flags the library changed.

        Safe to call twice, after a failed load and after a failed take.
        """
        self._model = None
