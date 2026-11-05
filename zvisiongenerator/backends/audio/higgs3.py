"""Higgs TTS 3 speech backend on torch: our own model class on the official weights and transformers' codec.

The network is :class:`~zvisiongenerator.backends.audio.higgs3_model.Higgs3Model`, filled from the official
``bosonai/higgs-tts-3-4b`` snapshot, and the codes of a part come from ``higgs3_generate.generate_speech_codes``. The audio
codec is transformers' ``HiggsAudioV2TokenizerModel`` at the pinned revision, loaded without remote code on the preset's
``codec_device``. Both repositories are read from the Hugging Face cache only; loading never downloads. Parts, the voice,
progress and the join are :mod:`higgs3_takes`'; this backend is its torch engine.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.backends.audio.higgs3_config import read_higgs3_config
from zvisiongenerator.backends.audio.higgs3_generate import generate_speech_codes
from zvisiongenerator.backends.audio.higgs3_prompt import load_tokenizer, special_ids
from zvisiongenerator.backends.audio.higgs3_takes import SAMPLE_RATE, read_speech
from zvisiongenerator.core.audio_backend import raise_if_cancelled

if TYPE_CHECKING:
    from zvisiongenerator.backends.audio.higgs3_prompt import SpecialIds
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["SAMPLE_RATE", "Higgs3Backend"]

_CODEC_MODULE = "transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer"
"""The module of transformers' codec, whose conv-layer cache :func:`_clear_codec_cache` clears."""


class Higgs3Backend:
    """Speech from Higgs TTS 3 on a torch device, with long text read in parts and an optional voice to follow (see :mod:`higgs3_takes`)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._ids: SpecialIds | None = None
        self._codec: Any = None
        self._device = ""
        self._codec_device = "cpu"
        self._boc_id = 0
        self._limits: AudioLimits | None = None
        self._settings: Mapping[str, Any] = {}

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the model onto *device* and the codec onto the preset's ``codec_device`` (the CPU on CUDA and MPS on a Mac by default), from the Hugging Face cache.

        Nothing is downloaded: both repositories must already be in the cache.

        Args:
            preset: The ``higgs3`` preset: its pinned model and codec files, its limits and its settings for *device*.
            device: ``cuda`` or ``mps``.
            cancelled: Called between the load steps (tokenizer, network, codec), while the network and the codec are built and
                before each tensor of the model is copied; ``True`` stops the load.

        Raises:
            FileNotFoundError: If a repository is not in the cache with every listed file.
            AudioCancelled: If *cancelled* returned ``True``; everything loaded so far is freed.
        """
        model_files, codec_files = preset.files
        official, codec_dir = snapshot_dir(model_files), snapshot_dir(codec_files)
        self.close()
        self._limits = preset.limits
        self._settings = preset.settings_for(device)
        self._device = device
        self._codec_device = str(self._settings.get("codec_device", "mps" if device == "mps" else "cpu"))
        try:
            self._load_models(official, codec_dir, cancelled)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take; see :func:`higgs3_takes.read_speech`.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            RuntimeError: If the model made no sound.
        """
        return read_speech(self, request, self._limits, self._settings, on_progress, cancelled)

    def close(self) -> None:
        """Drop the model and the codec; the runner returns the accelerator memory and the torch flags.

        Safe to call again, after a failed load and after a failed take.
        """
        self._model = self._tokenizer = self._ids = self._codec = None
        _clear_codec_cache()

    def _load_models(self, official: Path, codec_dir: Path, cancelled: CancelCheck | None = None) -> None:
        """Load the tokenizer, the model and the codec from their cached folders."""
        import torch
        from transformers import HiggsAudioV2TokenizerModel

        from zvisiongenerator.backends.audio.higgs3_model import load_higgs3_model

        raise_if_cancelled(cancelled)  # the imports above take seconds
        config = read_higgs3_config(official / "config.json")
        tokenizer = load_tokenizer(official)
        ids = special_ids(tokenizer)
        raise_if_cancelled(cancelled)
        model = load_higgs3_model(config, official / "model.safetensors", self._device, cancelled)
        raise_if_cancelled(cancelled)
        # No remote code: the codec class is native to transformers.
        with stop_while_building(cancelled):
            codec = HiggsAudioV2TokenizerModel.from_pretrained(str(codec_dir), dtype=torch.float32)
        raise_if_cancelled(cancelled)
        codec = codec.to(self._codec_device).eval()
        raise_if_cancelled(cancelled)
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        self._boc_id = config.boc_id
        self._tokenizer, self._ids, self._model, self._codec = tokenizer, ids, model, codec

    def read_part(
        self,
        text: str,
        seed: int,
        reference: tuple[Any, str | None] | None,
        max_new_tokens: int,
        on_frame: Callable[[], None],
        on_decode: Callable[[int], None],
    ) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut, generated with *seed* and following *reference* (codes and transcript) when given."""
        import torch

        torch.manual_seed(seed)
        codes, transcript = reference if reference is not None else (None, None)
        result = generate_speech_codes(
            self._model,
            self._tokenizer,
            self._ids,
            text,
            reference_codes=codes,
            reference_text=transcript,
            max_new_tokens=max_new_tokens,
            temperature=self._settings["temperature"],
            top_p=self._settings["top_p"],
            on_frame=on_frame,
        )
        if not result.codes.shape[0]:  # no decode is reported for a part without sound
            return np.zeros(0, dtype=np.float32), result.cut
        on_decode(0)
        wave = self._decode(result.codes)
        on_decode(1)
        return wave, result.cut

    def encode(self, wave: np.ndarray) -> Any:
        """Encode a 24 kHz mono waveform with the codec into ``[frames, codebooks]`` int64 codes on the CPU."""
        import torch

        if wave.shape[0] < SAMPLE_RATE:  # a reference shorter than a second is padded to one
            wave = np.pad(wave, (0, SAMPLE_RATE - wave.shape[0]))
        batch = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).reshape(1, 1, -1).to(self._codec_device, torch.float32)
        with torch.no_grad():
            codes = self._codec.encode(batch).audio_codes
        return codes.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    def _decode(self, codes: Any) -> np.ndarray:
        """Decode raw codes ``[frames, codebooks]`` into a mono 24 kHz float32 waveform on the codec's device."""
        import torch

        with torch.no_grad():
            codes = torch.where(codes >= self._boc_id, torch.zeros_like(codes), codes)  # the two ids above the codec's values are stream markers
            batch = codes.transpose(0, 1).unsqueeze(0).to(self._codec_device, torch.long)
            audio = self._codec.decode(batch).audio_values.squeeze(0).squeeze(0).detach().cpu().float()
        return np.asarray(audio.numpy(), dtype=np.float32)


def _clear_codec_cache() -> None:
    """Clear transformers' cache of the codec's conv layers, so no codec outlives :meth:`Higgs3Backend.close`.

    ``HiggsAudioV2TokenizerPreTrainedModel._get_conv1d_layers`` is an ``lru_cache`` on a method: its keys hold every
    codec that has encoded audio (a voice recording, or the first part's take that later parts follow), with all of its
    weights. Transformers is only looked up, never imported: when it is not loaded, nothing is cached. A transformers
    without that cache gets a warning, so ``close()`` never raises.
    """
    module = sys.modules.get(_CODEC_MODULE)
    if module is None:
        return
    cache_clear = getattr(getattr(getattr(module, "HiggsAudioV2TokenizerPreTrainedModel", None), "_get_conv1d_layers", None), "cache_clear", None)
    if cache_clear is None:
        warnings.warn("The Higgs codec's conv-layer cache was not found; the codec may stay in memory after the job.", stacklevel=2)
        return
    cache_clear()
