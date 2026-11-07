"""Qwen3-TTS speech backend on CUDA, through faster-qwen3-tts (CUDA graphs).

The model has two checkpoints: CustomVoice reads in one of the built-in voices, Base follows a voice recording. The backend
loads the one checkpoint its preset's ``files`` name, once, when the batch loads; the runner decides which by handing it a preset
whose ``files`` are the Base files for a batch that follows a recording. Parts, the voice, progress and the join are
:mod:`speech_takes`'; this backend is its engine for Qwen.

Loading reads the Hugging Face cache only; it never downloads. ``qwen_tts`` needs a one-time patch for the locked
transformers (:mod:`qwen3_tts_shim`), applied before the first load.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.backends.audio.qwen3_tts_shim import apply_qwen_shim
from zvisiongenerator.backends.audio.qwen3_tts_instruct import delivery_instruction
from zvisiongenerator.backends.audio.speech_takes import REFERENCE_SILENCE_SECONDS, read_speech, read_voice
from zvisiongenerator.core.audio_backend import raise_if_cancelled
from zvisiongenerator.utils.library_noise import quiet_qwen_graph_capture

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest, SpeechDelivery, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioPreset

__all__ = ["SAMPLE_RATE", "Qwen3TtsBackend"]

SAMPLE_RATE = 24000
"""The model's sample rate, and the rate of every take."""

_FRAMES_PER_SECOND = 12.5
"""The codec's frame rate: tokens of speech per second of sound."""


class Qwen3TtsBackend:
    """Speech from Qwen3-TTS 1.7B on a CUDA card: built-in voices, or a voice recording to follow, with long text read in parts.

    Implements :class:`zvisiongenerator.core.audio_backend.AudioBackend` and :class:`~zvisiongenerator.backends.audio.speech_takes.SpeechEngine`.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._preset: AudioPreset | None = None
        self._settings: Mapping[str, Any] = {}
        self._device = "cuda"

    @property
    def sample_rate(self) -> int:
        """The sample rate of every part and of the take."""
        return SAMPLE_RATE

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the checkpoint in ``preset.files[0]`` onto *device* and capture its CUDA graphs, for every take to come.

        Args:
            preset: The ``qwen3_tts`` preset, its ``files`` naming the checkpoint to load (CustomVoice, or Base for a batch that
                follows a recording), with its limits and its settings for *device*.
            device: ``cuda``.
            cancelled: Called while the checkpoint loads; ``True`` stops it and nothing stays loaded.

        Raises:
            FileNotFoundError: If the checkpoint is not downloaded.
            AudioCancelled: If *cancelled* returned ``True``.
        """
        self.close()
        self._preset = preset
        self._settings = preset.settings_for(device)
        self._device = device
        folder = snapshot_dir(preset.files[0])
        try:
            raise_if_cancelled(cancelled)
            self._model = self._load_model(str(folder), cancelled)
            raise_if_cancelled(cancelled)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take; see :func:`speech_takes.read_speech`.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            RuntimeError: If the model made no sound.
        """
        return read_speech(self, request, self._preset, self._preset.limits, self._settings, on_progress, cancelled)

    def close(self) -> None:
        """Drop the model; the runner returns the accelerator memory and the torch flags.

        Safe to call again, after a failed load and after a failed take.
        """
        self._model = None

    def prepare_voice(self, voice: VoiceRef) -> Any:
        """Return the cloning prompt of the voice recording (its sound and transcript), made once per take.

        Needs the Base checkpoint, which :meth:`load` has loaded for a batch with a recording.
        """
        wave = read_voice(voice, float(self._preset.limits.max_voice_seconds), SAMPLE_RATE)
        padded = np.concatenate([wave, np.zeros(int(REFERENCE_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)])
        return self._model.model.create_voice_clone_prompt(ref_audio=(padded, SAMPLE_RATE), ref_text=voice.text)

    def read_part(
        self,
        text: str,
        seed: int,
        *,
        speaker: str | None,
        reference: Any,
        delivery: SpeechDelivery | None,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut, generated with *seed*.

        A *reference* (from :meth:`prepare_voice`) is followed as recorded, with no instruction (the Base checkpoint takes none);
        otherwise the part is read in the built-in *speaker* with *delivery* as one instruction sentence. The waveform is empty when the model made no sound; it is cut when it filled *max_new_tokens* frames.
        """
        import torch

        torch.manual_seed(seed)
        language = str(self._settings["language"])
        if reference is not None:
            waves, _ = self._model.generate_voice_clone(text=text, language=language, voice_clone_prompt=reference, max_new_tokens=max_new_tokens)
        else:
            waves, _ = self._model.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=delivery_instruction(delivery), max_new_tokens=max_new_tokens)
        wave = np.asarray(waves[0], dtype=np.float32).reshape(-1)
        if wave.shape[0] <= 1:  # the library answers a part without codes with a single zero sample
            return np.zeros(0, dtype=np.float32), False
        # One frame of tolerance: the part of a cloned take is cut out of the reference and the part together.
        return wave, wave.shape[0] / SAMPLE_RATE >= (max_new_tokens - 1) / _FRAMES_PER_SECOND

    def _load_model(self, folder: str, cancelled: CancelCheck | None) -> Any:
        """Load the checkpoint in *folder* onto the device in bfloat16 and capture its CUDA graphs."""
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        apply_qwen_shim()
        raise_if_cancelled(cancelled)  # the imports above take seconds
        with stop_while_building(cancelled):
            model = FasterQwen3TTS.from_pretrained(folder, device=self._device, dtype=torch.bfloat16, local_files_only=True)
        raise_if_cancelled(cancelled)
        with quiet_qwen_graph_capture():  # the library prints a line for each warm-up and graph capture
            model.warmup()
        return model
