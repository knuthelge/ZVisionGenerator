"""Qwen3-TTS speech backend on macOS, through mlx-audio.

The same model, preset and take logic as :mod:`qwen3_tts` on CUDA: the checkpoint is the one the preset's ``files`` name,
loaded once when the batch loads (CustomVoice for the built-in voices, Base for a batch that follows a recording, the runner
choosing by the preset it passes), and parts, the voice, progress and the join are :mod:`speech_takes`'. mlx-audio reads the
official ``Qwen/…`` snapshots directly (it converts the PyTorch convolution layouts while loading), so the files are the CUDA
engine's and the transformers shim is not involved.

Loading reads the Hugging Face cache only; it never downloads. Only mlx-audio's public API is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.qwen3_tts_instruct import delivery_instruction
from zvisiongenerator.backends.audio.speech_takes import REFERENCE_SILENCE_SECONDS, read_speech, read_voice
from zvisiongenerator.core.audio_backend import raise_if_cancelled

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest, SpeechDelivery, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioPreset

__all__ = ["SAMPLE_RATE", "Qwen3TtsMlxBackend"]

SAMPLE_RATE = 24000
"""The model's sample rate, and the rate of every take."""


class Qwen3TtsMlxBackend:
    """Speech from Qwen3-TTS 1.7B on Apple silicon: built-in voices, or a voice recording to follow, with long text read in parts.

    Implements :class:`zvisiongenerator.core.audio_backend.AudioBackend` and :class:`~zvisiongenerator.backends.audio.speech_takes.SpeechEngine`.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._preset: AudioPreset | None = None
        self._settings: Mapping[str, Any] = {}

    @property
    def sample_rate(self) -> int:
        """The sample rate of every part and of the take."""
        return SAMPLE_RATE

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the checkpoint in ``preset.files[0]`` with its weights resident, for every take to come.

        Args:
            preset: The ``qwen3_tts`` preset as macOS runs it, its ``files`` naming the checkpoint to load (CustomVoice, or Base
                for a batch that follows a recording), with its limits and its settings.
            device: ``mps``; MLX runs on the Mac's GPU whatever it is.
            cancelled: Checked before the load and after it; ``True`` stops and nothing stays loaded. mlx-audio's load is one call
                that cannot be stopped part-way.

        Raises:
            FileNotFoundError: If the checkpoint is not downloaded.
            AudioCancelled: If *cancelled* returned ``True``.
        """
        self.close()
        self._preset = preset
        self._settings = preset.settings_for(device)
        folder = snapshot_dir(preset.files[0])
        try:
            raise_if_cancelled(cancelled)
            self._model = self._load_model(str(folder))
            raise_if_cancelled(cancelled)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take; see :func:`speech_takes.read_speech`.

        MLX's free-buffer cache is off for the take, so each part's freed key-value cache does not stay beside the next part's.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            RuntimeError: If the model made no sound.
        """
        from zvisiongenerator.backends.mac.memory import mlx_cache_limit

        with mlx_cache_limit(0):
            return read_speech(self, request, self._preset, self._preset.limits, self._settings, on_progress, cancelled)

    def close(self) -> None:
        """Drop the model; the runner returns MLX's memory. Safe to call again, after a failed load and after a failed take."""
        self._model = None

    def prepare_voice(self, voice: VoiceRef) -> tuple[Any, str]:
        """Return the voice recording as the reference mlx-audio follows: its mono wave with half a second of silence, and its transcript.

        mlx-audio encodes the wave once and keeps the result for the take's remaining parts. Needs the Base checkpoint, which
        :meth:`load` has loaded for a batch with a recording.
        """
        import mlx.core as mx

        wave = read_voice(voice, float(self._preset.limits.max_voice_seconds), SAMPLE_RATE)
        padded = np.concatenate([wave, np.zeros(int(REFERENCE_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)])
        return mx.array(padded), voice.text

    def read_part(
        self,
        text: str,
        seed: int,
        *,
        speaker: str | None,
        reference: tuple[Any, str] | None,
        delivery: SpeechDelivery | None,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut, generated with *seed*.

        A *reference* (from :meth:`prepare_voice`) is followed as recorded, with no instruction (the Base checkpoint takes none);
        otherwise the part is read in the built-in *speaker* with *delivery* as one instruction sentence. The waveform is empty when the model made no sound; it is cut when it filled *max_new_tokens* tokens.
        mlx-audio's own splitting of the text at line breaks is off: :mod:`speech_takes` has already split it into parts.
        """
        import mlx.core as mx

        mx.random.seed(seed)  # mlx-audio has no seed argument; its sampling draws from the global key
        ref_audio, ref_text = reference if reference is not None else (None, None)
        results = self._model.generate(
            text=text,
            voice=speaker,
            instruct=None if reference is not None else delivery_instruction(delivery),
            lang_code=str(self._settings["language"]),
            ref_audio=ref_audio,
            ref_text=ref_text,
            split_pattern="",
            max_tokens=max_new_tokens,
        )
        waves: list[np.ndarray] = []
        tokens = 0
        for result in results:
            waves.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            tokens += int(result.token_count)
        if not waves:  # the model ended before its first token
            return np.zeros(0, dtype=np.float32), False
        return np.concatenate(waves), tokens >= max_new_tokens

    @staticmethod
    def _load_model(folder: str) -> Any:
        """Load the checkpoint in *folder* with MLX's free-buffer cache off and make its weights resident, as the image models are."""
        from mlx_audio.tts.utils import load

        from zvisiongenerator.backends.mac.memory import load_resident

        return load_resident(lambda: load(folder))
