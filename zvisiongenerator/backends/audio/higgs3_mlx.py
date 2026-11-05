"""Higgs TTS 3 speech backend on MLX (macOS): mlx-audio's ``higgs_audio_v3`` model on the official weights.

mlx-audio loads the pinned ``bosonai/higgs-tts-3-4b`` snapshot from the Hugging Face cache only, with the codec read from
the same checkpoint. Its own ``generate`` reports nothing until a part is done, so :func:`_generate_rows` runs the same
loop from the model's parts (following mlx-audio 0.6.2's ``Model.generate``, MIT) and calls back once per frame, which is how progress and Stop reach it; mlx-audio's fades
are left out, as the join does its own. Parts, the voice and the join are :mod:`higgs3_takes`'; this backend is its MLX
engine. mlx and mlx-audio are imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.higgs3_takes import read_speech
from zvisiongenerator.core.audio_backend import raise_if_cancelled

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["Higgs3MlxBackend"]


def _generate_rows(
    model: Any,
    text: str,
    reference: tuple[Any, str | None] | None,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    on_frame: Callable[[], None],
) -> tuple[list[Any], bool]:
    """Return the delayed code rows of one part and whether it was cut, sampling as mlx-audio's ``generate`` does.

    Args:
        model: mlx-audio's ``higgs_audio_v3`` model.
        text: The part's text.
        reference: The delayed reference codes and their transcript, or ``None`` for the model's own voice.
        seed: The part's seed.
        max_new_tokens: The most frames to run.
        temperature: The sampling temperature.
        top_p: The nucleus mass.
        on_frame: Called once per frame, right after the model's head has run, the frame that ends the part included.

    Returns:
        The rows, and ``True`` when the loop ran *max_new_tokens* frames without the part ending.
    """
    import mlx.core as mx
    from mlx_audio.lm.models.cache import make_prompt_cache
    from mlx_audio.tts.models.higgs_audio_v3.generation import HiggsSamplerState, step
    from mlx_audio.tts.models.higgs_audio_v3.prompt import ReferenceCodes

    mx.random.seed(seed)
    references = [ReferenceCodes(codes=reference[0], text=reference[1])] if reference is not None else []
    embeds, _ = model._build_prompt_embeddings(text, references)
    cache = make_prompt_cache(model)
    hidden = model.backbone(mx.zeros((1, embeds.shape[1]), dtype=mx.int32), cache=cache, input_embeddings=embeds)[:, -1, :]
    config = model.config
    state = HiggsSamplerState(num_codebooks=config.audio_num_codebooks)
    rows: list[Any] = []
    for _ in range(max_new_tokens):
        logits = model._audio_logits(hidden)[0]
        on_frame()
        codes = step(logits, state, temperature=temperature, top_p=top_p, top_k=None, boc_id=config.audio_boc_token_id, eoc_id=config.audio_eoc_token_id)
        rows.append(codes)
        if state.generation_done:
            return rows, False
        hidden = model.backbone(mx.zeros((1, 1), dtype=mx.int32), cache=cache, input_embeddings=model._embed_audio_codes(codes)[None])[:, -1, :]
    return rows, True


class Higgs3MlxBackend:
    """Speech from Higgs TTS 3 on MLX, with long text read in parts and an optional voice to follow (see :mod:`higgs3_takes`)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._limits: AudioLimits | None = None
        self._settings: Mapping[str, Any] = {}

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the model and its codec from the cached snapshot of the preset's one repository; nothing is downloaded.

        The weights are built with MLX's free-buffer cache off and made resident, as the image models are.

        Args:
            preset: The ``higgs3`` preset as macOS runs it: the pinned model files, the limits and the settings.
            device: ``mps``; MLX runs on the Mac's GPU whatever it is.
            cancelled: Checked before and after the load; ``True`` stops it and frees the model.

        Raises:
            FileNotFoundError: If the repository is not in the cache with every listed file.
            AudioCancelled: If *cancelled* returned ``True``.
        """
        (model_files,) = preset.files
        folder = snapshot_dir(model_files)
        self.close()
        self._limits = preset.limits
        self._settings = preset.settings_for(device)
        try:
            raise_if_cancelled(cancelled)
            from mlx_audio.tts.utils import load

            from zvisiongenerator.backends.mac.memory import load_resident

            self._model = load_resident(lambda: load(str(folder)))
            raise_if_cancelled(cancelled)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take; see :func:`higgs3_takes.read_speech`.

        MLX's free-buffer cache is off for the take, so each part's freed key-value cache does not stay beside the next part's.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            RuntimeError: If the model made no sound.
        """
        from zvisiongenerator.backends.mac.memory import mlx_cache_limit

        with mlx_cache_limit(0):
            return read_speech(self, request, self._limits, self._settings, on_progress, cancelled)

    def close(self) -> None:
        """Drop the model and its codec; the runner returns MLX's memory. Safe to call again, after a failed load and after a failed take."""
        self._model = None

    def read_part(
        self,
        text: str,
        seed: int,
        reference: tuple[Any, str | None] | None,
        max_new_tokens: int,
        on_frame: Callable[[], None],
        on_decode: Callable[[int], None],
    ) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut, generated with *seed* and following *reference* when given."""
        import mlx.core as mx

        rows, cut = _generate_rows(
            self._model,
            text,
            reference,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=float(self._settings["temperature"]),
            top_p=float(self._settings["top_p"]),
            on_frame=on_frame,
        )
        if len(rows) < self._model.config.audio_num_codebooks:  # no decode is reported for a part without sound
            return np.zeros(0, dtype=np.float32), cut
        on_decode(0)
        boc_id = self._model.config.audio_boc_token_id
        rows = [mx.where(row >= boc_id, 0, row) for row in rows]  # the two ids above the codec's values are stream markers
        audio = self._model._decode_audio(rows)
        mx.eval(audio)
        on_decode(1)
        return np.asarray(audio, dtype=np.float32), cut

    def encode(self, wave: np.ndarray) -> Any:
        """Encode a 24 kHz mono waveform with the model's codec into the delayed reference codes the prompt takes."""
        return self._model.encode_reference_audio(np.ascontiguousarray(wave, dtype=np.float32))
