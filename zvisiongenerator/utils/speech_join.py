"""Join the parts of a long speech take into one take.

Each part is trimmed of long edge silence, brought close to the parts' common speech level, faded at both edges and
followed by a pause; there is no crossfade, which would blend the words across the join. The functions here are pure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["join_speech_parts"]

# Silence and level are measured on blocks of this many milliseconds.
_BLOCK_MS = 10
# A block is speech when it is louder than this far below the 95th percentile of the blocks, and above the floor.
_SPEECH_RANGE_DB = 30.0
_SPEECH_FLOOR_DB = -60.0


def join_speech_parts(parts: Sequence[np.ndarray], rate: int, settings: Mapping[str, Any]) -> np.ndarray:
    """Join mono *parts* into one mono take.

    For each part: the silence before and after its speech is cut to at most ``edge_silence_max_ms``; its speech level
    is moved toward the mean level of the parts by at most ``part_level_match_db`` (never into clipping); ``join_fade_ms``
    fades are applied at both edges. ``join_pause_ms`` of silence separates the parts. Empty parts are left out, and a
    part with no speech above the silence floor is kept as it is.

    Args:
        parts: Float32 waveforms shaped ``(samples,)``, in reading order.
        rate: Their sample rate in Hz.
        settings: The preset settings ``join_pause_ms``, ``join_fade_ms``, ``part_level_match_db`` and ``edge_silence_max_ms``.

    Returns:
        The take as a float32 waveform shaped ``(samples,)``: the trimmed parts and ``len(parts) - 1`` pauses.

    Raises:
        ValueError: If a setting is missing, or no part holds samples.
    """
    pause = _samples(_setting(settings, "join_pause_ms"), rate)
    fade = _samples(_setting(settings, "join_fade_ms"), rate)
    edge = _samples(_setting(settings, "edge_silence_max_ms"), rate)
    cap_db = float(_setting(settings, "part_level_match_db"))
    waves = [np.asarray(part, dtype=np.float32) for part in parts if len(part)]
    if not waves:
        raise ValueError("There is no speech to join.")
    block = max(1, round(rate * _BLOCK_MS / 1000))
    measures = [_measure(wave, block) for wave in waves]
    levels = [level for _, _, level in measures if level is not None]
    mean_db = float(np.mean(levels)) if levels else 0.0
    pieces: list[np.ndarray] = []
    for index, (wave, (first, last, level)) in enumerate(zip(waves, measures, strict=True)):
        if level is not None:
            wave = wave[max(0, first * block - edge) : min(len(wave), (last + 1) * block + edge)]
            wave = wave * _gain(wave, float(np.clip(mean_db - level, -cap_db, cap_db)))
        pieces.append(_faded(wave, fade))
        if index < len(waves) - 1:
            pieces.append(np.zeros(pause, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _setting(settings: Mapping[str, Any], key: str) -> Any:
    """Return the setting *key*, or raise ``ValueError`` naming it."""
    if key not in settings:
        raise ValueError(f"The speech join needs the setting '{key}'.")
    return settings[key]


def _samples(milliseconds: float, rate: int) -> int:
    """Return *milliseconds* as a whole number of samples at *rate*."""
    return max(0, round(rate * float(milliseconds) / 1000))


def _measure(wave: np.ndarray, block: int) -> tuple[int, int, float | None]:
    """Return the first and last speech block of *wave* and its speech level in dB, or ``(0, 0, None)`` when it has no speech.

    The level is the mean power of the speech blocks.
    """
    padded = np.pad(wave, (0, -len(wave) % block))
    power = (padded.reshape(-1, block).astype(np.float64) ** 2).mean(axis=1)
    db = 10 * np.log10(power + 1e-12)
    threshold = max(float(np.percentile(db, 95)) - _SPEECH_RANGE_DB, _SPEECH_FLOOR_DB)
    speech = np.flatnonzero(db > threshold)
    if speech.size == 0:
        return 0, 0, None
    return int(speech[0]), int(speech[-1]), float(10 * np.log10(power[speech].mean()))


def _gain(wave: np.ndarray, gain_db: float) -> float:
    """Return the linear gain for *gain_db*; a boost stops short of clipping the loudest sample."""
    gain = 10 ** (gain_db / 20)
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if gain > 1 and peak > 0:
        gain = min(gain, max(1.0, 1 / peak))
    return gain


def _faded(wave: np.ndarray, fade: int) -> np.ndarray:
    """Return a copy of *wave* with linear fades of *fade* samples (at most half the wave) at both edges."""
    out = wave.astype(np.float32, copy=True)
    count = min(fade, len(out) // 2)
    if count:
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        out[:count] *= ramp
        out[-count:] *= ramp[::-1]
    return out
