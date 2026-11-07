"""Qwen3-TTS's way of taking a Delivery: one instruction sentence per call.

Qwen3-TTS (the CustomVoice checkpoint) steers its delivery with a free-text instruction, not with tags. This module turns the
five Delivery settings into that sentence, ``<Speak|Whisper|Shout> [emotion][, pace][, pitch][, expressiveness].``, such as
``Speak sadly, in a low and sorrowful voice, slowly.`` Settings at their neutral value add nothing, so a neutral Delivery has no
instruction at all. The functions here are pure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import SpeechDelivery

__all__ = ["delivery_instruction"]

_LEAD = {"natural": "Speak", "whisper": "Whisper", "shout": "Shout"}
_LEAD_ALONE = {"whisper": "Whisper softly", "shout": "Shout loudly"}
"""What a manner says when no other setting adds a phrase."""

_EMOTION = {
    "elated": "with elation, in a thrilled and overjoyed voice",
    "amused": "with amusement, in a light and playful voice",
    "enthusiastic": "with excitement, bright and full of energy",
    "proud": "proudly, in a confident and satisfied voice",
    "content": "contentedly, in a warm and relaxed voice",
    "affectionate": "affectionately, in a soft and tender voice",
    "relieved": "with relief, in a calm and lightened voice",
    "longing": "with longing, in a wistful and yearning voice",
    "thoughtful": "thoughtfully, in a calm and reflective voice",
    "determined": "with determination, in a firm and resolute voice",
    "awed": "with awe, in a wondering and amazed voice",
    "surprised": "with surprise, in a startled and astonished voice",
    "confused": "with confusion, in a hesitant and puzzled voice",
    "sad": "sadly, in a low and sorrowful voice",
    "bitter": "bitterly, in a resentful and sour voice",
    "ashamed": "with shame, in a quiet and embarrassed voice",
    "helpless": "helplessly, in a weak and defeated voice",
    "afraid": "fearfully, in a trembling and frightened voice",
    "angry": "angrily, in a sharp and furious voice",
    "disgusted": "with disgust, in a repulsed and contemptuous voice",
}
_PACE = {"very-slow": "very slowly", "slow": "slowly", "fast": "quickly", "very-fast": "very quickly"}
_PITCH = {"low": "in a low-pitched voice", "high": "in a high-pitched voice"}
_EXPRESSIVENESS = {"low": "with little expression", "high": "with lots of expression"}


def delivery_instruction(delivery: SpeechDelivery | None) -> str | None:
    """Return the instruction sentence for *delivery*, or ``None`` when it asks for nothing (no Delivery, or every setting neutral).

    Args:
        delivery: The Delivery of one call, or ``None`` for the model's own manner.

    Returns:
        One sentence ending in a full stop, such as ``Whisper softly.`` or ``Speak with excitement, bright and full of energy, quickly.``
    """
    if delivery is None:
        return None
    phrases = [
        phrase
        for phrase in (
            _EMOTION.get(delivery.emotion),
            _PACE.get(delivery.pace),
            _PITCH.get(delivery.pitch),
            _EXPRESSIVENESS.get(delivery.expressiveness),
        )
        if phrase
    ]
    if not phrases:
        alone = _LEAD_ALONE.get(delivery.manner)
        return f"{alone}." if alone else None
    return f"{_LEAD.get(delivery.manner, 'Speak')} {', '.join(phrases)}."
