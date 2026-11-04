"""The tokenizer and the prompt of Higgs TTS 3.

``build_prompt_ids`` is adapted from sgl-project/sglang-omni (Apache License 2.0, Copyright 2025-2026 SGLang-Omni Team) at
upstream commit ``d80161dc34d7b900e7a0869481fa79d35ed618d9``: ``sglang_omni/models/higgs_tts/text_tokenizer.py``
(``HiggsTokenizerAdapter.build_prompt``). Changed: it is a function over :class:`SpecialIds` instead of a method of an
adapter object, and the ``<|ref_text|>`` token is required instead of optional. See THIRD_PARTY_LICENSES.md.

transformers is imported inside :func:`load_tokenizer`, so importing the module is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Qwen2Tokenizer

__all__ = ["AUDIO_SLOT", "SpecialIds", "build_prompt_ids", "load_tokenizer", "special_ids"]

AUDIO_SLOT = -100
"""Placeholder in the prompt ids where a reference voice's audio embedding replaces the text embedding."""

_TTS = "<|tts|>"
_REF_TEXT = "<|ref_text|>"
_REF_AUDIO = "<|ref_audio|>"
_TEXT = "<|text|>"
_AUDIO = "<|audio|>"


@dataclass(frozen=True)
class SpecialIds:
    """The ids of the five structural tokens that frame a prompt."""

    tts: int
    ref_text: int
    ref_audio: int
    text: int
    audio: int


def load_tokenizer(folder: Path) -> Qwen2Tokenizer:
    """Return the tokenizer held by *folder* (``tokenizer.json`` and ``tokenizer_config.json``).

    The class is named here instead of resolved through an Auto class, so the folder's ``config.json`` is never read.

    Args:
        folder: The model folder.
    """
    from transformers import Qwen2Tokenizer

    return Qwen2Tokenizer.from_pretrained(str(folder))


def special_ids(tokenizer: Qwen2Tokenizer) -> SpecialIds:
    """Return the ids of the structural tokens from *tokenizer*'s added vocabulary."""
    vocab = tokenizer.get_added_vocab()
    return SpecialIds(tts=vocab[_TTS], ref_text=vocab[_REF_TEXT], ref_audio=vocab[_REF_AUDIO], text=vocab[_TEXT], audio=vocab[_AUDIO])


def build_prompt_ids(tokenizer: Qwen2Tokenizer, ids: SpecialIds, text: str, *, reference_rows: int, reference_text: str | None) -> list[int]:
    """Return the prompt ids for reading *text*, with the voice of a reference recording when *reference_rows* is above zero.

    The layout has no start token and no chat template::

        <|tts|> [<|ref_text|> reference_text <|ref_audio|> SLOT x rows] <|text|> text <|audio|>

    ``<|ref_text|>`` and the transcript are left out when the transcript is empty or ``None``; the whole bracket is left
    out without a reference.

    Args:
        tokenizer: The model's tokenizer.
        ids: The structural token ids.
        text: The text to read.
        reference_rows: The number of delayed reference rows, each replaced by one :data:`AUDIO_SLOT`; 0 for no reference.
        reference_text: The transcript of the reference recording.
    """
    prompt = [ids.tts]
    if reference_rows > 0 and reference_text:
        prompt.append(ids.ref_text)
        prompt.extend(tokenizer.encode(reference_text, add_special_tokens=False))
    if reference_rows > 0:
        prompt.append(ids.ref_audio)
        prompt.extend([AUDIO_SLOT] * reference_rows)
    prompt.append(ids.text)
    prompt.extend(tokenizer.encode(text, add_special_tokens=False))
    prompt.append(ids.audio)
    return prompt
