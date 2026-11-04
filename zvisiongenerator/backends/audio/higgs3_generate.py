"""The Higgs TTS 3 generation loop: one prompt in, the codes of one take out.

The loop prefills the prompt once, then samples one frame of N codes per iteration and feeds it back through the cached
backbone, until the frame rule ends the take or ``max_new_tokens`` iterations have run. torch is imported inside the
function, so importing the module is cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zvisiongenerator.backends.audio.higgs3_codes import FrameState, advance_frame, delay_codes, sample_codebooks, undelay_codes
from zvisiongenerator.backends.audio.higgs3_prompt import build_prompt_ids

if TYPE_CHECKING:
    import torch
    from transformers import Qwen2Tokenizer

    from zvisiongenerator.backends.audio.higgs3_model import Higgs3Model
    from zvisiongenerator.backends.audio.higgs3_prompt import SpecialIds

__all__ = ["SpeechCodes", "generate_speech_codes"]


@dataclass(frozen=True)
class SpeechCodes:
    """The codes of one take."""

    codes: torch.Tensor
    """``[T, N]`` long, on the CPU; ``T == 0`` means the take has no sound."""
    cut: bool
    """Whether the loop ran ``max_new_tokens`` iterations without the take finishing."""


def generate_speech_codes(
    model: Higgs3Model,
    tokenizer: Qwen2Tokenizer,
    ids: SpecialIds,
    text: str,
    *,
    reference_codes: torch.Tensor | None,
    reference_text: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    on_frame: Callable[[], None],
) -> SpeechCodes:
    """Return the codes that read *text*, in the voice of *reference_codes* when given.

    Args:
        model: The loaded network.
        tokenizer: The model's tokenizer.
        ids: The structural token ids.
        text: The text to read.
        reference_codes: The codec codes ``[T, N]`` of a recording whose voice to follow, on the CPU; ``None`` for the model's own voice.
        reference_text: The transcript of the recording; ``None`` or empty for none.
        max_new_tokens: The most iterations to run.
        temperature: The sampling temperature; 0 takes the most likely code every time.
        top_p: The nucleus mass; 1 keeps every code.
        on_frame: Called once per iteration, right after the model's head has run, the iteration that ends the take included.
            It may raise to stop the take; the exception propagates.

    Returns:
        The codes of the take, with ``cut`` set when the loop ran out of iterations. A take of fewer than N frames has no sound.
    """
    import torch

    num_codebooks = model.audio_head.num_codebooks
    boc, eoc = model.audio_head.codebook_size - 2, model.audio_head.codebook_size - 1
    with torch.no_grad():
        reference_rows = delay_codes(reference_codes, boc, eoc) if reference_codes is not None else None
        prompt = build_prompt_ids(tokenizer, ids, text, reference_rows=0 if reference_rows is None else reference_rows.shape[0], reference_text=reference_text)
        hidden, cache = model.prefill(model.embed_prompt(prompt, reference_rows))
        state = FrameState(num_codebooks)
        kept: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            logits = model.audio_head(hidden).to(torch.float32)[0]
            on_frame()
            codes = sample_codebooks(logits, temperature=temperature, top_p=top_p)
            advance_frame(state, codes, boc=boc, eoc=eoc)
            if state.done:
                break
            kept.append(codes.cpu())
            hidden = model.step(codes, cache)
        rows = torch.stack(kept) if kept else torch.empty((0, num_codebooks), dtype=torch.long)
        return SpeechCodes(codes=undelay_codes(rows), cut=not state.done)
