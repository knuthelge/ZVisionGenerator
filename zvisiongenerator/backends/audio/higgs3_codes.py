"""The code rules of Higgs TTS 3: the delay pattern, the sampler and the rule that ends a take.

The delay and de-delay transforms, the sampler and the frame rule are adapted from sgl-project/sglang-omni (Apache License
2.0, Copyright 2025-2026 SGLang-Omni Team) at upstream commit ``d80161dc34d7b900e7a0869481fa79d35ed618d9``:
``sglang_omni/models/higgs_tts/utils.py`` (``apply_delay_pattern``), ``sglang_omni/utils/codec_delay.py``
(``reverse_delay_pattern``) and ``sglang_omni/models/higgs_tts/sampler.py`` (``sample_independent`` and ``step``, the
single-request path). Changed: the stream markers are arguments instead of module constants, a de-delay of a short input
always returns an empty result, and the sampler and the frame rule are split into two functions over plain values (no
top-k, and the state is a small dataclass). See THIRD_PARTY_LICENSES.md.

Every function here is pure torch over small tensors. torch is imported inside the functions, so importing the module is
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["FrameState", "advance_frame", "delay_codes", "sample_codebooks", "undelay_codes"]

# At or below this temperature the sampler takes the highest logit instead of drawing, which avoids dividing by ~0.
_GREEDY_TEMPERATURE = 1e-5


def delay_codes(codes: torch.Tensor, boc: int, eoc: int) -> torch.Tensor:
    """Shift codebook ``c`` of *codes* down by ``c`` rows, padding with *boc* before and *eoc* after.

    Args:
        codes: Raw codes, ``[T, N]``.
        boc: The id that begins a codebook's stream.
        eoc: The id that ends it.

    Returns:
        The delayed codes, ``[T + N - 1, N]``: row ``r`` of codebook ``c`` is *boc* for ``r < c``, ``codes[r - c, c]`` for
        ``c <= r < c + T`` and *eoc* after that.
    """
    import torch

    length, num_codebooks = codes.shape
    out = torch.full((length + num_codebooks - 1, num_codebooks), eoc, device=codes.device, dtype=codes.dtype)
    rows = torch.arange(length + num_codebooks - 1, device=codes.device)
    for c in range(num_codebooks):
        out[rows < c, c] = boc
        out[c : c + length, c] = codes[:, c]
    return out


def undelay_codes(rows: torch.Tensor) -> torch.Tensor:
    """Undo :func:`delay_codes`: take ``rows[c : c + L - N + 1, c]`` of each codebook ``c``.

    Args:
        rows: Delayed codes, ``[L, N]``.

    Returns:
        The raw codes, ``[L - N + 1, N]``, or ``[0, N]`` when *rows* is shorter than ``N``.
    """
    length, num_codebooks = rows.shape
    kept = length - (num_codebooks - 1)
    if kept <= 0:
        return rows.new_empty((0, num_codebooks))
    out = rows.new_empty((kept, num_codebooks))
    for c in range(num_codebooks):
        out[:, c] = rows[c : c + kept, c]
    return out


def sample_codebooks(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    """Return one code per codebook, each drawn independently from its row of *logits*.

    The operations run in a fixed order, so a seeded take is reproducible bit for bit: greedy at a temperature of about
    zero (no random draw); otherwise the logits divided by *temperature*, a nucleus cut when *top_p* is below 1 (the top
    code always stays), and one ``multinomial`` over the whole ``[N, V]`` tensor.

    Args:
        logits: ``[N, V]`` float32 logits of the N codebooks.
        temperature: The sampling temperature.
        top_p: The nucleus mass; 1 or more keeps every code.

    Returns:
        The sampled codes, ``[N]`` long.
    """
    import torch

    if temperature <= _GREEDY_TEMPERATURE:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # shifted right, so the code that crosses top_p stays
        remove[..., 0] = False
        mask = torch.zeros_like(remove)
        mask.scatter_(-1, sorted_indices, remove)
        logits = torch.where(mask, float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1).to(torch.long)


@dataclass
class FrameState:
    """Where a take is in its start-up ramp and its wind-down; mutated by :func:`advance_frame`."""

    num_codebooks: int
    delay: int = 0
    """Frames sampled so far while still inside the ramp (at most ``num_codebooks``)."""
    countdown: int | None = None
    """Frames left after codebook 0 ended, or ``None`` before it ended."""
    done: bool = False


def advance_frame(state: FrameState, codes: torch.Tensor, *, boc: int, eoc: int) -> None:
    """Apply the frame rule to the sampled *codes* of one frame and move *state*.

    During the first N frames (the ramp) the codebooks after frame ``k`` are overwritten with *boc*, so they start one
    row later each, and an *eoc* in codebook 0 is ignored. After the ramp, an *eoc* in codebook 0 starts a countdown of
    ``N - 2`` frames (the take ends at once when N is 2 or less); when it reaches zero the take is done. The frame that
    ends the take is not part of it.

    Args:
        state: The take's state, changed in place.
        codes: The sampled codes of this frame, ``[N]``, changed in place during the ramp.
        boc: The id that begins a codebook's stream.
        eoc: The id that ends it.
    """
    num_codebooks = state.num_codebooks
    if state.delay < num_codebooks:
        following = state.delay + 1
        if following < num_codebooks:
            codes[following:] = boc
        state.delay += 1
    elif state.countdown is not None:
        state.countdown -= 1
        if state.countdown <= 0:
            state.done = True
    elif int(codes[0].item()) == eoc:
        if num_codebooks <= 2:
            state.done = True
        else:
            state.countdown = num_codebooks - 2
