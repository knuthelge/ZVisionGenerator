"""Encode a job's prompts ahead on macOS (mflux), then free the text encoder for the rest of the job.

A job's prompts are known once its plan is made, so every text the image model's encoder will see can be encoded
right after the model loads. The encoder's memory is then returned for the denoising steps. Each family does it its own way:

- FLUX.2 Klein, Z-Image and Krea 2: :class:`_TextEncoderReplay` takes the encoder's place on the model, records its outputs
  while the prompts are encoded through the model's own encode method, and replays them afterwards. Replays are keyed
  by the encoder's inputs (the token ids), so they serve every generation path that reaches the encoder: txt2img,
  img2img, the references view and the upscale refine at any guidance.
- Ideogram 4: the encoder runs the steps of ``generate_image`` that precede its encode, which fills mflux's own
  ``prompt_cache``. The model's ``text_encoder`` is then ``None``, so a prompt that was not encoded fails with mflux's
  own error.

Anything that stops the encoding (Quit, an unexpected error) puts the real text encoder back on the model. Prompt-cache
entries made before then stay: they are valid encodings of the same encoder.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

from zvisiongenerator.core import JobPrompt
from zvisiongenerator.backends.mac.memory import evaluate, release_memory
from zvisiongenerator.core.ideogram4 import wrap_ideogram4_prompt

__all__ = ["prepare_prompts"]

_BLANK = " "

# The model method that encodes exactly one text when called with ``negative_prompt=None, guidance=1.0``.
_ENCODE_TEXT = {
    "flux2_klein": "_encode_prompt_pair",
    "zimage": "_encode_prompts",
    "krea2": "_encode_prompts",
}

_FREED_MISS = "The text encoder was freed after this job's prompts were encoded, and this prompt was not among them."


def _encoder_texts(family: str, prompts: Sequence[JobPrompt]) -> tuple[str, ...]:
    """Return the distinct texts *family*'s text encoder sees for *prompts*, in first-seen order.

    Klein encodes each prompt and the blank ``" "`` (its implicit negative). Z-Image and Krea 2 also encode each
    non-blank negative prompt as written.

    Args:
        family: A family in ``_ENCODE_TEXT``.
        prompts: The job's prompts.

    Returns:
        The texts to encode.
    """
    texts = [job_prompt.prompt for job_prompt in prompts]
    if family != "flux2_klein":
        texts.extend(job_prompt.negative_prompt for job_prompt in prompts if job_prompt.negative_prompt and job_prompt.negative_prompt.strip())
    texts.append(_BLANK)
    return tuple(dict.fromkeys(texts))


def _array_key(array: Any) -> tuple[Any, ...]:
    """Return a hashable identity for an MLX array: its shape, dtype and bytes."""
    import numpy as np

    return ("array", tuple(array.shape), str(array.dtype), np.array(array).tobytes())


def _call_key(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Return a hashable key for one encoder call: the method and its arguments, arrays as shape, dtype and bytes."""
    import mlx.core as mx

    def keyed(value: Any) -> Any:
        return _array_key(value) if isinstance(value, mx.array) else value

    return (method, tuple(keyed(arg) for arg in args), tuple(sorted((name, keyed(value)) for name, value in kwargs.items())))


class _TextEncoderReplay:
    """Stand in for a model's text encoder: record its outputs, then replay them once the encoder is released.

    Each output is evaluated as it is recorded, so the next text is encoded from a settled graph.

    A plain object on purpose: assigning it to ``model.text_encoder`` takes the encoder out of the model's module tree,
    so the real encoder's parameters are no longer reachable from the model.
    """

    def __init__(self, real: Any) -> None:
        self._real: Any | None = real
        self._outputs: dict[tuple[Any, ...], Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._answer("__call__", args, kwargs)

    def get_prompt_embeds(self, *args: Any, **kwargs: Any) -> Any:
        """Return the recorded embeddings for these inputs, asking the real encoder when it is still attached."""
        return self._answer("get_prompt_embeds", args, kwargs)

    def release(self) -> None:
        """Let go of the real encoder; calls with inputs that were not recorded now raise."""
        self._real = None

    def _answer(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        key = _call_key(method, args, kwargs)
        if key in self._outputs:
            return self._outputs[key]
        if self._real is None:
            raise RuntimeError(_FREED_MISS)
        target = self._real if method == "__call__" else getattr(self._real, method)
        output = evaluate(target(*args, **kwargs))
        self._outputs[key] = output
        return output


def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
    return cancelled is not None and cancelled()


def _encode_with_replay(model: Any, family: str, prompts: Sequence[JobPrompt], real: Any, cancelled: Callable[[], bool] | None) -> bool:
    """Encode every text through a recording stand-in, then release it; return ``False`` when cancelled.

    On a cancel the stand-in is left on the model for the caller to restore.
    """
    replay = _TextEncoderReplay(real)
    model.text_encoder = replay
    encode = getattr(model, _ENCODE_TEXT[family])
    texts = _encoder_texts(family, prompts)
    print(f"Encoding {len(texts)} text(s)...", flush=True)
    for text in texts:
        if _is_cancelled(cancelled):
            return False
        encode(prompt=text, negative_prompt=None, guidance=1.0)
    replay.release()
    return True


def _encode_ideogram4(model: Any, prompts: Sequence[JobPrompt], width: int, height: int, real: Any, cancelled: Callable[[], bool] | None) -> bool:
    """Fill mflux's ``prompt_cache`` with every prompt at the job's size, then drop the encoder; return ``False`` when cancelled.

    A prompt whose caption is empty is skipped, and so is every prompt when mflux rejects the job's size: those images
    fail later with mflux's own error.
    """
    from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator
    from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder

    try:
        Ideogram4LatentCreator.validate_dimensions(width=width, height=height)
    except ValueError:
        model.text_encoder = None
        return True
    print(f"Encoding {len({job_prompt.prompt for job_prompt in prompts})} text(s)...", flush=True)
    for job_prompt in prompts:
        if _is_cancelled(cancelled):
            return False
        try:
            caption = Ideogram4PromptEncoder.resolve_prompt(wrap_ideogram4_prompt(job_prompt.prompt), strict_caption_validation=False, warn_on_caption_issues=False)
        except ValueError:
            continue
        features = Ideogram4PromptEncoder.build_inputs(model.tokenizers["ideogram4"], [caption], height=height, width=width)
        Ideogram4PromptEncoder.encode_prompt(
            prompt=caption,
            width=width,
            height=height,
            inputs=features,
            text_encoder=real,
            prompt_cache=model.prompt_cache,
        )
    model.text_encoder = None
    return True


def prepare_prompts(
    model: Any,
    family: str,
    prompts: Sequence[JobPrompt],
    *,
    width: int,
    height: int,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Encode *prompts* with *model*'s text encoder, then free it for the rest of the job.

    Nothing happens for a family without an entry or an empty *prompts*. When *cancelled* returns true, or encoding
    raises, the model's text encoder is put back (a raised error is reported as a warning) and the job runs with it in
    memory. Otherwise ``model.text_encoder`` no longer holds the real encoder.

    Args:
        model: A loaded mflux model of *family*.
        family: Image model family, as in ``ImageModelInfo.family``.
        prompts: Every prompt the job's generations will use.
        width: The width the job generates at; only Ideogram 4 keys its encoding on it.
        height: The height the job generates at; only Ideogram 4 keys its encoding on it.
        cancelled: Polled before each encode; returning true stops the encoding.
    """
    if (family not in _ENCODE_TEXT and family != "ideogram4") or not prompts:
        return
    real = model.text_encoder
    try:
        if family == "ideogram4":
            completed = _encode_ideogram4(model, prompts, width, height, real, cancelled)
        else:
            completed = _encode_with_replay(model, family, prompts, real, cancelled)
    except Exception as exc:
        model.text_encoder = real
        warnings.warn(f"Could not encode the job's prompts ahead: {exc}. The text encoder stays loaded for this job.", stacklevel=2)
        return
    if not completed:
        model.text_encoder = real
        return
    del real  # this frame must not keep the last reference to the encoder; the model no longer holds it
    release_memory()
