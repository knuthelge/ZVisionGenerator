"""Ideogram 4 Instant on the diffusers/CUDA image backend: load, placement, the first-step sigma and the pipeline call.

The model is the built Ideogram 4 Instant folder: the base model's NF4 text encoder and one distilled transformer,
in BF16 or as a q4 (NF4) or q8 (FP8) stored copy. Instant has its guidance built in, so the pipeline's unconditional
branch gets a stand-in without weights whose output the guidance of 1.0 multiplies by zero. The transformer stays in
system memory and streams to the GPU block by block, and the text encoder is on the GPU only while it encodes the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.block_streaming import group_offload, move_outside, stream_blocks, unload_before
from zvisiongenerator.backends.cuda.quant import load_fp8_transformer, load_nf4_transformer
from zvisiongenerator.core.ideogram4 import IDEOGRAM4_INITIAL_SIGMA, INSTANT_GUIDANCE, Ideogram4Sampler, caption_text
from zvisiongenerator.utils.file_links import link_tree

if TYPE_CHECKING:
    import torch

__all__ = ["first_sigma_override", "generate", "load_pipeline", "save_quantized", "zero_unconditional_transformer"]

# Ideogram 4's own default: the prompt is padded to this many tokens for the text encoder.
_MAX_SEQUENCE_LENGTH = 2048
# Components a stored copy shares with the folder it was made from (the text encoder is already NF4 there).
_SHARED_COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "vae")
# Where the pipeline was loaded from, so a stored copy can link the components it shares.
_SOURCE_ATTRIBUTE = "_ziv_source_path"


def zero_unconditional_transformer(dtype: torch.dtype) -> Any:
    """Return a stand-in for the unconditional transformer diffusers' Ideogram 4 pipeline always calls: no weights, zero output.

    The pipeline blends ``g * conditional + (1 - g) * unconditional``; at Instant's guidance of 1.0 the stand-in's
    output is multiplied by zero, so the result is the distilled transformer's alone. It reports *dtype*, which the
    pipeline casts the unconditional inputs to.
    """
    import torch

    class ZeroUnconditionalTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("_dtype_anchor", torch.empty(0, dtype=dtype), persistent=False)

        @property
        def dtype(self) -> torch.dtype:
            return self._dtype_anchor.dtype

        def forward(self, *, hidden_states: torch.Tensor, **_kwargs: Any) -> tuple[torch.Tensor]:
            return (torch.zeros_like(hidden_states),)

    return ZeroUnconditionalTransformer()


def load_pipeline(model_path: str, torch_dtype: torch.dtype, bits: int | None, *, prequantized: bool) -> Any:
    """Load Ideogram 4 Instant from the folder at *model_path* and place it for a small GPU.

    Args:
        model_path: The built Instant folder, or a stored quant of it.
        torch_dtype: The compute dtype.
        bits: ``4`` for an NF4 transformer, ``8`` for FP8 weight storage, ``None`` for BF16.
        prequantized: Whether *model_path* is a stored copy whose transformer is already at *bits*.

    Raises:
        RuntimeError: If *model_path* is a base Ideogram 4 model (it has an unconditional transformer).
    """
    import torch
    from diffusers import AutoModel as DiffusersAutoModel
    from diffusers import Ideogram4Pipeline
    from transformers import AutoModel as HFAutoModel

    _require_instant(model_path)
    # The text encoder is the base repo's NF4 copy at every level; it loads onto the GPU, so it moves off before the transformer loads.
    text_encoder = HFAutoModel.from_pretrained(model_path, subfolder="text_encoder", dtype=torch_dtype)
    text_encoder.to("cpu")
    if bits == 4:
        transformer = load_nf4_transformer(model_path, torch_dtype, prequantized=prequantized)
        transformer.to("cpu")
    elif bits == 8:
        transformer = load_fp8_transformer(model_path, torch_dtype)
    else:
        transformer = DiffusersAutoModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch_dtype)
    pipeline = Ideogram4Pipeline.from_pretrained(model_path, text_encoder=text_encoder, transformer=transformer, unconditional_transformer=None, torch_dtype=torch_dtype)
    # The model index lists no unconditional transformer, so the stand-in is not recorded there (nor saved with a stored copy).
    pipeline.unconditional_transformer = zero_unconditional_transformer(torch_dtype)
    setattr(pipeline, _SOURCE_ATTRIBUTE, str(Path(model_path).expanduser()))
    _place(pipeline, torch.device("cuda"), nf4=bits == 4)
    return pipeline


def save_quantized(pipeline: Any, path: str) -> None:
    """Write the stored copy of a pipeline loaded at q4: its NF4 transformer, and hard links to the components it shares.

    transformers cannot save a text encoder that was loaded already quantized, and it would be the same bytes as
    the source's, so it is linked like the scheduler, tokenizer and VAE.
    """
    target = Path(path)
    source = Path(getattr(pipeline, _SOURCE_ATTRIBUTE))
    pipeline.transformer.save_pretrained(str(target / "transformer"))
    for name in _SHARED_COMPONENTS:
        link_tree(source / name, target / name)


def _require_instant(model_path: str) -> None:
    """Raise unless *model_path* is a local Ideogram 4 Instant folder (the base model lists an unconditional transformer)."""
    index_path = Path(model_path).expanduser() / "model_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Ideogram 4 runs from the Ideogram 4 Instant folder built on first use; use 'ideo' instead of '{model_path}'.")
    entry = json.loads(index_path.read_text(encoding="utf-8")).get("unconditional_transformer")
    if isinstance(entry, list) and entry and entry[0] is not None:
        raise RuntimeError(f"'{model_path}' is the base Ideogram 4 model, which is no longer supported; use 'ideo' (Ideogram 4 Instant).")


def _place(pipeline: Any, device: torch.device, *, nf4: bool) -> None:
    """Stream the transformer's blocks to *device*, keep the VAE there, and send the text encoder away once it has run."""
    pipeline.vae.to(device)
    transformer = pipeline.transformer
    if nf4:
        # Group offloading leaves bitsandbytes' quantization state on the CPU; swapping whole parameters carries it.
        stream_blocks(transformer.layers, device)
        move_outside(transformer, transformer.layers, device)
    else:
        group_offload(transformer, device, "block_level")
    pipeline.unconditional_transformer.to(device)
    # The pipeline calls the text encoder's layers directly, so it cannot offload itself: it leaves before the first denoising step.
    unload_before(transformer, [pipeline.text_encoder])
    _offload_text_encoder_after_encoding(pipeline)


def _offload_text_encoder_after_encoding(pipeline: Any) -> None:
    """Send the text encoder to the CPU as soon as it has produced its hidden states.

    Right after them the pipeline builds a float32 conditioning tensor over every image token (about 53,000 features
    each, 2 GiB at 2 MP and more beyond), which does not fit beside the 4.7 GiB text encoder on a 10 GB card.
    """
    original = pipeline._get_text_encoder_hidden_states  # a static method; the instance attribute shadows it

    def encode_then_offload(*args: Any, **kwargs: Any) -> Any:
        import torch

        try:
            return original(*args, **kwargs)
        finally:
            pipeline.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    pipeline._get_text_encoder_hidden_states = encode_then_offload


@contextmanager
def first_sigma_override(scheduler: Any, sigma: float | None) -> Iterator[None]:
    """Make the next ``scheduler.set_timesteps(sigmas=...)`` start at *sigma* instead of its first sigma, until the block ends.

    The pipeline builds its noise schedule inside ``__call__`` and takes no argument for the first sigma, so the
    scheduler's call is wrapped. ``None`` leaves the schedule alone.
    """
    if sigma is None:
        yield
        return
    original = scheduler.set_timesteps

    def set_timesteps(*args: Any, sigmas: Any = None, **kwargs: Any) -> Any:
        if sigmas is None:  # a call by step count or timesteps has no sigmas to start elsewhere
            return original(*args, **kwargs)
        sigmas = list(sigmas)
        sigmas[0] = sigma
        return original(*args, sigmas=sigmas, **kwargs)

    scheduler.set_timesteps = set_timesteps
    try:
        yield
    finally:
        del scheduler.set_timesteps  # back to the class's method


def generate(
    pipeline: Any,
    prompt: str,
    *,
    width: int,
    height: int,
    generator: torch.Generator,
    sampler: Ideogram4Sampler,
    first_sigma: float | None,
    callback_kwargs: dict[str, Any],
) -> Any:
    """Run one text-to-image generation and return the pipeline's output.

    Args:
        pipeline: The pipeline from :func:`load_pipeline`.
        prompt: A plain prompt or a JSON caption; plain text is wrapped into a minimal caption.
        width: Image width in pixels.
        height: Image height in pixels.
        generator: Seeded generator for the initial noise.
        sampler: Steps and noise schedule.
        first_sigma: The first-step sigma for this run, or ``None`` for the default.
        callback_kwargs: The step-end callback arguments (progress, skip).
    """
    import torch

    pipeline.text_encoder.to(torch.device("cuda"))
    sigma = first_sigma if first_sigma is not None else IDEOGRAM4_INITIAL_SIGMA
    try:
        with first_sigma_override(pipeline.scheduler, sigma):
            return pipeline(
                caption_text(prompt),
                height=height,
                width=width,
                num_inference_steps=sampler.steps,
                guidance_scale=INSTANT_GUIDANCE,
                guidance_schedule=None,
                mu=sampler.mu,
                std=sampler.std,
                max_sequence_length=_MAX_SEQUENCE_LENGTH,
                generator=generator,
                **callback_kwargs,
            )
    finally:
        pipeline.text_encoder.to("cpu")  # already there once the first step ran; this covers a run that failed before it
