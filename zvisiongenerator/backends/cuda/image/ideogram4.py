"""Ideogram 4 on the diffusers/CUDA image backend: load, placement, the first-step sigma and the pipeline call.

The model is the NF4 diffusers repo (Unsloth's ungated mirror of the official one): a text encoder and two transformers (the conditional one and the
unconditional one that asymmetric guidance needs) in bitsandbytes NF4, about 15 GB in all. They do not fit a 10 GB card
together, so the transformers stay in system memory and stream to the GPU block by block, and the text encoder is on the
GPU only while it encodes the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.block_streaming import move_outside, stream_blocks, unload_before
from zvisiongenerator.core.ideogram4 import IDEOGRAM4_INITIAL_SIGMA, Ideogram4Sampler, caption_text

if TYPE_CHECKING:
    import torch

__all__ = ["first_sigma_override", "generate", "load_pipeline"]

# Ideogram 4's own default: the prompt is padded to this many tokens for the text encoder.
_MAX_SEQUENCE_LENGTH = 2048


def load_pipeline(model_path: str, torch_dtype: torch.dtype) -> Any:
    """Load the Ideogram 4 pipeline from the NF4 repo at *model_path* and place it for a small GPU.

    Each NF4 component loads onto the GPU, so it is moved to system memory before the next one loads.

    Raises:
        RuntimeError: If *model_path* is not an NF4 copy (the macOS FP8 repo does not fit a consumer GPU).
    """
    import torch
    from diffusers import AutoModel as DiffusersAutoModel
    from diffusers import Ideogram4Pipeline
    from transformers import AutoModel as HFAutoModel

    _require_nf4(model_path)
    text_encoder = HFAutoModel.from_pretrained(model_path, subfolder="text_encoder", dtype=torch_dtype)
    text_encoder.to("cpu")
    transformer = DiffusersAutoModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch_dtype)
    transformer.to("cpu")
    unconditional = DiffusersAutoModel.from_pretrained(model_path, subfolder="unconditional_transformer", torch_dtype=torch_dtype)
    unconditional.to("cpu")
    pipeline = Ideogram4Pipeline.from_pretrained(model_path, text_encoder=text_encoder, transformer=transformer, unconditional_transformer=unconditional, torch_dtype=torch_dtype)
    _place(pipeline, torch.device("cuda"))
    return pipeline


def _require_nf4(model_path: str) -> None:
    """Raise unless the transformer of *model_path* carries a bitsandbytes quantization config."""
    local = Path(model_path).expanduser() / "transformer" / "config.json"
    if local.is_file():
        config_path = local
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(model_path, "transformer/config.json"))
    quantization = json.loads(config_path.read_text(encoding="utf-8")).get("quantization_config") or {}
    if quantization.get("quant_method") != "bitsandbytes":
        raise RuntimeError("Ideogram 4 on Windows and Linux needs an NF4 copy such as unsloth/ideogram-4-nf4-diffusers; the model at '" + model_path + "' is not quantized with bitsandbytes.")


def _place(pipeline: Any, device: torch.device) -> None:
    """Stream both transformers' blocks to *device*, keep the VAE there, and send the text encoder away once it has run."""
    pipeline.vae.to(device)
    for transformer in (pipeline.transformer, pipeline.unconditional_transformer):
        stream_blocks(transformer.layers, device)
        move_outside(transformer, transformer.layers, device)
    # The pipeline calls the text encoder's layers directly, so it cannot offload itself: it leaves before the first denoising step.
    unload_before(pipeline.transformer, [pipeline.text_encoder])
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
        sampler: Steps, guidance and noise schedule.
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
                guidance_scale=sampler.guidance_scale,
                guidance_schedule=None if sampler.guidance_schedule is None else list(sampler.guidance_schedule),
                mu=sampler.mu,
                std=sampler.std,
                max_sequence_length=_MAX_SEQUENCE_LENGTH,
                generator=generator,
                **callback_kwargs,
            )
    finally:
        pipeline.text_encoder.to("cpu")  # already there once the first step ran; this covers a run that failed before it
