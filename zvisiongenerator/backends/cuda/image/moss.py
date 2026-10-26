"""MOSS-Image on the diffusers/CUDA image backend: load, placement, the sampling run and the stored quantized copies.

The model is the built ``moss`` (Turbo) or ``moss-sft`` folder, or a stored q4/q8 copy of one. Turbo is the SFT
checkpoint with a 4-step LoRA that is applied unmerged next to the weights. The text encoder is NF4 at every level and
is on the GPU only while it encodes the prompt. The transformer stays on the GPU at q4 (and goes to the CPU only while
the encoder runs); at q8 and unquantized it streams block by block. The sampling loop is :mod:`.moss_sampler`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.block_streaming import group_offload
from zvisiongenerator.backends.cuda.quant import load_fp8_transformer, load_nf4_text_encoder, load_nf4_transformer, save_components
from zvisiongenerator.core.moss import MAX_SEQUENCE_LENGTH, MossSampler
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.moss_build import built_moss_base, shared_quant_source

if TYPE_CHECKING:
    import torch
    from PIL import Image

__all__ = ["KEEP_IN_COMPUTE_DTYPE", "fp8_keep_patterns", "generate", "load_pipeline", "nf4_keep_modules", "write_stored_copy"]

# The GPU the model runs on (a module constant so tests can run the same code on the CPU).
_DEVICE = "cuda"
# Components a stored copy shares with the folder it was made from; its transformer and text encoder are its own (or the sibling copy's).
_SHARED_COMPONENTS = ("vae", "tokenizer", "scheduler")
# Latent channels and the pixel size of one latent cell: the patchified FLUX.2 VAE latents.
_LATENT_CHANNELS = 128
_LATENT_SCALE = 16
# The Linears whose ``weight.dtype`` the vendored DiT casts activations to; NF4/FP8 storage there corrupts activations.
# ``*`` stands for a layer index.
KEEP_IN_COMPUTE_DTYPE: tuple[str, ...] = (
    "layers.*.proj.q",
    "text_embedder.proj",
    "video_embedder",
    "time_embedding.0",
    "shared_block_modulation",
    "video_head_modulation",
    "final_linear_video",
)
_LAYER_WILDCARD = "layers.*.proj.q"


def load_pipeline(
    model_path: str,
    torch_dtype: torch.dtype,
    bits: int | None,
    *,
    prequantized: bool,
    turbo_lora: Path | None,
    lora_paths: list[str] | None,
    lora_weights: list[float] | None,
) -> Any:
    """Load MOSS-Image from the built folder at *model_path* and place it for a small GPU.

    The text encoder is NF4 at every level and waits on the CPU. The adapters (the Turbo LoRA first, then the user's)
    are added to the transformer before it is placed.

    Args:
        model_path: A built ``moss`` / ``moss-sft`` folder, or a stored quant of one.
        torch_dtype: The compute dtype.
        bits: ``4`` for an NF4 transformer on the GPU, ``8`` for FP8 weight storage, ``None`` for BF16.
        prequantized: Whether *model_path* is a stored copy whose transformer and text encoder are already quantized.
        turbo_lora: The Turbo LoRA file, or ``None`` for SFT.
        lora_paths: User LoRA files, installed after the Turbo LoRA.
        lora_weights: Strength per user LoRA; 1.0 each when ``None``.

    Raises:
        RuntimeError: If *model_path* is not a built MOSS-Image folder or a stored copy of one.
    """
    import torch
    from diffusers import AutoencoderKLFlux2
    from moss_image.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
    from transformers import AutoTokenizer

    from zvisiongenerator.backends.cuda.image.moss_lora import install_loras

    _require_built(model_path)
    device = torch.device(_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = MOSSImageScheduler.from_pretrained(model_path, subfolder="scheduler")
    vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).eval().to(device)
    # The text encoder is the NF4 copy at every level; it loads onto the GPU, so it leaves before the transformer loads.
    text_encoder = load_nf4_text_encoder(model_path, torch_dtype, prequantized=prequantized)
    text_encoder.to("cpu")
    if bits == 4:
        transformer = load_nf4_transformer(model_path, torch_dtype, prequantized=prequantized, model_class=MOSSImageTransformer, keep_modules=nf4_keep_modules(_layer_count(Path(model_path))))
    elif bits == 8:
        transformer = load_fp8_transformer(model_path, torch_dtype, model_class=MOSSImageTransformer, keep_patterns=fp8_keep_patterns())
    else:
        transformer = MOSSImageTransformer.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch_dtype)
    transformer.to("cpu")
    if files := _lora_files(turbo_lora, lora_paths, lora_weights):
        install_loras(transformer, files, device=device)
    pipeline = MOSSImagePipeline(transformer=transformer, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae)
    pipeline._ziv_resident = bits == 4
    pipeline._ziv_memo = None
    if bits != 4:
        group_offload(transformer, device, "block_level")
    return pipeline


def generate(
    pipeline: Any,
    prompt: str,
    *,
    width: int,
    height: int,
    seed: int,
    sampler: MossSampler,
    make_callback: Callable[[int], Callable[..., Any] | None],
    image: Image.Image | None = None,
    strength: float = 1.0,
) -> Image.Image | None:
    """Run one generation and return the image, or ``None`` when the step callback interrupted it.

    At q4 the resident transformer steps aside twice: while the text encoder runs and, once denoising has ended (finished,
    skipped or failed), so the FP32 decode, or the next run's VAE encode, has the card to itself. It returns to the GPU once the starting latents
    exist (after any image-to-image VAE encode), just before denoising. The Turbo LoRA factors are plain tensors that ``.to()`` does not
    move, so they stay on the GPU throughout.

    Args:
        pipeline: The pipeline from :func:`load_pipeline`.
        prompt: The prompt.
        width: Image width in pixels (a multiple of 16).
        height: Image height in pixels (a multiple of 16).
        seed: Seed of the initial noise.
        sampler: Steps, guidance and the negative prompt.
        make_callback: Given the number of steps this run takes, returns the diffusers-style step-end callback (or
            ``None``). The callback sets ``pipeline._interrupt`` to stop the run.
        image: A reference image for image-to-image; ``None`` starts from noise.
        strength: How far image-to-image departs from *image* (1.0 = from noise).
    """
    import torch

    from zvisiongenerator.backends.cuda.image.moss_sampler import decode, denoise, encode_image, initial_noise, noised, set_schedule

    device = torch.device(_DEVICE)
    scheduler = pipeline.scheduler
    start = set_schedule(scheduler, sampler.steps, strength, device)
    callback = make_callback(len(scheduler.timesteps) - start)
    cond, uncond = _conditioning(pipeline, prompt, sampler, device)
    noise = initial_noise((1, _LATENT_CHANNELS, height // _LATENT_SCALE, width // _LATENT_SCALE), torch.Generator("cpu").manual_seed(seed), device)
    latents = noise if image is None else noised(encode_image(pipeline.vae, pipeline.image_processor, image, width, height), noise, float(scheduler.sigmas[start]))
    if pipeline._ziv_resident:
        pipeline.transformer.to(device)

    def on_step(index: int, timestep: torch.Tensor, current: torch.Tensor) -> bool:
        if callback is not None:
            callback(pipeline, index, timestep, {"latents": current})
        return pipeline._interrupt

    pipeline._interrupt = False
    try:
        result = denoise(pipeline.transformer, scheduler, latents, cond=cond, uncond=uncond, guidance=sampler.guidance, skip_layers=sampler.skip_layers, start=start, on_step=on_step)
    finally:
        # However denoising ended, the next run's VAE encode must not find the transformer on the GPU.
        if pipeline._ziv_resident:
            pipeline.transformer.to("cpu")
            torch.cuda.empty_cache()
    if result is None:
        pipeline._interrupt = False
        torch.cuda.empty_cache()
        return None
    decoded = decode(pipeline.vae, pipeline.image_processor, result)
    torch.cuda.empty_cache()
    return decoded


def write_stored_copy(source: Path, target: Path, bits: int, torch_dtype: torch.dtype, backend_format: str, cancelled: Callable[[], bool] | None) -> None:
    """Write the stored q4 (NF4) or q8 (FP8) copy of the built folder *source* into the new folder *target*, from its files.

    Turbo and SFT hold the same transformer and text encoder, so when the other alias already has a current copy at
    *bits* its two components are hard-linked and nothing is quantized. Otherwise the transformer is quantized, then the
    NF4 text encoder, each saved and released before the next. No LoRA is loaded. The VAE, tokenizer, scheduler and
    license are linked from *source*. When *cancelled* turns true the write stops, leaving an incomplete folder for the
    caller to discard.

    Raises:
        RuntimeError: If *source* is not a built MOSS-Image folder.
    """
    _require_built(str(source))
    target.mkdir(parents=True, exist_ok=True)
    shared = shared_quant_source(source, bits, backend_format)
    if shared is not None:
        for name in ("transformer", "text_encoder"):
            link_tree(shared / name, target / name)
    elif not _quantize_components(source, target, bits, torch_dtype, cancelled):
        return
    for name in _SHARED_COMPONENTS:
        link_tree(source / name, target / name)
    if (source / "LICENSE").is_file():
        link_tree(source / "LICENSE", target / "LICENSE")


def _quantize_components(source: Path, target: Path, bits: int, torch_dtype: torch.dtype, cancelled: Callable[[], bool] | None) -> bool:
    """Quantize and save *source*'s transformer and text encoder into *target*, one at a time; return False if stopped."""
    from moss_image.pipeline_moss_image import MOSSImageTransformer

    def load_transformer() -> Any:
        if bits == 4:
            return load_nf4_transformer(str(source), torch_dtype, prequantized=False, model_class=MOSSImageTransformer, keep_modules=nf4_keep_modules(_layer_count(source)))
        return load_fp8_transformer(str(source), torch_dtype, model_class=MOSSImageTransformer, keep_patterns=fp8_keep_patterns())

    loaders = {"transformer": load_transformer, "text_encoder": lambda: load_nf4_text_encoder(str(source), torch_dtype, prequantized=False)}
    return save_components(source, target, loaders, cancelled)


def _conditioning(pipeline: Any, prompt: str, sampler: MossSampler, device: torch.device) -> tuple[Any, Any]:
    """Return the prompt's conditioning and, for a guided run, the negative prompt's; reuse the last run's when unchanged.

    The text encoder takes turns with the transformer on the GPU: at q4 the resident transformer steps aside while it
    encodes (and :func:`generate` brings it back). A repeated prompt and negative prompt skip the turn (and both moves).
    """
    import torch

    from zvisiongenerator.backends.cuda.image.moss_sampler import encode_prompt

    key = (prompt, sampler.negative_prompt)
    memo = pipeline._ziv_memo
    if memo is not None and memo[0] == key:
        return memo[1], memo[2]
    if pipeline._ziv_resident:
        pipeline.transformer.to("cpu")
    pipeline.text_encoder.to(device)
    try:
        cond = encode_prompt(pipeline.tokenizer, pipeline.text_encoder, prompt, device=device, max_length=MAX_SEQUENCE_LENGTH)
        uncond = encode_prompt(pipeline.tokenizer, pipeline.text_encoder, sampler.negative_prompt, device=device, max_length=MAX_SEQUENCE_LENGTH) if sampler.do_cfg else None
    finally:
        pipeline.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    pipeline._ziv_memo = (key, cond, uncond)
    return cond, uncond


def nf4_keep_modules(num_layers: int) -> tuple[str, ...]:
    """Return the full module names NF4 quantisation leaves in the compute dtype (bitsandbytes matches whole dotted names)."""
    per_layer = tuple(_LAYER_WILDCARD.replace("*", str(i)) for i in range(num_layers))
    return per_layer + tuple(name for name in KEEP_IN_COMPUTE_DTYPE if name != _LAYER_WILDCARD)


def fp8_keep_patterns() -> tuple[str, ...]:
    """Return the keep list as anchored regexes for diffusers' layerwise casting (``*`` matches a layer index)."""
    return tuple("^" + name.replace(".", r"\.").replace("*", r"\d+") + "$" for name in KEEP_IN_COMPUTE_DTYPE)


def _lora_files(turbo_lora: Path | None, lora_paths: list[str] | None, lora_weights: list[float] | None) -> list[Any]:
    """Return the adapters to install: the Turbo LoRA at full strength in float32, then the user's at theirs in bfloat16."""
    import torch

    from zvisiongenerator.backends.cuda.image.moss_lora import LoraFile

    paths = lora_paths or []
    weights = lora_weights if lora_weights is not None else [1.0] * len(paths)
    files = [LoraFile(str(turbo_lora), 1.0, torch.float32)] if turbo_lora is not None else []
    return files + [LoraFile(path, weight, torch.bfloat16) for path, weight in zip(paths, weights, strict=True)]


def _layer_count(model_dir: Path) -> int:
    """Return the number of transformer layers in the model at *model_dir*, from the transformer's config."""
    config = json.loads((model_dir.expanduser() / "transformer" / "config.json").read_text(encoding="utf-8"))
    return int(config["num_layers"])


def _require_built(model_path: str) -> None:
    """Raise unless *model_path* is a built MOSS-Image folder (``moss``, ``moss-sft``) or a stored copy of one."""
    if built_moss_base(Path(model_path)) is None:
        raise RuntimeError(f"MOSS-Image runs from the folders built on first use; use -m moss (Turbo) or -m moss-sft instead of '{model_path}'.")
