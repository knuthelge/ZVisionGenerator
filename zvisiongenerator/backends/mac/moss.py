"""MOSS-Image on macOS (mflux): staged loading, prompt encoding, the denoising loop and the stored copies.

The model loads in two stages that are never resident together: the Qwen3.5 text encoder, which encodes every prompt of
the job and is then freed, and the denoiser (transformer and VAE, with the Turbo and user LoRAs unmerged). The loop is
our own, driving mflux's transformer, scheduler and VAE, because mflux's ``generate_image`` takes no start image: it
runs text-to-image, image-to-image and the upscale refine, and reports each step through the model's mflux callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from zvisiongenerator.backends.mac.memory import load_resident, release_memory, upcast_weights
from zvisiongenerator.backends.mac.moss_lora import apply_fused_lokr, moss_lora_mapping, needs_preparing, prepare_lora
from zvisiongenerator.core import JobPrompt
from zvisiongenerator.core.moss import MossSampler, img2img_run_steps
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.safetensors_stream import read_header

__all__ = [
    "MossModel",
    "encode_prompts",
    "ensure_denoiser",
    "generate",
    "job_texts",
    "link_saved_copy",
    "load",
    "save",
]

_TEXT_ENCODER = ("text_encoder",)
_DENOISER = ("transformer", "vae")
_TOKENIZER = "qwen3_5"
# Above this many output pixels the VAE runs without the transformer: a start image is encoded in tiles before the
# transformer loads, and the transformer is dropped before the decode and loaded again for the next image. The VAE's own
# peak grows with the image (decode on q8: 1.2 GiB above the loop's at 1 MP, 5.6 GiB at 4 MP; a whole-image encode at
# 8 MP asks for 62 GB), while a reload costs about 2 s; at 2 MP and below the memory saved never changes what fits.
_DROP_TRANSFORMER_PIXELS = 2 * 1024 * 1024
# The folders mflux's ModelSaver writes for MOSS-Image: the three components and the tokenizer.
_SAVED_FOLDERS = ("text_encoder", "transformer", "vae", "tokenizer")


@dataclass
class MossModel:
    """A MOSS-Image model whose two stages load on demand: the text encoder, then the denoiser.

    Attributes:
        path: The model folder (a built ``moss`` / ``moss-sft`` folder or a stored copy of one).
        quantize: The level the stages are quantized to at load; ``None`` keeps a stored copy's own level.
        lora_paths: The LoRA files the denoiser loads with, Turbo's first.
        lora_scales: The scale of each LoRA.
        upcast: The components cast to float32 as they load.
        tokenizers: mflux's tokenizers, by name.
        callbacks: mflux's callback registry, which every generation reports its steps to.
        prompt_cache: Text embeddings by prompt.
        text_encoder: The loaded text encoder, or ``None``.
        transformer: The loaded transformer, or ``None``.
        vae: The loaded VAE, or ``None``.
    """

    path: str
    quantize: int | None
    lora_paths: list[str]
    lora_scales: list[float]
    upcast: tuple[str, ...]
    tokenizers: dict[str, Any]
    callbacks: Any
    prompt_cache: dict[str, Any] = field(default_factory=dict)
    text_encoder: Any | None = None
    transformer: Any | None = None
    vae: Any | None = None


@dataclass
class _RunConfig:
    """The run settings mflux callbacks read (``num_inference_steps``, ``init_time_step``, ``scheduler`` and the size)."""

    width: int
    height: int
    num_inference_steps: int
    init_time_step: int
    time_steps: range
    scheduler: Any = None


def load(
    path: str,
    quantize: int | None,
    *,
    lora_paths: Sequence[str] = (),
    lora_scales: Sequence[float] = (),
    upcast: tuple[str, ...] = (),
) -> MossModel:
    """Load the tokenizer and the text encoder stage of the MOSS-Image model at *path*.

    The denoiser loads once the prompts are encoded (:func:`ensure_denoiser`), so the two never share memory.

    Args:
        path: The model folder or a stored copy.
        quantize: The quantize level for both stages, or ``None``.
        lora_paths: LoRA files the denoiser applies, unmerged.
        lora_scales: The scale of each LoRA.
        upcast: The components cast to float32 as they load.
    """
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

    model = MossModel(
        path=path,
        quantize=quantize,
        lora_paths=list(lora_paths),
        lora_scales=list(lora_scales),
        upcast=upcast,
        tokenizers=TokenizerLoader.load_all(definitions=MOSSImageWeightDefinition.get_tokenizers(), model_path=path),
        callbacks=CallbackRegistry(),
    )
    _load_stage(model, _TEXT_ENCODER)
    return model


def job_texts(prompts: Sequence[JobPrompt], *, turbo: bool) -> tuple[str, ...]:
    """Return the distinct texts the text encoder sees for *prompts*, in first-seen order.

    SFT also encodes each negative prompt and the empty one, its default negative under guidance; Turbo has no negative pass.
    """
    texts = [job_prompt.prompt for job_prompt in prompts]
    if not turbo:
        texts += [job_prompt.negative_prompt or "" for job_prompt in prompts]
        texts.append("")
    return tuple(dict.fromkeys(texts))


def encode_prompts(model: MossModel, texts: Sequence[str], cancelled: Callable[[], bool] | None = None) -> bool:
    """Encode the *texts* not yet in the prompt cache, then free the text encoder; return ``False`` when cancelled.

    The text encoder loads again when it is needed and was freed; on a cancel it stays loaded.
    """
    from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_prompt_encoder import MOSSImagePromptEncoder

    import mlx.core as mx

    missing = [text for text in dict.fromkeys(texts) if text not in model.prompt_cache]
    if not missing:
        return True
    if model.text_encoder is None:
        # Keep the stages apart: the transformer goes first and loads again for the next image.
        model.transformer = None
        release_memory()
        _load_stage(model, _TEXT_ENCODER)
    print(f"Encoding {len(missing)} text(s)...", flush=True)
    for text in missing:
        if cancelled is not None and cancelled():
            return False
        embeds = MOSSImagePromptEncoder.encode(text, model.tokenizers[_TOKENIZER], model.text_encoder)
        mx.eval(embeds)
        model.prompt_cache[text] = embeds
    model.text_encoder = None
    release_memory()
    return True


def ensure_denoiser(model: MossModel) -> None:
    """Load the transformer (with the LoRAs unmerged) and the VAE, whichever is not loaded."""
    missing = tuple(name for name in _DENOISER if getattr(model, name) is None)
    if missing:
        _load_stage(model, missing)


def _apply_loras(transformer: Any, paths: Sequence[str], scales: Sequence[float]) -> None:
    """Apply the LoRAs to *transformer* unmerged: through mflux with :func:`moss_lora_mapping`, then any fused ``qkv`` LoKr.

    mflux reads LoRAs only from files, so a file :func:`prepare_lora` has to sort is handed over as a filtered copy, with
    the file's metadata, in a temporary folder that is removed once mflux has read it.
    """
    import tempfile

    import mlx.core as mx
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader

    fused: list[tuple[dict[int, dict[str, Any]], float]] = []
    readable_paths: list[str] = []
    readable_scales: list[float] = []
    with tempfile.TemporaryDirectory(prefix="ziv-moss-lora-") as folder:
        for index, (path, scale) in enumerate(zip(paths, scales, strict=True)):
            _length, header = read_header(Path(path))
            if not needs_preparing([name for name in header if name != "__metadata__"]):
                readable_paths.append(path)
                readable_scales.append(scale)
                continue
            tensors, metadata = mx.load(path, return_metadata=True)
            prepared = prepare_lora(dict(tensors.items()), Path(path).name)
            fused.append((prepared.fused_lokr, scale))
            if prepared.rest:
                copy = str(Path(folder) / f"{index}-{Path(path).name}")
                mx.save_safetensors(copy, prepared.rest, metadata=metadata)
                readable_paths.append(copy)
                readable_scales.append(scale)
        if readable_paths:
            LoRALoader.load_and_apply_lora(lora_mapping=moss_lora_mapping(), transformer=transformer, lora_paths=readable_paths, lora_scales=readable_scales, bake_lora=False)
    for layers, scale in fused:
        apply_fused_lokr(transformer, layers, scale)


def generate(
    model: MossModel,
    prompt: str,
    *,
    width: int,
    height: int,
    seed: int,
    sampler: MossSampler,
    image: Image.Image | None = None,
    strength: float = 1.0,
) -> Image.Image:
    """Generate one image: from noise, or from *image* at *strength* (the share of the schedule that runs).

    Each step is reported to ``model.callbacks``; a callback stops the run by raising mflux's
    ``StopImageGenerationException``, which propagates. Above :data:`_DROP_TRANSFORMER_PIXELS` the transformer is
    freed while the VAE encodes a start image and while it decodes, and loads again when it is needed.

    Args:
        model: A loaded model.
        prompt: The prompt.
        width: Image width, a multiple of 16.
        height: Image height, a multiple of 16.
        seed: The noise seed.
        sampler: Steps, guidance, negative prompt and skipped layers.
        image: A start image, or ``None`` for text-to-image.
        strength: How much of the schedule an image-to-image run denoises over, from 0 to 1.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator
    from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler
    from mflux.utils.image_util import ImageUtil

    import mlx.core as mx

    MOSSImageLatentCreator.check_size(height=height, width=width)
    negative = sampler.negative_prompt if sampler.do_cfg else None
    encode_prompts(model, [prompt] if negative is None else [prompt, negative])
    large = width * height > _DROP_TRANSFORMER_PIXELS
    clean = None
    if image is not None:
        if large:
            model.transformer = None
            release_memory()
        if model.vae is None:
            _load_stage(model, ("vae",))
        clean = _encode_image(model.vae, image, width, height, tiled=large)
        mx.eval(clean)
    ensure_denoiser(model)
    start = 0 if image is None else sampler.steps - img2img_run_steps(sampler.steps, strength)
    run = _RunConfig(width=width, height=height, num_inference_steps=sampler.steps, init_time_step=start, time_steps=range(start, sampler.steps))
    run.scheduler = MOSSImageScheduler(run)
    sigmas = run.scheduler.sigmas
    timesteps = run.scheduler.timesteps.tolist()

    latents = MOSSImageLatentCreator.create_noise(seed=seed, height=height, width=width)
    if clean is not None:
        sigma = sigmas[start]
        latents = (1.0 - sigma) * clean + sigma * latents
    positive = model.prompt_cache[prompt]
    uncond = model.prompt_cache[negative] if negative is not None else None
    skip_layers = frozenset(sampler.skip_layers)

    context = model.callbacks.start(seed=seed, prompt=prompt, config=run)
    context.before_loop(latents)
    for t in run.time_steps:
        noise = _predict(model.transformer, latents, timesteps[t], positive, uncond, sampler.guidance, skip_layers)
        latents = run.scheduler.step(noise=noise, timestep=t, latents=latents)
        context.in_loop(t, latents)
        mx.eval(latents)
    context.after_loop(latents)
    if large:
        model.transformer = None
        release_memory()
    decoded = model.vae.decode_packed_latents(latents.astype(ModelConfig.precision))
    return ImageUtil.to_pil(decoded)


def save(model: MossModel, path: str) -> None:
    """Write *model*'s weights at its quantize level to *path* in mflux's format, one stage at a time, without LoRAs.

    The loaded text encoder is written as it is (unless it was upcast); the denoiser is loaded for the write without
    LoRAs. Each stage is freed after its write, so the model is left with nothing loaded.

    Args:
        model: A model loaded at a quantize level.
        path: Destination directory.
    """
    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

    model.transformer = model.vae = None

    def write_and_free() -> None:
        ModelSaver.save_model(model=model, bits=model.quantize, base_path=path, weight_definition=MOSSImageWeightDefinition)
        model.text_encoder = model.transformer = model.vae = None
        release_memory()

    if model.text_encoder is None or "text_encoder" in model.upcast:
        model.text_encoder = None
        _load_stage(model, _TEXT_ENCODER, for_saving=True)
    write_and_free()
    _load_stage(model, _DENOISER, for_saving=True)
    write_and_free()


def link_saved_copy(shared: Path, path: Path) -> None:
    """Hard-link the saved folders of the stored copy *shared* into *path*: the other alias's copy holds the same weights."""
    for name in _SAVED_FOLDERS:
        link_tree(shared / name, path / name)


def _stage_definition(names: tuple[str, ...]) -> Any:
    """Return mflux's MOSS-Image weight definition limited to the components *names*."""
    from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

    components = [component for component in MOSSImageWeightDefinition.get_components() if component.name in names]
    return type("MossStageDefinition", (MOSSImageWeightDefinition,), {"get_components": staticmethod(lambda: components)})


def _load_stage(model: MossModel, names: tuple[str, ...], *, for_saving: bool = False) -> None:
    """Load the components *names* of *model* from its path and set them on it once every weight is evaluated.

    The transformer gets the LoRAs and the components in ``model.upcast`` are cast to float32, unless the stage is loaded
    for a stored copy, which holds the weights as quantized and nothing else.
    """
    parts = load_resident(lambda: _build_stage(model, names, for_saving=for_saving))
    for name in names:
        setattr(model, name, getattr(parts, name))


def _build_stage(model: MossModel, names: tuple[str, ...], *, for_saving: bool) -> Any:
    """Return a module holding the components *names* with their weights applied (see :func:`_load_stage`)."""
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mlx import nn

    definition = _stage_definition(names)
    weights = WeightLoader.load(weight_definition=definition, model_path=model.path)
    parts = nn.Module()
    for name in names:
        setattr(parts, name, _new_component(name))
    WeightApplier.apply_and_quantize(weights=weights, quantize_arg=model.quantize, weight_definition=definition, models={name: getattr(parts, name) for name in names})
    if not for_saving:
        if "transformer" in names and model.lora_paths:
            _apply_loras(parts.transformer, model.lora_paths, model.lora_scales)
        upcast_weights(parts, [name for name in names if name in model.upcast])
    return parts


def _new_component(name: str) -> Any:
    """Return an empty MOSS-Image component: the Qwen3.5 text encoder, the transformer, or the FLUX.2 VAE (small decoder)."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common_models.qwen3_5 import Qwen35TextModel
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
    from mflux.models.moss_image.moss_image_initializer import MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS

    config = ModelConfig.moss_image_sft()
    if name == "text_encoder":
        return Qwen35TextModel(**config.text_encoder_overrides)
    if name == "transformer":
        return MOSSImageTransformer(**config.transformer_overrides)
    vae = Flux2VAE(decoder_block_out_channels=MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS)
    # Decode in strips: the whole-image decode peaks about 11 GiB above the weights at 1024x1024, the strips about 1 GiB,
    # for the same image (PSNR 59.6 dB) and half a second more.
    vae.decode_mode = Flux2VAE.DECODE_LOW_MEMORY
    return vae


def _predict(transformer: Any, latents: Any, timestep: float, positive: Any, negative: Any | None, guidance: float, skip_layers: frozenset[int]) -> Any:
    """Return the guided velocity: the conditional pass, combined with the unconditional one under guidance.

    The two passes run separately, as in mflux: their text lengths differ and the unconditional one skips layers.
    """
    import mlx.core as mx

    cond = transformer(latents, timestep, positive).astype(mx.float32)
    if negative is None:
        return cond
    uncond = transformer(latents, timestep, negative, skip_layers=skip_layers).astype(mx.float32)
    return uncond + guidance * (cond - uncond)


def _encode_image(vae: Any, image: Image.Image, width: int, height: int, *, tiled: bool) -> Any:
    """Return *image* at *width* x *height* as normalised, patchified FP32 latents (the layout the transformer denoises).

    With *tiled*, the encoder runs on overlapping 512-pixel tiles: its attention over the whole image does not fit memory
    at upscale sizes.
    """
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.utils.image_util import ImageUtil

    import mlx.core as mx

    resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    pixels = ImageUtil.to_array(resized).astype(mx.float32)
    encoded = VAEUtil.encode(vae=vae, image=pixels, tiling_config=TilingConfig(vae_encode_tiled=True) if tiled else None)
    if encoded.ndim == 5:  # the tiled encode keeps a frame axis
        encoded = encoded[:, :, 0]
    packed = Flux2LatentCreator.patchify_latents(encoded).astype(mx.float32)
    mean = vae.bn.running_mean.astype(mx.float32).reshape(1, -1, 1, 1)
    std = mx.sqrt(vae.bn.running_var.astype(mx.float32).reshape(1, -1, 1, 1) + vae.bn.eps)
    return (packed - mean) / std
