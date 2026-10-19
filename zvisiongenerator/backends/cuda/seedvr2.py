"""Windows/Linux SeedVR2 upscaler using the vendored PyTorch core (``packages/seedvr2_torch``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from zvisiongenerator.backends.cuda.driver import require_cuda
from zvisiongenerator.backends.cuda.memory import configure_allocator
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, SEEDVR2_MODELS
from zvisiongenerator.utils.upscale import seedvr2_input_image

# Every block of the DiT streams from system memory: on a 10 GB GPU this costs almost no time, and it leaves the
# whole GPU to the activations, which set the size limit. The core caps the count at the model's block total.
_BLOCKS_TO_SWAP = 36
# The core's default VAE tiles, which hold the encode and decode under about 5.3 GB at any output size.
_TILE = (1024, 1024)
_TILE_OVERLAP = (128, 128)
_COLOR_CORRECTION = "lab"
# The core seeds the VAE with seed + 1_000_000 through numpy, which accepts seeds up to 2**32 - 1.
_CORE_SEED_RANGE = 2**32 - 1_000_000
_DEVICE = "cuda:0"
_OFFLOAD = "cpu"


@dataclass(frozen=True)
class TorchSeedVR2Model:
    """The files of one SeedVR2 tier in its weights folder; the core builds the models from them per run."""

    model: str
    dit_file: str
    vae_file: str
    model_dir: str


class TorchSeedVR2Upscaler:
    """Vendored PyTorch SeedVR2 for CUDA — implements the SeedVR2Upscaler Protocol."""

    weights = FP8_WEIGHTS

    def load_model(self, model: str, *, quantize: int | None, model_path: str) -> TorchSeedVR2Model:
        """Download *model*'s FP8 transformer and its VAE into the folder *model_path*, where the core reads both.

        Files already in the folder are not downloaded again. They are already 8-bit, so *quantize* must be
        ``None``. The models are built for each upscale and freed between its phases, so nothing is held in
        memory here.

        Raises:
            ValueError: If *model* is not a known SeedVR2 variant or *quantize* is set.
            RuntimeError: If CUDA is not available, checked before anything is downloaded.
        """
        if model not in SEEDVR2_MODELS:
            raise ValueError(f"Unknown SeedVR2 model '{model}'. Supported: {', '.join(SEEDVR2_MODELS)}")
        if quantize is not None:
            raise ValueError("SeedVR2 on CUDA reads FP8 weights and does not quantize while loading.")
        _require_cuda()
        from huggingface_hub import hf_hub_download

        (dit_repo, dit_file), (vae_repo, vae_file) = self.weights.files[model]
        for repo, name in ((dit_repo, dit_file), (vae_repo, vae_file)):
            if not os.path.isfile(os.path.join(model_path, name)):
                hf_hub_download(repo, name, local_dir=model_path)
        return TorchSeedVR2Model(model=model, dit_file=dit_file, vae_file=vae_file, model_dir=model_path)

    def save_model(self, model: Any, path: Any) -> None:
        """Refuse: CUDA reads the downloaded FP8 files and keeps no stored copy."""
        raise RuntimeError("SeedVR2 on CUDA keeps no stored copy.")

    def upscale_image(self, model: TorchSeedVR2Model, image_path: str, *, width: int, height: int, seed: int, softness: float) -> Image.Image:
        """Return *image_path* enlarged to *width* × *height*, pre-blurred by *softness* (0-1).

        The image is prepared at the target size as on macOS, so the core's own resize leaves it unchanged, and
        its LAB colour correction matches mflux's. It is read as stored, without EXIF rotation, as the upscale
        stage sized it.
        """
        with Image.open(image_path) as opened:
            source = seedvr2_input_image(opened, width, height, softness)
        return _run(model, source, seed=_core_seed(seed))


def _require_cuda() -> None:
    """Raise when this PyTorch cannot use an NVIDIA GPU, with a driver hint where an old driver is the likely cause.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    configure_allocator()
    import torch

    require_cuda(torch, "The SeedVR2 upscale on Windows and Linux")


def _core_seed(seed: int) -> int:
    """Return *seed* folded into the range the core can use; seeds already in range are unchanged."""
    return seed % _CORE_SEED_RANGE


def _run(model: TorchSeedVR2Model, source: Image.Image, *, seed: int) -> Image.Image:
    """Run the core's encode, upscale, decode and colour correction on the prepared *source*."""
    configure_allocator()
    import numpy as np
    import torch
    from seedvr2_torch.src.core.generation_phases import decode_all_batches, encode_all_batches, postprocess_all_batches, upscale_all_batches
    from seedvr2_torch.src.core.generation_utils import compute_generation_info, load_text_embeddings, prepare_runner, setup_generation_context
    from seedvr2_torch.src.utils.constants import get_script_directory
    from seedvr2_torch.src.utils.debug import Debug

    debug = Debug(enabled=False)
    resolution = min(source.size)
    frames = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0)[None]
    ctx = setup_generation_context(dit_device=_DEVICE, vae_device=_DEVICE, dit_offload_device=_OFFLOAD, vae_offload_device=_OFFLOAD, tensor_offload_device=_OFFLOAD, debug=debug)
    runner, cache_context = prepare_runner(
        dit_model=model.dit_file,
        vae_model=model.vae_file,
        model_dir=model.model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=False,
        vae_cache=False,
        block_swap_config={"blocks_to_swap": _BLOCKS_TO_SWAP, "swap_io_components": False, "offload_device": _OFFLOAD},
        encode_tiled=True,
        encode_tile_size=_TILE,
        encode_tile_overlap=_TILE_OVERLAP,
        decode_tiled=True,
        decode_tile_size=_TILE,
        decode_tile_overlap=_TILE_OVERLAP,
        attention_mode="sdpa",
    )
    ctx["cache_context"] = cache_context
    ctx["text_embeds"] = load_text_embeddings(get_script_directory(), ctx["dit_device"], ctx["compute_dtype"], debug)
    frames, _info = compute_generation_info(ctx=ctx, images=frames, resolution=resolution, max_resolution=0, batch_size=1, seed=seed, debug=debug)
    ctx = encode_all_batches(runner, ctx=ctx, images=frames, debug=debug, batch_size=1, seed=seed, resolution=resolution, max_resolution=0, input_noise_scale=0.0, color_correction=_COLOR_CORRECTION)
    ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, seed=seed, latent_noise_scale=0.0, cache_model=False)
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, cache_model=False)
    ctx = postprocess_all_batches(ctx=ctx, debug=debug, color_correction=_COLOR_CORRECTION, batch_size=1)
    pixels = ctx["final_video"][0].float().clamp(0, 1).cpu().numpy()
    return Image.fromarray((pixels * 255).round().astype(np.uint8))
