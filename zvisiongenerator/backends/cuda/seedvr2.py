"""Windows/Linux SeedVR2 upscaler using the vendored PyTorch core (``packages/seedvr2_torch``)."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from zvisiongenerator.backends.cuda.driver import require_cuda
from zvisiongenerator.backends.cuda.memory import configure_allocator
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, SEEDVR2_MODELS
from zvisiongenerator.utils.upscale import seedvr2_input_image

if TYPE_CHECKING:
    import numpy as np
    import torch

# Every block of the DiT streams from system memory: on a 10 GB GPU this costs almost no time, and it leaves the
# whole GPU to the activations, which set the size limit. The core caps the count at the model's block total.
_BLOCKS_TO_SWAP = 36
# The core's default VAE tiles, which hold the encode and decode of one frame under about 5.3 GB at any output size.
_TILE = (1024, 1024)
_TILE_OVERLAP = (128, 128)
# A video batch is up to 25 frames, so the VAE's activations per pixel are many times an image's. Video frames go through
# the VAE in 512-pixel tiles at every size, which holds the card under its measured peak whatever the output size.
_VIDEO_TILE = (512, 512)
# A video goes through the core in chunks of this many batches, so the frames held in system memory stay flat for any length.
_BATCHES_PER_CHUNK = 2
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

    def upscale_video(
        self,
        model: TorchSeedVR2Model,
        frames: Iterable[np.ndarray],
        *,
        width: int,
        height: int,
        total_frames: int,
        seed: int,
        softness: float,
        batch_size: int,
        progress: Callable[[int, int], None] | None = None,
        interrupt: Callable[[], None] | None = None,
    ) -> Iterator[np.ndarray]:
        """Return an iterator over *frames* (HxWx3 uint8) enlarged to *width* × *height*, pre-blurred by *softness* (0-1).

        The frames stream: they are read two batches at a time, each chunk is prepared at the target size as
        :meth:`upscale_image` does and goes through the core, and its upscaled frames are yielded in order, so
        system memory does not grow with the clip. The runner is built once and the DiT and VAE stay loaded
        between chunks; they are freed when the iterator ends, raises or is closed. The core splits a chunk into
        temporal batches of *batch_size* and uses the same seed for each.

        *progress* gets ``(batches_done, batches_total)`` after each decoded batch; the total starts as the batches
        *total_frames* needs and grows if the stream turns out longer. *interrupt* runs at each core step and
        stops the upscale by raising.

        Raises:
            ValueError: If *batch_size* is below 1.
            RuntimeError: If the GPU runs out of memory, naming the output size.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        return _stream_upscaled_frames(
            model, frames, width=width, height=height, total_frames=total_frames, seed=_core_seed(seed), softness=softness, batch_size=batch_size, progress=progress, interrupt=interrupt
        )


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
    import numpy as np
    import torch

    frames = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0)[None]
    runner, ctx, debug = _prepare_runner(model, _TILE)
    pixels = _run_core(runner, ctx, debug, frames, seed=seed, resolution=min(source.size), batch_size=1, cache_model=False)
    return Image.fromarray(_to_uint8_frames(pixels)[0])


def _video_batch_tensor(frames: list[np.ndarray], width: int, height: int, softness: float) -> torch.Tensor:
    """Return the HxWx3 uint8 *frames*, each prepared at *width* × *height*, as a float [T, H, W, 3] tensor in [0, 1].

    The frames are written into one array that is then scaled in place, so no second float copy of the batch is made.
    """
    import numpy as np
    import torch

    batch = np.empty((len(frames), height, width, 3), dtype=np.float32)
    for index, frame in enumerate(frames):
        batch[index] = np.asarray(seedvr2_input_image(Image.fromarray(frame), width, height, softness))
    return torch.from_numpy(batch).div_(255.0)


def _to_uint8_frames(pixels: torch.Tensor) -> list[np.ndarray]:
    """Return the float [T, H, W, 3] *pixels* in [0, 1] as a list of HxWx3 uint8 arrays (views of one array).

    A float32 *pixels* is clamped and scaled in place, so it holds the result of the call and should not be used again.
    """
    import torch

    values = pixels.float().clamp_(0, 1).mul_(255).round_()
    return list(values.to(torch.uint8).cpu().numpy())


def _prepare_runner(model: TorchSeedVR2Model, tile: tuple[int, int]) -> tuple[Any, dict[str, Any], Any]:
    """Return the core's runner, its generation context and the debug logger for *model*.

    The VAE encodes and decodes frames larger than *tile* (height, width) in overlapping tiles of that size. The
    models themselves are built by the phases, not here. The text embeddings are left unset: the upscale phase loads them
    for each run and frees them again, so a context shared between chunks must not hold them.
    """
    configure_allocator()
    from seedvr2_torch.src.core.generation_utils import prepare_runner, setup_generation_context
    from seedvr2_torch.src.utils.debug import Debug

    debug = Debug(enabled=False)
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
        encode_tile_size=tile,
        encode_tile_overlap=_TILE_OVERLAP,
        decode_tiled=True,
        decode_tile_size=tile,
        decode_tile_overlap=_TILE_OVERLAP,
        attention_mode="sdpa",
    )
    ctx["cache_context"] = cache_context
    return runner, ctx, debug


def _run_core(
    runner: Any,
    base_ctx: dict[str, Any],
    debug: Any,
    frames: torch.Tensor,
    *,
    seed: int,
    resolution: int,
    batch_size: int,
    cache_model: bool,
    on_decoded: Callable[[int, int, int, str], None] | None = None,
) -> torch.Tensor:
    """Run the core's encode, upscale, decode and colour correction on the float [T, H, W, 3] *frames* with a fresh copy of *base_ctx*; return the result.

    The core splits *frames* into batches of *batch_size*. With *cache_model* the DiT and VAE stay loaded for the next
    run; otherwise each is freed after its phase. *on_decoded* is called after each decoded batch.
    """
    from seedvr2_torch.src.core.generation_phases import decode_all_batches, encode_all_batches, postprocess_all_batches, upscale_all_batches
    from seedvr2_torch.src.core.generation_utils import compute_generation_info

    ctx = dict(base_ctx)
    frames, _info = compute_generation_info(ctx=ctx, images=frames, resolution=resolution, max_resolution=0, batch_size=batch_size, seed=seed, debug=debug)
    ctx = encode_all_batches(
        runner, ctx=ctx, images=frames, debug=debug, batch_size=batch_size, seed=seed, resolution=resolution, max_resolution=0, input_noise_scale=0.0, color_correction=_COLOR_CORRECTION
    )
    ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, seed=seed, latent_noise_scale=0.0, cache_model=cache_model)
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=on_decoded, cache_model=cache_model)
    ctx = postprocess_all_batches(ctx=ctx, debug=debug, color_correction=_COLOR_CORRECTION, batch_size=batch_size)
    return ctx["final_video"]


def _stream_upscaled_frames(
    model: TorchSeedVR2Model,
    frames: Iterable[np.ndarray],
    *,
    width: int,
    height: int,
    total_frames: int,
    seed: int,
    softness: float,
    batch_size: int,
    progress: Callable[[int, int], None] | None,
    interrupt: Callable[[], None] | None,
) -> Iterator[np.ndarray]:
    """Yield the upscaled frames of *frames*, one chunk of whole batches through the core at a time (see :meth:`TorchSeedVR2Upscaler.upscale_video`)."""
    import torch

    expected_batches = math.ceil(total_frames / batch_size)
    batches_done = 0
    runner, base_ctx, debug = _prepare_runner(model, _VIDEO_TILE)
    base_ctx["interrupt_fn"] = interrupt
    try:
        for chunk in itertools.batched(frames, _BATCHES_PER_CHUNK * batch_size):
            chunk_batches = 0

            def on_decoded(current: int, _total: int, _frames: int, _phase: str) -> None:
                nonlocal chunk_batches
                chunk_batches = current
                if progress is not None:
                    progress(batches_done + current, max(expected_batches, batches_done + current))

            try:
                pixels = _run_core(
                    runner,
                    base_ctx,
                    debug,
                    _video_batch_tensor(list(chunk), width, height, softness),
                    seed=seed,
                    resolution=min(width, height),
                    batch_size=batch_size,
                    cache_model=True,
                    on_decoded=on_decoded,
                )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"Not enough GPU memory to upscale to {width}×{height}. Free GPU memory, or upscale a smaller video.") from error
            batches_done += chunk_batches
            yield from _to_uint8_frames(pixels)
            del pixels
    finally:
        from seedvr2_torch.src.optimization.memory_manager import complete_cleanup

        complete_cleanup(runner, debug, dit_cache=False, vae_cache=False)
