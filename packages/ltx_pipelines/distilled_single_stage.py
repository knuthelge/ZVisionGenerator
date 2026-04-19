from __future__ import annotations

from collections.abc import Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.types import ModalitySpec


class DistilledSingleStagePipeline:
    """Single-stage distilled video generation pipeline.

    Generates video at full target resolution in a single diffusion pass,
    mirroring the Mac TextToVideoPipeline behavior on CUDA.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path, gemma_root, self.dtype, self.device, registry=registry
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        sigmas: torch.Tensor = DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        sigmas = sigmas.to(dtype=torch.float32, device=self.device)
        conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=ModalitySpec(context=audio_context),
            streaming_prefetch_count=streaming_prefetch_count,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio
