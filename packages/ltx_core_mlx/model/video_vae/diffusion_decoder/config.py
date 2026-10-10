"""Configuration of the LTX-2.5 diffusion video decoder (``NADiffusionDecoder``).

The values live in the safetensors metadata of ``vae_decoder_av.safetensors`` (mlx-forge copies
upstream's ``config.vae`` verbatim); :data:`LTX_2_5_DIFFUSION_DECODER` mirrors them so the
decoder can be built without the file too. ``stage_kernels[4]`` is never read upstream: the
diffusion stage kernel comes from ``stage5_kernel`` only.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

Kernel = tuple[int, int, int]
Upsample = tuple[Kernel, int]  # (stride (t, h, w), channel reduction)


@dataclass(frozen=True)
class DiffusionDecoderConfig:
    """Architecture of the diffusion video decoder.

    Attributes:
        in_channels: Latent channels (128).
        out_channels: Pixel channels (3).
        patch_size: Spatial patch of the pixel grid at stage 5 (4).
        head_dim: Attention head size (64).
        stage_channels: Channels of stages 1-4 (det) and 5 (diffusion).
        stage_depths: Blocks per stage.
        stage_kernels: Neighborhood kernels (t, h, w) of the four det stages.
        upsamples: ``((stride_t, stride_h, stride_w), channel_reduction)`` after each det stage.
        stage5_kernel: Neighborhood kernel of the diffusion stage.
        t_embed_hidden: Hidden size of the timestep MLP (384).
        timestep_scale_multiplier: Multiplies ``t`` before the sinusoidal embedding (1000).
        t_freq_dim: Sinusoidal embedding size fed to the timestep MLP (256).
    """

    in_channels: int
    out_channels: int
    patch_size: int
    head_dim: int
    stage_channels: tuple[int, int, int, int, int]
    stage_depths: tuple[int, int, int, int, int]
    stage_kernels: tuple[Kernel, Kernel, Kernel, Kernel]
    upsamples: tuple[Upsample, Upsample, Upsample, Upsample]
    stage5_kernel: Kernel
    t_embed_hidden: int
    timestep_scale_multiplier: float
    t_freq_dim: int = 256

    def __post_init__(self) -> None:
        for i, ((_stride, reduction), c_in, c_out) in enumerate(
            zip(self.upsamples, self.stage_channels[:-1], self.stage_channels[1:], strict=True)
        ):
            if c_in % reduction or c_in // reduction != c_out:
                raise ValueError(
                    f"stage_channels[{i + 1}]={c_out} must equal stage_channels[{i}]={c_in} / reduction {reduction}"
                )
        for c in self.stage_channels:
            if c % self.head_dim:
                raise ValueError(f"stage channel {c} is not a multiple of head_dim {self.head_dim}")

    @property
    def num_det_stages(self) -> int:
        return 4

    @property
    def diff_channels(self) -> int:
        return self.stage_channels[-1]

    @property
    def diff_depth(self) -> int:
        return self.stage_depths[-1]

    @property
    def adaln_dim(self) -> int:
        return 7 * self.diff_channels

    def heads(self, channels: int) -> int:
        return channels // self.head_dim

    def cumulative_strides(self) -> list[Kernel]:
        """Stride (t, h, w) relative to the latent grid at the input of stages 0..4."""
        out: list[Kernel] = [(1, 1, 1)]
        for (st, sh, sw), _ in self.upsamples:
            t, h, w = out[-1]
            out.append((t * st, h * sh, w * sw))
        return out

    def min_latent_shape(self) -> Kernel:
        """Smallest latent (F, H, W) such that every stage grid is at least its kernel."""
        strides = self.cumulative_strides()
        kernels = [*self.stage_kernels, self.stage5_kernel]
        return tuple(  # type: ignore[return-value]
            max(math.ceil(k[a] / s[a]) for k, s in zip(kernels, strides, strict=True)) for a in range(3)
        )

    def ghost_pad_frames(self) -> int:
        """Trailing latent frames replicated before stage 1 (``(kt0 // 2) * 2``)."""
        return (self.stage_kernels[0][0] // 2) * 2

    @classmethod
    def from_safetensors_metadata(cls, path: str | Path) -> DiffusionDecoderConfig:
        """Read ``config.vae.decoder`` from a safetensors header written by mlx-forge."""
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        meta = header.get("__metadata__") or {}
        if "config" not in meta:
            raise ValueError(f"{path}: no 'config' entry in safetensors metadata")
        dec = json.loads(meta["config"]).get("vae", {}).get("decoder", {})
        if dec.get("_class_name") != "NADiffusionDecoder":
            raise ValueError(f"{path}: decoder class is {dec.get('_class_name')!r}, expected 'NADiffusionDecoder'")
        return cls(
            in_channels=int(dec["in_channels"]),
            out_channels=int(dec["out_channels"]),
            patch_size=int(dec["patch_size"]),
            head_dim=int(dec["head_dim"]),
            stage_channels=tuple(int(c) for c in dec["stage_channels"]),  # type: ignore[arg-type]
            stage_depths=tuple(int(d) for d in dec["stage_depths"]),  # type: ignore[arg-type]
            stage_kernels=tuple(tuple(int(v) for v in k) for k in dec["stage_kernels"][:4]),  # type: ignore[arg-type]
            upsamples=tuple((tuple(int(v) for v in s), int(r)) for s, r in dec["upsamples"]),  # type: ignore[arg-type]
            stage5_kernel=tuple(int(v) for v in dec["stage5_kernel"]),  # type: ignore[arg-type]
            t_embed_hidden=384,
            timestep_scale_multiplier=float(dec.get("timestep_scale_multiplier", 1000.0)),
        )


LTX_2_5_DIFFUSION_DECODER = DiffusionDecoderConfig(
    in_channels=128,
    out_channels=3,
    patch_size=4,
    head_dim=64,
    stage_channels=(2048, 1024, 512, 512, 256),
    stage_depths=(4, 6, 4, 2, 8),
    stage_kernels=((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
    upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
    stage5_kernel=(11, 11, 11),
    t_embed_hidden=384,
    timestep_scale_multiplier=1000.0,
)
