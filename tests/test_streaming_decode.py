"""Tests for decoding LTX-2 latents a few frames at a time, against diffusers' own frame-wise decode."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video  # noqa: E402

from zvisiongenerator.backends.streaming_decode import decode_frames  # noqa: E402


class _FakeVae:
    """A tiny VAE with the LTX-2 tiling attributes and a deterministic decoder (8x in time, 2x in space)."""

    temporal_compression_ratio = 8
    spatial_compression_ratio = 2
    tile_sample_min_num_frames = 16
    tile_sample_stride_num_frames = 8
    tile_sample_min_height = 8
    tile_sample_min_width = 8
    tile_sample_stride_height = 6
    tile_sample_stride_width = 6
    dtype = torch.float32

    def __init__(self, use_tiling: bool):
        self.use_tiling = use_tiling
        self.decoded_spans: list[int] = []

    def decoder(self, tile, temb=None, causal=None):
        self.decoded_spans.append(tile.shape[2])
        frames = (tile.shape[2] - 1) * self.temporal_compression_ratio + 1
        signal = tile.mean(dim=1, keepdim=True)
        signal = torch.nn.functional.interpolate(signal, size=(frames, tile.shape[3] * 2, tile.shape[4] * 2), mode="trilinear", align_corners=True)
        return torch.tanh(signal.expand(-1, 3, -1, -1, -1).contiguous())

    # diffusers' own blending and spatial tiling, run on this fake
    blend_t = AutoencoderKLLTX2Video.blend_t
    blend_v = AutoencoderKLLTX2Video.blend_v
    blend_h = AutoencoderKLLTX2Video.blend_h
    tiled_decode = AutoencoderKLLTX2Video.tiled_decode


def _reference_frames(vae, latents):
    """Decode with diffusers' frame-wise decode and map to uint8 frames the way its image output does."""
    video = AutoencoderKLLTX2Video._temporal_tiled_decode(vae, latents, None, return_dict=False)[0]
    video = (video[0] * 0.5 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8)
    return list(video.permute(1, 2, 3, 0).numpy())


@pytest.mark.parametrize("latent_frames", [1, 2, 3, 5, 16, 31])
@pytest.mark.parametrize("use_tiling", [False, True])
def test_frames_equal_diffusers_frame_wise_decode(latent_frames, use_tiling):
    torch.manual_seed(latent_frames)
    latents = torch.randn(1, 4, latent_frames, 6, 6)

    frames = decode_frames(_FakeVae(use_tiling), latents)
    expected = _reference_frames(_FakeVae(use_tiling), latents)

    assert len(frames) == (latent_frames - 1) * 8 + 1 == len(expected)
    assert all((frame == reference).all() for frame, reference in zip(frames, expected, strict=True))
    assert frames[0].shape == (12, 12, 3) and frames[0].dtype.name == "uint8"


def test_each_tile_spans_a_few_latent_frames_however_long_the_clip():
    vae = _FakeVae(use_tiling=False)

    decode_frames(vae, torch.randn(1, 4, 31, 4, 4))

    assert max(vae.decoded_spans) == 3  # tile_sample_min_num_frames // 8 + 1, never the whole clip
