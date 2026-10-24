"""Tests for the ported H3 latent upscaler, with tiny networks and no real weights."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from zvisiongenerator.backends.cuda.video.h3.latent_upsampler import (  # noqa: E402
    LatentResizer3D,
    load_latent_upsampler,
    upscale_latents,
)

_TINY = partial(LatentResizer3D, channels=32, in_blocks=2, out_blocks=2)
_MEAN = [0.5 * (i % 3) for i in range(24)]
_STD = [1.0 + 0.25 * (i % 4) for i in range(24)]


class _RecordingNet(torch.nn.Module):
    """A stand-in network: records its input and returns a fixed function of it."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))  # gives the device and dtype
        self.seen: list[tuple[torch.Tensor, float, tuple[int, int, int]]] = []

    def forward(self, x, scale, size):
        self.seen.append((x.clone(), scale, size))
        return torch.nn.functional.interpolate(x, size=size, mode="nearest") * 2


def _save_tiny(path: Path, *, drop: str | None = None) -> dict[str, torch.Tensor]:
    state = {name: tensor.detach().clone() for name, tensor in _TINY().state_dict().items()}
    if drop is not None:
        del state[drop]
    safetensors_torch.save_file(state, str(path))
    return state


def test_the_pinned_architecture_has_the_published_parameter_count() -> None:
    with torch.device("meta"):
        net = LatentResizer3D()
    assert sum(parameter.numel() for parameter in net.parameters()) == 345_280_216


def test_a_tiny_network_doubles_height_and_width_and_keeps_time() -> None:
    net = _TINY().eval()
    with torch.inference_mode():
        out = net(torch.randn(1, 24, 3, 4, 6), 2.0, (3, 8, 12))
    assert out.shape == (1, 24, 3, 8, 12)


def test_the_latents_are_normalised_again_before_the_network_and_reverted_after_it() -> None:
    net = _RecordingNet()
    latents = torch.randn(1, 24, 3, 4, 6)
    out = upscale_latents(net, latents, _MEAN, _STD)
    mean = torch.tensor(_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(_STD).view(1, -1, 1, 1, 1)
    ((seen, scale, size),) = net.seen
    assert torch.allclose(seen, (latents - mean) / std, atol=1e-6)
    assert scale == 2.0
    assert size == (3, 8, 12)
    expected = torch.nn.functional.interpolate((latents - mean) / std, size=size, mode="nearest") * 2 * std + mean
    assert torch.allclose(out, expected, atol=1e-5)


def test_the_result_is_float32_on_the_cpu() -> None:
    out = upscale_latents(_RecordingNet().to(torch.float16), torch.randn(1, 24, 2, 2, 2), _MEAN, _STD)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_a_file_missing_one_tensor_raises_naming_the_config_key(tmp_path: Path) -> None:
    path = tmp_path / "net.safetensors"
    _save_tiny(path, drop="conv_out.bias")
    with pytest.raises(RuntimeError, match=r"video_model_presets\.h3\.diffusers\.latent_upsampler"):
        load_latent_upsampler(path, "cpu", build=_TINY)


def test_a_file_with_the_wrong_shapes_raises_naming_the_config_key(tmp_path: Path) -> None:
    path = tmp_path / "net.safetensors"
    _save_tiny(path)
    with pytest.raises(RuntimeError, match=r"video_model_presets\.h3\.diffusers\.latent_upsampler"):
        load_latent_upsampler(path, "cpu", build=partial(LatentResizer3D, channels=64, in_blocks=2, out_blocks=2))


def test_the_state_dict_round_trips_and_loads_as_fp16_without_gradients(tmp_path: Path) -> None:
    path = tmp_path / "net.safetensors"
    state = _save_tiny(path)
    net = load_latent_upsampler(path, "cpu", build=_TINY)
    assert not net.training
    for name, parameter in net.named_parameters():
        assert parameter.dtype == torch.float16
        assert not parameter.requires_grad
        assert torch.equal(parameter, state[name].to(torch.float16))
