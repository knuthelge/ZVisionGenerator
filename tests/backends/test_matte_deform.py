"""Tests for the grid_sample stand-in of BiRefNet's deformable convolution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from zvisiongenerator.backends.matte.deform_conv import GridSampleDeformableConv2d, deform_conv2d_gs, replace_deformable_convs


def _ramp() -> torch.Tensor:
    """Return a 1 x 1 x 3 x 3 input holding 1..9."""
    return torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)


def _kernel() -> torch.Tensor:
    """Return a 1 x 1 x 3 x 3 kernel with distinct taps."""
    return torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])


class TestDeformConv2dGs:
    def test_zero_offsets_and_unit_modulator_equal_a_plain_convolution(self):
        x, weight = _ramp(), _kernel()
        offset, mask = torch.zeros(1, 18, 3, 3), torch.ones(1, 9, 3, 3)

        got = deform_conv2d_gs(x, offset, mask, weight, None, stride=(1, 1), padding=(1, 1))

        assert torch.allclose(got, F.conv2d(x, weight, padding=1))

    def test_a_column_offset_of_one_shifts_the_input_by_one_with_zero_fill(self):
        x = _ramp()
        weight = torch.zeros(1, 1, 1, 1) + 1.0
        offset = torch.zeros(1, 2, 3, 3)
        offset[:, 1] = 1.0  # tap 0: column shift +1

        got = deform_conv2d_gs(x, offset, torch.ones(1, 1, 3, 3), weight, None, stride=(1, 1), padding=(0, 0))

        expected = torch.tensor([[[[2.0, 3.0, 0.0], [5.0, 6.0, 0.0], [8.0, 9.0, 0.0]]]])
        assert torch.allclose(got, expected)

    def test_a_modulator_of_one_half_halves_the_output(self):
        x, weight = _ramp(), _kernel()
        offset = torch.zeros(1, 18, 3, 3)

        full = deform_conv2d_gs(x, offset, torch.ones(1, 9, 3, 3), weight, None, stride=(1, 1), padding=(1, 1))
        half = deform_conv2d_gs(x, offset, torch.full((1, 9, 3, 3), 0.5), weight, None, stride=(1, 1), padding=(1, 1))

        assert torch.allclose(half, full / 2)

    def test_the_bias_is_added_per_output_channel(self):
        x = _ramp()
        weight = torch.zeros(2, 1, 1, 1)
        bias = torch.tensor([1.0, -2.0])

        got = deform_conv2d_gs(x, torch.zeros(1, 2, 3, 3), torch.ones(1, 1, 3, 3), weight, bias, stride=(1, 1), padding=(0, 0))

        assert torch.all(got[:, 0] == 1.0) and torch.all(got[:, 1] == -2.0)

    @pytest.mark.parametrize(("kernel", "stride", "padding"), [(3, 1, 1), (7, 1, 3), (1, 1, 0), (3, 2, 1)])
    def test_matches_torchvision_at_the_birefnet_kernel_shapes(self, kernel, stride, padding):
        deform_conv2d = pytest.importorskip("torchvision.ops").deform_conv2d
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(2, 16, 37, 41, generator=generator)
        weight = torch.randn(8, 16, kernel, kernel, generator=generator)
        bias = torch.randn(8, generator=generator)
        out_h = (37 + 2 * padding - kernel) // stride + 1
        out_w = (41 + 2 * padding - kernel) // stride + 1
        offset = torch.randn(2, 2 * kernel * kernel, out_h, out_w, generator=generator) * 3
        mask = torch.rand(2, kernel * kernel, out_h, out_w, generator=generator) * 2

        expected = deform_conv2d(x, offset, weight, bias, stride=(stride, stride), padding=(padding, padding), mask=mask)
        got = deform_conv2d_gs(x, offset, mask, weight, bias, stride=(stride, stride), padding=(padding, padding))

        assert (got - expected).abs().max() <= 1e-4 * expected.abs().max()


class _FakeDeformable(nn.Module):
    """Mimic the upstream module's attributes without torchvision."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.stride = (1, 1)
        self.padding = 1
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)
        self.modulator_conv = nn.Conv2d(channels, 9, 3, padding=1)
        self.regular_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)


_FakeDeformable.__name__ = "DeformableConv2d"


class _Tree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top = _FakeDeformable()
        self.block = nn.Sequential(nn.ReLU(), _FakeDeformable())


class TestReplaceDeformableConvs:
    def test_every_instance_is_replaced_and_counted_and_the_forward_still_runs(self):
        tree = _Tree()

        count = replace_deformable_convs(tree, GridSampleDeformableConv2d)

        assert count == 2
        assert isinstance(tree.top, GridSampleDeformableConv2d)
        assert isinstance(tree.block[1], GridSampleDeformableConv2d)
        assert tree.top(torch.randn(1, 4, 6, 6)).shape == (1, 4, 6, 6)

    def test_a_model_without_deformable_convs_gives_zero(self):
        assert replace_deformable_convs(nn.Sequential(nn.Conv2d(1, 1, 1)), GridSampleDeformableConv2d) == 0

    def test_state_dict_keys_and_values_are_unchanged(self):
        tree = _Tree()
        before = {key: value.clone() for key, value in tree.state_dict().items()}

        replace_deformable_convs(tree, GridSampleDeformableConv2d)

        assert list(tree.state_dict()) == list(before)
        tree.load_state_dict(before, strict=True)

    def test_the_vendored_module_is_swapped_without_touching_its_state_dict(self):
        pytest.importorskip("torchvision")
        from birefnet.birefnet import DeformableConv2d

        parent = nn.Sequential(DeformableConv2d(8, 8, 3, 1, 1))
        original = {key: value.clone() for key, value in parent.state_dict().items()}
        x = torch.randn(1, 8, 9, 9)
        expected = parent(x)

        count = replace_deformable_convs(parent, GridSampleDeformableConv2d)

        assert count == 1
        assert list(parent.state_dict()) == list(original)
        parent.load_state_dict(original, strict=True)
        assert torch.allclose(parent(x), expected, atol=1e-4)
