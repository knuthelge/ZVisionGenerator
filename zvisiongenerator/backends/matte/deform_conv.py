"""Modulated deformable convolution built from ``grid_sample``, and the swap that puts it into BiRefNet.

torchvision's ``deform_conv2d`` has no MPS kernel and torchvision's CUDA kernel is a second code path to carry, so
BiRefNet's ``DeformableConv2d`` modules are replaced after construction by :class:`GridSampleDeformableConv2d`. It
computes with ``F.grid_sample`` on every platform and matches torchvision to about 1e-4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["GridSampleDeformableConv2d", "deform_conv2d_gs", "replace_deformable_convs"]

_UPSTREAM_CLASS_NAME = "DeformableConv2d"


def deform_conv2d_gs(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    """Return a modulated deformable convolution (DCNv2, one offset group, no dilation), one kernel tap at a time.

    Each tap samples the input bilinearly at its shifted position (zero outside the input) and scales the sample by
    the modulator. Offsets are laid out as torchvision's: ``offset[:, 2k]`` is the row shift and
    ``offset[:, 2k + 1]`` the column shift of tap ``k``, taps running row-major over the kernel.

    Args:
        x: Input of shape ``(N, C, H, W)``.
        offset: Tap shifts of shape ``(N, 2 * kh * kw, out_h, out_w)``.
        mask: Modulator of shape ``(N, kh * kw, out_h, out_w)``.
        weight: Kernel of shape ``(O, C, kh, kw)``.
        bias: Bias of shape ``(O,)``, or ``None``.
        stride: Row and column stride.
        padding: Row and column zero padding.

    Returns:
        The result of shape ``(N, O, out_h, out_w)``.
    """
    batch, _, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = offset.shape[-2:]
    rows = torch.arange(out_h, device=x.device, dtype=x.dtype) * stride[0] - padding[0]
    cols = torch.arange(out_w, device=x.device, dtype=x.dtype) * stride[1] - padding[1]
    base_y, base_x = torch.meshgrid(rows, cols, indexing="ij")
    result = x.new_zeros(batch, out_channels, out_h, out_w)
    for tap in range(kernel_h * kernel_w):
        i, j = divmod(tap, kernel_w)
        y = base_y + i + offset[:, 2 * tap]
        x_pos = base_x + j + offset[:, 2 * tap + 1]
        grid = torch.stack((x_pos / max(width - 1, 1) * 2 - 1, y / max(height - 1, 1) * 2 - 1), dim=-1)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled * mask[:, tap : tap + 1]
        result = result + torch.einsum("oc,nchw->nohw", weight[:, :, i, j], sampled)
    if bias is not None:
        result = result + bias.view(1, -1, 1, 1)
    return result


class GridSampleDeformableConv2d(nn.Module):
    """Stand-in for BiRefNet's ``DeformableConv2d`` that keeps its three convolutions and computes with ``grid_sample``.

    The submodules keep the upstream names (``offset_conv``, ``modulator_conv``, ``regular_conv``), so the
    ``state_dict`` keys do not change and the checkpoint loads with ``strict=True``.
    """

    def __init__(self, original: nn.Module) -> None:
        """Take over the convolutions, the stride (a tuple) and the padding (an int) of *original*."""
        super().__init__()
        self.offset_conv = original.offset_conv
        self.modulator_conv = original.modulator_conv
        self.regular_conv = original.regular_conv
        self.stride = original.stride
        self.padding = original.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deformable convolution of *x* with offsets and a ``2 * sigmoid`` modulator predicted from *x*."""
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        return deform_conv2d_gs(x, offset, modulator, self.regular_conv.weight, self.regular_conv.bias, stride=self.stride, padding=padding)


def replace_deformable_convs(model: nn.Module, cls: Callable[[nn.Module], nn.Module]) -> int:
    """Replace every upstream ``DeformableConv2d`` in *model* with ``cls(original)`` and return how many were replaced.

    Call it after the model is constructed and before its weights load. The upstream class is found by name, so the
    caller needs no torchvision import.
    """
    targets = [(name, module) for name, module in model.named_modules() if type(module).__name__ == _UPSTREAM_CLASS_NAME]
    for name, module in targets:
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, cls(module))
    return len(targets)
