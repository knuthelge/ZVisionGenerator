"""Tests for streaming model blocks to the GPU, run on the meta device so they need no GPU."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda import block_streaming  # noqa: E402
from zvisiongenerator.backends.cuda.block_streaming import move_outside, offload_type, stream_blocks  # noqa: E402

_DEVICE = torch.device("meta")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(4, 4)
        self.transformer_blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(2))
        self.transformer_blocks[0].register_buffer("table", torch.ones(4))
        self.scale = torch.nn.Parameter(torch.ones(4))

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        return x * self.scale


@pytest.fixture(autouse=True)
def _no_pinning():
    # Pinning needs a GPU; the swap itself does not.
    with patch.object(block_streaming, "_pin_parameters"):
        yield


def _record_devices(block, seen):
    block.register_forward_pre_hook(lambda module, args: seen.append(module.weight.device))


class TestStreamBlocks:
    def test_a_block_runs_with_its_parameters_on_the_device_and_gets_its_cpu_ones_back(self):
        model = _Model()
        original = [(block.weight, block.bias, block.weight.detach().clone()) for block in model.transformer_blocks]
        stream_blocks(model.transformer_blocks, _DEVICE)
        seen: list[torch.device] = []
        for block in model.transformer_blocks:
            _record_devices(block, seen)  # registered after the streaming hook, so it sees the swapped parameters
        move_outside(model, model.transformer_blocks, _DEVICE)

        model(torch.ones(1, 4, device=_DEVICE))

        assert seen == [_DEVICE, _DEVICE]
        for block, (weight, bias, values) in zip(model.transformer_blocks, original, strict=True):
            assert block.weight is weight and block.bias is bias
            assert block.weight.device.type == "cpu"
            assert torch.equal(block.weight, values)

    def test_buffers_move_to_the_device_for_good(self):
        model = _Model()

        stream_blocks(model.transformer_blocks, _DEVICE)

        assert model.transformer_blocks[0].table.device == _DEVICE

    def test_the_returned_handles_remove_the_streaming(self):
        model = _Model()
        handles = stream_blocks(model.transformer_blocks, _DEVICE)
        for handle in handles:
            handle.remove()

        model(torch.ones(1, 4))  # all on the CPU again: no device mismatch

        assert len(handles) == 4


class TestMoveOutside:
    def test_everything_outside_the_blocks_moves_and_the_blocks_stay(self):
        model = _Model()

        move_outside(model, model.transformer_blocks, _DEVICE)

        assert model.proj_in.weight.device == _DEVICE
        assert model.scale.device == _DEVICE  # the model's own parameters too
        assert all(param.device.type == "cpu" for param in model.transformer_blocks.parameters())
        assert model.transformer_blocks[0].table.device.type == "cpu"


class TestOffloadType:
    def test_a_top_level_layer_list_streams_by_block(self):
        assert offload_type(_Model()) == "block_level"

    def test_nested_layers_stream_by_leaf(self):
        outer = torch.nn.Module()
        outer.model = _Model()
        outer.lm_head = torch.nn.Linear(4, 4)

        assert offload_type(outer) == "leaf_level"
