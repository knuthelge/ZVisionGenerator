"""The torch flag snapshot: capturing and restoring every flag it names, including a state PyTorch calls mixed."""

from __future__ import annotations

import pytest
import torch

from tests._torch_flags import current_flags, set_app_flags, set_kit_flags
from zvisiongenerator.utils.torch_flags import capture_torch_flags, default_dtype, preserved_torch_flags, restore_torch_flags


def test_restoring_a_snapshot_brings_back_every_flag(torch_flags_guard):
    set_app_flags()
    snapshot = capture_torch_flags()
    set_kit_flags()
    assert current_flags() != snapshot

    restore_torch_flags(snapshot)

    assert current_flags() == snapshot
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_a_mixed_legacy_and_new_state_is_restored_exactly(torch_flags_guard):
    set_app_flags()
    torch.backends.cuda.matmul.allow_tf32 = False  # the legacy flag after the new call: get_float32_matmul_precision() now raises
    with pytest.raises(RuntimeError):
        torch.get_float32_matmul_precision()
    mixed = capture_torch_flags()
    assert mixed["float32_matmul_precision"] is None
    set_kit_flags()

    restore_torch_flags(mixed)

    assert current_flags() == mixed
    with pytest.raises(RuntimeError):
        torch.get_float32_matmul_precision()


def test_the_flags_come_back_after_a_block_that_changed_them_even_when_it_fails(torch_flags_guard):
    set_app_flags()
    before = current_flags()

    with pytest.raises(ValueError, match="boom"), preserved_torch_flags():
        set_kit_flags()
        raise ValueError("boom")

    assert current_flags() == before


def test_the_default_dtype_holds_for_the_block_only_and_comes_back_after_a_failure():
    previous = torch.get_default_dtype()

    with default_dtype(torch.float16):
        assert torch.nn.Linear(2, 2).weight.dtype == torch.float16
    assert torch.get_default_dtype() == previous
    with pytest.raises(ValueError, match="boom"), default_dtype(torch.bfloat16):
        raise ValueError("boom")
    assert torch.get_default_dtype() == previous
