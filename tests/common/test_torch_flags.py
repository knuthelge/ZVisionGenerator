"""The torch flag snapshot: capturing and restoring every flag it names, including a state PyTorch calls mixed."""

from __future__ import annotations

import pytest
import torch

from tests._torch_flags import current_flags, set_app_flags, set_kit_flags
from zvisiongenerator.utils.torch_flags import capture_torch_flags, restore_torch_flags


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


def test_the_kit_and_app_states_differ(torch_flags_guard):
    set_app_flags()
    app = capture_torch_flags()
    set_kit_flags()

    assert capture_torch_flags() != app


def test_a_mixed_legacy_and_new_state_is_restored_exactly(torch_flags_guard):
    set_app_flags()
    torch.backends.cuda.matmul.allow_tf32 = False  # the legacy flag after the new call: get_float32_matmul_precision() now raises
    with pytest.raises(RuntimeError):
        torch.get_float32_matmul_precision()
    mixed = capture_torch_flags()
    assert mixed.float32_matmul_precision is None
    set_kit_flags()

    restore_torch_flags(mixed)

    assert current_flags() == mixed
    with pytest.raises(RuntimeError):
        torch.get_float32_matmul_precision()
