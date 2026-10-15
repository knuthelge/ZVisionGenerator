"""Tests for the MLX unified-memory helpers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call

import pytest

from zvisiongenerator.backends.mac.memory import BUFFER_CACHE_LIMIT_BYTES, apply_buffer_cache_policy, materialize_weights, memory_size_bytes, mlx_cache_limit


@pytest.fixture
def mx(monkeypatch):
    """Replace ``mlx.core`` with a mock whose ``set_cache_limit`` returns a sentinel previous limit."""
    fake_mx = MagicMock(name="mx")
    fake_mx.set_cache_limit.return_value = "previous"
    fake_mlx = MagicMock(core=fake_mx)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return fake_mx


class TestMlxCacheLimit:
    def test_sets_the_limit_then_restores_the_previous_one(self, mx):
        with mlx_cache_limit(0):
            assert mx.set_cache_limit.call_args_list == [call(0)]

        assert mx.set_cache_limit.call_args_list == [call(0), call("previous")]

    def test_restores_the_previous_limit_when_the_block_raises(self, mx):
        with pytest.raises(RuntimeError, match="boom"):
            with mlx_cache_limit(0):
                raise RuntimeError("boom")

        assert mx.set_cache_limit.call_args_list == [call(0), call("previous")]


class TestMemorySizeBytes:
    @pytest.mark.parametrize(
        ("info", "expected"),
        [
            ({"memory_size": 32 * 1024**3}, 32 * 1024**3),
            ({"memory_size": 0}, None),
            ({"memory_size": "32"}, None),
            ({}, None),
        ],
    )
    def test_reads_memory_size_from_the_device_info(self, mx, info, expected):
        mx.device_info.return_value = info

        assert memory_size_bytes() == expected


class TestBufferCachePolicy:
    def test_materialize_weights_evaluates_the_model_parameters(self, mx):
        model = MagicMock()

        materialize_weights(model)

        mx.eval.assert_called_once_with(model.parameters.return_value)

    def test_policy_caps_the_cache_then_clears_it(self, mx):
        apply_buffer_cache_policy()

        assert [c[0] for c in mx.method_calls] == ["set_cache_limit", "clear_cache"]
        mx.set_cache_limit.assert_called_once_with(BUFFER_CACHE_LIMIT_BYTES)

    def test_materializing_precedes_the_policy_when_both_run(self, mx):
        materialize_weights(MagicMock())
        apply_buffer_cache_policy()

        assert [c[0] for c in mx.method_calls if c[0] in {"eval", "set_cache_limit", "clear_cache"}] == ["eval", "set_cache_limit", "clear_cache"]

    def test_buffer_cache_limit_is_four_gib(self):
        assert BUFFER_CACHE_LIMIT_BYTES == 4 * 1024**3
