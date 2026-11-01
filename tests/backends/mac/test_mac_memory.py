"""Tests for the MLX unified-memory helpers."""

from __future__ import annotations

import contextlib
import sys
from unittest.mock import MagicMock, call

import pytest

from zvisiongenerator.backends.mac import memory
from zvisiongenerator.backends.mac.memory import (
    BUFFER_CACHE_LIMIT_BYTES,
    apply_buffer_cache_policy,
    load_resident,
    materialize_weights,
    memory_size_bytes,
    mlx_cache_limit,
)


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
    def test_materialize_weights_evaluates_each_parameter_array_separately(self, mx, monkeypatch):
        first, second, third = object(), object(), object()
        fake_utils = MagicMock(tree_flatten=MagicMock(return_value=[("a.weight", first), ("a.bias", second), ("b.weight", third)]))
        monkeypatch.setitem(sys.modules, "mlx.utils", fake_utils)
        model = MagicMock()

        materialize_weights(model)

        fake_utils.tree_flatten.assert_called_once_with(model.parameters.return_value)
        assert mx.eval.call_args_list == [call(first), call(second), call(third)]

    def test_evaluate_computes_its_argument_and_returns_it(self, mx):
        value = object()

        assert memory.evaluate(value) is value
        mx.eval.assert_called_once_with(value)

    def test_policy_caps_the_cache_then_clears_it(self, mx):
        apply_buffer_cache_policy()

        assert [c[0] for c in mx.method_calls] == ["set_cache_limit", "clear_cache"]
        mx.set_cache_limit.assert_called_once_with(BUFFER_CACHE_LIMIT_BYTES)


class TestLoadResident:
    @pytest.fixture
    def order(self, monkeypatch):
        """Record the cache-limit block and the materialize and policy calls, in order, on one mock."""
        recorder = MagicMock()

        @contextlib.contextmanager
        def scope(limit):
            recorder.enter(limit)
            try:
                yield
            finally:
                recorder.exit()

        monkeypatch.setattr(memory, "mlx_cache_limit", scope)
        monkeypatch.setattr(memory, "materialize_weights", recorder.materialize)
        monkeypatch.setattr(memory, "apply_buffer_cache_policy", recorder.policy)
        return recorder

    def test_builds_and_materializes_with_the_cache_disabled_then_applies_the_policy(self, order):
        build = order.build

        loaded = load_resident(build)

        assert loaded is build.return_value
        assert order.mock_calls == [call.enter(0), call.build(), call.materialize(build.return_value), call.exit(), call.policy()]

    def test_restores_the_cache_limit_and_skips_the_policy_when_the_build_fails(self, order):
        order.build.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            load_resident(order.build)

        assert order.mock_calls == [call.enter(0), call.build(), call.exit()]

    def test_restores_the_cache_limit_and_skips_the_policy_when_materializing_fails(self, order):
        order.materialize.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            load_resident(order.build)

        assert order.mock_calls == [call.enter(0), call.build(), call.materialize(order.build.return_value), call.exit()]


@pytest.fixture
def torch_absent(monkeypatch):
    """Make ``sys.modules`` hold no torch, as in a process that never imported it."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)


@pytest.fixture
def fake_torch(monkeypatch):
    """Put a fake ``torch`` with an ``mps`` namespace into ``sys.modules``."""
    torch = MagicMock(name="torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class TestFreeBytes:
    @pytest.fixture
    def budget(self, mx, monkeypatch):
        mx.get_active_memory.return_value = 5
        mx.get_cache_memory.return_value = 3
        monkeypatch.setattr(memory, "memory_budget_bytes", lambda: 20)
        return mx

    def test_subtracts_active_cached_and_mps_allocated_memory(self, budget, fake_torch):
        fake_torch.mps.current_allocated_memory.return_value = 4

        assert memory.free_bytes() == 8

    def test_without_torch_loaded_the_mps_part_is_zero(self, budget, torch_absent):
        assert memory.free_bytes() == 12

    def test_a_negative_result_clamps_to_zero(self, budget, fake_torch):
        fake_torch.mps.current_allocated_memory.return_value = 100

        assert memory.free_bytes() == 0

    def test_an_unknown_budget_gives_none(self, budget, monkeypatch, torch_absent):
        monkeypatch.setattr(memory, "memory_budget_bytes", lambda: None)

        assert memory.free_bytes() is None


class TestReleaseMemory:
    @pytest.fixture
    def cleanup(self, monkeypatch):
        aggressive_cleanup = MagicMock(name="aggressive_cleanup")
        monkeypatch.setitem(sys.modules, "ltx_core_mlx", MagicMock())
        monkeypatch.setitem(sys.modules, "ltx_core_mlx.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "ltx_core_mlx.utils.memory", MagicMock(aggressive_cleanup=aggressive_cleanup))
        return aggressive_cleanup

    def test_empties_the_mps_cache_once_after_the_mlx_cleanup_when_torch_is_loaded(self, cleanup, fake_torch):
        order = MagicMock()
        order.attach_mock(cleanup, "cleanup")
        order.attach_mock(fake_torch.mps.empty_cache, "empty_cache")

        memory.release_memory()

        assert [c[0] for c in order.mock_calls] == ["cleanup", "empty_cache"]

    def test_never_imports_torch_itself(self, cleanup, torch_absent, monkeypatch):
        monkeypatch.setattr("builtins.__import__", _refuse_torch_import(__import__))

        memory.release_memory()

        cleanup.assert_called_once_with()
        assert "torch" not in sys.modules


def _refuse_torch_import(real_import):
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("release_memory imported torch")
        return real_import(name, *args, **kwargs)

    return guarded
