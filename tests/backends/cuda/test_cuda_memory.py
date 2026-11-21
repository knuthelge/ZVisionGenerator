"""Tests for the CUDA allocator configuration."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.backends.cuda import memory as memory_module
from zvisiongenerator.backends.cuda import nvml
from zvisiongenerator.backends.cuda.memory import PIN_ROUND_SETTING, configure_allocator, free_bytes, memory_usage, merged_alloc_conf
from zvisiongenerator.core.memory_sampler import MemoryUsage

_DEFAULTS = "expandable_segments:True,garbage_collection_threshold:0.8"


class TestMergedAllocConf:
    def test_nothing_set_yields_only_the_added_option(self):
        assert merged_alloc_conf(None, None, PIN_ROUND_SETTING) == PIN_ROUND_SETTING

    def test_defaults_gain_the_added_option_last(self):
        assert merged_alloc_conf(_DEFAULTS, None, PIN_ROUND_SETTING) == f"{_DEFAULTS},{PIN_ROUND_SETTING}"

    def test_custom_options_are_kept(self):
        assert merged_alloc_conf("max_split_size_mb:128", None, PIN_ROUND_SETTING) == f"max_split_size_mb:128,{PIN_ROUND_SETTING}"

    def test_a_user_pin_value_is_kept(self):
        assert merged_alloc_conf(None, "pinned_max_round_threshold_mb:64", PIN_ROUND_SETTING) == "pinned_max_round_threshold_mb:64"
        assert merged_alloc_conf("pinned_max_round_threshold_mb:8,a:1", None, PIN_ROUND_SETTING) == "pinned_max_round_threshold_mb:8,a:1"

    def test_alloc_conf_overrides_cuda_conf_in_place(self):
        merged = merged_alloc_conf("a:1,b:2", "b:3,c:4", PIN_ROUND_SETTING)

        assert merged == f"a:1,b:3,c:4,{PIN_ROUND_SETTING}"

    def test_blank_pairs_and_whitespace_are_dropped(self):
        assert merged_alloc_conf(" a:1 ,, ", "", PIN_ROUND_SETTING) == f"a:1,{PIN_ROUND_SETTING}"

    def test_merging_its_own_result_changes_nothing(self):
        once = merged_alloc_conf(_DEFAULTS, None, PIN_ROUND_SETTING)

        assert merged_alloc_conf(_DEFAULTS, once, PIN_ROUND_SETTING) == once


class TestConfigureAllocator:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    def test_sets_the_same_string_in_both_variables(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == f"{_DEFAULTS},{PIN_ROUND_SETTING}"
        assert os.environ["PYTORCH_ALLOC_CONF"] == os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    def test_the_linux_default_keeps_expandable_segments(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")

        configure_allocator()

        assert "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",")

    def test_the_windows_default_has_no_expandable_segments(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

        configure_allocator()

        expected = f"garbage_collection_threshold:0.8,{PIN_ROUND_SETTING}"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == expected
        assert os.environ["PYTORCH_ALLOC_CONF"] == expected

    def test_a_user_alloc_conf_alone_is_not_shadowed_by_the_defaults(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")

        configure_allocator()

        expected = f"max_split_size_mb:64,{PIN_ROUND_SETTING}"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == expected
        assert os.environ["PYTORCH_ALLOC_CONF"] == expected

    def test_a_user_pin_value_wins_in_both_variables(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "pinned_max_round_threshold_mb:32")

        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "pinned_max_round_threshold_mb:32"
        assert os.environ["PYTORCH_ALLOC_CONF"] == "pinned_max_round_threshold_mb:32"

    def test_a_user_cuda_conf_is_left_as_written(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "custom:value")

        configure_allocator()

        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "custom:value"
        assert os.environ["PYTORCH_ALLOC_CONF"] == f"custom:value,{PIN_ROUND_SETTING}"

    def test_the_gpu_order_is_cudas_own_unless_the_user_set_one(self, monkeypatch):
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "unset")
        monkeypatch.delenv("CUDA_DEVICE_ORDER")  # leaves the variable absent and restores the original afterwards

        configure_allocator()

        assert "CUDA_DEVICE_ORDER" not in os.environ

        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        configure_allocator()

        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_is_idempotent(self):
        configure_allocator()
        first = (os.environ["PYTORCH_CUDA_ALLOC_CONF"], os.environ["PYTORCH_ALLOC_CONF"])
        configure_allocator()

        assert (os.environ["PYTORCH_CUDA_ALLOC_CONF"], os.environ["PYTORCH_ALLOC_CONF"]) == first

    def test_torch_sees_the_pin_setting(self):
        """A subprocess with CUDA unavailable skips; otherwise the allocator's own settings include the pin."""
        import subprocess
        import sys

        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        code = (
            "from zvisiongenerator.backends.cuda.memory import configure_allocator\n"
            "configure_allocator()\n"
            "import torch\n"
            "torch.zeros(1, device='cuda')\n"
            "print(torch._C._accelerator_getAllocatorSettings())\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=True)

        assert "pinned_max_round_threshold_mb" in result.stdout


class TestFreeBytes:
    def test_returns_the_first_value_of_the_driver_query(self, monkeypatch):
        torch = MagicMock()
        torch.cuda.mem_get_info.return_value = (3 * 1024**3, 10 * 1024**3)
        monkeypatch.setitem(sys.modules, "torch", torch)

        assert free_bytes() == 3 * 1024**3

    def test_a_failing_query_gives_none(self, monkeypatch):
        torch = MagicMock()
        torch.cuda.mem_get_info.side_effect = RuntimeError("no CUDA")
        monkeypatch.setitem(sys.modules, "torch", torch)

        assert free_bytes() is None


def _fake_torch(*, initialized: bool, free: int = 4 * 1024**3, total: int = 10 * 1024**3) -> MagicMock:
    torch = MagicMock(name="torch")
    torch.cuda.is_initialized.return_value = initialized
    torch.cuda.mem_get_info.return_value = (free, total)
    return torch


@pytest.fixture
def machine_memory(monkeypatch):
    """Serve a fake ``psutil`` whose machine has 32 GiB of RAM, 12 GiB of it available."""
    psutil = MagicMock(name="psutil")
    psutil.virtual_memory.return_value = SimpleNamespace(total=32 * 1024**3, available=12 * 1024**3)
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    return psutil


class TestMemoryUsage:
    @pytest.fixture(autouse=True)
    def _smi(self, monkeypatch):
        """Serve nvidia-smi's budget as 9.5 GiB (what the memory tag judges against) and forget what an earlier test cached."""
        monkeypatch.setattr(memory_module, "_smi_budget", None)
        monkeypatch.setattr(memory_module, "_gpu_budget_bytes", lambda: int(9.5 * 1024**3))
        monkeypatch.setattr(nvml, "card_memory", lambda: None)

    def test_system_memory_is_the_whole_machines_load(self, machine_memory, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        assert memory_usage() == MemoryUsage(system_used_bytes=20 * 1024**3, gpu_used_bytes=None, system_total_bytes=32 * 1024**3, gpu_total_bytes=int(9.5 * 1024**3), gpu_unknown=True)

    def test_an_unreadable_machine_gives_none(self, machine_memory):
        machine_memory.virtual_memory.side_effect = OSError("no /proc")

        assert memory_usage() is None

    def test_an_initialized_cuda_reports_the_whole_cards_use_against_the_budget(self, machine_memory, monkeypatch):
        torch = _fake_torch(initialized=True, free=3 * 1024**3, total=10 * 1024**3)
        monkeypatch.setitem(sys.modules, "torch", torch)

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_total_bytes, usage.gpu_unknown) == (7 * 1024**3, int(9.5 * 1024**3), False)
        torch.cuda.mem_get_info.assert_called_once_with(0)

    def test_without_nvidia_smi_the_drivers_own_total_is_used(self, machine_memory, monkeypatch):
        monkeypatch.setattr(memory_module, "_gpu_budget_bytes", lambda: None)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(initialized=True, total=int(9.64 * 1024**3)))

        assert memory_usage().gpu_total_bytes == int(9.64 * 1024**3)

    def test_without_nvidia_smi_and_before_cuda_starts_the_total_is_unknown(self, machine_memory, monkeypatch):
        monkeypatch.setattr(memory_module, "_gpu_budget_bytes", lambda: None)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(initialized=False))

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_total_bytes, usage.gpu_unknown) == (None, None, True)

    def test_nvidia_smi_is_asked_once(self, machine_memory, monkeypatch):
        asked = []
        monkeypatch.setattr(memory_module, "_gpu_budget_bytes", lambda: asked.append("smi") or 10 * 1024**3)
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        memory_usage()
        memory_usage()

        assert asked == ["smi"]

    def test_an_uninitialized_cuda_reports_an_unknown_gpu_figure_without_querying_the_driver(self, machine_memory, monkeypatch):
        torch = _fake_torch(initialized=False)
        monkeypatch.setitem(sys.modules, "torch", torch)

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_unknown) == (None, True)
        torch.cuda.mem_get_info.assert_not_called()

    def test_a_failing_driver_query_reports_an_unknown_gpu_figure(self, machine_memory, monkeypatch):
        torch = _fake_torch(initialized=True)
        torch.cuda.mem_get_info.side_effect = RuntimeError("driver error")
        monkeypatch.setitem(sys.modules, "torch", torch)

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_unknown) == (None, True)
        assert usage.system_used_bytes == 20 * 1024**3

    def test_nvml_reports_the_whole_cards_use_before_cuda_starts(self, machine_memory, monkeypatch):
        gib = 1024**3
        monkeypatch.setattr(nvml, "card_memory", lambda: nvml.CardMemory(total=10 * gib, free=5 * gib))
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_total_bytes, usage.gpu_unknown) == (
            int(4.5 * gib),
            int(9.5 * gib),
            False,
        )  # the budget less the free memory: the driver's reserve is not counted as use

    def test_nvml_counts_other_programs_that_the_cuda_query_does_not(self, machine_memory, monkeypatch):
        gib = 1024**3
        monkeypatch.setattr(nvml, "card_memory", lambda: nvml.CardMemory(total=10 * gib, free=5 * gib))
        torch = _fake_torch(initialized=True, free=int(8.88 * gib), total=10 * gib)
        monkeypatch.setitem(sys.modules, "torch", torch)

        assert memory_usage().gpu_used_bytes == int(4.5 * gib)
        torch.cuda.mem_get_info.assert_not_called()

    def test_nvml_without_nvidia_smi_measures_against_its_own_total(self, machine_memory, monkeypatch):
        gib = 1024**3
        monkeypatch.setattr(memory_module, "_gpu_budget_bytes", lambda: None)
        monkeypatch.setattr(nvml, "card_memory", lambda: nvml.CardMemory(total=10 * gib, free=4 * gib))
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        usage = memory_usage()

        assert (usage.gpu_used_bytes, usage.gpu_total_bytes) == (6 * gib, 10 * gib)

    def test_torch_is_not_imported_when_it_is_absent(self, machine_memory, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        memory_usage()

        assert "torch" not in sys.modules


class TestBudget:
    @pytest.fixture(autouse=True)
    def _card(self, monkeypatch):
        """CUDA's device 0 sits at PCI bus id 0000:01:00.0, so nvidia-smi is asked for that card."""
        monkeypatch.setattr(nvml, "card_bus_id", lambda: "0000:01:00.0")

    @staticmethod
    def _smi(monkeypatch, outputs: dict[str, str], asked: list[list[str]] | None = None):
        """Serve nvidia-smi with *outputs* by queried fields, recording each command in *asked*; a field set that is not listed fails as an unknown field does."""
        import subprocess

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")

        def run(command, **_kwargs):
            if asked is not None:
                asked.append(command)
            fields = next(part for part in command if part.startswith("--query-gpu=")).removeprefix("--query-gpu=")
            if fields not in outputs:
                raise subprocess.CalledProcessError(2, command)
            return SimpleNamespace(stdout=outputs[fields])

        monkeypatch.setattr("subprocess.run", run)

    def test_the_budget_is_the_total_less_the_drivers_reserve(self, monkeypatch):
        self._smi(monkeypatch, {"memory.total,memory.reserved": "10240, 394\n"})

        assert memory_module._gpu_budget_bytes() == (10240 - 394) * 1024**2

    def test_the_card_total_keeps_the_drivers_reserve(self, monkeypatch):
        self._smi(monkeypatch, {"memory.total,memory.reserved": "10240, 394\n"})

        assert memory_module.card_total_bytes() == 10240 * 1024**2

    def test_without_nvidia_smi_there_is_no_card_total(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert memory_module.card_total_bytes() is None

    def test_a_driver_that_does_not_report_the_reserve_leaves_the_total(self, monkeypatch):
        self._smi(monkeypatch, {"memory.total,memory.reserved": "10240, [N/A]\n"})

        assert memory_module._gpu_budget_bytes() == 10240 * 1024**2

    def test_a_driver_that_cannot_query_the_reserve_gives_no_budget(self, monkeypatch):
        self._smi(monkeypatch, {"memory.total": "10240\n"})

        assert memory_module._gpu_budget_bytes() is None

    def test_the_budget_is_read_from_the_card_at_the_bus_id_cuda_names_cuda_0(self, monkeypatch):
        monkeypatch.setattr(nvml, "card_bus_id", lambda: "0000:41:00.0")
        asked: list[list[str]] = []
        self._smi(monkeypatch, {"memory.total,memory.reserved": "10240, 394\n"}, asked)

        assert memory_module._gpu_budget_bytes() == (10240 - 394) * 1024**2
        assert asked[0][:3] == ["/usr/bin/nvidia-smi", "-i", "0000:41:00.0"]
        assert len(asked) == 1  # one query, whatever the driver reports

    def test_without_a_card_for_cuda_there_is_no_budget_and_nvidia_smi_is_not_run(self, monkeypatch):
        monkeypatch.setattr(nvml, "card_bus_id", lambda: None)
        asked: list[list[str]] = []
        self._smi(monkeypatch, {"memory.total,memory.reserved": "10240, 394\n"}, asked)

        assert memory_module._gpu_budget_bytes() is None and memory_module.card_total_bytes() is None
        assert asked == []

    def test_without_nvidia_smi_there_is_no_budget(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert memory_module._gpu_budget_bytes() is None
