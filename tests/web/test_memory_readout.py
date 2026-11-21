"""Tests for the memory readings the Web UI shows beside a running job."""

from __future__ import annotations

from zvisiongenerator import backends
from zvisiongenerator.core.memory_sampler import MemoryUsage
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import memory_readout

UNIFIED = MemoryBudget(gpu_bytes=24 * GIB, unified_bytes=32 * GIB)
DISCRETE = MemoryBudget(gpu_bytes=10 * GIB, system_bytes=32 * GIB)


class TestReadingPayload:
    def test_unified_reports_the_machines_memory_in_use_against_the_total_memory_with_no_status(self):
        payload = memory_readout.reading_payload(MemoryUsage(system_used_bytes=18 * GIB), UNIFIED)

        assert payload == {"kind": "unified", "used_gb": 18.0, "budget_gb": 32.0}

    def test_unified_without_a_budget_has_no_reading(self):
        assert memory_readout.reading_payload(MemoryUsage(system_used_bytes=18 * GIB), None) is None

    def test_discrete_reports_gpu_and_system_memory_apart_with_no_status(self):
        payload = memory_readout.reading_payload(MemoryUsage(system_used_bytes=21 * GIB, gpu_used_bytes=7 * GIB), DISCRETE)

        assert payload == {"kind": "discrete", "gpu_gb": 7.0, "gpu_budget_gb": 10.0, "system_gb": 21.0, "system_budget_gb": 32.0}

    def test_discrete_without_gpu_memory_reads_zero(self):
        assert memory_readout.reading_payload(MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=0), DISCRETE)["gpu_gb"] == 0.0

    def test_a_figure_beyond_its_total_is_shown_as_it_is(self):
        payload = memory_readout.reading_payload(MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=11 * GIB), DISCRETE)

        assert (payload["gpu_gb"], payload["gpu_budget_gb"]) == (11.0, 10.0)

    def test_the_totals_a_reading_carries_win_over_the_budget(self):
        usage = MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=3 * GIB, system_total_bytes=16 * GIB, gpu_total_bytes=12 * GIB)

        payload = memory_readout.reading_payload(usage, DISCRETE)

        assert (payload["gpu_budget_gb"], payload["system_budget_gb"]) == (12.0, 16.0)

    def test_a_card_whose_use_is_not_known_yet_reports_no_gpu_figure_but_keeps_its_total(self):
        usage = MemoryUsage(system_used_bytes=21 * GIB, system_total_bytes=32 * GIB, gpu_total_bytes=12 * GIB, gpu_unknown=True)

        payload = memory_readout.reading_payload(usage, DISCRETE)

        assert payload == {"kind": "discrete", "gpu_gb": None, "gpu_budget_gb": 12.0, "system_gb": 21.0, "system_budget_gb": 32.0}

    def test_an_unknown_card_total_falls_back_to_the_budget_and_never_reads_as_unified(self):
        payload = memory_readout.reading_payload(MemoryUsage(system_used_bytes=21 * GIB, gpu_unknown=True), DISCRETE)

        assert payload == {"kind": "discrete", "gpu_gb": None, "gpu_budget_gb": 10.0, "system_gb": 21.0, "system_budget_gb": 32.0}

    def test_without_a_budget_the_reading_carries_its_own_totals(self):
        usage = MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=3 * GIB, system_total_bytes=16 * GIB, gpu_total_bytes=12 * GIB)

        assert memory_readout.reading_payload(usage, None) == {"kind": "discrete", "gpu_gb": 3.0, "gpu_budget_gb": 12.0, "system_gb": 2.0, "system_budget_gb": 16.0}

    def test_discrete_with_an_unknown_total_has_no_reading(self):
        assert memory_readout.reading_payload(MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=0), None) is None
        assert memory_readout.reading_payload(MemoryUsage(system_used_bytes=2 * GIB, gpu_used_bytes=0, system_total_bytes=16 * GIB), None) is None


class TestMachineProbe:
    def test_reads_are_the_backends_own_reader(self):
        assert memory_readout.machine_probe().read is backends.memory_usage

    def test_budget_is_the_accelerator_budget(self):
        assert memory_readout.machine_probe().budget is memory_readout.accelerator_memory_budget_or_none
