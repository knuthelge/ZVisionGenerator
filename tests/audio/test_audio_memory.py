"""Tests for the audio memory need that words the out-of-memory message."""

from __future__ import annotations

import dataclasses

import pytest

from tests.audio._helpers import preset

from zvisiongenerator.utils.audio_memory import audio_memory_need, audio_verdict
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

QWEN = preset("qwen3_tts")
_CUDA_PEAK_GIB, _MAC_PEAK_GIB = QWEN.memory_gb_for("cuda"), QWEN.memory_gb_for("mps")


def _card(gb: float) -> MemoryBudget:
    return MemoryBudget(round(gb * GIB), system_bytes=64 * GIB)


class TestNeed:
    @pytest.mark.parametrize("card_gb", [9, 6.1, 5.5])
    def test_the_recorded_peak_is_worded_against_the_cards_memory_whatever_the_verdict(self, card_gb):
        text = audio_memory_need(QWEN, "cuda", _card(card_gb))

        assert text == f"it needs about {round((_CUDA_PEAK_GIB * GIB + CUDA_OFFSET_BYTES) / GIB, 1):g} GB and the GPU has {round(card_gb, 1):g} GB"

    def test_a_mac_is_worded_against_its_total_memory_which_its_verdict_is_graded_against(self):
        assert audio_memory_need(QWEN, "mps", MemoryBudget(16 * GIB, unified_bytes=24 * GIB)) == f"it needs about {round(_MAC_PEAK_GIB, 1):g} GB and this Mac has 24 GB"

    def test_a_model_without_a_measured_peak_or_a_machine_without_a_budget_has_no_need(self):
        unmeasured = dataclasses.replace(QWEN, platforms={key: value for key, value in QWEN.platforms.items() if key != "mps"})

        assert audio_memory_need(unmeasured, "mps", MemoryBudget(16 * GIB, unified_bytes=24 * GIB)) is None
        assert audio_memory_need(QWEN, "cuda", None) is None


class TestMacBudgetWithoutTotalMemory:
    def test_a_mac_budget_without_the_total_unified_memory_has_no_verdict_and_no_need(self):
        budget = MemoryBudget(16 * GIB)

        assert audio_verdict(QWEN, "mps", budget) is None
        assert audio_memory_need(QWEN, "mps", budget) is None
