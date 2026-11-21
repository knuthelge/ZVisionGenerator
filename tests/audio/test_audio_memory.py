"""Tests for the audio memory need that words the out-of-memory message."""

from __future__ import annotations

import dataclasses

import pytest

from tests._factories import load_packaged_config
from tests.audio._helpers import preset

from zvisiongenerator.utils import memory_verdict
from zvisiongenerator.utils.audio_memory import audio_memory_need, audio_verdict
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_LINUX_BYTES, CUDA_OFFSET_WINDOWS_BYTES
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

QWEN = preset("qwen3_tts")
_CUDA_PEAK_GIB, _MAC_PEAK_GIB = QWEN.memory_gb_for("cuda"), QWEN.memory_gb_for("mps")


def _card(gb: float) -> MemoryBudget:
    return MemoryBudget(round(gb * GIB), system_bytes=64 * GIB)


class TestNeed:
    @pytest.mark.parametrize("card_gb", [9, 6.1, 5.5])
    def test_the_recorded_peak_is_worded_against_the_cards_memory_whatever_the_verdict(self, card_gb):
        text = audio_memory_need(QWEN, "cuda", _card(card_gb))

        assert text == f"it needs about {round((_CUDA_PEAK_GIB * GIB + CUDA_OFFSET_LINUX_BYTES) / GIB, 1):g} GB and the GPU has {round(card_gb, 1):g} GB"

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


class TestWindowsFigure:
    @staticmethod
    def _shipped(model_id: str):
        return next(item for item in parse_audio_presets(load_packaged_config()) if item.id == model_id)

    @staticmethod
    def _on(monkeypatch, platform: str, offset: int) -> None:
        monkeypatch.setattr("sys.platform", platform)
        monkeypatch.setattr(memory_verdict, "CUDA_OFFSET_BYTES", offset)

    def test_stable_audio_judges_its_windows_peak_plus_the_windows_offset_on_windows(self, monkeypatch):
        self._on(monkeypatch, "win32", CUDA_OFFSET_WINDOWS_BYTES)

        verdict = audio_verdict(self._shipped("sa3"), "cuda", _card(12))

        assert verdict is not None
        assert verdict.gpu_bytes == round(6.46 * GIB) + CUDA_OFFSET_WINDOWS_BYTES

    def test_stable_audio_still_judges_its_cuda_peak_plus_the_linux_offset_on_linux(self, monkeypatch):
        self._on(monkeypatch, "linux", CUDA_OFFSET_LINUX_BYTES)

        verdict = audio_verdict(self._shipped("sa3"), "cuda", _card(12))

        assert verdict is not None
        assert verdict.gpu_bytes == round(5.27 * GIB) + CUDA_OFFSET_LINUX_BYTES

    def test_a_preset_without_a_windows_figure_falls_back_to_its_cuda_figure_on_windows(self, monkeypatch):
        self._on(monkeypatch, "win32", CUDA_OFFSET_WINDOWS_BYTES)
        yue2 = self._shipped("yue2")

        verdict = audio_verdict(yue2, "cuda", _card(12))

        assert yue2.memory_gb_for("windows") is None
        assert verdict is not None
        assert verdict.gpu_bytes == round(yue2.memory_gb_for("cuda") * GIB) + CUDA_OFFSET_WINDOWS_BYTES

    def test_a_mac_never_reads_the_windows_figure(self, monkeypatch):
        self._on(monkeypatch, "win32", CUDA_OFFSET_WINDOWS_BYTES)
        sa3 = self._shipped("sa3")

        verdict = audio_verdict(sa3, "mps", MemoryBudget(16 * GIB, unified_bytes=24 * GIB))

        assert verdict is not None
        assert verdict.gpu_bytes == round(sa3.memory_gb_for("mps") * GIB)
