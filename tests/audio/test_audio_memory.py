"""Tests for the audio memory fit shared by the Web UI badge, the CLI warning and the out-of-memory message."""

from __future__ import annotations

import dataclasses

import pytest

from tests.audio._helpers import preset

from zvisiongenerator.utils.audio_memory import audio_memory_fit, memory_need_text, memory_warning_text
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

QWEN = preset("qwen3_tts")  # measured at 6.0 GiB on CUDA, not yet on a Mac


def _card(gb: float) -> MemoryBudget:
    return MemoryBudget(round(gb * GIB), system_bytes=64 * GIB)


class TestFit:
    @pytest.mark.parametrize(("card_gb", "status"), [(9, "fits"), (6.1, "tight"), (5.5, "too_large")])
    def test_the_recorded_peak_is_judged_against_the_cards_memory(self, card_gb, status):
        fit = audio_memory_fit(QWEN, "cuda", _card(card_gb))

        assert fit == {"status": status, "required_gb": 6.0, "budget_gb": round(card_gb, 1), "kind": "discrete"}

    def test_a_model_without_a_measured_peak_or_a_machine_without_a_budget_has_no_fit(self):
        unmeasured = dataclasses.replace(QWEN, platforms={key: value for key, value in QWEN.platforms.items() if key != "mps"})

        assert audio_memory_fit(unmeasured, "mps", MemoryBudget(16 * GIB)) is None
        assert audio_memory_fit(QWEN, "cuda", None) is None


class TestWords:
    def test_a_card_that_is_too_small_gets_a_warning_with_the_need_and_the_size(self):
        text = memory_warning_text("Qwen", audio_memory_fit(QWEN, "cuda", _card(5.5)))

        assert text is not None and "6 GB" in text and "5.5 GB" in text and "Qwen" in text

    def test_a_tight_card_is_warned_too(self):
        assert memory_warning_text("Qwen", audio_memory_fit(QWEN, "cuda", _card(6.1))) is not None

    def test_a_card_that_fits_or_is_unknown_gets_no_warning(self):
        assert memory_warning_text("Qwen", audio_memory_fit(QWEN, "cuda", _card(9))) is None
        assert memory_warning_text("Qwen", None) is None

    def test_the_need_text_names_both_sizes(self):
        assert memory_need_text({"required_gb": 5.8, "budget_gb": 10.0}) == "it needs about 5.8 GB and the GPU has 10 GB"
