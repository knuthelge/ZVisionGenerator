"""Tests for the audio memory fit shared by the Web UI badge, the CLI warning and the out-of-memory message."""

from __future__ import annotations

import dataclasses

import pytest

from tests.audio._helpers import preset

from zvisiongenerator.utils.audio_memory import audio_memory_fit, memory_need_text, memory_warning_text
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

HIGGS3 = preset("higgs3")  # measured at 8.2 GiB on CUDA and 11.0 GiB on MPS


def _card(gb: float) -> MemoryBudget:
    return MemoryBudget(round(gb * GIB), system_bytes=64 * GIB)


class TestFit:
    @pytest.mark.parametrize(("card_gb", "status"), [(12, "fits"), (8.5, "tight"), (8, "too_large")])
    def test_the_recorded_peak_is_judged_against_the_cards_memory(self, card_gb, status):
        fit = audio_memory_fit(HIGGS3, "cuda", _card(card_gb))

        assert fit == {"status": status, "required_gb": 8.2, "budget_gb": round(card_gb, 1), "kind": "discrete"}

    def test_a_model_without_a_measured_peak_or_a_machine_without_a_budget_has_no_fit(self):
        unmeasured = dataclasses.replace(HIGGS3, platforms={key: value for key, value in HIGGS3.platforms.items() if key != "mps"})

        assert audio_memory_fit(unmeasured, "mps", MemoryBudget(16 * GIB)) is None
        assert audio_memory_fit(HIGGS3, "cuda", None) is None


class TestWords:
    def test_a_card_that_is_too_small_gets_a_warning_with_the_need_and_the_size(self):
        text = memory_warning_text("Higgs", audio_memory_fit(HIGGS3, "cuda", _card(8)))

        assert text is not None and "8.2 GB" in text and "8 GB" in text and "Higgs" in text

    def test_a_tight_card_is_warned_too(self):
        assert memory_warning_text("Higgs", audio_memory_fit(HIGGS3, "cuda", _card(8.5))) is not None

    def test_a_card_that_fits_or_is_unknown_gets_no_warning(self):
        assert memory_warning_text("Higgs", audio_memory_fit(HIGGS3, "cuda", _card(12))) is None
        assert memory_warning_text("Higgs", None) is None

    def test_the_need_text_names_both_sizes(self):
        assert memory_need_text({"required_gb": 8.2, "budget_gb": 10.0}) == "it needs about 8.2 GB and the GPU has 10 GB"
