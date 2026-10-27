"""Tests for the shared beta-distribution sigma grid."""

from __future__ import annotations

import pytest

from zvisiongenerator.schedulers.beta_sigmas import beta_sigmas


class TestBetaSigmas:
    def test_length_matches_steps(self):
        assert len(beta_sigmas(8)) == 8

    def test_starts_at_one_and_decreases(self):
        sigmas = beta_sigmas(10)
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas == sorted(sigmas, reverse=True)
        assert 0.0 < sigmas[-1] < 0.2

    def test_concentrates_steps_at_both_ends(self):
        sigmas = beta_sigmas(10)
        gaps = [a - b for a, b in zip(sigmas, sigmas[1:])]
        assert gaps[0] < gaps[len(gaps) // 2]
