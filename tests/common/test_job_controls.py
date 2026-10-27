"""Tests for zvisiongenerator.job_controls — the seed range shared by the runners and the preflight."""

from __future__ import annotations

import pytest

from zvisiongenerator.job_controls import seed_range


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (4, 2**32 - 1)),
        ({"generation": {"seed_min": 1, "seed_max": 100}}, (1, 100)),
    ],
)
def test_seed_range_comes_from_the_generation_section_with_defaults(config, expected):
    assert seed_range(config) == expected
