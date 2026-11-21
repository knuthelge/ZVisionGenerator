"""Tests for the thresholds every memory verdict shares."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS, CUDA_RAM_RATIOS, FITS, MAC_RATIOS, TIGHT, TOO_LARGE, grade

_GIB = 1024**3


@pytest.mark.parametrize(
    ("need", "status"),
    [(9.0, FITS), (9.0 + 1e-6, TIGHT), (10.0, TIGHT), (10.0 + 1e-6, TOO_LARGE)],
)
def test_a_cuda_gpu_fits_up_to_90_percent_is_tight_up_to_the_whole_card_and_too_large_beyond(need, status):
    assert grade(need * _GIB, 10 * _GIB, CUDA_GPU_RATIOS) == status


@pytest.mark.parametrize(("need", "status"), [(24.0, FITS), (24.0 + 1e-6, TIGHT), (30.0, TIGHT), (30.0 + 1e-6, TOO_LARGE)])
def test_cuda_ram_fits_up_to_80_percent_is_tight_up_to_all_of_it_and_too_large_beyond(need, status):
    assert grade(need * _GIB, 30 * _GIB, CUDA_RAM_RATIOS) == status


@pytest.mark.parametrize(("need", "status"), [(12.0, FITS), (12.0 + 1e-6, TIGHT), (16.0, TIGHT), (16.0 + 1e-6, TOO_LARGE)])
def test_a_mac_fits_up_to_75_percent_of_its_memory_is_tight_up_to_all_of_it_and_too_large_beyond(need, status):
    assert grade(need * _GIB, 16 * _GIB, MAC_RATIOS) == status
