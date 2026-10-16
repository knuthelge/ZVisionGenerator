"""Tests for the CUDA video backend's header-based system-memory estimate."""

from __future__ import annotations

from pathlib import Path

import pytest

from zvisiongenerator.backends.cuda.video import memory_estimate
from zvisiongenerator.utils.model_memory import estimate_ltx_cuda_system_memory

_GIB = 1024**3
_CONFIG = {"video_model_presets": {"ltx": {"diffusers": {"transformer": {"repo": "owner/ckpt", "filename": "ckpt.safetensors"}}}}}


@pytest.fixture()
def header_counts(monkeypatch):
    """Replace the header readers with fixed counts, recording the paths they were asked about."""
    paths: list[Path] = []

    def _transformer(source):
        paths.append(source)
        return 10 * _GIB, _GIB // 2

    monkeypatch.setattr(memory_estimate.weights, "transformer_elements", _transformer)
    monkeypatch.setattr(memory_estimate.weights, "component_elements", lambda folder: 3 * _GIB)
    monkeypatch.setattr(memory_estimate.weights, "text_encoder_bytes", lambda folder: 12 * _GIB)
    return paths


class TestLtxSystemEstimate:
    @pytest.mark.parametrize("quantize", [4, 8])
    def test_counts_the_headers_the_way_the_load_time_check_does(self, header_counts, quantize):
        estimate = memory_estimate.ltx_system_estimate(Path("/repo"), Path("/ckpt"), quantize)

        expected = estimate_ltx_cuda_system_memory(block_matrices=10 * _GIB, transformer_rest=_GIB // 2, connectors=3 * _GIB, text_encoder=12 * _GIB, quantize=quantize)
        assert estimate == expected
        assert header_counts == [Path("/ckpt")]


class TestCachedLtxSystemEstimate:
    def test_reads_the_cached_single_file_transformer(self, header_counts, monkeypatch, tmp_path):
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo_id, filename: str(tmp_path / filename) if (repo_id, filename) == ("owner/ckpt", "ckpt.safetensors") else None)

        estimate = memory_estimate.cached_ltx_system_estimate(_CONFIG, "owner/ltx", 4, find_local_dir=lambda reference: tmp_path)

        assert estimate is not None and estimate.quantize == 4
        assert header_counts == [tmp_path / "ckpt.safetensors"]

    def test_uses_the_repos_own_transformer_when_no_single_file_is_configured(self, header_counts, tmp_path):
        estimate = memory_estimate.cached_ltx_system_estimate({}, "owner/ltx", 8, find_local_dir=lambda reference: tmp_path)

        assert estimate is not None and estimate.quantize == 8
        assert header_counts == [tmp_path / "transformer"]

    def test_is_unknown_when_the_repo_is_not_on_disk(self, header_counts):
        assert memory_estimate.cached_ltx_system_estimate(_CONFIG, "owner/ltx", 4, find_local_dir=lambda reference: None) is None
        assert header_counts == []

    def test_is_unknown_when_the_single_file_transformer_is_not_cached(self, header_counts, monkeypatch, tmp_path):
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo_id, filename: None)

        assert memory_estimate.cached_ltx_system_estimate(_CONFIG, "owner/ltx", 4, find_local_dir=lambda reference: tmp_path) is None
        assert header_counts == []

    def test_is_unknown_when_a_header_cannot_be_read(self, monkeypatch, tmp_path):
        def _broken(source):
            raise OSError("unreadable")

        monkeypatch.setattr(memory_estimate.weights, "transformer_elements", _broken)

        assert memory_estimate.cached_ltx_system_estimate({}, "owner/ltx", 4, find_local_dir=lambda reference: tmp_path) is None
