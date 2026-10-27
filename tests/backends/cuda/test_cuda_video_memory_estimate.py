"""Tests for the CUDA video backend's header-based system-memory estimate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.backends.cuda._helpers import _converted_folder, _patch_memory

from zvisiongenerator.backends.cuda.video import memory_estimate
from zvisiongenerator.backends.cuda.video.backend import _check_system_memory
from zvisiongenerator.utils.model_memory import estimate_ltx_cuda_system_memory

_GIB = 1024**3
_CONFIG = {"video_model_presets": {"ltx": {"diffusers": {"transformer": {"repo": "owner/ckpt", "filename": "ckpt.safetensors"}}}}}
_VERSION_CONFIG = {
    "video_model_presets": {
        "ltx": {
            "diffusers": {"transformer": {"repo": "owner/ckpt", "filename": "ckpt.safetensors"}},
            "versions": {"ltx25": {"aliases": ["ltx25-4"], "diffusers": {"default_repo": "owner/ltx25@abc123", "transformer": None, "quantize": {"ltx25-4": 4}}}},
        }
    }
}


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

    def test_a_version_without_a_single_file_transformer_reads_the_repos_own(self, header_counts, monkeypatch, tmp_path):
        def _unexpected(repo_id, filename, **kwargs):
            raise AssertionError("no single-file transformer is configured for the version")

        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _unexpected)

        estimate = memory_estimate.cached_ltx_system_estimate(_VERSION_CONFIG, "owner/ltx25@abc123", 4, model_name="ltx25-4", find_local_dir=lambda reference: tmp_path)

        assert estimate is not None and estimate.quantize == 4
        assert header_counts == [tmp_path / "transformer"]

    def test_the_family_single_file_transformer_still_serves_a_2_3_alias_in_a_version_config(self, header_counts, monkeypatch, tmp_path):
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo_id, filename: str(tmp_path / filename) if (repo_id, filename) == ("owner/ckpt", "ckpt.safetensors") else None)

        estimate = memory_estimate.cached_ltx_system_estimate(_VERSION_CONFIG, "owner/ltx", 4, model_name="ltx-4", find_local_dir=lambda reference: tmp_path)

        assert estimate is not None
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


class TestCachedHubFile:
    def test_a_plain_repo_is_looked_up_without_a_revision(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: calls.append((args, kwargs)) or "/cache/ckpt.safetensors")

        assert memory_estimate._cached_hub_file("owner/ckpt", "ckpt.safetensors") == Path("/cache/ckpt.safetensors")
        assert calls == [(("owner/ckpt", "ckpt.safetensors"), {})]

    def test_a_pinned_repo_is_looked_up_at_its_revision(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: calls.append((args, kwargs)) or "/cache/ckpt.safetensors")

        assert memory_estimate._cached_hub_file("owner/ckpt@abc123", "ckpt.safetensors") == Path("/cache/ckpt.safetensors")
        assert calls == [(("owner/ckpt", "ckpt.safetensors"), {"revision": "abc123"})]

    def test_a_file_missing_from_the_cache_is_unknown(self, monkeypatch):
        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: None)

        assert memory_estimate._cached_hub_file("owner/ckpt@abc123", "ckpt.safetensors") is None


class TestLtx25Estimate:
    def test_the_q4_estimate_from_the_models_header_counts_is_21_08_gib_taking_turns(self, monkeypatch, tmp_path):
        monkeypatch.setattr(memory_estimate.weights, "transformer_elements", lambda source: (18_555_076_608, 432_782_592))
        monkeypatch.setattr(memory_estimate.weights, "component_elements", lambda folder: 3_172_227_584)
        monkeypatch.setattr(memory_estimate.weights, "text_encoder_bytes", lambda folder: 13_019_512_880)

        estimate = memory_estimate.cached_ltx_system_estimate(_VERSION_CONFIG, "owner/ltx25@abc123", 4, model_name="ltx25-4", find_local_dir=lambda reference: tmp_path)

        assert estimate is not None
        assert round(estimate.taking_turns / _GIB, 2) == 21.08
        assert round(estimate.together / _GIB, 2) == 30.81


class TestConvertedFolderEstimate:
    """A converted LTX folder is estimated from its manifest, like the alias it was made from."""

    def test_a_stock_converted_folder_is_estimated_like_the_alias(self, monkeypatch, tmp_path):
        _patch_memory(monkeypatch, total_gib=64, block_matrices=123, rest=45, connectors=6, text_encoder=7)

        stock = memory_estimate.ltx_system_estimate(Path("/fake/repo"), Path("/fake/ckpt"), 4)
        converted = memory_estimate.converted_ltx_system_estimate(tmp_path, {"bits": 4, "block_matrices": 123, "transformer_rest": 45})

        assert converted == stock

    def test_a_converted_q8_folder_that_does_not_fit_advises_converting_again_at_q4(self, monkeypatch, tmp_path):
        _patch_memory(monkeypatch, total_gib=12)
        estimate = memory_estimate.converted_ltx_system_estimate(_converted_folder(tmp_path, bits=8), {"bits": 8, "block_matrices": 10 * _GIB, "transformer_rest": 0})

        with pytest.raises(RuntimeError, match="--quantize 4") as raised:
            _check_system_memory(estimate, 8, converted=True)
        assert "ltx-4" not in str(raised.value)
