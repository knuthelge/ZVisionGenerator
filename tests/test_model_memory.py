"""Tests for header-only model memory estimation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_memory
from zvisiongenerator.utils.model_memory import (
    classify_memory_fit,
    estimate_image_memory,
    estimate_ltx_mlx_memory,
    read_safetensors_totals,
)
from zvisiongenerator.utils.model_memory import WeightTotals

_GIB = 1024**3
_IMAGE_MARGIN = model_memory._IMAGE_WORKING_BYTES
_VIDEO_MARGIN = model_memory._VIDEO_WORKING_BYTES


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """Write a header-only safetensors file; the estimator never reads tensor data."""
    header = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]} for name, (dtype, shape) in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    payload = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


class TestReadSafetensorsTotals:
    def test_summarises_tensors_by_how_they_load_and_skips_metadata(self, tmp_path):
        path = _write_safetensors(
            tmp_path / "w.safetensors",
            {"matrix": ("BF16", [4, 8]), "bias": ("F32", [8]), "fp8": ("F8_E4M3", [16]), "packed": ("U32", [2, 4])},
        )

        assert read_safetensors_totals(path) == WeightTotals(float_matrix_elements=32, float_other_elements=8, packed_bytes=16 + 32, prequantized=True)

    def test_rejects_a_header_that_is_not_an_object(self, tmp_path):
        payload = json.dumps([1, 2]).encode("utf-8")
        path = tmp_path / "list.safetensors"
        path.write_bytes(struct.pack("<Q", len(payload)) + payload)

        with pytest.raises(ValueError, match="Malformed"):
            read_safetensors_totals(path)

    def test_rejects_unknown_dtypes_instead_of_guessing(self, tmp_path):
        path = _write_safetensors(tmp_path / "w.safetensors", {"a": ("F8_E8M0", [4])})

        with pytest.raises(ValueError, match="F8_E8M0"):
            read_safetensors_totals(path)

    def test_rejects_truncated_file(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        path.write_bytes(b"\x01")

        with pytest.raises(ValueError):
            read_safetensors_totals(path)


class TestEstimateImageMemory:
    def test_float_weights_load_as_bfloat16_and_vae_as_float32(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("F32", [1000, 1000])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"w": ("BF16", [500, 1000])})
        _write_safetensors(tmp_path / "vae" / "model.safetensors", {"w": ("BF16", [100, 100])})

        expected = 1_000_000 * 2 + 500_000 * 2 + 10_000 * 4 + _IMAGE_MARGIN
        assert estimate_image_memory(tmp_path) == {None: expected}

    def test_text_encoder_lm_head_is_not_counted(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"lm_head.weight": ("BF16", [10, 10])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"w": ("BF16", [10, 10]), "lm_head.weight": ("BF16", [1000, 1000])})

        # Only the text encoder's LM head is skipped: mflux never loads it; other components keep every tensor.
        assert estimate_image_memory(tmp_path) == {None: 100 * 2 + 100 * 2 + _IMAGE_MARGIN}

    @pytest.mark.parametrize("bits", [4, 8])
    def test_quantize_packs_two_dimensional_weights_only(self, tmp_path, bits):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"linear": ("BF16", [1024, 1024]), "norm": ("BF16", [1024])})

        linear = 1024 * 1024 * (bits / 8 + 4 / 64)
        estimates = estimate_image_memory(tmp_path, (None, bits))
        assert estimates == {None: (1024 * 1024 + 1024) * 2 + _IMAGE_MARGIN, bits: int(linear + 1024 * 2) + _IMAGE_MARGIN}

    def test_text_encoder_stays_bfloat16_when_the_loader_does_not_quantize_it(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"linear": ("BF16", [1024, 1024])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"linear": ("BF16", [1024, 1024])})

        transformer_q8 = int(1024 * 1024 * (8 / 8 + 4 / 64))
        estimates = estimate_image_memory(tmp_path, (8,), quantize_text_encoder=False)
        assert estimates == {8: transformer_q8 + 1024 * 1024 * 2 + _IMAGE_MARGIN}

    def test_fp8_and_prequantized_weights_stay_as_stored(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("F8_E4M3", [1000, 1000])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"packed": ("U32", [1000, 125]), "scales": ("BF16", [1000, 16])})

        stored = 1_000_000 + 1000 * 125 * 4 + 1000 * 16 * 2
        assert estimate_image_memory(tmp_path, (4,)) == {4: stored + _IMAGE_MARGIN}

    def test_returns_none_without_weights_or_with_unreadable_weights(self, tmp_path):
        assert estimate_image_memory(tmp_path) is None
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "broken.safetensors").write_bytes(b"")
        assert estimate_image_memory(tmp_path) is None

    def test_full_repo_download_counts_only_model_index_components(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Model"], "scheduler": ["diffusers", "FlowMatchScheduler"]}))
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [1000, 1000])})
        _write_safetensors(tmp_path / "flux1-dev.safetensors", {"w": ("BF16", [1000, 1000])})  # single-file duplicate
        _write_safetensors(tmp_path / "extras" / "other.safetensors", {"w": ("BF16", [1000, 1000])})

        assert estimate_image_memory(tmp_path) == {None: 2_000_000 + _IMAGE_MARGIN}

    def test_unknown_dtype_is_not_estimated(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("F4", [1000, 1000])})

        assert estimate_image_memory(tmp_path) is None


class TestEstimateLtxMlxMemory:
    def _make_ltx(self, root: Path, *, transformer_name: str = "transformer-distilled.safetensors") -> tuple[Path, Path]:
        model_dir = root / "ltx"
        _write_safetensors(model_dir / transformer_name, {"w": ("U32", [1000, 1000])})  # 4 MB packed
        _write_safetensors(model_dir / "connector.safetensors", {"w": ("BF16", [1000, 1000])})  # 2 MB
        _write_safetensors(model_dir / "vae_decoder.safetensors", {"w": ("F32", [500, 1000])})  # 1 MB as bf16
        text_encoder_dir = root / "gemma"
        _write_safetensors(text_encoder_dir / "model-00001-of-00001.safetensors", {"w": ("U32", [1000, 250])})  # 1 MB
        return model_dir, text_encoder_dir

    def test_low_memory_peak_keeps_the_connector_loaded_with_the_larger_stage(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        gemma, connector, denoise = 1_000_000, 2_000_000, 4_000_000 + 1_000_000
        # The pipeline frees Gemma before the transformer loads, but the connector stays resident throughout.
        assert estimate_ltx_mlx_memory(model_dir, text_encoder_dir) == connector + max(gemma, denoise) + _VIDEO_MARGIN

    def test_without_low_memory_every_stage_stays_loaded(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        assert estimate_ltx_mlx_memory(model_dir, text_encoder_dir, low_memory=False) == 3_000_000 + 5_000_000 + _VIDEO_MARGIN

    def test_returns_none_for_non_ltx_layout(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [10, 10])})

        assert estimate_ltx_mlx_memory(tmp_path, tmp_path) is None


@pytest.mark.parametrize(
    ("required", "expected"),
    [(11 * _GIB, "fits"), (11 * _GIB + 1, "tight"), (15 * _GIB, "tight"), (15 * _GIB + 1, "too_large")],
)
def test_classify_memory_fit(required, expected):
    assert classify_memory_fit(required, 10 * _GIB) == expected
