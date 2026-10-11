"""Tests for header-only model memory estimation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_memory
from zvisiongenerator.utils.model_memory import (
    CudaMemoryEstimate,
    MemoryBudget,
    classify_cuda_memory_fit,
    classify_memory_fit,
    estimate_cuda_image_memory,
    estimate_image_memory,
    estimate_ltx_mlx_memory,
    estimate_ltx_mlx_window_memory,
    read_safetensors_totals,
    window_memory_fit,
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

    def test_krea2_text_encoder_stays_bfloat16_when_quantized(self, tmp_path):
        index = {"_class_name": "Krea2Pipeline", "transformer": ["diffusers", "Krea2Transformer2DModel"], "text_encoder": ["transformers", "Qwen3VLModel"]}
        (tmp_path / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"linear": ("BF16", [1024, 1024])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"linear": ("BF16", [1024, 1024])})

        transformer_q8 = int(1024 * 1024 * (8 / 8 + 4 / 64))
        estimates = estimate_image_memory(tmp_path, (8,))
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


class TestEstimateLtxMlxWindowMemory:
    _make_ltx = TestEstimateLtxMlxMemory._make_ltx
    # The measured reference window: 73 frames (10 latent frames) at 704x448 (22 x 14 tokens per latent frame).
    _REFERENCE = {"width": 704, "height": 448, "frames": 73}

    def test_equals_the_plain_estimate_at_the_reference_window(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        assert model_memory._REFERENCE_TOKENS == 3080
        assert estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, **self._REFERENCE) == estimate_ltx_mlx_memory(model_dir, text_encoder_dir)

    def test_a_smaller_window_keeps_the_allowance(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        small = estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, width=512, height=256, frames=33)

        assert small == estimate_ltx_mlx_memory(model_dir, text_encoder_dir)

    def test_a_larger_window_scales_the_working_allowance_by_its_tokens(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)
        base = estimate_ltx_mlx_memory(model_dir, text_encoder_dir)

        # 145 frames is 19 latent frames, 1.9x the reference window's 10.
        large = estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, width=704, height=448, frames=145)

        assert large == base - _VIDEO_MARGIN + int(_VIDEO_MARGIN * (19 * 22 * 14) / 3080)
        assert large > base

    def test_grows_with_the_window(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)
        sizes = [estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, width=1280, height=704, frames=frames) for frames in (73, 121, 241)]

        assert sizes == sorted(sizes) and len(set(sizes)) == 3

    def test_low_memory_off_keeps_every_stage_resident(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        resident = estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, low_memory=False, **self._REFERENCE)

        assert resident == estimate_ltx_mlx_memory(model_dir, text_encoder_dir, low_memory=False)

    def test_returns_none_for_non_ltx_layout(self, tmp_path):
        assert estimate_ltx_mlx_window_memory(tmp_path, tmp_path, **self._REFERENCE) is None


class TestWindowMemoryFit:
    _make_ltx = TestEstimateLtxMlxMemory._make_ltx
    _WINDOW = {"width": 704, "height": 448, "frames": 73, "low_memory": True}

    @pytest.fixture(autouse=True)
    def _text_encoder_repo(self, monkeypatch):
        monkeypatch.setattr(model_memory, "ltx_mlx_text_encoder_repo", lambda: "example/gemma")

    def _finder(self, model_dir: Path | None, text_encoder_dir: Path | None):
        return lambda reference: {"example/model": model_dir, "example/gemma": text_encoder_dir}.get(reference)

    def _required(self, model_dir: Path, text_encoder_dir: Path) -> int:
        required = estimate_ltx_mlx_window_memory(model_dir, text_encoder_dir, width=704, height=448, frames=73)
        assert required is not None
        return required

    def test_classifies_the_window_estimate_against_the_budget(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)
        required = self._required(model_dir, text_encoder_dir)
        finder = self._finder(model_dir, text_encoder_dir)

        fits = window_memory_fit("example/model", budget_bytes=required, find_local_dir=finder, **self._WINDOW)
        too_large = window_memory_fit("example/model", budget_bytes=required // 2, find_local_dir=finder, **self._WINDOW)

        assert fits == {"status": "fits", "required_gb": round(required / _GIB, 1), "budget_gb": round(required / _GIB, 1)}
        assert too_large is not None and too_large["status"] == "too_large"

    def test_a_tight_budget_is_tight(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)
        required = self._required(model_dir, text_encoder_dir)

        result = window_memory_fit("example/model", budget_bytes=int(required / 1.3), find_local_dir=self._finder(model_dir, text_encoder_dir), **self._WINDOW)

        assert result is not None and result["status"] == "tight"

    def test_looks_the_model_and_its_text_encoder_up_by_reference(self, tmp_path):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)
        seen: list[str] = []

        def finder(reference: str) -> Path | None:
            seen.append(reference)
            return self._finder(model_dir, text_encoder_dir)(reference)

        window_memory_fit("example/model", budget_bytes=_GIB, find_local_dir=finder, **self._WINDOW)

        assert seen == ["example/model", "example/gemma"]

    @pytest.mark.parametrize("budget", [None, 0])
    def test_no_budget_gives_none(self, tmp_path, budget):
        model_dir, text_encoder_dir = self._make_ltx(tmp_path)

        assert window_memory_fit("example/model", budget_bytes=budget, find_local_dir=self._finder(model_dir, text_encoder_dir), **self._WINDOW) is None

    def test_a_model_that_is_not_downloaded_gives_none(self, tmp_path):
        _, text_encoder_dir = self._make_ltx(tmp_path)

        assert window_memory_fit("example/model", budget_bytes=_GIB, find_local_dir=self._finder(None, text_encoder_dir), **self._WINDOW) is None

    def test_a_missing_text_encoder_gives_none(self, tmp_path):
        model_dir, _ = self._make_ltx(tmp_path)

        assert window_memory_fit("example/model", budget_bytes=_GIB, find_local_dir=self._finder(model_dir, None), **self._WINDOW) is None

    def test_a_model_without_the_ltx_layout_gives_none(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [10, 10])})

        assert window_memory_fit("example/model", budget_bytes=_GIB, find_local_dir=self._finder(tmp_path, tmp_path), **self._WINDOW) is None


@pytest.mark.parametrize(
    ("required", "expected"),
    [(11 * _GIB, "fits"), (11 * _GIB + 1, "tight"), (15 * _GIB, "tight"), (15 * _GIB + 1, "too_large")],
)
def test_classify_memory_fit(required, expected):
    assert classify_memory_fit(required, 10 * _GIB) == expected


class TestEstimateCudaImageMemory:
    """Mirror DiffusersBackend: unquantized and q8 stream from system memory, q4 holds NF4 on the GPU."""

    def _model(self, tmp_path: Path, pipeline: str = "ZImagePipeline") -> Path:
        index = {"_class_name": pipeline, "transformer": ["diffusers", "T"], "text_encoder": ["transformers", "E"], "vae": ["diffusers", "V"]}
        (tmp_path / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"linear": ("BF16", [4096, 1024]), "norm": ("BF16", [1024])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"linear": ("BF16", [1024, 1024])})
        _write_safetensors(tmp_path / "vae" / "model.safetensors", {"conv": ("BF16", [64, 64, 3, 3])})
        return tmp_path

    def test_streamed_levels_hold_weights_in_system_memory(self, tmp_path):
        estimates = estimate_cuda_image_memory(self._model(tmp_path), (None, 8))

        vae = 64 * 64 * 9 * 2
        bf16 = (4096 * 1024 + 1024 + 1024 * 1024) * 2
        fp8 = (4096 * 1024 + 1024 * 1024) + 1024 * 2
        assert estimates[None].gpu_bytes == estimates[8].gpu_bytes == vae + model_memory._CUDA_STREAMED_GPU_BYTES
        assert estimates[None].system_bytes == bf16 + model_memory._CUDA_SYSTEM_WORKING_BYTES
        assert estimates[8].system_bytes == fp8 + model_memory._CUDA_SYSTEM_WORKING_BYTES

    def test_nf4_holds_transformer_and_text_encoder_on_the_gpu(self, tmp_path):
        estimate = estimate_cuda_image_memory(self._model(tmp_path), (4,))[4]

        nf4 = (4096 * 1024 + 1024 * 1024) * model_memory._NF4_BYTES_PER_WEIGHT + 1024 * 2
        assert estimate.gpu_bytes == int(nf4 + 64 * 64 * 9 * 2 + model_memory._CUDA_RESIDENT_WORKING_BYTES)
        assert estimate.system_bytes == model_memory._CUDA_SYSTEM_WORKING_BYTES

    def test_krea2_nf4_holds_one_component_on_the_gpu_at_a_time(self, tmp_path):
        estimate = estimate_cuda_image_memory(self._model(tmp_path, "Krea2Pipeline"), (4,))[4]

        transformer = 4096 * 1024 * model_memory._NF4_BYTES_PER_WEIGHT + 1024 * 2
        text_encoder = 1024 * 1024 * model_memory._NF4_BYTES_PER_WEIGHT
        assert estimate.gpu_bytes == int(transformer + 64 * 64 * 9 * 2 + model_memory._CUDA_RESIDENT_WORKING_BYTES)
        assert estimate.system_bytes == int(text_encoder + model_memory._CUDA_SYSTEM_WORKING_BYTES)

    def test_stored_q4_copy_holds_its_weights_on_the_gpu(self, tmp_path):
        (tmp_path / "stored").mkdir()
        (tmp_path / "plain").mkdir()
        model_dir = self._model(tmp_path / "stored")
        (model_dir / "ziv-quant.json").write_text(json.dumps({"version": 1, "bits": 4}), encoding="utf-8")

        estimates = estimate_cuda_image_memory(model_dir)

        assert estimates[None] == estimate_cuda_image_memory(self._model(tmp_path / "plain"), (4,))[4]

    def test_returns_none_without_weights(self, tmp_path):
        assert estimate_cuda_image_memory(tmp_path) is None


@pytest.mark.parametrize(
    ("gpu", "system", "expected"),
    [
        (4 * _GIB, 20 * _GIB, "fits"),
        (4 * _GIB, 25 * _GIB, "tight"),  # over 80% of system memory: weights stream from disk
        (9.5 * _GIB, 4 * _GIB, "tight"),  # over 90% of the GPU
        (10 * _GIB + 1, 4 * _GIB, "too_large"),
    ],
)
def test_classify_cuda_memory_fit(gpu, system, expected):
    assert classify_cuda_memory_fit(CudaMemoryEstimate(int(gpu), int(system)), MemoryBudget(10 * _GIB, 30 * _GIB)) == expected
