"""Tests for header-only model memory estimation."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_memory
from zvisiongenerator.utils.model_memory import (
    CudaMemoryEstimate,
    MemoryBudget,
    classify_cuda_memory_fit,
    classify_memory_fit,
    clip_memory_fit,
    estimate_cuda_image_memory,
    estimate_image_memory,
    estimate_ltx_cuda_clip_memory,
    estimate_ltx_cuda_system_memory,
    estimate_ltx_mlx_clip_memory,
    ltx_decode_budget_bytes,
    ltx_video_tokens,
    read_safetensors_totals,
)
from zvisiongenerator.utils.model_memory import WeightTotals

_GIB = 1024**3
_IMAGE_MARGIN = model_memory._IMAGE_WORKING_BYTES


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

        with pytest.raises(ValueError, match="not a safetensors file"):
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

    def test_fp8_matrices_are_one_byte_unquantized_and_pack_when_quantized_while_prequantized_weights_stay_as_stored(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("F8_E4M3", [1000, 1000])})
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"packed": ("U32", [1000, 125]), "scales": ("BF16", [1000, 16])})

        prequantized = 1000 * 125 * 4 + 1000 * 16 * 2
        estimates = estimate_image_memory(tmp_path, (None, 4))
        assert estimates[None] == 1_000_000 + prequantized + _IMAGE_MARGIN
        assert estimates[4] == int(1_000_000 * (4 / 8 + 4 / 64) + prequantized) + _IMAGE_MARGIN

    def test_ideogram4_adds_its_own_working_memory_at_every_level(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline", "transformer": ["diffusers", "Ideogram4Transformer2DModel"]}), encoding="utf-8")
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("F8_E4M3", [1000, 1000])})

        estimates = estimate_image_memory(tmp_path, (None, 4))

        assert estimates[None] == 1_000_000 + model_memory._IDEOGRAM4_MLX_WORKING_BYTES
        assert estimates[4] == int(1_000_000 * (4 / 8 + 4 / 64)) + model_memory._IDEOGRAM4_MLX_WORKING_BYTES

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


def _write_sized(path: Path, gib: float) -> Path:
    """Write a header-only safetensors file whose weights load as *gib* GiB of bfloat16."""
    return _write_safetensors(path, {"w": ("BF16", [int(gib * _GIB / 2 / 1024), 1024])})


_DECODER_GIB = 0.758


def _make_ltx(root: Path, *, gemma: float, connector: float, transformer: float, upsampler: float | None = None) -> tuple[Path, Path]:
    """Build a fake LTX MLX model and text-encoder directory with the given weight sizes in GiB."""
    model_dir = root / "ltx"
    _write_sized(model_dir / "transformer-distilled-1.1.safetensors", transformer)
    _write_sized(model_dir / "connector.safetensors", connector)
    _write_sized(model_dir / "vae_decoder.safetensors", _DECODER_GIB)  # Counted by the measured decode curve; adds to the tiler's cap.
    if upsampler is not None:
        _write_sized(model_dir / "spatial_upscaler_x2_v1_1.safetensors", upsampler)
    text_encoder_dir = root / "gemma"
    _write_sized(text_encoder_dir / "model-00001-of-00001.safetensors", gemma)
    return model_dir, text_encoder_dir


def _make_ltx_25(root: Path, *, encoder: float) -> tuple[Path, Path]:
    """Build a fake LTX-2.5 MLX pack bundling its text encoder, plus a Gemma dir holding the same encoder file."""
    model_dir = root / "ltx25"
    _write_sized(model_dir / "transformer-distilled-1.1.safetensors", 10.535)
    _write_sized(model_dir / "connector.safetensors", 5.909)
    _write_sized(model_dir / "vae_decoder_conv.safetensors", _DECODER_GIB)
    _write_sized(model_dir / "text_encoder.safetensors", encoder)
    (model_dir / "text_encoder_config.json").write_text("{}", encoding="utf-8")
    text_encoder_dir = root / "gemma"
    _write_sized(text_encoder_dir / "model-00001-of-00001.safetensors", encoder)
    return model_dir, text_encoder_dir


# The weights (GiB) and the peaks MLX measured per run on the machine the estimate was fit on (32 GiB, q4 distilled).
_MEASURED_WEIGHTS = {"gemma": 7.477, "connector": 5.909, "transformer": 10.535, "upsampler": 0.927}
_MEASURED_UNIFIED = 32 * _GIB
# (width, height, frames, low_memory, upscale, measured overall peak in GiB)
_MEASURED_RUNS = [
    pytest.param(512, 256, 121, True, False, 14.23, id="run1-small"),
    pytest.param(704, 448, 121, True, False, 14.81, id="run2-reference"),
    pytest.param(704, 448, 241, True, False, 15.91, id="run3-long"),
    pytest.param(960, 512, 121, True, False, 15.39, id="run4-large"),
    pytest.param(704, 448, 121, False, False, 35.14, id="run5-resident"),
    pytest.param(704, 448, 121, True, True, 14.80, id="run6-upscale"),
]


class TestEstimateLtxMlxClipMemory:
    @pytest.mark.parametrize(("width", "height", "frames", "low_memory", "upscale", "measured"), _MEASURED_RUNS)
    def test_reproduces_the_measured_peaks_without_going_below_them(self, tmp_path, width, height, frames, low_memory, upscale, measured):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)

        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=width, height=height, frames=frames, low_memory=low_memory, upscale=upscale, unified_bytes=_MEASURED_UNIFIED, env={})

        assert required is not None
        # Header sizes are rounded to the GiB thousandth, so allow for that; the fit is otherwise never under.
        # A long clip reaches the tiler's cap, where the decoder weights sit on top of its budget (up to 0.76 GiB above what was measured).
        assert measured - 0.02 <= required / _GIB <= measured + 1.4

    def test_a_bigger_clip_needs_more_memory(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)

        sizes = [
            estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=width, height=height, frames=frames, low_memory=True, env={})
            for width, height, frames in ((512, 256, 121), (704, 448, 121), (704, 448, 241), (1280, 704, 241))
        ]

        assert sizes == sorted(sizes) and len(set(sizes)) == 4

    def test_the_decode_peak_is_capped_by_half_of_unified_memory(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "env": {}}

        uncapped = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        capped = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, unified_bytes=32 * _GIB, **clip)
        small_mac = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, unified_bytes=24 * _GIB, **clip)

        assert uncapped is not None and uncapped / _GIB > 16 + _DECODER_GIB
        assert capped == pytest.approx((16 + _DECODER_GIB) * _GIB, abs=2**20)
        assert small_mac == pytest.approx((12 + _DECODER_GIB) * _GIB, abs=2**20)

    def test_a_capped_decode_adds_the_decoder_weights_to_the_tiler_budget(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        _write_sized(model_dir / "audio_vae.safetensors", 0.1)
        _write_sized(model_dir / "vocoder.safetensors", 0.2)
        _write_sized(model_dir / "vae_encoder.safetensors", 5)  # Not loaded for a decode.
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "env": {}, "unified_bytes": 16 * _GIB}

        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert required == pytest.approx((8 + _DECODER_GIB + 0.1 + 0.2) * _GIB, abs=2**20)

    def test_the_decode_budget_variable_overrides_half_of_unified_memory(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "unified_bytes": 64 * _GIB}

        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, env={"LTX2_VAE_DECODE_BUDGET_GB": "10"}, **clip)

        assert required == pytest.approx((10 + _DECODER_GIB) * _GIB, abs=2**20)

    def test_the_process_environment_is_read_when_none_is_given(self, tmp_path, monkeypatch):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "9")

        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=1280, height=704, frames=241, low_memory=True)

        assert required == pytest.approx((9 + _DECODER_GIB) * _GIB, abs=2**20)

    def test_upscale_adds_the_upsample_stage_only_when_the_model_has_an_upsampler(self, tmp_path):
        with_file, text_encoder_dir = _make_ltx(tmp_path / "with", gemma=1, connector=1, transformer=1, upsampler=20)
        without_file, _ = _make_ltx(tmp_path / "without", gemma=1, connector=1, transformer=1)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, "env": {}}

        plain = estimate_ltx_mlx_clip_memory(with_file, text_encoder_dir, **clip)
        upscaled = estimate_ltx_mlx_clip_memory(with_file, text_encoder_dir, upscale=True, **clip)
        no_upsampler = estimate_ltx_mlx_clip_memory(without_file, text_encoder_dir, upscale=True, **clip)

        assert upscaled == pytest.approx((1 + 20 + 0.5) * _GIB, abs=1)  # transformer + upsampler + working memory
        assert plain is not None and plain < upscaled
        assert no_upsampler == plain

    def test_without_low_memory_every_weight_stays_resident(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=2, connector=3, transformer=4, upsampler=1)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": False, "env": {}}

        resident = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        upscaled = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, upscale=True, **clip)

        # 4928 tokens: denoise working memory 1.7 GiB, decode 12.9 + 0.00067 * 2880 less the 2.83 GiB already counted.
        working = 12.9 + 0.00067 * (4928 - 2048) - 2.83
        assert resident == pytest.approx((2 + 3 + 4 + working) * _GIB, abs=1)
        assert upscaled == pytest.approx((2 + 3 + 4 + 1 + working) * _GIB, abs=1)

    def test_without_low_memory_the_working_memory_is_the_larger_of_denoise_and_decode(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=2, connector=3, transformer=4)

        # A tight decode budget leaves the denoise working memory (1.7 GiB at 4928 tokens) as the larger term.
        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=704, height=448, frames=121, low_memory=False, env={"LTX2_VAE_DECODE_BUDGET_GB": "3"})

        assert required == pytest.approx((2 + 3 + 4 + 1.7) * _GIB, abs=1)

    def test_ltx_2_3_estimates_are_pinned_to_the_bytes(self, tmp_path):
        """The 2.5 pack support leaves every 2.3 estimate unchanged."""
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)
        _write_sized(model_dir / "audio_vae.safetensors", 0.1)
        _write_sized(model_dir / "vocoder.safetensors", 0.2)
        clip = {"width": 768, "height": 448, "frames": 121, "env": {}}

        def estimate(low_memory: bool, unified_bytes: int | None) -> int | None:
            return estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, low_memory=low_memory, unified_bytes=unified_bytes, **clip)

        assert estimate(True, None) == 16245456099
        assert estimate(True, _MEASURED_UNIFIED) == 16245456099
        assert estimate(True, 16 * _GIB) == 15285786214
        assert estimate(False, None) == 38891742353
        assert estimate(False, _MEASURED_UNIFIED) == 38891742353
        assert estimate(False, 16 * _GIB) == 32372236206

    def test_a_bundled_pack_estimates_without_a_text_encoder_dir_and_matches_a_gemma_dir_holding_the_same_file(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx_25(tmp_path, encoder=7.477)
        clip = {"width": 768, "height": 448, "frames": 121, "low_memory": True, "env": {}}

        bundled = estimate_ltx_mlx_clip_memory(model_dir, None, **clip)
        with_dir = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert bundled is not None
        assert bundled == with_dir
        # 5376 tokens: the decode stage (12.9 + 0.00067 * 3328 GiB) is the peak, about 15.1 GiB.
        assert bundled / _GIB == pytest.approx(12.9 + 0.00067 * (5376 - 2048), abs=0.01)

    def test_the_bundled_encoder_is_counted_resident_without_low_memory(self, tmp_path):
        model_dir, _ = _make_ltx_25(tmp_path, encoder=3)
        _write_sized(model_dir / "connector.safetensors", 2)
        _write_sized(model_dir / "transformer-distilled-1.1.safetensors", 4)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": False, "env": {}}

        resident = estimate_ltx_mlx_clip_memory(model_dir, None, **clip)

        assert resident == pytest.approx((3 + 2 + 4 + 12.9 + 0.00067 * (4928 - 2048) - 2.83) * _GIB, abs=1)

    def test_a_2_3_pack_without_a_text_encoder_dir_gives_none(self, tmp_path):
        model_dir, _ = _make_ltx(tmp_path, gemma=1, connector=1, transformer=1)

        assert estimate_ltx_mlx_clip_memory(model_dir, None, width=704, height=448, frames=121, low_memory=True) is None

    def test_the_conv_decoder_bytes_are_counted_in_the_capped_decode(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "env": {}, "unified_bytes": 16 * _GIB}
        plain = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        _write_sized(model_dir / "vae_decoder_conv.safetensors", 0.5)

        with_conv = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert plain == pytest.approx((8 + _DECODER_GIB) * _GIB, abs=2**20)
        assert with_conv == pytest.approx((8 + 0.5) * _GIB, abs=2**20)  # The conv decoder replaces vae_decoder.

    def test_returns_none_for_non_ltx_layout(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [10, 10])})

        assert estimate_ltx_mlx_clip_memory(tmp_path, tmp_path, width=704, height=448, frames=121, low_memory=True) is None

    def test_returns_none_for_unreadable_weights(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=1, connector=1, transformer=1)
        (model_dir / "connector.safetensors").write_bytes(b"")

        assert estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=704, height=448, frames=121, low_memory=True) is None


class TestLtxDecodeBudget:
    def test_defaults_to_half_of_unified_memory(self):
        assert ltx_decode_budget_bytes(32 * _GIB, {}) == 16 * _GIB

    def test_the_variable_is_in_gib_and_wins(self):
        assert ltx_decode_budget_bytes(32 * _GIB, {"LTX2_VAE_DECODE_BUDGET_GB": "20.5"}) == int(20.5 * _GIB)
        assert ltx_decode_budget_bytes(None, {"LTX2_VAE_DECODE_BUDGET_GB": "8"}) == 8 * _GIB

    def test_is_unknown_without_either(self):
        assert ltx_decode_budget_bytes(None, {}) is None

    def test_an_unusable_variable_falls_back_to_the_default(self):
        assert ltx_decode_budget_bytes(32 * _GIB, {"LTX2_VAE_DECODE_BUDGET_GB": "lots"}) == 16 * _GIB


def test_ltx_video_tokens_count_latent_frames_times_32_pixel_cells():
    assert ltx_video_tokens(704, 448, 121) == 16 * 22 * 14
    assert ltx_video_tokens(512, 256, 121) == 2048


class TestClipMemoryFit:
    _CLIP = {"width": 704, "height": 448, "frames": 121, "low_memory": True}

    @pytest.fixture(autouse=True)
    def _text_encoder_repo(self, monkeypatch):
        monkeypatch.setattr(model_memory, "ltx_mlx_text_encoder_repo", lambda: "example/gemma")

    def _finder(self, model_dir: Path | None, text_encoder_dir: Path | None):
        return lambda reference: {"example/model": model_dir, "example/gemma": text_encoder_dir}.get(reference)

    def _required(self, model_dir: Path, text_encoder_dir: Path) -> int:
        required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, env={}, **self._CLIP)
        assert required is not None
        return required

    def test_classifies_the_clip_estimate_against_the_budget(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)
        required = self._required(model_dir, text_encoder_dir)
        finder = self._finder(model_dir, text_encoder_dir)

        fits = clip_memory_fit("example/model", budget=MemoryBudget(required), find_local_dir=finder, env={}, **self._CLIP)
        too_large = clip_memory_fit("example/model", budget=MemoryBudget(required // 2), find_local_dir=finder, env={}, **self._CLIP)

        assert fits == {"status": "fits", "required_gb": round(required / _GIB, 1), "budget_gb": round(required / _GIB, 1)}
        assert too_large is not None and too_large["status"] == "too_large"

    def test_a_tight_budget_is_tight(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)
        required = self._required(model_dir, text_encoder_dir)

        result = clip_memory_fit("example/model", budget=MemoryBudget(int(required / 1.3)), find_local_dir=self._finder(model_dir, text_encoder_dir), env={}, **self._CLIP)

        assert result is not None and result["status"] == "tight"

    def test_the_budgets_unified_memory_caps_the_decode_peak(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "env": {}}
        finder = self._finder(model_dir, text_encoder_dir)

        capped = clip_memory_fit("example/model", budget=MemoryBudget(20 * _GIB, unified_bytes=24 * _GIB), find_local_dir=finder, **clip)
        unknown = clip_memory_fit("example/model", budget=MemoryBudget(20 * _GIB), find_local_dir=finder, **clip)

        assert capped is not None and capped["required_gb"] == round(12 + _DECODER_GIB, 1)
        assert unknown is not None and unknown["required_gb"] > 16

    @pytest.mark.parametrize("budget", [None, MemoryBudget(0)])
    def test_no_memory_budget_means_no_estimate_and_no_filesystem_lookup(self, budget):
        finder = MagicMock(return_value=None)

        assert clip_memory_fit("example/model", budget=budget, find_local_dir=finder, **self._CLIP) is None
        finder.assert_not_called()

    def test_upscale_is_passed_to_the_estimate(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=1, connector=1, transformer=1, upsampler=20)
        finder = self._finder(model_dir, text_encoder_dir)

        plain = clip_memory_fit("example/model", budget=MemoryBudget(40 * _GIB), find_local_dir=finder, env={}, **self._CLIP)
        upscaled = clip_memory_fit("example/model", budget=MemoryBudget(40 * _GIB), find_local_dir=finder, upscale=True, env={}, **self._CLIP)

        assert plain is not None and upscaled is not None
        assert upscaled["required_gb"] > plain["required_gb"]

    def test_looks_the_model_and_its_text_encoder_up_by_reference(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)
        seen: list[str] = []

        def finder(reference: str) -> Path | None:
            seen.append(reference)
            return self._finder(model_dir, text_encoder_dir)(reference)

        clip_memory_fit("example/model", budget=MemoryBudget(_GIB), find_local_dir=finder, env={}, **self._CLIP)

        assert seen == ["example/model", "example/gemma"]

    def test_a_model_that_is_not_downloaded_gives_none(self, tmp_path):
        _, text_encoder_dir = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)

        assert clip_memory_fit("example/model", budget=MemoryBudget(_GIB), find_local_dir=self._finder(None, text_encoder_dir), **self._CLIP) is None

    def test_a_bundled_pack_needs_no_gemma_lookup(self, tmp_path):
        model_dir, _ = _make_ltx_25(tmp_path, encoder=7.477)
        seen: list[str] = []

        def finder(reference: str) -> Path | None:
            seen.append(reference)
            return self._finder(model_dir, None)(reference)

        result = clip_memory_fit("example/model", budget=MemoryBudget(40 * _GIB), find_local_dir=finder, env={}, **self._CLIP)

        assert result is not None and result["status"] == "fits"
        assert seen == ["example/model"]

    def test_a_missing_text_encoder_gives_none(self, tmp_path):
        model_dir, _ = _make_ltx(tmp_path, **_MEASURED_WEIGHTS)

        assert clip_memory_fit("example/model", budget=MemoryBudget(_GIB), find_local_dir=self._finder(model_dir, None), **self._CLIP) is None

    def test_a_model_without_the_ltx_layout_gives_none(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [10, 10])})

        assert clip_memory_fit("example/model", budget=MemoryBudget(_GIB), find_local_dir=self._finder(tmp_path, tmp_path), **self._CLIP) is None


class TestLtxClipMemoryFitGolden:
    """G5: the LTX MLX clip estimates, pinned so the family dispatch cannot move them."""

    _FAMILY_ARGS = [pytest.param({}, id="default"), pytest.param({"family": "ltx"}, id="ltx")]
    _BUDGET = MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB)
    _CLIPS = [
        ({"width": 704, "height": 448, "frames": 121, "low_memory": True}, {"status": "fits", "required_gb": 14.8, "budget_gb": 24.0}),
        ({"width": 960, "height": 512, "frames": 241, "low_memory": True}, {"status": "fits", "required_gb": 16.8, "budget_gb": 24.0}),
        ({"width": 704, "height": 448, "frames": 121, "low_memory": False, "upscale": True}, {"status": "too_large", "required_gb": 36.8, "budget_gb": 24.0}),
    ]

    @pytest.fixture(autouse=True)
    def _text_encoder_repo(self, monkeypatch):
        monkeypatch.setattr(model_memory, "ltx_mlx_text_encoder_repo", lambda: "example/gemma")

    def test_a_family_without_an_estimate_gives_none_without_a_filesystem_lookup(self):
        finder = MagicMock(return_value=None)

        result = clip_memory_fit("example/model", budget=self._BUDGET, find_local_dir=finder, env={}, family="h3", **self._CLIPS[0][0])

        assert result is None
        finder.assert_not_called()

    @pytest.mark.parametrize("family", _FAMILY_ARGS)
    def test_three_clips_keep_their_estimates(self, tmp_path, family):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=7.477, connector=5.909, transformer=10.535, upsampler=0.927)
        finder = lambda reference: {"example/model": model_dir, "example/gemma": text_encoder_dir}.get(reference)  # noqa: E731

        results = [clip_memory_fit("example/model", budget=self._BUDGET, find_local_dir=finder, env={}, **clip, **family) for clip, _ in self._CLIPS]

        assert results == [expected for _, expected in self._CLIPS]


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

    def test_ideogram4_streams_its_transformer_at_every_level_and_counts_the_nf4_text_encoder_as_stored(self, tmp_path):
        model_dir = self._model(tmp_path, "Ideogram4Pipeline")
        _write_safetensors(model_dir / "text_encoder" / "model.safetensors", {"packed": ("U8", [512 * 1024, 1]), "embed": ("BF16", [1024, 64])})

        estimates = estimate_cuda_image_memory(model_dir, (None, 4, 8))

        vae = 64 * 64 * 9 * 2
        text_encoder = 512 * 1024 + 1024 * 64 * 2
        gpu = max(text_encoder + model_memory._CUDA_RESIDENT_WORKING_BYTES, model_memory._CUDA_STREAMED_GPU_BYTES) + vae
        assert estimates[None].gpu_bytes == estimates[4].gpu_bytes == estimates[8].gpu_bytes == gpu
        transformer = {None: 4096 * 1024 * 2, 4: 4096 * 1024 * model_memory._NF4_BYTES_PER_WEIGHT, 8: 4096 * 1024}
        for level, weights in transformer.items():
            assert estimates[level].system_bytes == int(weights + 1024 * 2 + text_encoder + model_memory._CUDA_SYSTEM_WORKING_BYTES)

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


class TestEstimateLtxCudaSystemMemory:
    _COUNTS = {"block_matrices": 18 * _GIB, "transformer_rest": _GIB // 2, "connectors": 3 * _GIB, "text_encoder": 12 * _GIB}

    def test_q8_stores_the_transformer_layers_text_encoder_and_connectors_in_one_byte(self):
        q8 = estimate_ltx_cuda_system_memory(**self._COUNTS, quantize=8)

        assert (q8.transformer_bytes, q8.text_encoder_bytes, q8.resident_bytes) == (18 * _GIB + 2 * (_GIB // 2), 12 * _GIB, 3 * _GIB)

    def test_q4_packs_only_the_block_matrices(self):
        q4 = estimate_ltx_cuda_system_memory(**self._COUNTS, quantize=4)

        assert q4.transformer_bytes == int(18 * _GIB * model_memory._NF4_BYTES_PER_WEIGHT + 2 * (_GIB // 2))
        assert q4.text_encoder_bytes == 12 * _GIB

    def test_taking_turns_holds_the_larger_one_and_together_holds_both_with_the_connectors(self):
        runtime = model_memory._LTX_CUDA_RUNTIME_BYTES
        estimate = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)

        assert estimate.taking_turns == 15 * _GIB + runtime
        assert estimate.together == 25 * _GIB + runtime

    def test_the_ltx_2_5_q4_estimate_from_its_header_counts(self):
        estimate = estimate_ltx_cuda_system_memory(block_matrices=18_555_076_608, transformer_rest=432_782_592, connectors=3_172_227_584, text_encoder=13_019_512_880, quantize=4)

        assert round(estimate.taking_turns / _GIB, 2) == 21.08
        assert round(estimate.together / _GIB, 2) == 30.81

    @pytest.mark.parametrize(
        ("quantize", "taking_turns", "together"),
        [(4, 22_815_172_976, 33_261_754_346), (8, 29_037_278_208, 42_237_772_656)],
    )
    def test_the_ltx_2_3_estimates_are_unchanged(self, quantize, taking_turns, together):
        estimate = estimate_ltx_cuda_system_memory(block_matrices=18_555_076_608, transformer_rest=433_761_536, connectors=3_172_227_584, text_encoder=13_200_494_448, quantize=quantize)

        assert (estimate.taking_turns, estimate.together) == (taking_turns, together)


class TestLtxCudaSystemMemory:
    _CONFIG = {
        "video_model_presets": {
            "ltx": {
                "diffusers": {"quantize": {"ltx-4": 4, "ltx-8": 8}},
                "versions": {"ltx25": {"aliases": ["ltx25-4"], "diffusers": {"quantize": {"ltx25-4": 4}}}},
            }
        }
    }

    @pytest.mark.parametrize(("model_name", "quantize"), [("ltx-4", 4), ("ltx-8", 8), ("ltx25-4", 4)])
    def test_the_model_name_and_its_level_reach_the_estimate(self, monkeypatch, model_name, quantize):
        import zvisiongenerator.backends as backends

        calls: list[tuple] = []
        monkeypatch.setattr(backends, "estimate_ltx_system_memory", lambda *args, **kwargs: calls.append((args, kwargs)) or "estimate")

        assert model_memory.ltx_cuda_system_memory("owner/ltx", model_name, self._CONFIG, lambda reference: None) == "estimate"

        ((args, kwargs),) = calls
        assert args[1:] == ("owner/ltx", quantize)
        assert kwargs["model_name"] == model_name


class TestEstimateLtxCudaClipMemory:
    _SYSTEM = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)

    def _clip(self, system=None, **overrides):
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, **overrides}
        return estimate_ltx_cuda_clip_memory(system or self._SYSTEM, **clip)

    @pytest.mark.parametrize("quantize", [4, 8])
    def test_the_gpu_peak_is_flat_for_clips_below_the_knee(self, quantize):
        system = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB, quantize=quantize)
        flat = model_memory._LTX_CUDA_GPU_PLATEAU_BYTES[quantize] + model_memory._LTX_CUDA_GPU_HEADROOM_BYTES

        assert ltx_video_tokens(704, 448, 121) < model_memory._LTX_CUDA_GPU_KNEE_TOKENS[quantize]
        assert self._clip(system).gpu_bytes == self._clip(system, frames=9).gpu_bytes == flat

    def test_q8_peaks_lower_than_q4_for_a_small_clip(self):
        q8 = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=18 * _GIB, quantize=8)

        assert self._clip(q8).gpu_bytes < self._clip().gpu_bytes

    def test_a_longer_clip_never_needs_less_gpu_memory(self):
        sizes = [self._clip(width=1408, height=896, frames=frames).gpu_bytes for frames in (121, 241, 289, 361)]

        assert sizes == sorted(sizes)

    def test_the_xl_clip_that_ran_out_of_memory_on_a_10_gb_card_does_not_fit_it(self):
        xl_15_seconds = self._clip(width=1408, height=896, frames=361)

        assert classify_cuda_memory_fit(xl_15_seconds, MemoryBudget(10 * _GIB, 30 * _GIB)) == "too_large"

    @pytest.mark.parametrize("frames", [241, 289, 337])
    def test_the_xl_clips_that_ran_on_a_10_gb_card_are_not_too_large_for_it(self, frames):
        xl = self._clip(width=1408, height=896, frames=frames)

        assert classify_cuda_memory_fit(xl, MemoryBudget(10 * _GIB, 32 * _GIB)) != "too_large"

    def test_the_upscale_holds_the_upsampler_on_the_gpu_for_a_small_clip_and_adds_nothing_for_a_large_one(self):
        small = self._clip()
        small_upscaled = self._clip(upscale=True)
        large = self._clip(width=1408, height=896, frames=241)
        large_upscaled = self._clip(width=1408, height=896, frames=241, upscale=True)

        assert small_upscaled.gpu_bytes == small.gpu_bytes + model_memory._LTX_CUDA_UPSCALE_GPU_BYTES
        assert large_upscaled.gpu_bytes == large.gpu_bytes

    def test_without_low_memory_the_transformer_stays_on_the_gpu(self):
        assert self._clip(low_memory=False).gpu_bytes == self._clip().gpu_bytes + self._SYSTEM.transformer_bytes

    def test_system_memory_is_the_weights_peak_and_the_upscale_adds_its_components(self):
        plain = self._clip()

        assert plain.system_bytes == self._SYSTEM.taking_turns
        assert self._clip(upscale=True).system_bytes == plain.system_bytes + model_memory._LTX_CUDA_UPSCALE_SYSTEM_BYTES


class TestClipMemoryFitOnCuda:
    _CLIP = {"width": 704, "height": 448, "frames": 121, "low_memory": True}
    _SYSTEM = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)

    @pytest.fixture()
    def weights(self, monkeypatch):
        calls: list[tuple] = []

        def _system(resolved_model, model_name, app_config, find_local_dir):
            calls.append((resolved_model, model_name))
            return self._SYSTEM

        monkeypatch.setattr(model_memory, "ltx_cuda_system_memory", _system)
        return calls

    def test_reports_the_gpu_when_it_is_the_tighter_memory(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(7 * _GIB, 64 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None
        assert (result["kind"], result["resource"], result["budget_gb"]) == ("discrete", "gpu", 7.0)
        assert result["status"] == "tight"  # 6.9 GiB with headroom is over 90% of 7
        assert weights == [("repo/ltx", "ltx-4")]

    def test_reports_system_memory_when_that_is_the_tighter_memory(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(24 * _GIB, 24 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None
        assert (result["resource"], result["budget_gb"]) == ("system", 24.0)
        assert result["status"] == "tight"  # the weights' peak is over 80% of 24

    def test_a_gpu_smaller_than_the_clip_needs_is_too_large(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(4 * _GIB, 64 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None and result["status"] == "too_large"

    def test_weights_beyond_system_memory_are_too_large_since_the_load_is_refused(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(24 * _GIB, 16 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None
        assert (result["status"], result["resource"]) == ("too_large", "system")

    def test_a_clip_out_of_gpu_memory_names_the_gpu_even_when_system_memory_is_fuller(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(4 * _GIB, 16 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None
        assert (result["status"], result["resource"], result["budget_gb"]) == ("too_large", "gpu", 4.0)

    def test_a_clip_that_fits_says_so(self, weights):
        result = clip_memory_fit("repo/ltx", budget=MemoryBudget(24 * _GIB, 64 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP)

        assert result is not None and result["status"] == "fits"

    def test_an_unknown_model_has_no_estimate(self, monkeypatch):
        monkeypatch.setattr(model_memory, "ltx_cuda_system_memory", lambda *args: None)

        assert clip_memory_fit("repo/ltx", budget=MemoryBudget(10 * _GIB, 30 * _GIB), model_name="ltx-4", app_config={}, **self._CLIP) is None

    def test_a_model_without_a_quantize_level_has_no_estimate_and_no_filesystem_lookup(self):
        finder = MagicMock(return_value=None)

        assert clip_memory_fit("repo/ltx", budget=MemoryBudget(10 * _GIB, 30 * _GIB), find_local_dir=finder, model_name="not-ltx", app_config={}, **self._CLIP) is None
        finder.assert_not_called()
