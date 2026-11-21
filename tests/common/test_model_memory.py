"""Tests for header-only model memory estimation."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from zvisiongenerator.utils import model_memory
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryWeights, judge
from zvisiongenerator.utils.model_memory import (
    MemoryBudget,
    WeightTotals,
    estimate_ltx_cuda_clip_memory,
    estimate_ltx_cuda_system_memory,
    level_component_bytes,
    ltx_decode_budget_bytes,
    ltx_mlx_clip_bytes,
    ltx_mlx_weights,
    ltx_video_tokens,
    mlx_image_bytes,
    read_image_components,
    read_safetensors_totals,
    turbo_lora_bytes,
)

_GIB = 1024**3
_GEMMA = "mlx-community/gemma-3-12b-it-4bit"  # the Gemma 3 repo the model metadata declares for the macOS LTX-2.3 packs
_IMAGE_MARGIN = model_memory.IMAGE_WORKING_BYTES


def _detected_family(model_dir: Path) -> str | None:
    """Return the image family the app detects in *model_dir*, ``None`` for a folder without a model index."""
    try:
        return detect_image_model(str(model_dir)).family
    except FileNotFoundError, ValueError:
        return None


def estimate_image_memory(model_dir: Path, levels: tuple[int | None, ...] = (None,)) -> dict[int | None, int] | None:
    """Return the MLX estimate of the image model in *model_dir* at each level, as the memory verdict builds it from the files; ``None`` without readable weights."""
    components = read_image_components(model_dir, unified=True)
    if components is None:
        return None
    detected = _detected_family(model_dir)
    family = detected if detected in {"moss", "ideogram4", "krea2"} else None
    lora_bytes = turbo_lora_bytes(model_dir) if family == "moss" else 0
    return {level: mlx_image_bytes(components, level, family=family, lora_bytes=lora_bytes) for level in levels}


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


class TestMlxImageMemory:
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


class TestLevelComponentBytes:
    """A CUDA image component's bytes at each level: the 2-D weights packed, the text encoder's embedding table left in bfloat16."""

    _TOTALS = WeightTotals(float_matrix_elements=1000, float_other_elements=100, embedding_elements=300)

    def test_unquantized_floats_are_two_bytes_each(self):
        assert level_component_bytes("text_encoder", self._TOTALS, None) == 1000 * 2 + 100 * 2
        assert level_component_bytes("transformer", self._TOTALS, None) == 1000 * 2 + 100 * 2

    def test_q8_packs_the_matrices_to_one_byte_but_a_text_encoders_embedding_table_stays_two(self):
        assert level_component_bytes("text_encoder", self._TOTALS, 8) == 1000 * 1 + 300 * 1 + 100 * 2
        assert level_component_bytes("transformer", self._TOTALS, 8) == 1000 * 1 + 100 * 2

    def test_q4_packs_to_nf4_and_a_text_encoders_embedding_table_stays_two_bytes(self):
        nf4 = model_memory._NF4_BYTES_PER_WEIGHT

        assert level_component_bytes("text_encoder", self._TOTALS, 4) == pytest.approx(1000 * nf4 + 300 * (2 - nf4) + 100 * 2)
        assert level_component_bytes("transformer", self._TOTALS, 4) == pytest.approx(1000 * nf4 + 100 * 2)

    def test_fp8_and_packed_bytes_stay_as_stored(self):
        stored = WeightTotals(fp8_matrix_elements=50, packed_bytes=7)

        assert [level_component_bytes("text_encoder", stored, level) for level in (None, 8)] == [57, 57]

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="quantize level"):
            level_component_bytes("transformer", self._TOTALS, 3)


class TestLtxLoraBytes:
    def test_the_mlx_estimate_counts_them_with_the_transformer(self):
        weights = model_memory.LtxMlxWeights(gemma=2 * _GIB, connector=_GIB, transformer=14 * _GIB, decoders=_GIB)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, "unified_bytes": 64 * _GIB, "env": {}}

        assert ltx_mlx_clip_bytes(weights, lora_bytes=_GIB, **clip) - ltx_mlx_clip_bytes(weights, **clip) == pytest.approx(_GIB, abs=2)

    def test_the_cuda_estimate_counts_them_in_system_memory_and_on_the_gpu_only_when_the_transformer_stays_there(self):
        system = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=4 * _GIB, transformer_bytes=10 * _GIB)
        clip = {"width": 704, "height": 448, "frames": 121}

        streamed = estimate_ltx_cuda_clip_memory(system, low_memory=True, **clip)
        streamed_loaded = estimate_ltx_cuda_clip_memory(system, low_memory=True, lora_bytes=_GIB, **clip)
        resident = estimate_ltx_cuda_clip_memory(system, low_memory=False, **clip)
        resident_loaded = estimate_ltx_cuda_clip_memory(system, low_memory=False, lora_bytes=_GIB, **clip)

        assert streamed_loaded.system_bytes - streamed.system_bytes == _GIB
        assert streamed_loaded.gpu_bytes == streamed.gpu_bytes
        assert resident_loaded.gpu_bytes - resident.gpu_bytes == _GIB


class TestTotalsFromHeader:
    def test_summarises_a_header_without_a_file(self):
        header = {"matrix": {"dtype": "BF16", "shape": [4, 8]}, "bias": {"dtype": "F32", "shape": [8]}, "lm_head.weight": {"dtype": "BF16", "shape": [2, 2]}}

        totals = model_memory.totals_from_header(header, ("lm_head.",))

        assert totals == WeightTotals(float_matrix_elements=32, float_other_elements=8)

    def test_rejects_unknown_dtypes_naming_what_it_read(self):
        with pytest.raises(ValueError, match="the repo"):
            model_memory.totals_from_header({"w": {"dtype": "F4", "shape": [2]}}, label="the repo")


class TestImageComponentTotals:
    _HEADER = {"model.w": {"dtype": "BF16", "shape": [2, 2]}, "lm_head.weight": {"dtype": "BF16", "shape": [3, 3]}, "visual.w": {"dtype": "BF16", "shape": [5, 5]}}

    def test_a_text_encoder_leaves_out_its_lm_head_and_for_moss_on_a_mac_its_vision_tower(self):
        assert model_memory.image_component_totals("text_encoder", self._HEADER, unified=False).float_matrix_elements == 4 + 25
        assert model_memory.image_component_totals("text_encoder", self._HEADER, unified=True).float_matrix_elements == 4

    def test_another_component_counts_everything(self):
        assert model_memory.image_component_totals("transformer", self._HEADER, unified=True).float_matrix_elements == 4 + 9 + 25


class TestReadImageComponents:
    def test_a_moss_text_encoder_drops_its_vision_tower_only_on_a_unified_backend(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline", "text_encoder": ["transformers", "Model"]}))
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"w": ("BF16", [2, 2]), "visual.w": ("BF16", [5, 5])})

        assert model_memory.read_image_components(tmp_path, unified=False)["text_encoder"].float_matrix_elements == 29
        assert model_memory.read_image_components(tmp_path, unified=True)["text_encoder"].float_matrix_elements == 4

    def test_another_model_keeps_everything_on_either(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline", "text_encoder": ["transformers", "Model"]}))
        _write_safetensors(tmp_path / "text_encoder" / "model.safetensors", {"w": ("BF16", [2, 2]), "visual.w": ("BF16", [5, 5])})

        assert model_memory.read_image_components(tmp_path, unified=True)["text_encoder"].float_matrix_elements == 29

    def test_none_without_weights(self, tmp_path):
        assert model_memory.read_image_components(tmp_path, unified=False) is None


class TestLtxMlxWeights:
    def test_sizes_come_from_the_callers_when_given(self, tmp_path):
        pack, encoder = tmp_path / "pack", tmp_path / "gemma"
        for name in ("connector.safetensors", "transformer-distilled.safetensors", "vae_decoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"):
            (pack / name).parent.mkdir(parents=True, exist_ok=True)
            (pack / name).touch()
        (encoder / "model.safetensors").parent.mkdir()
        (encoder / "model.safetensors").touch()
        sizes = {
            "connector.safetensors": 1,
            "transformer-distilled.safetensors": 2,
            "vae_decoder.safetensors": 3,
            "audio_vae.safetensors": 4,
            "vocoder.safetensors": 5,
            "spatial_upscaler_x2_v1_1.safetensors": 6,
            "model.safetensors": 7,
        }

        weights = model_memory.ltx_mlx_weights(pack, encoder, file_bytes=lambda path: sizes[path.name])
        no_upscale = model_memory.ltx_mlx_weights(pack, encoder, upscale=False, file_bytes=lambda path: sizes[path.name])

        assert (weights.gemma, weights.connector, weights.transformer, weights.decoders, weights.upsampler) == (7, 1, 2, 12, 6)
        assert no_upscale.upsampler is None

    def test_a_pack_without_a_bundled_text_encoder_needs_the_gemma_folder(self, tmp_path):
        (tmp_path / "connector.safetensors").touch()
        (tmp_path / "transformer-distilled.safetensors").touch()

        assert model_memory.ltx_mlx_weights(tmp_path, None, file_bytes=lambda path: 1) is None


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


# The weights (GiB) of the two packs; the measured peak footprints below are per clip.
_LTX4_WEIGHTS = {"gemma": 7.477, "connector": 5.909, "transformer": 10.535, "upsampler": 0.927}
_LTX8_WEIGHTS = {"gemma": 7.477, "connector": 5.909, "transformer": 19.173, "upsampler": 0.927}
_MEASURED_UNIFIED = 32 * _GIB
# (weights, width, height, frames, upscale, mode, measured overall peak in GiB)
_MEASURED_RUNS = [
    pytest.param(_LTX4_WEIGHTS, 768, 448, 121, False, "t2v", 16.46, id="ltx4-small"),
    pytest.param(_LTX4_WEIGHTS, 1024, 576, 121, False, "t2v", 16.46, id="ltx4-medium"),
    pytest.param(_LTX4_WEIGHTS, 768, 448, 121, True, "t2v", 16.46, id="ltx4-upscale"),
    pytest.param(_LTX4_WEIGHTS, 768, 448, 121, False, "i2v", 17.02, id="ltx4-image-to-video"),
    pytest.param(_LTX4_WEIGHTS, 576, 320, 97, False, "extend", 20.66, id="ltx4-extend"),
    pytest.param(_LTX8_WEIGHTS, 768, 448, 121, False, "t2v", 22.59, id="ltx8-small"),
    pytest.param(_LTX8_WEIGHTS, 1024, 576, 121, False, "t2v", 23.54, id="ltx8-medium"),
]


def _mlx_clip_memory(model_dir: Path, text_encoder_dir: Path | None, *, upscale: bool = False, **clip) -> int | None:
    """Return the peak of an LTX MLX clip from the pack in *model_dir*: its weights read from the files, then the estimate; ``None`` when the weights cannot be read."""
    weights = ltx_mlx_weights(model_dir, text_encoder_dir, upscale=upscale)
    return None if weights is None else ltx_mlx_clip_bytes(weights, **clip)


class TestLtxMlxClipBytes:
    @pytest.mark.parametrize(("weights", "width", "height", "frames", "upscale", "mode", "measured"), _MEASURED_RUNS)
    def test_reproduces_the_measured_peaks_without_going_below_them(self, tmp_path, weights, width, height, frames, upscale, mode, measured):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **weights)

        required = _mlx_clip_memory(model_dir, text_encoder_dir, width=width, height=height, frames=frames, low_memory=True, upscale=upscale, mode=mode, unified_bytes=_MEASURED_UNIFIED, env={})

        assert required is not None
        # Header sizes are rounded to the GiB thousandth, so allow for that; the fit is otherwise never under.
        # A long clip reaches the tiler's cap, where the decoder weights sit on top of its budget (up to 0.3 GiB above what was measured).
        assert measured - 0.02 <= required / _GIB <= measured + 1.4

    def test_a_bigger_clip_needs_more_memory_where_the_denoise_stage_sets_the_peak(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_LTX8_WEIGHTS)

        sizes = [
            _mlx_clip_memory(model_dir, text_encoder_dir, width=width, height=height, frames=frames, low_memory=True, env={})
            for width, height, frames in ((512, 256, 121), (704, 448, 121), (704, 448, 241), (1280, 704, 241))
        ]

        assert sizes == sorted(sizes) and len(set(sizes)) == 4

    def test_the_decode_peak_is_capped_by_half_of_unified_memory(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "env": {}}

        uncapped = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        capped = _mlx_clip_memory(model_dir, text_encoder_dir, unified_bytes=32 * _GIB, **clip)
        small_mac = _mlx_clip_memory(model_dir, text_encoder_dir, unified_bytes=28 * _GIB, **clip)

        assert uncapped is not None and uncapped / _GIB > 16 + _DECODER_GIB
        assert capped == pytest.approx((16 + _DECODER_GIB) * _GIB, abs=2**20)
        assert small_mac == pytest.approx((14 + _DECODER_GIB) * _GIB, abs=2**20)

    @pytest.mark.parametrize("unified_gib", [16, 24])
    @pytest.mark.parametrize(
        ("width", "height", "frames", "upscale", "mode"), [(768, 448, 121, False, "t2v"), (1280, 704, 241, True, "t2v"), (768, 448, 121, False, "i2v"), (576, 320, 97, False, "extend")]
    )
    def test_a_streaming_mac_needs_one_figure_per_pack_whatever_the_clip(self, tmp_path, unified_gib, width, height, frames, upscale, mode):
        gemma3, gemma3_dir = _make_ltx(tmp_path / "gemma3", **_LTX4_WEIGHTS)
        big, big_dir = _make_ltx(tmp_path / "big", **_LTX8_WEIGHTS)
        clip = {"width": width, "height": height, "frames": frames, "low_memory": True, "upscale": upscale, "mode": mode, "unified_bytes": unified_gib * _GIB, "env": {}}

        small_pack = _mlx_clip_memory(gemma3, gemma3_dir, **clip)
        big_pack = _mlx_clip_memory(big, big_dir, **clip)

        assert small_pack == big_pack
        assert small_pack is not None and small_pack / _GIB == pytest.approx(14.77, abs=0.001)

    def test_a_streaming_mac_reads_the_bundled_encoder_pack_higher(self):
        bundled = model_memory.LtxMlxWeights(gemma=_GIB, connector=_GIB, transformer=_GIB, decoders=_GIB, bundled_encoder=True)
        plain = replace(bundled, bundled_encoder=False)
        clip = {"width": 768, "height": 448, "frames": 121, "low_memory": True, "unified_bytes": 16 * _GIB, "env": {}}

        assert ltx_mlx_clip_bytes(bundled, **clip) > ltx_mlx_clip_bytes(plain, **clip)

    @pytest.mark.parametrize(("low_memory", "unified_gib"), [(False, 16), (True, 32), (True, 25)])
    def test_the_streaming_figure_applies_only_with_low_memory_on_a_mac_of_24_gib_or_less(self, tmp_path, low_memory, unified_gib):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, **_LTX4_WEIGHTS)
        clip = {"width": 768, "height": 448, "frames": 121, "low_memory": low_memory, "unified_bytes": unified_gib * _GIB, "env": {}}

        required = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert required is not None and required / _GIB != pytest.approx(14.77, abs=0.001)

    def test_a_capped_decode_adds_the_decoder_weights_to_the_tiler_budget(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        _write_sized(model_dir / "audio_vae.safetensors", 0.1)
        _write_sized(model_dir / "vocoder.safetensors", 0.2)
        _write_sized(model_dir / "vae_encoder.safetensors", 5)  # Not loaded for a decode.
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, "env": {"LTX2_VAE_DECODE_BUDGET_GB": "8"}}

        required = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert required == pytest.approx((8 + _DECODER_GIB + 0.1 + 0.2) * _GIB, abs=2**20)

    def test_the_decode_budget_variable_overrides_half_of_unified_memory(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 1280, "height": 704, "frames": 241, "low_memory": True, "unified_bytes": 64 * _GIB}

        required = _mlx_clip_memory(model_dir, text_encoder_dir, env={"LTX2_VAE_DECODE_BUDGET_GB": "10"}, **clip)

        assert required == pytest.approx((10 + _DECODER_GIB) * _GIB, abs=2**20)

    def test_the_process_environment_is_read_when_none_is_given(self, tmp_path, monkeypatch):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "9")

        required = _mlx_clip_memory(model_dir, text_encoder_dir, width=1280, height=704, frames=241, low_memory=True)

        assert required == pytest.approx((9 + _DECODER_GIB) * _GIB, abs=2**20)

    def test_upscale_adds_the_upsample_stage_only_when_the_model_has_an_upsampler(self, tmp_path):
        with_file, text_encoder_dir = _make_ltx(tmp_path / "with", gemma=1, connector=1, transformer=1, upsampler=20)
        without_file, _ = _make_ltx(tmp_path / "without", gemma=1, connector=1, transformer=1)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, "env": {}}

        plain = _mlx_clip_memory(with_file, text_encoder_dir, **clip)
        upscaled = _mlx_clip_memory(with_file, text_encoder_dir, upscale=True, **clip)
        no_upsampler = _mlx_clip_memory(without_file, text_encoder_dir, upscale=True, **clip)

        assert upscaled == pytest.approx((1 + 20 + 0.5) * _GIB, abs=1)  # transformer + upsampler + working memory
        assert plain is not None and plain < upscaled
        assert no_upsampler == plain

    def test_without_low_memory_every_weight_stays_resident(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=2, connector=3, transformer=4, upsampler=1)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": False, "env": {}}

        resident = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        upscaled = _mlx_clip_memory(model_dir, text_encoder_dir, upscale=True, **clip)

        # 4928 tokens: denoise working memory 1.7 GiB, decode 12.9 + 0.00067 * 2880 less the 2.83 GiB already counted.
        working = 12.9 + 0.00067 * (4928 - 2048) - 2.83
        assert resident == pytest.approx((2 + 3 + 4 + working) * _GIB, abs=1)
        assert upscaled == pytest.approx((2 + 3 + 4 + 1 + working) * _GIB, abs=1)

    def test_without_low_memory_the_working_memory_is_the_larger_of_denoise_and_decode(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=2, connector=3, transformer=4)

        # A tight decode budget leaves the denoise working memory (3.41 GiB at 5376 tokens, 0.000247 less per token below) as the larger term.
        required = _mlx_clip_memory(model_dir, text_encoder_dir, width=704, height=448, frames=121, low_memory=False, env={"LTX2_VAE_DECODE_BUDGET_GB": "3"})

        assert required == pytest.approx((2 + 3 + 4 + 3.41 + 0.000247 * (4928 - 5376)) * _GIB, abs=1)

    def test_a_bundled_pack_estimates_without_a_text_encoder_dir_and_matches_a_gemma_dir_holding_the_same_file(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx_25(tmp_path, encoder=7.477)
        clip = {"width": 768, "height": 448, "frames": 121, "low_memory": True, "env": {}}

        bundled = _mlx_clip_memory(model_dir, None, **clip)
        with_dir = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert bundled is not None
        assert bundled == with_dir
        # The bundled Gemma 4's prompt stage (the encoder, the connector and 4.27 GiB) is the peak, above the decode stage (about 15.1 GiB at 5376 tokens).
        assert bundled / _GIB == pytest.approx(7.477 + 5.909 + 4.27, abs=0.01)
        assert bundled / _GIB > 12.9 + 0.00067 * (5376 - 2048)

    def test_the_bundled_encoder_is_counted_resident_without_low_memory(self, tmp_path):
        model_dir, _ = _make_ltx_25(tmp_path, encoder=3)
        _write_sized(model_dir / "connector.safetensors", 2)
        _write_sized(model_dir / "transformer-distilled-1.1.safetensors", 4)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": False, "env": {}}

        resident = _mlx_clip_memory(model_dir, None, **clip)

        assert resident == pytest.approx((3 + 2 + 4 + 12.9 + 0.00067 * (4928 - 2048) - 2.83) * _GIB, abs=1)

    def test_a_2_3_pack_without_a_text_encoder_dir_gives_none(self, tmp_path):
        model_dir, _ = _make_ltx(tmp_path, gemma=1, connector=1, transformer=1)

        assert _mlx_clip_memory(model_dir, None, width=704, height=448, frames=121, low_memory=True) is None

    def test_the_conv_decoder_bytes_are_counted_in_the_capped_decode(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=0.1, connector=0.1, transformer=0.1)
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, "env": {"LTX2_VAE_DECODE_BUDGET_GB": "8"}}
        plain = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)
        _write_sized(model_dir / "vae_decoder_conv.safetensors", 0.5)

        with_conv = _mlx_clip_memory(model_dir, text_encoder_dir, **clip)

        assert plain == pytest.approx((8 + _DECODER_GIB) * _GIB, abs=2**20)
        assert with_conv == pytest.approx((8 + 0.5) * _GIB, abs=2**20)  # The conv decoder replaces vae_decoder.

    def test_returns_none_for_non_ltx_layout(self, tmp_path):
        _write_safetensors(tmp_path / "transformer" / "model.safetensors", {"w": ("BF16", [10, 10])})

        assert _mlx_clip_memory(tmp_path, tmp_path, width=704, height=448, frames=121, low_memory=True) is None

    def test_returns_none_for_unreadable_weights(self, tmp_path):
        model_dir, text_encoder_dir = _make_ltx(tmp_path, gemma=1, connector=1, transformer=1)
        (model_dir / "connector.safetensors").write_bytes(b"")

        assert _mlx_clip_memory(model_dir, text_encoder_dir, width=704, height=448, frames=121, low_memory=True) is None


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


class TestEstimateMlxMossMemory:
    """MOSS-Image on MLX: the text encoder stage and the denoiser stage are never resident together."""

    def _model(self, root: Path, *, lora: int = 0, encoder_rows: int = 1024, transformer_rows: int = 4096) -> Path:
        model_dir = root / "moss"
        index = {"_class_name": "MOSSImagePipeline", "transformer": ["x", "T"], "text_encoder": ["x", "E"], "vae": ["x", "V"]}
        model_dir.mkdir(parents=True)
        (model_dir / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        _write_safetensors(
            model_dir / "text_encoder" / "model.safetensors",
            # The names of the pinned checkpoint's text encoder: the language model and the vision tower mflux skips.
            {"language_model.layers.0.mlp.weight": ("BF16", [encoder_rows, 1024]), "visual.blocks.0.mlp.weight": ("BF16", [8192, 1024])},
        )
        _write_safetensors(model_dir / "transformer" / "model.safetensors", {"linear": ("BF16", [transformer_rows, 1024])})
        _write_safetensors(model_dir / "vae" / "model.safetensors", {"conv": ("F32", [64, 64, 3, 3])})
        if lora:
            (model_dir / "turbo_lora.safetensors").write_bytes(b"\0" * lora)
        return model_dir

    def test_the_peak_is_the_larger_stage_with_the_vision_tower_left_out(self, tmp_path):
        estimates = estimate_image_memory(self._model(tmp_path, lora=4000), (None, 4))

        denoiser = 4096 * 1024 * 2 + 64 * 64 * 9 * 4 + 4000
        assert estimates[None] == denoiser + model_memory._MOSS_MLX_DENOISE_WORKING_BYTES
        q4 = 4096 * 1024 * (4 / 8 + model_memory._QUANT_GROUP_OVERHEAD_BYTES) + 64 * 64 * 9 * 4 + 4000
        assert estimates[4] == int(q4 + model_memory._MOSS_MLX_DENOISE_WORKING_BYTES)

    def test_a_larger_text_encoder_stage_sets_the_peak(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_memory, "_MOSS_MLX_DENOISE_WORKING_BYTES", 0)
        estimates = estimate_image_memory(self._model(tmp_path, encoder_rows=8192, transformer_rows=1024))

        assert estimates[None] == 8192 * 1024 * 2 + model_memory._MOSS_MLX_TEXT_ENCODER_WORKING_BYTES


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
        runtime = model_memory._LTX_CUDA_RUNTIME_BYTES[4]
        estimate = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)

        assert estimate.taking_turns == 15 * _GIB + runtime
        assert estimate.together == 25 * _GIB + runtime

    def test_q8_takes_less_runtime_than_q4(self):
        q4 = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB, quantize=4)
        q8 = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB, quantize=8)

        saved = model_memory._LTX_CUDA_RUNTIME_BYTES[4] - model_memory._LTX_CUDA_RUNTIME_BYTES[8]

        assert saved > 0
        assert (q4.taking_turns - q8.taking_turns, q4.together - q8.together) == (saved, saved)

    def test_the_q8_clip_estimate_matches_the_measured_system_memory(self):
        """A 5 s LTX-2.3 q8 clip (ltx8-s) held 24.13 GiB of system memory (the highest anonymous plus shared memory in one sample); the estimate is that figure."""
        system = estimate_ltx_cuda_system_memory(block_matrices=18_555_076_608, transformer_rest=433_761_536, connectors=3_172_227_584, text_encoder=13_200_494_448, quantize=8)

        clip = model_memory.estimate_ltx_cuda_clip_memory(system, width=768, height=448, frames=121, low_memory=True)

        assert round(clip.system_bytes / _GIB, 2) == pytest.approx(24.13, abs=0.05)

    @pytest.mark.parametrize("quantize", [4, 8])
    def test_the_ltx_2_3_estimates_are_the_weights_plus_the_measured_runtime(self, quantize):
        estimate = estimate_ltx_cuda_system_memory(block_matrices=18_555_076_608, transformer_rest=433_761_536, connectors=3_172_227_584, text_encoder=13_200_494_448, quantize=quantize)
        runtime = model_memory._LTX_CUDA_RUNTIME_BYTES[quantize]

        assert estimate.taking_turns == max(estimate.text_encoder_bytes, estimate.transformer_bytes) + estimate.resident_bytes + runtime
        assert estimate.together == estimate.text_encoder_bytes + estimate.transformer_bytes + estimate.resident_bytes + runtime

    def test_ltx_2_3_needs_less_system_memory_at_q4_than_at_q8(self):
        counts = {"block_matrices": 18_555_076_608, "transformer_rest": 433_761_536, "connectors": 3_172_227_584, "text_encoder": 13_200_494_448}

        q4 = estimate_ltx_cuda_system_memory(**counts, quantize=4)
        q8 = estimate_ltx_cuda_system_memory(**counts, quantize=8)

        assert q4.taking_turns < q8.taking_turns


class TestLtxCudaSystemMemory:
    @pytest.mark.parametrize(("model_name", "quantize"), [("ltx-4", 4), ("ltx-8", 8), ("ltx25-4", 4)])
    def test_the_model_name_and_its_level_reach_the_estimate(self, monkeypatch, model_name, quantize):
        import zvisiongenerator.backends as backends

        calls: list[tuple] = []
        monkeypatch.setattr(backends, "estimate_ltx_system_memory", lambda *args, **kwargs: calls.append((args, kwargs)) or "estimate")

        assert model_memory.ltx_cuda_system_memory("owner/ltx", model_name, lambda reference: None) == "estimate"

        ((args, kwargs),) = calls
        assert args == ("owner/ltx", quantize)
        assert kwargs["model_name"] == model_name


class TestEstimateLtxCudaClipMemory:
    _SYSTEM = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)

    def _clip(self, system=None, **overrides):
        clip = {"width": 704, "height": 448, "frames": 121, "low_memory": True, **overrides}
        return estimate_ltx_cuda_clip_memory(system or self._SYSTEM, **clip)

    @pytest.mark.parametrize("quantize", [4, 8])
    def test_the_gpu_peak_is_flat_for_clips_below_the_knee(self, quantize):
        system = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB, quantize=quantize)
        flat = model_memory._LTX_CUDA_GPU_PLATEAU_BYTES[quantize]

        assert ltx_video_tokens(704, 448, 121) < model_memory._LTX_CUDA_GPU_KNEE_TOKENS[quantize]
        assert self._clip(system).gpu_bytes == self._clip(system, frames=9).gpu_bytes == flat

    def test_q8_peaks_lower_than_q4_for_a_small_clip(self):
        q8 = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=18 * _GIB, quantize=8)

        assert self._clip(q8).gpu_bytes < self._clip().gpu_bytes

    def test_a_longer_clip_never_needs_less_gpu_memory(self):
        sizes = [self._clip(width=1408, height=896, frames=frames).gpu_bytes for frames in (121, 241, 289, 361)]

        assert sizes == sorted(sizes)

    @pytest.mark.parametrize(("frames", "status"), [(241, "fits"), (337, "fits"), (361, "tight")])
    def test_the_xl_clips_that_ran_read_their_measured_status(self, frames, status):
        """1600x896 at 10, 14 and 15 s (ltx4-xl-10s, ltx4-xl-14s, ltx4-xl-15s) ran, needing 6.72, 8.44 and 8.99 GiB (the middles of their hold-probe brackets)."""
        job = MemoryJob(media="video", family="ltx", width=1600, height=896, frames=frames, low_memory=True)

        assert judge(job, MemoryBudget(int(9.64 * _GIB), 32 * _GIB), MemoryWeights(ltx_cuda=self._SYSTEM)).status == status

    @pytest.mark.parametrize("frames", [337, 361])
    def test_the_long_xl_clips_are_too_large_for_a_card_a_probe_ran_out_of_memory_on(self, frames):
        """The 14 s and 15 s clips (ltx4-xl-14s, ltx4-xl-15s) ran out of memory under holds that left 8.37 and 8.85 GiB."""
        job = MemoryJob(media="video", family="ltx", width=1600, height=896, frames=frames, low_memory=True)

        assert judge(job, MemoryBudget(int(8.3 * _GIB), 32 * _GIB), MemoryWeights(ltx_cuda=self._SYSTEM)).status == "too_large"

    def test_q8_needs_more_than_q4_on_a_long_clip_and_less_on_a_short_one_as_measured(self):
        """q8 ran 1600x896 at 10 s on 6.86 GiB against 6.72 at q4 (ltx8-xl-10s, ltx4-xl-10s), and 768x448 at 5 s on 3.99 against 4.76 (ltx8-s, ltx4-s)."""
        q8 = model_memory.LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=18 * _GIB, quantize=8)

        assert self._clip(q8, width=1600, height=896, frames=241).gpu_bytes > self._clip(width=1600, height=896, frames=241).gpu_bytes
        assert self._clip(q8, width=768, height=448, frames=121).gpu_bytes < self._clip(width=768, height=448, frames=121).gpu_bytes

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
