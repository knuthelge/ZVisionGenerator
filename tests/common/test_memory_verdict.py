"""Tests for the memory verdict: the CUDA image engine against every bench case, and the wrapped video and audio estimators."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bench.memory.image import judge_forced_placement
from zvisiongenerator.utils.h3_memory import estimate_h3_cuda_clip_memory, estimate_h3_mlx_clip_memory, estimate_h3_mlx_seconds, h3_expected_seconds, h3_nominal_references
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.memory_verdict import (
    MemoryJob,
    MemoryWeights,
    RESIDENT,
    STREAMED,
    TURNS,
    cuda_image_gpu_bytes,
    cuda_image_placement,
    image_placement_candidates,
    judge,
)
from zvisiongenerator.utils.model_memory import (
    LtxCudaMemoryEstimate,
    LtxMlxWeights,
    MemoryBudget,
    WeightTotals,
    estimate_ltx_cuda_clip_memory,
    estimate_ltx_cuda_system_memory,
    ltx_mlx_clip_bytes,
)

_GIB = 1024**3
_ANCHOR_MP = 1024 * 1024 / 1e6
_CARD = MemoryBudget(10 * _GIB, int(30.3 * _GIB))

# Component GiB of each model from its local headers: (text encoder NF4, transformer NF4, text encoder FP8, transformer FP8,
# text encoder bf16, transformer bf16, VAE).
_COMPONENT_GIB = {
    "zit": (2.472, 2.961, 3.746, 5.733, 7.492, 11.464, 0.156),
    "klein4b": (2.472, 1.863, 3.746, 3.609, 7.492, 7.219, 0.157),
    "klein9b": (4.499, 4.365, 7.049, 8.455, 14.097, 16.910, 0.157),
    "krea2": (2.674, 6.164, 4.135, 11.940, 8.266, 23.879, 0.236),
}
_MODEL = {
    "zit": ("zimage", None),
    "klein4b": ("flux2_klein", "4b"),
    "klein9b": ("flux2_klein", "9b"),
    "krea2": ("krea2", None),
    "ideo": ("ideogram4", None),
    "moss": ("moss", None),
}
_NF4_PER_WEIGHT = 4.13 / 8


def _status(need: float, limit: float, fits: float, tight: float) -> int:
    """Rank a need against a limit: 0 fits (up to *fits* of it), 1 tight (up to *tight*), 2 too large."""
    return 0 if need <= limit * fits else 1 if need <= limit * tight else 2


def _cuda_status(gpu: int, system: int | None, budget: MemoryBudget) -> str:
    """The status the bands give a CUDA estimate: the GPU figure (the estimate plus the CUDA offset) at 90% and the whole card, the RAM at 80% and all of it; the worse of the two."""
    worst = max(_status(gpu + CUDA_OFFSET_BYTES, budget.gpu_bytes, 0.9, 1.0), 0 if system is None else _status(system, budget.system_bytes, 0.8, 1.0))
    return ("fits", "tight", "too_large")[worst]


def _mac_status(need: int, budget: MemoryBudget) -> str:
    """The status the bands give a Mac estimate: it fits up to 75 % of the machine's memory, is tight up to all of it and too large beyond."""
    if need <= budget.unified_bytes * 0.75:
        return "fits"
    return "tight" if need <= budget.unified_bytes else "too_large"


def _elements(gib: float) -> int:
    return round(gib * _GIB)


def _component(nf4: float, fp8: float, bf16: float, *, embedding: bool = False) -> WeightTotals:
    """Return totals whose NF4, FP8 and bfloat16 sizes are the given GiB: matrices, other floats and, for a text encoder, its embedding table."""
    matrices = bf16 - fp8
    other = (fp8 - matrices) / 2
    table = max(0.0, (nf4 - _NF4_PER_WEIGHT * matrices - 2 * other) / (2 - _NF4_PER_WEIGHT)) if embedding else 0.0
    return WeightTotals(float_matrix_elements=_elements(matrices), float_other_elements=_elements(other), embedding_elements=_elements(table))


def _vae(gib: float, *, bytes_per_element: int = 2) -> WeightTotals:
    return WeightTotals(float_other_elements=_elements(gib / bytes_per_element))


def _weights(model: str) -> MemoryWeights:
    """Return synthetic weights for a bench model whose component sizes are the measured ones."""
    if model == "ideo":
        return MemoryWeights(components={"text_encoder": WeightTotals(packed_bytes=_elements(4.88)), "transformer": _component(8.0, 10.0, 20.0), "vae": _vae(0.156)})
    if model == "moss":
        return MemoryWeights(components={"vae": _vae(0.116, bytes_per_element=4)}, turbo_lora_bytes=_elements(1.108))
    te4, tr4, te8, tr8, te16, tr16, vae = _COMPONENT_GIB[model]
    return MemoryWeights(components={"text_encoder": _component(te4, te8, te16, embedding=True), "transformer": _component(tr4, tr8, tr16), "vae": _vae(vae)})


def _image_job(model: str, quantize: int | None, width: int, height: int, references: int = 0, **overrides) -> MemoryJob:
    family, size = _MODEL[model]
    return MemoryJob(media="image", family=family, model_size=size, quantize=quantize, width=width, height=height, reference_megapixels=(_ANCHOR_MP,) * references, **overrides)


# The recommended working sets of a 32 GB and a 16 GB Mac, as the Mac bench read them; a Mac is graded against its total memory.
_MAC_32 = MemoryBudget(26_800_603_136, None, 32 * _GIB)
_MAC_16 = MemoryBudget(11_452_547_072, None, 16 * _GIB)


def _mac_weights(model: str) -> MemoryWeights:
    """Return synthetic weights of a bench model as a Mac holds them: the Ideogram 4 transformer is an FP8 checkpoint, MOSS-Image's VAE float32."""
    if model == "ideo":
        return MemoryWeights(components={"text_encoder": WeightTotals(packed_bytes=_elements(1.0)), "transformer": WeightTotals(fp8_matrix_elements=_elements(13.0)), "vae": _vae(0.156)})
    return _weights(model)


# Every distinct case of the bench runs that ran or ran out of memory (a 10 GB card, 30.3 GB of RAM):
# (id, model, quantize, width, height, references, placement, measured allocated peak GiB, RAM estimate GiB).
# A case probed with holds gives the least memory it runs on (the middle of its bracket) less the CUDA offset in place of its free-card peak: the resident Z-Image and Klein 4B rows,
# whose VAE decode takes less when the card is tight.
# A case's references count the whole-picture edit's picture as one. A q8 text encoder keeps its embedding table in bfloat16, so the RAM of a
# streamed q8 model counts it at two bytes.
_BENCH_CASES = [
    ("zit-m", "zit", 8, 1024, 1024, 0, "streamed", 3.55, 12.03),
    ("zit-xl", "zit", 8, 1600, 1600, 0, "streamed", 3.55, 12.03),
    ("zit-q4-m", "zit", 4, 1024, 1024, 0, "resident", 7.14, 1.61),
    ("zit-q4-xl", "zit", 4, 1600, 1600, 0, "resident", 7.14, 1.61),
    ("zit-q4-m-turns", "zit", 4, 1024, 1024, 0, "turns", 3.60, 8.26),
    ("zit-q4-xl-turns", "zit", 4, 1600, 1600, 0, "turns", 4.28, 8.26),
    ("klein4b-q4-m-ref1", "klein4b", 4, 1024, 1024, 1, "turns", 4.43, 7.16),
    ("ideo-m", "ideo", 8, 1024, 1024, 0, "streamed", 5.71, 15.8),
    ("ideo-none-m", "ideo", None, 1024, 1024, 0, "streamed", 5.71, 25.7),
    ("ideo-q4-m", "ideo", 4, 1024, 1024, 0, "streamed", 5.86, 11.8),
    ("krea2-m", "krea2", 8, 1024, 1024, 0, "streamed", 2.44, 19.47),
    ("krea2-xl", "krea2", 8, 1600, 1600, 0, "streamed", 3.21, 19.47),
    ("krea2-q4-m", "krea2", 4, 1024, 1024, 0, "turns", 7.16, 11.67),
    ("krea2-q4-xl", "krea2", 4, 1600, 1600, 0, "turns", 7.92, 11.67),
    ("ideo-none-xl", "ideo", None, 1600, 1600, 0, "streamed", None, 25.7),
    ("klein4b-m", "klein4b", 8, 1024, 1024, 0, "streamed", 3.56, 9.91),
    ("klein4b-xl", "klein4b", 8, 1600, 1600, 0, "streamed", 3.56, 9.91),
    ("klein9b-m", "klein9b", 8, 1024, 1024, 0, "streamed", 3.56, 18.27),
    ("klein9b-xl", "klein9b", 8, 1600, 1600, 0, "streamed", 3.57, 18.27),
    ("klein9b-m-ref1", "klein9b", 8, 1024, 1024, 1, "streamed", 3.57, 18.27),
    ("klein9b-m-ref3", "klein9b", 8, 1024, 1024, 3, "streamed", 3.83, 18.27),
    ("klein9b-none-m", "klein9b", None, 1024, 1024, 0, "streamed", 3.56, 27.0),
    ("klein9b-none-xl", "klein9b", None, 1600, 1600, 0, "streamed", 3.80, 27.0),
    ("klein9b-q4-m", "klein9b", 4, 1024, 1024, 0, "turns", 5.44, 11.69),
    ("klein9b-q4-xl", "klein9b", 4, 1600, 1600, 0, "turns", 6.40, 11.69),
    ("klein9b-q4-m-ref1", "klein9b", 4, 1024, 1024, 1, "turns", 6.44, 11.69),
    ("klein9b-q4-m-ref3", "klein9b", 4, 1024, 1024, 3, "turns", 7.44, 11.69),
    ("klein9b-edit-2048", "klein9b", 8, 2048, 2048, 1, "streamed", 4.50, 18.27),
    ("klein9b-extend-right512", "klein9b", 8, 1488, 992, 1, "streamed", 3.46, 18.27),
    ("klein9b-q4-edit-2048", "klein9b", 4, 2048, 2048, 1, "turns", None, 11.69),
    ("klein9b-q4-extend-right512", "klein9b", 4, 1488, 992, 1, "turns", 6.38, 11.69),
    ("moss-m", "moss", 8, 1024, 1024, 0, "streamed", 6.82, 13.28),
    ("moss-xl", "moss", 8, 1600, 1600, 0, "streamed", 7.47, 13.28),
    ("moss-q4-m", "moss", 4, 1024, 1024, 0, "turns", 6.82, 11.76),
    ("moss-q4-xl", "moss", 4, 1600, 1600, 0, "turns", 7.47, 11.76),
    ("ideo-l", "ideo", 8, 1408, 1408, 0, "streamed", 7.43, 15.8),
    ("ideo-none-l", "ideo", None, 1408, 1408, 0, "streamed", None, 25.7),
    ("ideo-q4-l", "ideo", 4, 1408, 1408, 0, "streamed", 7.14, 11.8),
    ("klein9b-q4-l", "klein9b", 4, 1408, 1408, 0, "turns", 6.03, 11.69),
    ("klein9b-q4-xl-ref1", "klein9b", 4, 1600, 1600, 1, "turns", 7.07, 11.69),
    ("klein9b-q4-xl-ref3", "klein9b", 4, 1600, 1600, 3, "turns", None, 11.69),
    ("klein9b-q4-m-ref4", "klein9b", 4, 1024, 1024, 4, "turns", None, 11.69),
    ("klein9b-q4-2048", "klein9b", 4, 2048, 2048, 0, "turns", 7.44, 11.69),
    ("moss-q4-2048", "moss", 4, 2048, 2048, 0, "turns", 7.51, 11.76),
    ("moss-2048", "moss", 8, 2048, 2048, 0, "streamed", 7.51, 13.28),
    ("moss-none-m", "moss", None, 1024, 1024, 0, "streamed", 6.82, 16.8),
    ("moss-none-xl", "moss", None, 1600, 1600, 0, "streamed", 7.47, 16.8),
    ("moss-none-2048", "moss", None, 2048, 2048, 0, "streamed", 7.51, 16.8),
    ("klein4b-q4-m", "klein4b", 4, 1024, 1024, 0, "resident", 6.14, 1.61),
    ("klein4b-q4-xl", "klein4b", 4, 1600, 1600, 0, "resident", 6.14, 1.61),
    ("klein4b-q4-m-turns", "klein4b", 4, 1024, 1024, 0, "turns", 3.56, 7.16),
    ("klein4b-q4-xl-turns", "klein4b", 4, 1600, 1600, 0, "turns", 3.56, 7.16),
    # bench run cuda-r17: hold-bracket middles less the CUDA offset (resident and turns), allocated peaks (streamed)
    ("klein4b-q4-2048-resident", "klein4b", 4, 2048, 2048, 0, "resident", 6.64, 1.61),
    ("klein4b-q4-m-ref4-resident", "klein4b", 4, 1024, 1024, 4, "resident", 7.14, 1.61),
    ("klein4b-q4-xl-ref4-resident", "klein4b", 4, 1600, 1600, 4, "resident", 7.89, 1.61),
    ("klein4b-q4-2048-turns", "klein4b", 4, 2048, 2048, 0, "turns", 4.14, 7.16),
    ("klein4b-q4-xl-ref4-turns", "klein4b", 4, 1600, 1600, 4, "turns", 5.39, 7.16),
    ("klein4b-2048", "klein4b", 8, 2048, 2048, 0, "streamed", 3.56, 9.91),
    ("klein4b-m-ref4", "klein4b", 8, 1024, 1024, 4, "streamed", 3.57, 9.91),
    ("klein4b-xl-ref4", "klein4b", 8, 1600, 1600, 4, "streamed", 3.98, 9.91),
    ("klein4b-2048-ref4", "klein4b", 8, 2048, 2048, 4, "streamed", 4.76, 9.91),
    ("zit-xl-up2", "zit", 8, 1600, 1600, 0, "streamed", None, 12.03),
    ("ideo-xl", "ideo", 8, 1600, 1600, 0, "streamed", None, 15.8),
    ("ideo-q4-xl", "ideo", 4, 1600, 1600, 0, "streamed", None, 11.8),
]
# These ran out of memory on the card; every other case ran. A job that ran out of memory has no peak to fit, only the memory it had reached when it stopped.
_OUT_OF_MEMORY = {"ideo-none-xl", "ideo-none-l", "ideo-xl", "ideo-q4-xl", "klein9b-q4-edit-2048", "klein9b-q4-xl-ref3", "klein9b-q4-m-ref4"}
# Of those, the ones whose memory at the failure (the allocated memory plus the allocation that failed, 8.73, 8.75 and 8.99 GiB) is above what the card had beside the other programs
# (9.64 GiB less the 0.64 GiB they held) must read Too large. The others stopped below it, from fragmentation or at a peak the estimate puts under the line, and read Tight or Fits.
_OUT_OF_MEMORY_TOO_LARGE = {"ideo-xl", "ideo-none-xl", "ideo-q4-xl"}
_CARD_LESS_OTHERS = round((9.64 - 0.64) * _GIB)
# Unquantized weights that stream are mapped from disk and run past RAM, only slower: the RAM, not the card, reads Tight.
_RAM_TIGHT = {"klein9b-none-m", "klein9b-none-xl", "ideo-none-m", "ideo-none-l"}
# The bench measured these at the placement the app would not pick on this card, so they are forced.
_FORCED = {
    "klein4b-q4-m-turns": "turns",
    "klein4b-q4-xl-turns": "turns",
    "zit-q4-m-turns": "turns",
    "zit-q4-xl-turns": "turns",
    "klein4b-q4-m-ref1": "turns",
    "klein4b-q4-2048-resident": "resident",
    "klein4b-q4-m-ref4-resident": "resident",
    "klein4b-q4-xl-ref4-resident": "resident",
    "klein4b-q4-2048-turns": "turns",
    "klein4b-q4-xl-ref4-turns": "turns",
}


class TestBenchAppendix:
    @pytest.mark.parametrize(("case", "model", "quantize", "width", "height", "references", "placement", "measured", "ram"), _BENCH_CASES, ids=[row[0] for row in _BENCH_CASES])
    def test_reads_like_the_bench_case(self, case, model, quantize, width, height, references, placement, measured, ram):
        job = _image_job(model, quantize, width, height, references)

        result = judge_forced_placement(job, _CARD, _weights(model), _FORCED[case]) if case in _FORCED else judge(job, _CARD, _weights(model))

        assert result.placement == placement
        expected_ram = None if ram is None else ram * _GIB
        assert (result.ram_bytes is None) == (expected_ram is None)
        if expected_ram is not None:
            assert result.ram_bytes == pytest.approx(expected_ram, abs=0.06 * _GIB)
        if case in _RAM_TIGHT:
            assert (result.status, result.resource) == ("tight", "ram")
        if case in _OUT_OF_MEMORY:
            if case in _OUT_OF_MEMORY_TOO_LARGE:
                assert result.gpu_bytes > _CARD_LESS_OTHERS
            return
        # The bench's calibration rule: a job that ran never reads Too large, and its figure is within the larger of 10 % and 1 GiB of its allocated peak plus the CUDA offset.
        assert result.status != "too_large"
        if measured is not None:
            figure = measured * _GIB + CUDA_OFFSET_BYTES
            assert abs(result.gpu_bytes - figure) <= max(0.10 * figure, _GIB)


def _packed(text_encoder: float, transformer: float, vae: float = 0.1) -> dict[str, WeightTotals]:
    """Return components whose sizes are the given GiB at every level, so a phase test controls each phase's weights."""
    return {"text_encoder": WeightTotals(packed_bytes=_elements(text_encoder)), "transformer": WeightTotals(packed_bytes=_elements(transformer)), "vae": WeightTotals(packed_bytes=_elements(vae))}


def _gib(
    components: dict[str, WeightTotals],
    placement: str,
    quantize: int | None = 4,
    *,
    lora_gib: float = 0.0,
    megapixels: float = _ANCHOR_MP,
    reference_megapixels: float = 0.0,
    family: str = "flux2_klein",
    size: str | None = "9b",
) -> float:
    return (
        cuda_image_gpu_bytes(placement, components, quantize, family=family, model_size=size, lora_bytes=_elements(lora_gib), job_megapixels=megapixels, reference_megapixels=reference_megapixels)
        / _GIB
    )


class TestPhases:
    """The phases of a placement set the need: the largest of them is the peak, and each phase moves with what it holds."""

    def test_turns_take_the_encode_phase_when_the_text_encoder_is_large(self):
        base = _gib(_packed(6.0, 1.0), TURNS)

        assert _gib(_packed(7.0, 1.0), TURNS) == pytest.approx(base + 1.0)  # the encode phase holds the text encoder
        assert _gib(_packed(6.0, 2.0), TURNS) == pytest.approx(base)  # and not the transformer

    def test_turns_take_the_reference_encode_phase_when_references_are_held(self):
        components = _packed(6.0, 1.0)

        assert _gib(components, TURNS, reference_megapixels=_ANCHOR_MP) > _gib(components, TURNS)
        assert _gib(_packed(7.0, 1.0), TURNS, reference_megapixels=_ANCHOR_MP) == pytest.approx(_gib(components, TURNS, reference_megapixels=_ANCHOR_MP) + 1.0)

    def test_turns_take_the_denoise_phase_when_the_transformer_is_large(self):
        base = _gib(_packed(1.0, 8.0), TURNS)

        assert _gib(_packed(1.0, 9.0), TURNS) == pytest.approx(base + 1.0)
        assert _gib(_packed(2.0, 8.0), TURNS) == pytest.approx(base)

    def test_denoise_grows_with_the_size_beyond_one_megapixel_and_with_references(self):
        components = _packed(1.0, 8.0)

        sized = _gib(components, TURNS, megapixels=4.0)
        referenced = _gib(components, TURNS, reference_megapixels=_ANCHOR_MP)

        assert _gib(components, TURNS) < _gib(components, TURNS, megapixels=2.0) < sized
        assert _gib(components, TURNS) < referenced
        assert _gib(components, TURNS, megapixels=4.0, reference_megapixels=_ANCHOR_MP) > max(sized, referenced)

    def test_a_size_below_one_megapixel_adds_nothing(self):
        assert _gib(_packed(1.0, 8.0), TURNS, megapixels=0.25) == _gib(_packed(1.0, 8.0), TURNS)

    def test_loras_sit_with_the_transformer(self):
        assert _gib(_packed(1.0, 8.0), TURNS, lora_gib=1.0) == pytest.approx(_gib(_packed(1.0, 8.0), TURNS) + 1.0)
        assert _gib(_packed(0.3, 0.3), TURNS, lora_gib=0.2) == _gib(_packed(0.3, 0.3), TURNS)  # a small transformer leaves the decode as the peak

    def test_turns_take_the_decode_phase_when_the_weights_are_small(self):
        assert _gib(_packed(0.3, 0.3), TURNS) == _gib(_packed(0.5, 0.5), TURNS)

    def test_resident_holds_every_weight_beside_the_larger_of_decode_and_denoise(self):
        base = _gib(_packed(2.0, 3.0), RESIDENT)

        assert _gib(_packed(3.0, 3.0), RESIDENT) == pytest.approx(base + 1.0)
        assert _gib(_packed(2.0, 4.0), RESIDENT) == pytest.approx(base + 1.0)
        assert _gib(_packed(2.0, 3.0), RESIDENT, lora_gib=0.5) == pytest.approx(base + 0.5)
        assert _gib(_packed(2.0, 3.0), RESIDENT, megapixels=8.0) > base  # a large canvas lets the denoise outgrow the decode

    def test_resident_takes_the_reference_encode_phase_when_references_are_held(self):
        components = _packed(2.0, 3.0)
        without = _gib(components, RESIDENT, size="4b")
        with_reference = _gib(components, RESIDENT, size="4b", reference_megapixels=_ANCHOR_MP)

        assert with_reference > without  # the reference encode runs with every weight on the card and outgrows the decode
        assert with_reference == pytest.approx(2.0 + 3.0 + 0.1 + 1.78)
        assert _gib(_packed(3.0, 3.0), RESIDENT, size="4b", reference_megapixels=_ANCHOR_MP) == pytest.approx(with_reference + 1.0)

    def test_a_reference_can_move_a_resident_job_over_the_fits_line(self):
        components = _packed(2.0, 3.0)
        kwargs = {"family": "flux2_klein", "model_size": "4b", "job_megapixels": _ANCHOR_MP}
        without = cuda_image_gpu_bytes(RESIDENT, components, 4, **kwargs)
        with_reference = cuda_image_gpu_bytes(RESIDENT, components, 4, reference_megapixels=_ANCHOR_MP, **kwargs)
        budget = int((without + CUDA_OFFSET_BYTES) / 0.9) + 1024  # just over the Fits line without a reference

        assert cuda_image_placement(components, 4, budget, **kwargs)[0] == RESIDENT
        assert cuda_image_placement(components, 4, budget, reference_megapixels=_ANCHOR_MP, **kwargs)[0] == TURNS
        assert with_reference + CUDA_OFFSET_BYTES > 0.9 * budget

    def test_streamed_holds_only_the_vae_and_ignores_the_weights_and_loras(self):
        small, large = _packed(2.0, 3.0), _packed(20.0, 30.0)

        assert _gib(large, STREAMED) == _gib(small, STREAMED)
        assert _gib(large, STREAMED, lora_gib=5.0) == _gib(small, STREAMED)
        assert _gib(large, STREAMED, megapixels=4.19, reference_megapixels=_ANCHOR_MP) > _gib(small, STREAMED)

    def test_weights_count_in_the_format_of_their_level(self):
        components = {"text_encoder": WeightTotals(float_matrix_elements=_elements(2.0)), "transformer": WeightTotals(float_matrix_elements=_elements(4.0)), "vae": _vae(0.1)}

        bf16, fp8, nf4 = (_gib(components, RESIDENT, level) for level in (None, 8, 4))

        assert bf16 - fp8 == pytest.approx(6.0)  # bfloat16 is 2 bytes a weight, FP8 1
        assert fp8 - nf4 == pytest.approx(6.0 * (1 - _NF4_PER_WEIGHT))

    def test_rejects_an_unknown_placement_and_an_unknown_level(self):
        with pytest.raises(ValueError, match="placement"):
            _gib(_packed(1.0, 1.0), "offloaded")
        with pytest.raises(ValueError, match="quantize"):
            _gib(_packed(1.0, 1.0), RESIDENT, 5)


class TestReferenceMegapixels:
    def test_a_reference_counts_at_most_one_megapixel_and_a_smaller_one_counts_its_own_size(self):
        # At 1600x1600 the denoise phase, which reads the references, is the peak.
        weights = _weights("klein9b")

        def gib(*sizes: float) -> float:
            job = MemoryJob(media="image", family="flux2_klein", model_size="9b", quantize=4, width=1600, height=1600, reference_megapixels=sizes)
            return judge(job, _CARD, weights).gpu_bytes / _GIB

        assert gib(4.194304) == pytest.approx(gib(_ANCHOR_MP))  # a 2048x2048 edit picture is capped at 1024x1024
        assert gib(_ANCHOR_MP, 4.194304) == pytest.approx(gib(_ANCHOR_MP, _ANCHOR_MP))
        assert gib(0.25, 0.25) == pytest.approx(gib(0.5))  # the sum of the capped sizes
        assert gib(0.5) < gib(_ANCHOR_MP)

    def test_a_9b_model_reads_the_references_into_its_streamed_denoise(self):
        base = MemoryJob(media="image", family="flux2_klein", model_size="9b", quantize=8, width=2048, height=2048)
        weights = _weights("klein9b")

        plain = judge(base, _CARD, weights)
        referenced = judge(replace(base, reference_megapixels=(_ANCHOR_MP,) * 3), _CARD, weights)

        assert plain.placement == referenced.placement == "streamed"
        assert referenced.gpu_bytes > plain.gpu_bytes


class TestUnlistedFamilies:
    """A model with no measured row reads its weights in the placement plus the generic working memory: no size or reference terms."""

    @pytest.mark.parametrize("family", ["flux1", "flux2", "unknown", None])
    def test_a_family_with_no_row_does_not_grow_with_the_size_or_the_references(self, family):
        weights = _weights("klein9b")

        small = judge(MemoryJob(media="image", family=family, quantize=4, width=1024, height=1024), _CARD, weights)
        large = judge(MemoryJob(media="image", family=family, quantize=4, width=2048, height=2048, reference_megapixels=(_ANCHOR_MP,) * 3), _CARD, weights)

        assert small.gpu_bytes == large.gpu_bytes

    def test_klein_with_an_unknown_size_reads_like_a_family_with_no_row(self):
        weights = _weights("klein9b")

        unsized = judge(MemoryJob(media="image", family="flux2_klein", quantize=8, width=1600, height=1600), _CARD, weights)
        unlisted = judge(MemoryJob(media="image", family="flux1", quantize=8, width=1600, height=1600), _CARD, weights)

        assert unsized == unlisted

    def test_the_generic_estimate_grows_with_the_weights_of_its_placement(self):
        small, large = _packed(2.0, 3.0), _packed(3.0, 3.0)

        assert _gib(large, RESIDENT, family="flux1", size=None) == pytest.approx(_gib(small, RESIDENT, family="flux1", size=None) + 1.0)
        assert _gib(large, STREAMED, family="flux1", size=None) == _gib(small, STREAMED, family="flux1", size=None)


class TestPlacement:
    def test_only_q4_can_be_resident_and_the_other_levels_stream(self):
        assert image_placement_candidates(4) == ("resident", "turns")
        assert image_placement_candidates(8) == ("streamed",)
        assert image_placement_candidates(None) == ("streamed",)

    def test_takes_the_first_candidate_that_fits(self):
        weights = _weights("klein4b")
        resident = judge(_image_job("klein4b", 4, 1024, 1024), _CARD, weights)
        turns = judge(_image_job("klein4b", 4, 1024, 1024), MemoryBudget(6 * _GIB, int(30.3 * _GIB)), weights)

        assert resident.placement == "resident" and resident.status == "fits"
        assert turns.placement == "turns"

    def test_takes_the_candidate_that_needs_the_least_when_none_fits(self):
        weights = _weights("klein9b")

        result = judge(_image_job("klein9b", 4, 1024, 1024, references=4), MemoryBudget(4 * _GIB, int(30.3 * _GIB)), weights)

        assert result.placement == "turns"
        assert result.status == "too_large"

    @pytest.mark.parametrize("quantize", [8, None])
    def test_q8_and_unquantized_stream_even_on_a_card_that_could_hold_them(self, quantize):
        result = judge(_image_job("klein9b", quantize, 1024, 1024), MemoryBudget(24 * _GIB, int(30.3 * _GIB)), _weights("klein9b"))

        assert result.placement == "streamed"

    def test_a_forced_resident_placement_still_estimates_q8_for_the_bench(self):
        card = MemoryBudget(24 * _GIB, int(30.3 * _GIB))
        weights = _weights("klein9b")

        forced = judge_forced_placement(_image_job("klein9b", 8, 1024, 1024), card, weights, "resident")
        streamed = judge(_image_job("klein9b", 8, 1024, 1024), card, weights)

        assert forced.placement == "resident"
        assert forced.gpu_bytes > streamed.gpu_bytes  # every weight sits on the card
        assert forced.ram_bytes < streamed.ram_bytes  # and none is held in system memory

    def test_a_job_that_just_fits_the_card_keeps_the_resident_placement_and_one_that_just_does_not_takes_turns(self):
        weights = _weights("zit")
        need = cuda_image_placement(weights.components, 4, 10 * _GIB, family="zimage", job_megapixels=_ANCHOR_MP)[1]
        fits_line = (need + CUDA_OFFSET_BYTES) / 0.9

        at_the_line = judge(_image_job("zit", 4, 1024, 1024), MemoryBudget(int(fits_line) + 1, int(30.3 * _GIB)), weights)
        below_it = judge(_image_job("zit", 4, 1024, 1024), MemoryBudget(int(fits_line) - 1, int(30.3 * _GIB)), weights)

        assert (at_the_line.placement, at_the_line.status) == ("resident", "fits")
        assert below_it.placement == "turns"

    def test_a_stored_quant_loads_at_its_own_level_whatever_is_asked(self):
        weights = _weights("klein9b")
        stored = MemoryWeights(components=weights.components, stored_bits=4)

        asked_bf16 = judge(_image_job("klein9b", None, 1024, 1024), _CARD, stored)

        assert asked_bf16 == judge(_image_job("klein9b", 4, 1024, 1024), _CARD, weights)

    def test_a_forced_placement_is_taken_as_given(self):
        job, weights = _image_job("zit", 4, 1024, 1024), _weights("zit")
        result = judge_forced_placement(job, _CARD, weights, "turns")

        assert result.placement == "turns"
        assert result.gpu_bytes < judge_forced_placement(job, _CARD, weights, "resident").gpu_bytes


class TestKlein4bSizeAndReferences:
    """Klein 4B's figure follows the size and the references in every placement, so the tag and the placement follow them too."""

    _EIGHT_GB_CARD = MemoryBudget(int(7.71 * _GIB), int(30.3 * _GIB))

    @staticmethod
    def _figure(quantize: int | None, width: int, height: int, references: int = 0, placement: str | None = None) -> int:
        job, weights = _image_job("klein4b", quantize, width, height, references), _weights("klein4b")
        result = judge_forced_placement(job, _CARD, weights, placement) if placement else judge(job, _CARD, weights)
        return result.gpu_bytes

    @pytest.mark.parametrize(("quantize", "placement"), [(4, "resident"), (4, "turns"), (8, "streamed")])
    def test_the_figure_grows_with_the_references_and_with_the_size(self, quantize, placement):
        plain = self._figure(quantize, 1024, 1024, placement=placement)
        large = self._figure(quantize, 2048, 2048, placement=placement)
        with_references = self._figure(quantize, 1024, 1024, 4, placement)
        large_with_references = self._figure(quantize, 2048, 2048, 4, placement)

        assert large_with_references > max(large, with_references)
        assert max(large, with_references) >= plain
        assert large_with_references > plain

    @pytest.mark.parametrize("placement", ["resident", "turns"])
    def test_a_q4_job_grows_past_the_decode_and_the_reference_encode_with_the_size_and_the_references(self, placement):
        sizes = [self._figure(4, side, side, placement=placement) for side in (1024, 1600, 2048)]
        references = [self._figure(4, 1024, 1024, count, placement) for count in (0, 1, 2, 4)]

        assert sizes[0] <= sizes[1] < sizes[2]
        assert references[0] <= references[1] <= references[2] < references[3]  # the reference encode is the peak of one or two references

    def test_a_streamed_job_stays_on_the_decode_until_the_size_and_the_references_outgrow_it(self):
        plain = self._figure(8, 1024, 1024)

        assert self._figure(8, 2048, 2048) == plain  # the decode is the peak
        assert self._figure(8, 1024, 1024, 4) == plain
        assert self._figure(8, 1600, 1600, 4) > plain
        assert self._figure(8, 2048, 2048, 4) > self._figure(8, 1600, 1600, 4)

    def test_a_small_card_keeps_a_plain_job_resident_and_moves_a_large_or_referenced_one_to_turns(self):
        weights = _weights("klein4b")

        def verdict(width: int, height: int, references: int = 0):
            return judge(_image_job("klein4b", 4, width, height, references), self._EIGHT_GB_CARD, weights)

        plain = verdict(1024, 1024)
        large = verdict(2048, 2048)
        referenced = verdict(1600, 1600, 4)

        assert (plain.placement, plain.status) == ("resident", "fits")
        assert (large.placement, large.status) == ("turns", "fits")
        assert (referenced.placement, referenced.status) == ("turns", "fits")

    def test_resident_at_the_large_size_no_longer_reads_fits_on_a_small_card(self):
        job, weights = _image_job("klein4b", 4, 2048, 2048, 4), _weights("klein4b")

        forced = judge_forced_placement(job, self._EIGHT_GB_CARD, weights, "resident")

        assert forced.status != "fits"

    def test_the_largest_job_still_fits_a_ten_gigabyte_card(self):
        job, weights = _image_job("klein4b", 4, 2048, 2048, 4), _weights("klein4b")
        for card in (_CARD, MemoryBudget(int(9.64 * _GIB), int(30.3 * _GIB))):
            assert judge(job, card, weights).status != "too_large"


class TestStatus:
    def test_the_gpu_bands_are_ninety_percent_and_the_whole_card(self):
        def status(figure_gib: float) -> str:
            peak = int(figure_gib * _GIB) - CUDA_OFFSET_BYTES
            return judge(MemoryJob(media="audio"), _CARD, MemoryWeights(audio_peak_bytes=peak)).status

        assert [status(8.9), status(9.1), status(10.0), status(10.1)] == ["fits", "tight", "tight", "too_large"]

    @pytest.mark.parametrize(("ram_gib", "status"), [(25.0, "fits"), (22.0, "tight"), (18.0, "too_large")])
    def test_the_ram_thresholds_are_eighty_percent_and_all_of_it(self, ram_gib, status):
        result = judge(_image_job("klein9b", 8, 1024, 1024), MemoryBudget(10 * _GIB, int(ram_gib * _GIB)), _weights("klein9b"))

        assert (result.status, result.resource) == (status, "gpu" if status == "fits" else "ram")

    def test_unquantized_weights_mapped_from_disk_are_never_too_large_on_ram(self):
        result = judge(_image_job("klein9b", None, 1024, 1024), MemoryBudget(10 * _GIB, 16 * _GIB), _weights("klein9b"))

        assert (result.status, result.resource, result.placement) == ("tight", "ram", "streamed")

    def test_the_worse_status_wins_and_names_its_resource(self):
        gpu_worse = judge(_image_job("klein9b", 4, 1024, 1024, references=4), MemoryBudget(8 * _GIB, 30 * _GIB), _weights("klein9b"))
        ram_worse = judge(_image_job("klein9b", 4, 1024, 1024), MemoryBudget(10 * _GIB, 11 * _GIB), _weights("klein9b"))

        assert (gpu_worse.status, gpu_worse.resource) == ("too_large", "gpu")  # the GPU needs more than 8, the RAM 11.4 of 30
        assert (ram_worse.status, ram_worse.resource) == ("too_large", "ram")  # the GPU fits 10, the RAM needs 11.4 of 11

    def test_a_tie_goes_to_the_gpu(self):
        both_tight = judge(_image_job("klein9b", 4, 1024, 1024, references=3), MemoryBudget(int(8.2 * _GIB), 15 * _GIB), _weights("klein9b"))
        both_fit = judge(_image_job("klein9b", 4, 1024, 1024), MemoryBudget(10 * _GIB, 30 * _GIB), _weights("klein9b"))

        assert (both_tight.status, both_tight.resource) == ("tight", "gpu")
        assert (both_fit.status, both_fit.resource) == ("fits", "gpu")

    def test_the_cuda_offset_is_added_once_to_the_gpu_figure_of_every_medium(self):
        weights = _weights("zit")
        placement, need = cuda_image_placement(weights.components, 8, 10 * _GIB, family="zimage", job_megapixels=_ANCHOR_MP)
        image = judge(_image_job("zit", 8, 1024, 1024), _CARD, weights)
        audio = judge(MemoryJob(media="audio"), _CARD, MemoryWeights(audio_peak_bytes=_GIB))

        assert image.gpu_bytes == need + CUDA_OFFSET_BYTES
        assert audio.gpu_bytes == _GIB + CUDA_OFFSET_BYTES


class TestMac:
    _MAC = MemoryBudget(10 * _GIB, None, 16 * _GIB)

    @pytest.mark.parametrize(("peak_gib", "status"), [(11.9, "fits"), (12.1, "tight"), (15.9, "tight"), (16.1, "too_large")])
    def test_the_bands_are_three_quarters_of_the_memory_and_all_of_it(self, peak_gib, status):
        result = judge(MemoryJob(media="audio"), self._MAC, MemoryWeights(audio_peak_bytes=int(peak_gib * _GIB)))

        assert (result.status, result.resource, result.ram_bytes) == (status, "unified", None)
        assert result.gpu_bytes == int(peak_gib * _GIB)  # a Mac figure carries no offset

    def test_a_mac_that_does_not_know_its_memory_cannot_be_judged(self):
        with pytest.raises(ValueError, match="unified"):
            judge(MemoryJob(media="audio"), MemoryBudget(10 * _GIB, None), MemoryWeights(audio_peak_bytes=_GIB))

    def test_an_image_needs_its_weights_and_a_working_memory_and_grows_with_them_one_for_one(self):
        def need(transformer_elements_gib: float) -> int:
            components = {"transformer": WeightTotals(float_matrix_elements=_elements(transformer_elements_gib)), "vae": _vae(0.1, bytes_per_element=4)}
            return judge(_image_job("zit", None, 1024, 1024), self._MAC, MemoryWeights(components=components)).gpu_bytes

        assert need(2.0) > 4.0 * _GIB + 0.1 * _GIB  # more than the bfloat16 transformer and the float32 VAE: the working memory is on top
        assert need(3.0) - need(2.0) == pytest.approx(2.0 * _GIB, abs=1024)  # bfloat16: 2 bytes a weight
        assert judge(_image_job("zit", None, 1024, 1024), self._MAC, MemoryWeights(components={"transformer": WeightTotals(float_matrix_elements=_elements(2.0))})).placement == "unified"

    @pytest.mark.parametrize(("model", "quantize"), [("klein4b", 4), ("klein4b", 8), ("klein9b", 4), ("klein9b", 8), ("zit", 4), ("zit", 8), ("krea2", 4), ("krea2", 8), ("ideo", 4), ("ideo", 8)])
    def test_a_larger_size_needs_more_and_a_smaller_one_needs_no_less_than_one_megapixel(self, model, quantize):
        weights = _mac_weights(model)
        need = {size: judge(_image_job(model, quantize, *size), self._MAC, weights).gpu_bytes for size in ((768, 768), (1024, 1024), (1600, 1600), (2048, 2048))}

        flat_to_1600 = (model, quantize) == ("klein9b", 8)  # measured: its load phase holds the most up to 1600x1600
        assert need[(768, 768)] == need[(1024, 1024)]
        assert (need[(1024, 1024)] == need[(1600, 1600)]) if flat_to_1600 else (need[(1024, 1024)] < need[(1600, 1600)])
        assert need[(1600, 1600)] < need[(2048, 2048)]

    @pytest.mark.parametrize("model", ["klein9b", "zit"])
    def test_a_stored_quant_is_judged_at_its_own_level_whatever_is_asked(self, model):
        weights = _mac_weights(model)
        stored = MemoryWeights(components=weights.components, stored_bits=4)

        asked_bf16 = judge(_image_job(model, None, 1024, 1024), self._MAC, stored)

        assert asked_bf16 == judge(_image_job(model, 4, 1024, 1024), self._MAC, weights)
        assert asked_bf16.gpu_bytes < judge(_image_job(model, None, 1024, 1024), self._MAC, weights).gpu_bytes

    @pytest.mark.parametrize("quantize", [4, 8])
    def test_references_raise_the_need_of_klein_and_each_further_one_never_lowers_it(self, quantize):
        weights = _mac_weights("klein9b")
        need = [judge(_image_job("klein9b", quantize, 1024, 1024, references), self._MAC, weights).gpu_bytes for references in range(5)]

        assert need[0] < need[1] <= need[2] <= need[3] <= need[4]

    @pytest.mark.parametrize("model", ["klein4b", "klein9b", "zit", "krea2"])
    @pytest.mark.parametrize("size", [(1024, 1024), (1600, 1600)])
    def test_unquantized_needs_more_than_q8_which_needs_more_than_q4(self, model, size):
        weights = _mac_weights(model)
        need = [judge(_image_job(model, quantize, *size), self._MAC, weights).gpu_bytes for quantize in (4, 8, None)]

        assert need[0] < need[1] < need[2]

    def test_an_unlisted_level_reads_its_own_models_size_term_and_an_unlisted_model_has_none(self):
        weights = _mac_weights("krea2")
        krea_q8 = judge(_image_job("krea2", 8, 1600, 1600), self._MAC, weights).gpu_bytes
        krea_none = judge(_image_job("krea2", None, 1600, 1600), self._MAC, weights).gpu_bytes
        unlisted = [judge(MemoryJob(media="image", family="flux1", quantize=8, width=size, height=size), self._MAC, _mac_weights("klein9b")).gpu_bytes for size in (1024, 2048)]

        # Same growth with the size as q8, on a bigger set of weights.
        assert krea_none - krea_q8 == judge(_image_job("krea2", None, 1024, 1024), self._MAC, weights).gpu_bytes - judge(_image_job("krea2", 8, 1024, 1024), self._MAC, weights).gpu_bytes
        assert unlisted[0] == unlisted[1]

    @pytest.mark.parametrize("model", ["klein9b", "zit"])
    def test_a_job_that_starts_from_a_picture_needs_more_than_the_same_job_drawn_from_noise(self, model):
        weights = _mac_weights(model)
        plain = judge(_image_job(model, 8, 1600, 1600), self._MAC, weights)
        refined = judge(_image_job(model, 8, 1600, 1600, from_picture=True), self._MAC, weights)

        assert refined.gpu_bytes > plain.gpu_bytes

    # (model, quantize, width, height, references, from picture, status on a 32 GB Mac, status on a 16 GB Mac): jobs the Mac bench ran, away from a threshold.
    _MAC_JOBS = [
        ("klein4b", 4, 1024, 1024, 0, False, "fits", "fits"),
        ("klein4b", 4, 1600, 1600, 0, False, "fits", "fits"),
        ("klein4b", 4, 1024, 1024, 1, False, "fits", "fits"),
        ("zit", 4, 1600, 1600, 0, False, "fits", "fits"),
        ("klein9b", 4, 1024, 1024, 0, False, "fits", "fits"),
        ("klein9b", 4, 1024, 1024, 3, False, "fits", "tight"),
        ("klein9b", 4, 2048, 2048, 1, False, "fits", "too_large"),
        ("klein9b", 8, 1600, 1600, 0, False, "fits", "too_large"),
        ("klein9b", None, 1024, 1024, 0, False, "tight", "too_large"),  # measured 31.43 GiB at both sizes: under the whole 32 GB, though over its working set
        ("klein9b", None, 1600, 1600, 0, False, "tight", "too_large"),
        ("ideo", 4, 1024, 1024, 0, False, "fits", "fits"),
        ("ideo", 8, 1600, 1600, 0, False, "fits", "too_large"),
        ("moss", 8, 1600, 1600, 0, False, "fits", "fits"),
        ("krea2", 4, 1024, 1024, 0, False, "fits", "too_large"),
        ("krea2", 4, 1600, 1600, 0, False, "tight", "too_large"),
        ("krea2", 8, 1600, 1600, 0, False, "too_large", "too_large"),
        ("krea2", None, 1600, 1600, 0, False, "too_large", "too_large"),
    ]

    @pytest.mark.parametrize(("model", "quantize", "width", "height", "references", "from_picture", "on_32", "on_16"), _MAC_JOBS)
    def test_a_job_reads_as_on_a_32_gb_and_a_16_gb_mac(self, model, quantize, width, height, references, from_picture, on_32, on_16):
        job = _image_job(model, quantize, width, height, references, from_picture=from_picture)
        weights = _mac_weights(model)

        assert (judge(job, _MAC_32, weights).status, judge(job, _MAC_16, weights).status) == (on_32, on_16)

    def test_an_upscaled_z_image_does_not_fit_a_16_gb_mac(self):
        # Measured 23.7 GiB on a 32 GB Mac, close to that Mac's Fits line (24 GiB), so only the 16 GB reading is asserted.
        assert judge(_image_job("zit", 8, 1600, 1600, from_picture=True), _MAC_16, _mac_weights("zit")).status == "too_large"

    def test_moss_counts_its_loras_and_other_families_count_none(self):
        components = {"text_encoder": WeightTotals(float_matrix_elements=_elements(1.0)), "transformer": WeightTotals(float_matrix_elements=_elements(2.0)), "vae": _vae(0.1, bytes_per_element=4)}
        plain = judge(MemoryJob(media="image", family="moss", width=1024, height=1024), self._MAC, MemoryWeights(components=components, turbo_lora_bytes=_elements(1.0)))
        with_lora = judge(MemoryJob(media="image", family="moss", width=1024, height=1024, lora_bytes=_elements(1.0)), self._MAC, MemoryWeights(components=components, turbo_lora_bytes=_elements(1.0)))
        klein = MemoryWeights(components=components)
        klein_plain = judge(_image_job("klein9b", None, 1024, 1024), self._MAC, klein)
        klein_lora = judge(_image_job("klein9b", None, 1024, 1024, lora_bytes=_elements(1.0)), self._MAC, klein)

        assert with_lora.gpu_bytes - plain.gpu_bytes == pytest.approx(_GIB, abs=1024)
        assert klein_lora == klein_plain

    def test_ideogram_4_needs_its_weights_and_a_small_working_memory_at_one_megapixel(self):
        components = {"transformer": WeightTotals(float_matrix_elements=_elements(2.0))}

        ideogram = judge(MemoryJob(media="image", family="ideogram4", width=1024, height=1024), self._MAC, MemoryWeights(components=components))
        other = judge(_image_job("zit", None, 1024, 1024), self._MAC, MemoryWeights(components=components))

        assert 0 <= (ideogram.gpu_bytes - other.gpu_bytes) / _GIB < 2.0


class TestVideoAndAudio:
    _LTX_SYSTEM = estimate_ltx_cuda_system_memory(block_matrices=int(18e9), transformer_rest=int(1e9), connectors=int(0.5e9), text_encoder=int(11e9), quantize=4)

    @pytest.fixture(autouse=True)
    def _default_decode_budget(self, monkeypatch):
        monkeypatch.delenv("LTX2_VAE_DECODE_BUDGET_GB", raising=False)

    @pytest.mark.parametrize("low_memory", [True, False])
    def test_ltx_on_cuda_is_its_own_estimate_and_the_cuda_offset(self, low_memory):
        job = MemoryJob(media="video", family="ltx", width=768, height=448, frames=121, low_memory=low_memory, upscale=True)
        budget = MemoryBudget(10 * _GIB, 32 * _GIB)
        estimate = estimate_ltx_cuda_clip_memory(self._LTX_SYSTEM, width=768, height=448, frames=121, low_memory=low_memory, upscale=True)

        result = judge(job, budget, MemoryWeights(ltx_cuda=self._LTX_SYSTEM))

        assert (result.gpu_bytes, result.ram_bytes) == (estimate.gpu_bytes + CUDA_OFFSET_BYTES, estimate.system_bytes)
        assert result.placement == ("streamed" if low_memory else "resident")
        assert result.status == _cuda_status(estimate.gpu_bytes, estimate.system_bytes, budget)
        assert result.expected_seconds is None

    @pytest.mark.parametrize("system_gib", [64, 20, 16])
    def test_ltx_on_cuda_keeps_its_status_for_the_nearer_limit(self, system_gib):
        job = MemoryJob(media="video", family="ltx", width=1280, height=704, frames=241, low_memory=True)
        budget = MemoryBudget(8 * _GIB, system_gib * _GIB)
        estimate = estimate_ltx_cuda_clip_memory(self._LTX_SYSTEM, width=1280, height=704, frames=241, low_memory=True)

        result = judge(job, budget, MemoryWeights(ltx_cuda=self._LTX_SYSTEM))

        assert result.status == _cuda_status(estimate.gpu_bytes, estimate.system_bytes, budget)
        if result.status != "fits":
            gpu_rank, ram_rank = _status(estimate.gpu_bytes, budget.gpu_bytes, 0.9, 1.0), _status(estimate.system_bytes, budget.system_bytes, 0.8, 1.0)
            assert result.resource == ("ram" if ram_rank > gpu_rank else "gpu")

    def test_ltx_on_a_mac_is_the_measured_curve_and_counts_the_upsampler_only_when_upscaling(self):
        figures = LtxMlxWeights(gemma=_elements(6.0), connector=_elements(1.0), transformer=_elements(10.0), decoders=_elements(0.5), upsampler=_elements(1.0))
        mac = MemoryBudget(20 * _GIB, None, 32 * _GIB)
        job = MemoryJob(media="video", family="ltx", width=704, height=448, frames=121, low_memory=True)

        plain = judge(job, mac, MemoryWeights(ltx_mlx=figures))
        upscaled = judge(MemoryJob(media="video", family="ltx", width=704, height=448, frames=121, low_memory=True, upscale=True), mac, MemoryWeights(ltx_mlx=figures))

        assert plain.gpu_bytes == ltx_mlx_clip_bytes(
            LtxMlxWeights(figures.gemma, figures.connector, figures.transformer, figures.decoders), width=704, height=448, frames=121, low_memory=True, unified_bytes=32 * _GIB, env={}
        )
        assert upscaled.gpu_bytes == ltx_mlx_clip_bytes(figures, width=704, height=448, frames=121, low_memory=True, unified_bytes=32 * _GIB, env={})
        assert plain.status == _mac_status(plain.gpu_bytes, mac)
        assert (plain.resource, plain.ram_bytes) == ("unified", None)

    @pytest.mark.parametrize(
        ("upscale", "review", "mode", "references", "lora_gib"), [(False, False, "t2v", 0, 0.0), (True, False, "t2v", 0, 0.0), (True, True, "i2v", 0, 0.0), (False, False, "extend", 2, 0.5)]
    )
    def test_h3_on_cuda_is_its_own_estimate_and_expected_time(self, upscale, review, mode, references, lora_gib):
        loras = _elements(lora_gib)
        job = MemoryJob(media="video", family="h3", width=1024, height=576, frames=124, mode=mode, references=references, upscale=upscale, review=review, lora_bytes=loras, forwards=8)
        budget = MemoryBudget(12 * _GIB, 64 * _GIB)

        result = judge(job, budget, MemoryWeights())

        sizes = h3_nominal_references(references, 1024, 576)
        keep = result.placement == "kept"
        direct = estimate_h3_cuda_clip_memory(1024, 576, 124, mode=mode, loras_bytes=loras, upscale=upscale, keep_transformer=keep, references=sizes)
        assert (result.gpu_bytes, result.ram_bytes) == (direct.gpu_bytes + CUDA_OFFSET_BYTES, direct.system_bytes)
        assert result.status == _cuda_status(direct.gpu_bytes, direct.system_bytes, budget)
        seconds = h3_expected_seconds(1024, 576, 124, forwards=8, mode=mode, loras=loras > 0, upscale=upscale, keep_transformer=keep, references=sizes)
        assert result.expected_seconds == seconds

    def test_h3_on_a_mac_carries_its_expected_time(self):
        mac = MemoryBudget(24 * _GIB, None, 36 * _GIB)

        result = judge(MemoryJob(media="video", family="h3", width=768, height=448, frames=124, forwards=8), mac, MemoryWeights())

        assert result.expected_seconds == estimate_h3_mlx_seconds(768, 448, 124, forwards=8)
        assert result.gpu_bytes == estimate_h3_mlx_clip_memory(768, 448, 124)
        assert result.status == _mac_status(result.gpu_bytes, mac)
        assert result.resource == "unified"

    @pytest.mark.parametrize("low_memory", [True, False])
    def test_ltx_loras_on_cuda_ride_with_the_transformer(self, low_memory):
        job = MemoryJob(media="video", family="ltx", width=768, height=448, frames=121, low_memory=low_memory)
        budget = MemoryBudget(24 * _GIB, 64 * _GIB)
        weights = MemoryWeights(ltx_cuda=self._LTX_SYSTEM)

        plain = judge(job, budget, weights)
        loaded = judge(replace(job, lora_bytes=5 * _GIB), budget, weights)

        # The text encoder (11 GB) sets the system memory until the transformer and its adapters outgrow it.
        assert loaded.ram_bytes > plain.ram_bytes
        assert loaded.ram_bytes - plain.ram_bytes == max(self._LTX_SYSTEM.transformer_bytes + 5 * _GIB, self._LTX_SYSTEM.text_encoder_bytes) - max(
            self._LTX_SYSTEM.transformer_bytes, self._LTX_SYSTEM.text_encoder_bytes
        )
        # On the GPU only a transformer that stays there carries them; a streamed one moves its adapters a block at a time.
        assert loaded.gpu_bytes - plain.gpu_bytes == (0 if low_memory else 5 * _GIB)

    # The packs' components in GiB, as the bench's 32 GiB Mac read them.
    _LTX4 = LtxMlxWeights(gemma=_elements(7.477), connector=_elements(5.909), transformer=_elements(10.535), decoders=_elements(0.6), upsampler=_elements(0.927))
    _LTX8 = replace(_LTX4, transformer=_elements(19.173))
    _LTX25 = LtxMlxWeights(gemma=_elements(8.0), connector=_elements(1.0), transformer=_elements(11.0), decoders=_elements(0.6), upsampler=_elements(0.927), bundled_encoder=True)

    @pytest.mark.parametrize("mode", ["t2v", "i2v", "extend"])
    @pytest.mark.parametrize(
        ("pack", "unified_gib", "status"), [("ltx4", 16, "tight"), ("ltx8", 16, "tight"), ("ltx25", 16, "tight"), ("ltx4", 24, "fits"), ("ltx8", 24, "fits"), ("ltx25", 24, "fits")]
    )
    def test_ltx_on_a_streaming_mac_reads_the_same_status_for_every_job(self, pack, unified_gib, status, mode):
        job = MemoryJob(media="video", family="ltx", width=768, height=448, frames=121, low_memory=True, mode=mode)
        mac = MemoryBudget(unified_gib * _GIB, None, unified_gib * _GIB)

        result = judge(job, mac, MemoryWeights(ltx_mlx=getattr(self, f"_{pack.upper()}")))

        assert result.status == status
        assert result.resource == "unified"

    def test_ltx_on_a_32_gib_mac_keeps_the_resident_terms(self):
        job = MemoryJob(media="video", family="ltx", width=768, height=448, frames=121, low_memory=True)
        mac = MemoryBudget(32 * _GIB, None, 32 * _GIB)

        result = judge(job, mac, MemoryWeights(ltx_mlx=self._LTX4))

        assert result.gpu_bytes == ltx_mlx_clip_bytes(self._LTX4, width=768, height=448, frames=121, low_memory=True, unified_bytes=32 * _GIB, env={})
        assert result.gpu_bytes / _GIB > 16

    def test_ltx_loras_on_a_mac_count_with_the_transformer_in_every_stage_that_holds_it(self):
        figures = LtxMlxWeights(gemma=_elements(6.0), connector=_elements(1.0), transformer=_elements(14.0), decoders=_elements(0.5), upsampler=_elements(1.0))
        mac = MemoryBudget(40 * _GIB, None, 64 * _GIB)
        weights = MemoryWeights(ltx_mlx=figures)

        for low_memory in (True, False):
            job = MemoryJob(media="video", family="ltx", width=704, height=448, frames=121, low_memory=low_memory, upscale=True)
            plain = judge(job, mac, weights)
            loaded = judge(replace(job, lora_bytes=_elements(1.5)), mac, weights)

            assert loaded.gpu_bytes - plain.gpu_bytes == pytest.approx(1.5 * _GIB, abs=2)

    def test_h3_loras_on_a_mac_add_nothing_because_the_adapter_streams_from_disk(self):
        mac = MemoryBudget(24 * _GIB, None, 36 * _GIB)
        job = MemoryJob(media="video", family="h3", width=768, height=448, frames=124, forwards=8)

        assert judge(replace(job, lora_bytes=2 * _GIB), mac, MemoryWeights()) == judge(job, mac, MemoryWeights())

    @pytest.mark.parametrize(
        ("card_gib", "system_gib", "status", "resource"),
        [(5, 64, "tight", "gpu"), (24, 24, "tight", "ram"), (4, 64, "too_large", "gpu"), (24, 16, "too_large", "ram"), (4, 16, "too_large", "gpu"), (24, 64, "fits", "gpu")],
    )
    def test_ltx_on_cuda_names_the_resource_that_sets_the_status(self, card_gib, system_gib, status, resource):
        system = LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)
        job = MemoryJob(media="video", family="ltx", width=704, height=448, frames=121, low_memory=True)

        result = judge(job, MemoryBudget(card_gib * _GIB, system_gib * _GIB), MemoryWeights(ltx_cuda=system))

        # Weights beyond system memory are too large (the load is refused); a clip out of GPU memory names the GPU even when RAM is fuller.
        assert (result.status, result.resource) == (status, resource)

    _H3_MAC = MemoryBudget(24 * _GIB, None, 32 * _GIB)

    def test_h3_on_a_mac_reads_the_reference_clip_as_fitting_and_takes_23_minutes(self):
        result = judge(MemoryJob(media="video", family="h3", width=768, height=448, frames=124, forwards=8), self._H3_MAC, MemoryWeights())

        assert (result.status, result.gpu_bytes, result.expected_seconds) == ("fits", estimate_h3_mlx_clip_memory(768, 448, 124), 23 * 60)

    def test_h3_on_a_mac_takes_less_time_with_fewer_forwards(self):
        job = MemoryJob(media="video", family="h3", width=768, height=448, frames=124)

        assert judge(replace(job, forwards=4), self._H3_MAC, MemoryWeights()).expected_seconds < 23 * 60

    def test_h3_on_a_mac_needs_more_for_an_image_mode_and_for_references_at_a_small_clip(self):
        card = MemoryBudget(int(12.8 * _GIB), None, int(18.5 * _GIB))
        text = MemoryJob(media="video", family="h3", width=576, height=320, frames=73, forwards=8)

        plain, image, referenced = (judge(job, card, MemoryWeights()) for job in (text, replace(text, mode="i2v"), replace(text, references=1)))

        assert (plain.status, image.status, referenced.status) == ("fits", "tight", "tight")
        assert min(image.gpu_bytes, referenced.gpu_bytes) > plain.gpu_bytes
        assert referenced.expected_seconds > plain.expected_seconds

    def test_h3_refuses_a_length_it_cannot_take(self):
        with pytest.raises(ValueError):
            judge(MemoryJob(media="video", family="h3", width=768, height=448, frames=121), self._H3_MAC, MemoryWeights())

    _H3_CARD = MemoryBudget(10 * _GIB, 32 * _GIB)  # a clip's process memory is about 25 GiB, which reads Fits up to 80 % of 32 GiB

    def test_h3_on_cuda_reads_a_small_clip_as_fitting_in_under_five_minutes(self):
        result = judge(MemoryJob(media="video", family="h3", width=864, height=480, frames=124, forwards=8), self._H3_CARD, MemoryWeights())

        assert (result.status, result.resource) == ("fits", "gpu")
        assert abs(result.expected_seconds - 276) <= 2

    def test_h3_on_cuda_is_faster_with_fewer_forwards_and_needs_more_with_a_lora(self):
        plain = judge(MemoryJob(media="video", family="h3", width=864, height=480, frames=124, forwards=8), self._H3_CARD, MemoryWeights())
        tuned = judge(MemoryJob(media="video", family="h3", width=864, height=480, frames=124, forwards=4, mode="i2v", lora_bytes=_GIB), self._H3_CARD, MemoryWeights())

        assert tuned.expected_seconds < plain.expected_seconds
        assert tuned.gpu_bytes >= plain.gpu_bytes

    def test_h3_on_cuda_takes_longer_to_upscale_than_to_render_the_half_size_clip(self):
        upscaled = judge(MemoryJob(media="video", family="h3", width=1280, height=704, frames=124, forwards=8, upscale=True), self._H3_CARD, MemoryWeights())
        half = judge(MemoryJob(media="video", family="h3", width=640, height=352, frames=124, forwards=8), self._H3_CARD, MemoryWeights())

        assert upscaled.status == "fits"
        assert upscaled.expected_seconds > half.expected_seconds

    def test_only_h3_has_an_expected_time(self):
        assert judge(MemoryJob(media="audio"), _CARD, MemoryWeights(audio_peak_bytes=_GIB)).expected_seconds is None
        assert judge(_image_job("zit", 8, 1024, 1024), _CARD, _weights("zit")).expected_seconds is None

    def test_audio_is_its_measured_peak_and_the_cuda_offset_with_no_ram(self):
        peak = int(8.2 * _GIB)

        result = judge(MemoryJob(media="audio"), MemoryBudget(8 * _GIB, 16 * _GIB), MemoryWeights(audio_peak_bytes=peak))

        assert (result.gpu_bytes, result.ram_bytes, result.resource) == (peak + CUDA_OFFSET_BYTES, None, "gpu")
        assert result.status == "too_large"  # 8.2 GiB of an 8 GiB card


class TestMossRam:
    """MOSS-Image's RAM at every level, as the bench measured it (bench runs ``cuda-r11`` and ``cuda-r15``, a 10 GB card)."""

    @pytest.mark.parametrize("quantize", [4, 8, None])
    def test_every_level_reads_a_ram_figure(self, quantize):
        result = judge(_image_job("moss", quantize, 1024, 1024), _CARD, _weights("moss"))

        assert result.ram_bytes is not None

    def test_a_higher_level_needs_more_ram(self):
        need = {level: judge(_image_job("moss", level, 1024, 1024), _CARD, _weights("moss")).ram_bytes for level in (4, 8, None)}

        assert need[4] < need[8] < need[None]

    @pytest.mark.parametrize(("ram_gib", "status"), [(32.0, "fits"), (24.0, "fits"), (16.0, "tight"), (8.0, "tight")])
    def test_unquantized_reads_its_mapped_weights_and_is_never_too_large_on_ram(self, ram_gib, status):
        result = judge(_image_job("moss", None, 1024, 1024), MemoryBudget(10 * _GIB, int(ram_gib * _GIB)), _weights("moss"))

        assert (result.status, result.placement) == (status, "streamed")
        assert result.resource == ("gpu" if status == "fits" else "ram")

    def test_quantized_levels_read_too_large_where_unquantized_only_reads_tight(self):
        small = MemoryBudget(10 * _GIB, 10 * _GIB)

        assert judge(_image_job("moss", 8, 1024, 1024), small, _weights("moss")).status == "too_large"
        assert judge(_image_job("moss", None, 1024, 1024), small, _weights("moss")).status == "tight"

    def test_the_gpu_figure_does_not_depend_on_the_level(self):
        figures = {judge(_image_job("moss", level, 1600, 1600), _CARD, _weights("moss")).gpu_bytes for level in (4, 8, None)}

        assert len(figures) == 1

    def test_user_loras_sit_on_the_card_and_add_nothing_to_ram(self):
        plain = judge(_image_job("moss", 4, 1024, 1024), _CARD, _weights("moss"))
        loaded = judge(_image_job("moss", 4, 1024, 1024, lora_bytes=_elements(1.0)), _CARD, _weights("moss"))

        assert loaded.ram_bytes == plain.ram_bytes
        assert loaded.gpu_bytes - plain.gpu_bytes == pytest.approx(_GIB, abs=1024)


class TestLoraRam:
    """User LoRA files count in system memory wherever the weights are not on the card for good (bench run ``cuda-r15``)."""

    @pytest.mark.parametrize(("quantize", "placement"), [(4, "turns"), (8, "streamed")])
    def test_a_lora_adds_its_size_when_the_weights_wait_in_system_memory(self, quantize, placement):
        plain = judge(_image_job("klein9b", quantize, 1024, 1024), _CARD, _weights("klein9b"))
        loaded = judge(_image_job("klein9b", quantize, 1024, 1024, lora_bytes=_elements(1.0)), _CARD, _weights("klein9b"))

        assert loaded.placement == plain.placement == placement
        assert loaded.ram_bytes - plain.ram_bytes == pytest.approx(_GIB, abs=1024)

    def test_a_resident_job_keeps_a_share_of_its_lora_in_ram(self):
        plain = judge_forced_placement(_image_job("zit", 4, 1024, 1024), _CARD, _weights("zit"), "resident")
        loaded = judge_forced_placement(_image_job("zit", 4, 1024, 1024, lora_bytes=_elements(1.0)), _CARD, _weights("zit"), "resident")

        assert 0 < loaded.ram_bytes - plain.ram_bytes < _GIB

    def test_a_streamed_unquantized_job_counts_the_lora_whole_beside_its_mapped_weights(self):
        plain = judge(_image_job("klein9b", None, 1024, 1024), _CARD, _weights("klein9b"))
        loaded = judge(_image_job("klein9b", None, 1024, 1024, lora_bytes=_elements(1.0)), _CARD, _weights("klein9b"))

        assert loaded.ram_bytes - plain.ram_bytes == pytest.approx(_GIB, abs=1024)

    @pytest.mark.parametrize("quantize", [4, 8])
    def test_ideogram_4_counts_most_of_a_lora_in_ram_and_none_on_the_card(self, quantize):
        plain = judge(_image_job("ideo", quantize, 1024, 1024), _CARD, _weights("ideo"))
        loaded = judge(_image_job("ideo", quantize, 1024, 1024, lora_bytes=_elements(1.0)), _CARD, _weights("ideo"))

        assert loaded.gpu_bytes == plain.gpu_bytes
        assert 0.5 * _GIB < loaded.ram_bytes - plain.ram_bytes < _GIB

    def test_a_large_lora_can_make_the_ram_the_limit_for_ideogram_4_on_a_16_gb_machine(self):
        machine = MemoryBudget(10 * _GIB, 16 * _GIB)

        plain = judge(_image_job("ideo", 4, 1024, 1024), machine, _weights("ideo"))
        loaded = judge(_image_job("ideo", 4, 1024, 1024, lora_bytes=_elements(2.5)), machine, _weights("ideo"))

        assert plain.status == "fits"
        assert (loaded.status, loaded.resource) == ("tight", "ram")

    def test_a_large_lora_can_make_the_ram_the_limit_on_a_16_gb_machine(self):
        machine = MemoryBudget(10 * _GIB, 16 * _GIB)

        plain = judge(_image_job("klein9b", 4, 1024, 1024), machine, _weights("klein9b"))
        loaded = judge(_image_job("klein9b", 4, 1024, 1024, lora_bytes=_elements(2.5)), machine, _weights("klein9b"))

        assert plain.status == "fits"
        assert (loaded.status, loaded.resource, loaded.placement) == ("tight", "ram", "turns")


class TestDecodeTerms:
    def test_no_measured_family_reads_its_decode_as_needing_nothing(self):
        from zvisiongenerator.utils.memory_verdict import _IMAGE_TERMS

        assert all(terms.decode > 0 for terms in _IMAGE_TERMS.values())


class TestInvalidInput:
    def test_rejects_unknown_media(self):
        with pytest.raises(ValueError, match="media"):
            judge(MemoryJob(media="text"), _CARD, MemoryWeights())

    @pytest.mark.parametrize(
        ("job", "machine"),
        [
            (MemoryJob(media="audio"), _CARD),
            (MemoryJob(media="image", family="zimage"), _CARD),
            (MemoryJob(media="video", family="ltx"), _CARD),
            (MemoryJob(media="video", family="ltx"), MemoryBudget(10 * _GIB, None)),
            (MemoryJob(media="video", family="other"), _CARD),
        ],
    )
    def test_rejects_weights_the_job_cannot_be_judged_without(self, job, machine):
        with pytest.raises(ValueError):
            judge(job, machine, MemoryWeights())

    def test_rejects_an_unsupported_level(self):
        with pytest.raises(ValueError, match="quantize"):
            judge(_image_job("zit", 3, 1024, 1024), _CARD, _weights("zit"))
