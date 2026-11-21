"""Tests for the CUDA image placement the loader takes from the memory the card has free."""

from __future__ import annotations

import pytest

from tests.backends.cuda._image_weights import FAMILIES, image_components
from zvisiongenerator.backends.cuda.image.placement import choose_placement, model_label
from zvisiongenerator.utils.memory_grade import CUDA_GPU_RATIOS
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_BYTES
from zvisiongenerator.utils.memory_verdict import RESIDENT, STREAMED, TURNS, MemoryJob, cuda_image_gpu_bytes

_GIB = 1024**3
_IDLE_10_GB_CARD_FREE = round(9.40 * _GIB)  # the driver's 9.64 less the CUDA context, with no other program on the card
_TIGHT = CUDA_GPU_RATIOS[1]  # a candidate is taken when its figure is at most this share of the free memory: free memory already excludes other programs


def _choose(model: str, quantize: int | None, *, card_gib: float = 10, free_gib: float, job: MemoryJob | None = None):
    family, size = FAMILIES[model]
    return choose_placement(
        image_components(model), quantize, job=job or MemoryJob(media="image"), family=family, model_size=size, card_bytes=round(card_gib * _GIB), free_bytes=round(free_gib * _GIB)
    )


class TestQ4OnATenGigabyteCard:
    @pytest.mark.parametrize("model", ["zit", "klein4b"])
    def test_an_idle_card_keeps_the_nf4_pair_resident_without_a_warning(self, model):
        family, size = FAMILIES[model]
        choice = choose_placement(image_components(model), 4, job=MemoryJob(media="image"), family=family, model_size=size, card_bytes=10 * _GIB, free_bytes=_IDLE_10_GB_CARD_FREE)

        assert choice.placement == RESIDENT
        assert choice.warning is None

    @pytest.mark.parametrize("model", ["zit", "klein4b"])
    def test_with_less_free_the_text_encoder_and_transformer_take_turns_and_one_warning_says_so(self, model):
        choice = _choose(model, 4, free_gib=6.0)

        assert choice.placement == TURNS
        assert choice.warning is not None
        assert "6.0 GB of GPU memory is free" in choice.warning
        assert "take turns on the GPU" in choice.warning
        assert "may run out" not in choice.warning

    def test_the_estimate_plus_the_cuda_offset_must_not_be_above_the_free_memory(self):
        need = cuda_image_gpu_bytes("resident", image_components("zit"), 4, family="zimage", job_megapixels=MemoryJob(media="image").megapixels)
        taken_from = (need + CUDA_OFFSET_BYTES) / _TIGHT

        assert _choose("zit", 4, free_gib=taken_from / _GIB + 0.001).placement == RESIDENT
        assert _choose("zit", 4, free_gib=taken_from / _GIB - 0.001).placement == TURNS

    def test_a_figure_a_little_below_the_free_memory_is_taken_without_a_second_margin(self):
        need = cuda_image_gpu_bytes("resident", image_components("zit"), 4, family="zimage", job_megapixels=MemoryJob(media="image").megapixels)
        free = (need + CUDA_OFFSET_BYTES) * 1.05  # above the card's Fits line, but free memory already excludes other programs

        choice = _choose("zit", 4, free_gib=free / _GIB)

        assert choice.placement == RESIDENT
        assert choice.warning is None


class TestEveryCandidateAboveFreeMemory:
    def test_the_last_candidate_is_taken_and_the_warning_says_it_may_run_out(self):
        choice = _choose("klein4b", 4, free_gib=3.0)

        assert choice.placement == TURNS
        assert "take turns on the GPU" in choice.warning
        assert "may run out of GPU memory" in choice.warning

    def test_a_placement_the_tag_already_assumed_still_warns_that_it_may_run_out(self):
        """Klein 9B at q4 on a 6 GB card is too large resident, so the tag says turns; with too little free memory even that may fail."""
        choice = _choose("klein9b", 4, card_gib=6, free_gib=2.0)

        assert choice.placement == TURNS
        assert choice.warning.startswith("Only 2.0 GB of GPU memory is free")
        assert "may run out of GPU memory" in choice.warning

    def test_streaming_that_does_not_fit_the_free_memory_says_what_is_free_and_the_risk(self):
        choice = _choose("klein9b", 8, card_gib=24, free_gib=3.1)

        assert choice.placement == STREAMED
        assert choice.warning == "Only 3.1 GB of GPU memory is free, which is less than FLUX.2 Klein 9B needs. It may run out of GPU memory."


class TestNoteForTheJobCard:
    """The short sentence the running job's card shows, with the same free figure as the log line."""

    def test_stepping_down_says_the_job_runs_slower(self):
        assert _choose("zit", 4, free_gib=6.0).note == "Only 6.0 GB of GPU memory is free, so this runs slower."

    def test_nothing_that_fits_says_the_job_may_run_out_of_memory_whether_or_not_it_stepped_down(self):
        assert _choose("klein4b", 4, free_gib=3.0).note == "Only 3.0 GB of GPU memory is free. This may run out of memory."
        assert _choose("klein9b", 4, card_gib=6, free_gib=2.0).note == "Only 2.0 GB of GPU memory is free. This may run out of memory."
        assert _choose("klein9b", 8, card_gib=24, free_gib=3.1).note == "Only 3.1 GB of GPU memory is free. This may run out of memory."

    def test_a_placement_that_needs_no_warning_has_no_note(self):
        assert _choose("zit", 4, free_gib=_IDLE_10_GB_CARD_FREE / _GIB).note is None


class TestLevelsThatStream:
    def test_klein_9b_q8_streams_even_on_a_24_gb_card_without_a_warning(self):
        """Resident q8 and unquantized models are off until a run on a 16 to 24 GB card backs them."""
        choice = _choose("klein9b", 8, card_gib=24, free_gib=22.5)

        assert choice.placement == STREAMED
        assert choice.warning is None

    def test_klein_9b_q8_streams_on_a_10_gb_card_without_a_warning(self):
        """The tag already says streamed there, so nothing was stepped down from."""
        choice = _choose("klein9b", 8, free_gib=8.7)

        assert choice.placement == STREAMED
        assert choice.warning is None

    def test_streaming_that_fits_the_free_memory_needs_no_warning_however_little_is_free(self):
        choice = _choose("klein9b", 8, card_gib=24, free_gib=4.5)

        assert choice.placement == STREAMED
        assert choice.warning is None

    @pytest.mark.parametrize("card_gib", [10, 24, 48])
    def test_unquantized_weights_stream_like_q8_on_every_card(self, card_gib):
        assert _choose("klein9b", None, card_gib=card_gib, free_gib=card_gib - 2).placement == STREAMED
        assert _choose("klein9b", 8, card_gib=card_gib, free_gib=card_gib - 2).placement == STREAMED


class TestTheJobChangesTheNeed:
    def test_references_and_size_count_against_the_free_memory(self):
        components = image_components("klein9b")
        sizing = {"family": "flux2_klein", "model_size": "9b", "job_megapixels": 1.048576}
        small_job = MemoryJob(media="image")
        large_job = MemoryJob(media="image", width=2048, height=2048, reference_megapixels=(1.048576,) * 3)
        small_need = cuda_image_gpu_bytes("turns", components, 4, **sizing)
        free = (small_need + CUDA_OFFSET_BYTES) / _TIGHT / _GIB * 1.001  # what the small job needs to be taken

        small = _choose("klein9b", 4, free_gib=free, job=small_job)
        large = _choose("klein9b", 4, free_gib=free, job=large_job)

        assert small.warning is None
        assert large.placement == TURNS and "may run out of GPU memory" in large.warning

    def test_lora_bytes_count_with_the_transformer(self):
        assert _choose("zit", 4, free_gib=9.4).placement == RESIDENT
        assert _choose("zit", 4, free_gib=9.4, job=MemoryJob(media="image", lora_bytes=2 * _GIB)).placement == TURNS


@pytest.mark.parametrize(
    ("family", "size", "label"),
    [
        ("flux2_klein", "4b", "FLUX.2 Klein 4B"),
        ("flux2_klein", None, "FLUX.2 Klein"),
        ("zimage", None, "Z-Image"),
        ("krea2", None, "Krea 2"),
        ("flux1", None, "FLUX.1"),
        ("unknown", None, "this model"),
        (None, None, "this model"),
    ],
)
def test_model_label(family, size, label):
    assert model_label(family, size) == label
