"""Tests for the sentences and warnings a memory verdict is worded as."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.memory_verdict import MemoryVerdict
from zvisiongenerator.utils.memory_wording import (
    control_title,
    judged_details,
    judged_heading,
    judged_title,
    level_label,
    memory_figures,
    memory_rows_line,
    memory_sentence,
    memory_warning,
    status_word,
    table_heading,
)
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

_CARD = MemoryBudget(10 * GIB, system_bytes=30 * GIB)
_MAC = MemoryBudget(21 * GIB, unified_bytes=32 * GIB)


def _verdict(status: str, resource: str = "gpu", gpu_gb: float = 10.4, ram_gb: float | None = None) -> MemoryVerdict:
    return MemoryVerdict(round(gpu_gb * GIB), None if ram_gb is None else round(ram_gb * GIB), "resident", status, resource)


class TestSentence:
    @pytest.mark.parametrize(
        ("verdict", "sentence"),
        [
            (_verdict("tight"), "Fits, but close to the GPU's limit. It may run out of memory."),
            (_verdict("too_large", "gpu"), "Needs more GPU memory than this card has."),
            (_verdict("too_large", "ram"), "Needs more RAM than this computer has."),
            (_verdict("too_large", "unified"), "Needs more memory than this Mac has."),
        ],
    )
    def test_each_status_and_resource_has_its_sentence(self, verdict, sentence):
        assert memory_warning("Heading", verdict, _CARD).startswith(f"Heading: {sentence.rstrip('.')}")

    @pytest.mark.parametrize(
        ("resource", "sentence"),
        [
            ("gpu", "Fits, but close to the GPU's limit. It may run out of memory."),
            ("ram", "Fits, but close to the system memory limit. It may run slowly."),
            ("unified", "Fits, but macOS will compress or swap memory to run it. It may run slowly."),
        ],
    )
    def test_a_tight_job_names_the_resource_it_is_close_to(self, resource, sentence):
        assert memory_warning("Heading", _verdict("tight", resource), _CARD).startswith(f"Heading: {sentence.rstrip('.')}")


class TestFigures:
    def test_the_gpu_figures_are_the_need_and_the_card(self):
        assert memory_figures(_verdict("too_large", "gpu", gpu_gb=10.44), _CARD) == (10.4, 10.0)

    def test_the_ram_figures_are_the_ram_need_and_the_ram_total(self):
        assert memory_figures(_verdict("too_large", "ram", gpu_gb=5.0, ram_gb=31.46), _CARD) == (31.5, 30.0)

    def test_a_mac_reads_the_whole_need_against_its_total_memory(self):
        assert memory_figures(_verdict("too_large", "unified", gpu_gb=40.2), _MAC) == (40.2, 32.0)

    @pytest.mark.parametrize("machine", [_CARD, _MAC])
    def test_a_ram_verdict_without_a_ram_figure_has_none(self, machine):
        assert memory_figures(_verdict("too_large", "ram"), machine) is None


class TestWarning:
    def test_a_fitting_job_has_no_warning(self):
        assert memory_warning("Memory for this image", _verdict("fits", gpu_gb=5.0), _CARD) is None

    def test_a_too_large_job_reads_as_the_heading_the_sentence_and_the_figures(self):
        text = memory_warning("Memory for this image (1152×1728 · q4 · 3 references)", _verdict("too_large"), _CARD)

        assert text == "Memory for this image (1152×1728 · q4 · 3 references): Needs more GPU memory than this card has (about 10.4 of 10 GB). Continuing."

    def test_a_tight_job_keeps_both_sentences(self):
        text = memory_warning("Memory for this clip", _verdict("tight", gpu_gb=9.6), _CARD)

        assert text == "Memory for this clip: Fits, but close to the GPU's limit. It may run out of memory (about 9.6 of 10 GB). Continuing."

    def test_a_verdict_without_figures_still_warns(self):
        assert memory_warning("Memory for this track", _verdict("too_large", "ram"), _CARD) == "Memory for this track: Needs more RAM than this computer has. Continuing."


class TestRowsLine:
    _ROWS = [("none", _verdict("too_large", gpu_gb=17.04)), ("8", _verdict("tight", gpu_gb=9.6)), ("4", _verdict("fits", gpu_gb=6.0))]

    def test_the_other_quants_read_in_the_order_of_the_rows_with_their_status_and_need(self):
        assert memory_rows_line("quantize", self._ROWS, "none", _CARD) == "Other quants: q8 Tight (9.6 GB) · q4 Fits (6 GB)"

    def test_the_current_level_is_left_out_wherever_it_sits(self):
        assert memory_rows_line("quantize", self._ROWS, "8", _CARD) == "Other quants: None Too large (17 GB) · q4 Fits (6 GB)"

    def test_the_other_low_memory_setting_reads_as_its_own_status(self):
        rows = [("on", _verdict("fits", gpu_gb=8.1)), ("off", _verdict("too_large", gpu_gb=35.0))]

        assert memory_rows_line("low_memory", rows, "off", _CARD) == "Low memory on: Fits (8.1 GB)"

    def test_the_need_is_the_figure_of_the_resource_that_decided_the_row(self):
        rows = [("on", _verdict("too_large", "ram", gpu_gb=5.0, ram_gb=31.46)), ("off", _verdict("fits"))]

        assert memory_rows_line("low_memory", rows, "off", _CARD) == "Low memory on: Too large (31.5 GB)"

    def test_a_row_without_a_figure_names_only_its_status(self):
        rows = [("on", _verdict("too_large", "ram")), ("off", _verdict("fits"))]

        assert memory_rows_line("low_memory", rows, "off", _CARD) == "Low memory on: Too large"

    def test_a_job_with_one_row_has_no_line(self):
        assert memory_rows_line("quantize", [("4", _verdict("fits"))], "4", _CARD) is None
        assert memory_rows_line("quantize", [], "4", _CARD) is None


class TestWarningWithOtherChoices:
    def test_the_choices_follow_the_sentence(self):
        text = memory_warning("Memory for this image", _verdict("too_large"), _CARD, other_choices="Other quants: q4 Fits (6 GB)")

        assert text == "Memory for this image: Needs more GPU memory than this card has (about 10.4 of 10 GB). Continuing. Other quants: q4 Fits (6 GB)"

    def test_a_fitting_job_has_no_warning_whatever_its_choices(self):
        assert memory_warning("Memory for this image", _verdict("fits", gpu_gb=5.0), _CARD, other_choices="Other quants: q4 Fits (6 GB)") is None


class TestSentences:
    """The sentence of each status and resource, the words of the Web UI and the command lines alike."""

    def test_a_job_that_fits_reads_the_same_whatever_decided_it(self):
        assert {memory_sentence("fits", resource) for resource in ("gpu", "ram", "unified")} == {"Fits this computer."}

    @pytest.mark.parametrize(
        ("status", "resource", "sentence"),
        [
            ("tight", "gpu", "Fits, but close to the GPU's limit. It may run out of memory."),
            ("tight", "ram", "Fits, but close to the system memory limit. It may run slowly."),
            ("tight", "unified", "Fits, but macOS will compress or swap memory to run it. It may run slowly."),
            ("too_large", "gpu", "Needs more GPU memory than this card has."),
            ("too_large", "ram", "Needs more RAM than this computer has."),
            ("too_large", "unified", "Needs more memory than this Mac has."),
        ],
    )
    def test_each_status_and_resource_has_its_sentence(self, status, resource, sentence):
        assert memory_sentence(status, resource) == sentence

    def test_every_status_has_a_word(self):
        assert [status_word(status) for status in ("fits", "tight", "too_large")] == ["Fits", "Tight", "Too large"]


class TestLevels:
    def test_a_quant_level_is_named_as_its_control_names_it(self):
        assert [level_label("quantize", key) for key in ("none", "8", "4")] == ["None", "q8", "q4"]

    def test_a_low_memory_level_is_on_or_off(self):
        assert [level_label("low_memory", key) for key in ("on", "off")] == ["On", "Off"]

    def test_a_single_row_has_no_level_name_or_column_title(self):
        assert (level_label(None, "only"), control_title(None), table_heading(None, "clip")) == ("", None, None)

    def test_the_level_column_is_titled_by_its_control(self):
        assert (control_title("quantize"), control_title("low_memory")) == ("Quant", "Low memory")

    @pytest.mark.parametrize(
        ("control", "frame", "heading"),
        [
            ("quantize", "image", "Quants for this image"),
            ("quantize", "edit", "Quants for this edit"),
            ("quantize", "reference", "Quants at this size"),
            ("low_memory", "clip", "Low memory on or off for this clip"),
            ("low_memory", "reference", "Low memory on or off at this size"),
        ],
    )
    def test_the_table_lists_every_level_so_its_heading_does_not_say_other(self, control, frame, heading):
        assert table_heading(control, frame) == heading


class TestJudgedHeading:
    def test_an_image_names_its_size_level_references_and_loras_in_that_order(self):
        assert judged_heading("image", width=1152, height=1728, quantize=4, references=3, loras=2) == "Memory for this image (1152×1728 · q4 · 3 references · 2 LoRAs)"

    def test_a_clip_names_its_length_before_its_size_and_the_same_order_follows(self):
        assert judged_heading("clip", width=768, height=448, seconds=5.04, low_memory=True, references=1, loras=1) == "Memory for this clip (5 s · 768×448 · Low memory on · 1 reference · 1 LoRA)"

    def test_an_unquantized_job_names_no_level(self):
        assert judged_details("image", width=1024, height=1024) == "1024×1024"

    def test_a_job_with_nothing_to_name_is_its_title_alone(self):
        assert judged_heading("track") == "Memory for this track"

    def test_a_label_names_what_has_no_setting(self):
        assert judged_heading("track", label="Stable Audio 3") == "Memory for this track (Stable Audio 3)"

    def test_a_reference_job_has_its_size_in_the_title_and_not_in_the_details(self):
        assert judged_title("reference", width=768, height=448, seconds=5.0) == "Memory at 5 s · 768×448"
        assert judged_details("reference", width=768, height=448, seconds=5.0, quantize=4) == "q4"

    def test_a_reference_job_without_a_size_is_a_track(self):
        assert judged_title("reference") == "Memory for this track"
