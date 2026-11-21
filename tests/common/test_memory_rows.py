"""Tests for the rows a job is judged at: one per Quant level, or one per Low memory setting."""

from __future__ import annotations

from zvisiongenerator.utils import memory_rows
from zvisiongenerator.utils.memory_rows import judge_rows, level_key, low_memory_key, low_memory_rows, quantize_choices, quantize_rows, supports_quantize
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict, MemoryWeights
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget, WeightTotals

_CARD = MemoryBudget(10 * GIB, system_bytes=32 * GIB)


class TestQuantizeChoices:
    def test_an_unquantized_model_is_judged_unquantized_then_from_the_highest_level_down(self):
        assert quantize_choices(None, "flux2_klein", (4, 8)) == (None, 8, 4)

    def test_a_repeated_level_is_judged_once(self):
        assert quantize_choices(None, "flux2_klein", (4, 8, 4)) == (None, 8, 4)

    def test_a_model_stored_at_one_level_has_only_that_level(self):
        assert quantize_choices(4, "flux2_klein", (4, 8)) == (4,)

    def test_a_family_that_cannot_be_quantized_has_only_the_unquantized_level(self, monkeypatch):
        monkeypatch.setattr(memory_rows, "model_capabilities", lambda family: {"supports_quantize": False})

        assert quantize_choices(None, "flux2_klein", (4, 8)) == (None,)

    def test_a_family_nobody_knows_can_be_quantized(self):
        assert supports_quantize(None) is True
        assert quantize_choices(None, None, (4, 8)) == (None, 8, 4)


class TestRows:
    def test_a_job_is_repeated_once_per_level_with_only_the_level_changed(self):
        job = MemoryJob(media="image", width=1600, height=1600, family="flux2_klein", quantize=4)

        rows = quantize_rows(job, (None, 8, 4))

        assert [key for key, _ in rows] == ["none", "8", "4"]
        assert [row.quantize for _, row in rows] == [None, 8, 4]
        assert {(row.width, row.height, row.family) for _, row in rows} == {(1600, 1600, "flux2_klein")}

    def test_a_clip_is_repeated_with_low_memory_on_then_off(self):
        job = MemoryJob(media="video", family="ltx", width=704, height=448, frames=121, low_memory=False)

        rows = low_memory_rows(job)

        assert [(key, row.low_memory) for key, row in rows] == [("on", True), ("off", False)]
        assert {(row.width, row.frames) for _, row in rows} == {(704, 121)}

    def test_the_keys_are_what_the_web_ui_sends(self):
        assert [level_key(level) for level in (None, 8, 4)] == ["none", "8", "4"]
        assert (low_memory_key(True), low_memory_key(False)) == ("on", "off")


class TestJudgeRows:
    def test_each_row_is_judged_against_the_machine(self, monkeypatch):
        seen = []

        def judge(job, machine, weights):
            seen.append((job.quantize, machine, weights))
            return MemoryVerdict(GIB, None, "resident", "fits", "gpu")

        monkeypatch.setattr(memory_rows, "judge", judge)
        weights = MemoryWeights()

        rows = judge_rows([("none", MemoryJob(media="image")), ("4", MemoryJob(media="image", quantize=4))], _CARD, weights)

        assert rows is not None and [key for key, _ in rows] == ["none", "4"]
        assert seen == [(None, _CARD, weights), (4, _CARD, weights)]

    def test_a_job_the_engine_cannot_judge_has_no_rows(self):
        assert judge_rows([("none", MemoryJob(media="image", family="flux2_klein"))], _CARD, MemoryWeights()) is None

    def test_a_row_the_engine_cannot_judge_is_left_out_and_the_rest_are_kept(self):
        components = {"text_encoder": WeightTotals(float_matrix_elements=GIB), "transformer": WeightTotals(float_matrix_elements=GIB), "vae": WeightTotals(float_other_elements=GIB // 10)}
        weights = MemoryWeights(components=components)
        job = MemoryJob(media="image", family="flux2_klein", model_size="4b")

        rows = judge_rows(quantize_rows(job, (None, 8, 6, 4)), _CARD, weights)

        assert rows is not None and [key for key, _ in rows] == ["none", "8", "4"]

    def test_no_rows_remain_when_none_can_be_judged(self):
        job = MemoryJob(media="image", family="flux2_klein", model_size="4b")

        assert judge_rows(quantize_rows(job, (6, 5)), _CARD, MemoryWeights()) is None
