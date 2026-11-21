"""Tests for the memory job an image load is told about."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from PIL import Image

from zvisiongenerator.utils.memory_verdict import MemoryJob
from zvisiongenerator.processing.edit_region import edit_model_size
from zvisiongenerator.utils.image_memory_job import edit_memory_job, image_memory_job, run_memory_job, sized_edit_job, sized_image_job


def _picture(path: Path, width: int, height: int) -> str:
    Image.new("RGB", (width, height)).save(path)
    return str(path)


class TestMemoryJob:
    def test_the_default_is_the_one_megapixel_reference_job(self):
        job = MemoryJob(media="image")

        assert (job.width, job.height, job.reference_megapixels, job.lora_bytes) == (1024, 1024, (), 0)
        assert job.megapixels == pytest.approx(1.048576)


class TestImageJob:
    def test_it_carries_the_size_each_reference_picture_and_the_lora_bytes(self, tmp_path):
        references = [_picture(tmp_path / "a.png", 1000, 500), _picture(tmp_path / "b.png", 2000, 2000)]
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 300)

        job = image_memory_job(1344, 768, reference_paths=references, lora_paths=[str(lora)])

        assert (job.width, job.height) == (1344, 768)
        assert job.reference_megapixels == pytest.approx((0.5, 4.0))
        assert job.lora_bytes == 300

    def test_a_job_without_references_or_loras_carries_none(self):
        assert image_memory_job(832, 1216) == MemoryJob(media="image", width=832, height=1216)

    def test_a_reference_that_cannot_be_read_counts_nothing(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a picture")

        assert image_memory_job(1024, 1024, reference_paths=[str(broken), str(tmp_path / "missing.png")]).reference_megapixels == ()

    def test_a_lora_file_that_is_gone_counts_nothing(self, tmp_path):
        assert image_memory_job(1024, 1024, lora_paths=[str(tmp_path / "gone.safetensors")]).lora_bytes == 0


class TestSizedJobs:
    """The Web UI is told the sizes and the CLI and the loader measure files; both reach the same job."""

    def test_a_job_built_from_sizes_is_the_job_built_from_the_files(self, tmp_path):
        references = [_picture(tmp_path / "a.png", 1000, 500), _picture(tmp_path / "b.png", 2000, 2000)]
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 300)

        from_files = image_memory_job(1344, 768, reference_paths=references, lora_paths=[str(lora)], from_picture=True)
        from_sizes = sized_image_job(1344, 768, reference_megapixels=(1000 * 500 / 1e6, 2000 * 2000 / 1e6), lora_bytes=300, from_picture=True)

        assert from_sizes == from_files

    def test_an_edit_built_from_sizes_is_the_edit_built_from_the_files(self, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 64)

        assert sized_edit_job((3000, 2000), masked=True, extend=None, lora_bytes=64) == edit_memory_job((3000, 2000), masked=True, extend=None, lora_paths=[str(lora)])


class TestEditJob:
    @pytest.mark.parametrize(("masked", "extend"), [(False, None), (True, None), (False, ("right", 512))])
    def test_it_draws_the_edit_canvas_with_the_picture_as_its_one_reference(self, masked, extend):
        job = edit_memory_job((3000, 2000), masked=masked, extend=extend)

        assert (job.width, job.height) == edit_model_size(3000, 2000, masked=masked, extend=extend)
        assert job.reference_megapixels == pytest.approx((6.0,))

    @pytest.mark.parametrize(("masked", "extend", "from_picture"), [(False, None, False), (True, None, True), (False, ("right", 512), True)])
    def test_a_painted_edit_and_an_extend_start_from_the_picture_and_a_whole_picture_edit_does_not(self, masked, extend, from_picture):
        assert edit_memory_job((1024, 1024), masked=masked, extend=extend).from_picture is from_picture

    def test_it_carries_the_lora_bytes(self, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 64)

        assert edit_memory_job((1024, 1024), masked=False, extend=None, lora_paths=[str(lora)]).lora_bytes == 64


class TestRunJob:
    """One rule for the command line and the Web runner: an edit draws its own canvas, anything else the size the run asks for."""

    def test_a_plain_run_draws_the_size_it_asks_for_with_its_references_and_loras(self, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x" * 8)
        args = argparse.Namespace(edit_source=None)

        job = run_memory_job(args, size=lambda: (832, 1216), reference_paths=[_picture(tmp_path / "ref.png", 1000, 1000)], lora_paths=[str(lora)])

        assert (job.media, job.width, job.height, job.lora_bytes) == ("image", 832, 1216, 8)
        assert job.reference_megapixels == pytest.approx((1.0,))

    def test_an_edit_draws_its_canvas_and_never_asks_for_a_size(self):
        args = argparse.Namespace(edit_source="/in/a.png", edit_size=(3000, 2000), edit_paint=None, edit_extend=("right", 512))

        def no_size() -> tuple[int, int]:
            raise AssertionError("an edit has no ratio or size")

        job = run_memory_job(args, size=no_size, reference_paths=[], lora_paths=None)

        assert (job.width, job.height) == edit_model_size(3000, 2000, masked=False, extend=("right", 512))
        assert job.reference_megapixels == pytest.approx((6.0,))

    def test_a_painted_edit_is_masked(self):
        args = argparse.Namespace(edit_source="/in/a.png", edit_size=(3000, 2000), edit_paint=object(), edit_extend=None)

        job = run_memory_job(args, size=lambda: (1, 1), reference_paths=[], lora_paths=None)

        assert (job.width, job.height) == edit_model_size(3000, 2000, masked=True, extend=None)

    @pytest.mark.parametrize("fields", [{"upscale": 2}, {"upscale": 4}, {"image_path": "/in/start.png"}])
    def test_an_upscale_of_either_factor_or_a_start_image_starts_from_a_picture_at_the_final_size(self, fields):
        job = run_memory_job(argparse.Namespace(edit_source=None, **fields), size=lambda: (1600, 1600), reference_paths=[], lora_paths=None)

        assert (job.width, job.height, job.from_picture) == (1600, 1600, True)

    def test_a_run_with_neither_starts_from_noise(self):
        job = run_memory_job(argparse.Namespace(edit_source=None, upscale=None, image_path=None), size=lambda: (1600, 1600), reference_paths=[], lora_paths=None)

        assert job.from_picture is False

    def test_arguments_without_edit_fields_are_a_plain_run(self):
        assert run_memory_job(argparse.Namespace(), size=lambda: (640, 480), reference_paths=[], lora_paths=None) == MemoryJob(media="image", width=640, height=480)
