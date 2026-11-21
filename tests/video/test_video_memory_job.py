"""Tests for the memory job of a video clip or an extension window."""

from __future__ import annotations

from zvisiongenerator.utils.video_memory_job import clip_memory_job


class TestClipMemoryJob:
    def test_it_carries_the_size_frames_mode_and_the_counts_of_the_clip(self):
        job = clip_memory_job("h3", width=768, height=448, frames=124, low_memory=True, mode="i2v", references=2, lora_bytes=64)

        assert (job.media, job.family, job.width, job.height, job.frames) == ("video", "h3", 768, 448, 124)
        assert (job.low_memory, job.mode, job.references, job.lora_bytes) == (True, "i2v", 2, 64)

    def test_a_clip_without_steps_runs_the_default_forwards(self):
        assert clip_memory_job("h3", width=768, height=448, frames=124).forwards == 8
        assert clip_memory_job("h3", width=768, height=448, frames=124, steps=12).forwards == 12

    def test_only_an_upscaled_clip_is_reviewed(self):
        assert clip_memory_job("h3", width=768, height=448, frames=124, upscale=True, review=True).review is True
        assert clip_memory_job("h3", width=768, height=448, frames=124, upscale=False, review=True).review is False
