"""Tests for extending a clip, ending it on an image, and reviewing the half-size render of the upscale on the diffusers LTX backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.backends.cuda._helpers import (
    install_fake_weights,
    _FakeConditionPipeline,
    _FakeImagePipeline,
    _FakeTextPipeline,
    _PIN,
    _UPSAMPLER_SOURCE,
    _load_upscale_model,
    _make_runtime,
    _patch_runtime,
    _patch_upsampler_io,
    _text_to_video_args,
)
from tests._factories import make_extend_source

import zvisiongenerator.backends.cuda.video.backend as video_diffusers_module
import zvisiongenerator.backends.cuda.video.sampler as _sampler_module
from zvisiongenerator.backends.cuda.video.backend import (
    DiffusersVideoBackend,
)


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Fake the weights, placement and memory checks of ``load_model`` (see ``_cuda_helpers.install_fake_weights``)."""
    return install_fake_weights(monkeypatch)


class TestExtendVideo:
    """Continue a clip: pin its tail at the start of a window at its size and rate, write only the new part."""

    @pytest.fixture
    def segments(self, monkeypatch):
        """Replace the tail readers, the audio carry and the segment encoder; return what each received."""
        from contextlib import contextmanager

        calls = SimpleNamespace(tails=[], tail_audio=[], carries=[], pinned=[], restored=[], encoded=[], events=[])
        tail = np.zeros((25, 256, 512, 3), dtype=np.uint8)
        tail_audio = np.zeros((2, 16000), dtype=np.float32)
        carry = SimpleNamespace(token_count=26)

        def _read_tail(source):
            calls.tails.append(source)
            return tail

        def _read_tail_audio(source, sample_rate):
            calls.tail_audio.append((source, sample_rate))
            return tail_audio

        def _encode_carry(pipeline, waveform, token_count):
            calls.carries.append((pipeline, waveform, token_count))
            return carry

        @contextmanager
        def _pin(transformer, pinned):
            calls.pinned.append((transformer, pinned))
            calls.events.append("pin")
            yield
            calls.events.append("unpin")

        def _restore(audio_latents, restored):
            calls.restored.append((audio_latents, restored))
            calls.events.append("restore")

        def _write_frames(frames, command):
            calls.encoded.append((list(frames), command))

        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail", _read_tail)
        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail_audio", _read_tail_audio)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.encode_audio_carry", _encode_carry)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.pin_audio_prefix", _pin)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.restore_audio_prefix", _restore)
        monkeypatch.setattr("zvisiongenerator.utils.video_splice.write_segment_frames", _write_frames)
        calls.tail, calls.tail_audio_samples, calls.carry = tail, tail_audio, carry
        return calls

    @staticmethod
    def _load(monkeypatch, *, mode="extend", upscale=False):
        _patch_runtime(monkeypatch, _make_runtime())
        if upscale:
            _patch_upsampler_io(monkeypatch)
            return DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)[0]
        return DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode)[0]

    @staticmethod
    def _extend(model, tmp_path, source=None, **overrides):
        kwargs = {"extend_frames": 16, "seed": 7, "steps": 8, "segment_dir": tmp_path, "audio_output": True, **overrides}
        return DiffusersVideoBackend().extend_video(model, source or make_extend_source(), "She closes the umbrella.", **kwargs)

    def test_extend_mode_loads_the_condition_pipeline(self, monkeypatch):
        model = self._load(monkeypatch)
        (pipeline,) = _FakeConditionPipeline.loaded_instances
        assert model.extend is pipeline
        assert model.text_to_video is None and model.image_to_video is None

    def test_pins_the_tail_at_latent_zero_and_generates_the_window_at_the_source_size_and_rate(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source(fps=23.976, fps_text="24000/1001")
        self._extend(model, tmp_path, source)
        (call,) = model.extend.calls
        (condition,) = call["conditions"]
        assert condition.frames is segments.tail
        assert (condition.index, condition.strength) == (0, 1.0)
        assert (call["width"], call["height"], call["num_frames"]) == (512, 256, 41)
        assert call["frame_rate"] == 23.976
        assert segments.tails == [source]

    def test_writes_only_the_new_frames_with_the_source_rate_and_colour_tags(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source(fps_text="24/1", color_space="bt709", color_primaries="bt709", color_trc="bt709")
        segment = self._extend(model, tmp_path, source)
        ((frames, command),) = segments.encoded
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(25, 41))
        assert command[command.index("-r") + 1] == "24/1"
        assert command[command.index("-s") + 1] == "512x256"
        assert "colorspace=bt709" in command[command.index("-vf") + 1]
        assert command[-1] == str(tmp_path / "segment.mp4")
        assert segment.video_path == tmp_path / "segment.mp4"

    def test_the_sound_starts_a_lead_before_the_new_part_and_covers_it_exactly(self, monkeypatch, tmp_path, segments):
        import wave

        from zvisiongenerator.utils.video_extend import audio_lead_samples

        model = self._load(monkeypatch)
        segment = self._extend(model, tmp_path)
        lead = audio_lead_samples(24.0)
        assert segment.audio_path == tmp_path / "segment.wav"
        assert segment.lead_samples == lead
        with wave.open(str(segment.audio_path), "rb") as reader:
            assert reader.getframerate() == 48000
            assert reader.getnframes() == lead + round(16 / 24 * 48000)

    def test_the_sound_under_the_tail_is_pinned_during_the_window_and_restored_before_decoding(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source()
        self._extend(model, tmp_path, source)
        assert segments.tail_audio == [(source, 16000)]
        ((pipeline, waveform, token_count),) = segments.carries
        assert pipeline is model.extend and waveform is segments.tail_audio_samples
        assert token_count == round(25 / 24 * 25)
        ((transformer, pinned),) = segments.pinned
        assert transformer is model.extend.transformer_at_call[0] and pinned is segments.carry
        ((audio_latents, restored),) = segments.restored
        assert audio_latents == "audio-latent" and restored is segments.carry
        assert segments.events == ["pin", "unpin", "restore"]
        assert model.extend.audio_vae.decoded == ["audio-latent"]

    def test_the_sound_is_pinned_even_when_none_is_written(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        self._extend(model, tmp_path, audio_output=False)
        assert len(segments.pinned) == 1

    def test_a_silent_source_is_continued_without_a_pin(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        self._extend(model, tmp_path, make_extend_source(has_audio=False))
        assert segments.tail_audio == [] and segments.carries == [] and segments.pinned == []

    @pytest.mark.parametrize(("has_audio", "audio_output"), [(False, True), (True, False)])
    def test_no_sound_is_written_for_a_silent_source_or_without_audio_output(self, monkeypatch, tmp_path, segments, has_audio, audio_output):
        model = self._load(monkeypatch)
        segment = self._extend(model, tmp_path, make_extend_source(has_audio=has_audio), audio_output=audio_output)
        assert segment.audio_path is None
        assert not (tmp_path / "segment.wav").exists()

    @pytest.mark.parametrize("frames", [0, 12])
    def test_refuses_a_length_that_is_not_a_positive_multiple_of_8(self, monkeypatch, tmp_path, segments, frames):
        model = self._load(monkeypatch)
        with pytest.raises(ValueError, match="multiple of 8"):
            self._extend(model, tmp_path, extend_frames=frames)

    def test_refuses_a_model_loaded_for_another_mode(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch, mode="t2v")
        with pytest.raises(RuntimeError, match="extend"):
            self._extend(model, tmp_path)

    def test_refuses_an_upscale_model(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch, upscale=True)
        with pytest.raises(ValueError, match="upscale"):
            self._extend(model, tmp_path)

    def test_a_window_of_the_wrong_length_fails_before_anything_is_written(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        monkeypatch.setattr(_FakeConditionPipeline, "window", 40)
        with pytest.raises(RuntimeError, match="41"):
            self._extend(model, tmp_path)
        assert segments.encoded == []

    def test_text_to_video_still_runs_at_the_model_rate(self, monkeypatch, tmp_path):
        model = self._load(monkeypatch, mode="t2v")
        DiffusersVideoBackend().text_to_video(model, "prompt", 512, 256, 25, 7, 8, str(tmp_path / "out.mp4"))
        assert model.text_to_video.calls[0]["frame_rate"] == model.model_info.default_fps

    def test_an_extend_pass_never_builds_the_ancestral_scheduler_whatever_the_sampler(self, monkeypatch, tmp_path, segments):
        repo = "unsloth/LTX-2.5-FP8@" + _PIN

        def _never(*_args, **_kwargs):
            raise AssertionError("the ancestral scheduler was built for an Extend pass")

        monkeypatch.setattr(_sampler_module, "ancestral_scheduler", _never)
        _patch_runtime(monkeypatch, _make_runtime())
        model = DiffusersVideoBackend().load_model(repo, quantize=4, mode="extend", sampler="euler_ancestral")[0]
        original = model.extend.scheduler

        self._extend(model, tmp_path)

        assert model.sampler == "euler_ancestral"
        assert model.extend.schedulers_at_call == [original]
        assert model.extend.scheduler is original


class TestEndImage:
    """A clip that ends on a last-frame image runs on the condition pipeline, pinned at its first and last frames."""

    @staticmethod
    def _load(monkeypatch, repo="diffusers/LTX-2.3-Distilled-Diffusers", **kwargs):
        _patch_runtime(monkeypatch, _make_runtime())
        return DiffusersVideoBackend().load_model(repo, quantize=4, **{"mode": "i2v", "end_image": True, **kwargs})[0]

    @staticmethod
    def _images(tmp_path):
        Image.new("RGB", (16, 8), "blue").save(tmp_path / "first.png")
        Image.new("RGB", (8, 16), "red").save(tmp_path / "last.png")
        return str(tmp_path / "first.png"), str(tmp_path / "last.png")

    def _run(self, model, tmp_path, *, num_frames=49, **kwargs):
        first, last = self._images(tmp_path)
        return DiffusersVideoBackend().image_to_video(model, first, "she turns", 128, 64, num_frames, 7, 8, str(tmp_path / "out.mp4"), end_image=last, **kwargs)

    @pytest.mark.parametrize(
        ("mode", "end_image", "slot", "pipeline_class"),
        [("i2v", True, "image_to_video", _FakeConditionPipeline), ("i2v", False, "image_to_video", _FakeImagePipeline), ("t2v", True, "text_to_video", _FakeTextPipeline)],
    )
    def test_only_an_image_to_video_load_for_an_end_image_builds_the_condition_pipeline(self, monkeypatch, mode, end_image, slot, pipeline_class):
        model = self._load(monkeypatch, mode=mode, end_image=end_image)
        assert type(getattr(model, slot)) is pipeline_class
        assert model.end_image is (end_image and mode == "i2v")

    def test_pins_the_first_image_at_latent_zero_and_the_last_after_it_both_fitted_to_the_clip(self, monkeypatch, tmp_path):
        model = self._load(monkeypatch)
        self._run(model, tmp_path)
        (call,) = model.image_to_video.calls
        first, last = call["conditions"]
        assert [(c.index, c.strength) for c in (first, last)] == [(0, 1.0), (-1, 1.0)]
        assert first.frames.size == last.frames.size == (128, 64)
        assert first.frames.getpixel((0, 0)) == (0, 0, 255) and last.frames.getpixel((0, 0)) == (255, 0, 0)
        assert "image" not in call

    @pytest.mark.parametrize("num_frames", [9, 49, 121])
    def test_the_last_image_lands_on_the_last_frame_and_the_placement_is_undone_after(self, monkeypatch, tmp_path, num_frames):
        model = self._load(monkeypatch)
        self._run(model, tmp_path, num_frames=num_frames)
        assert model.image_to_video.keyframe_pixels == [num_frames - 1]
        assert "_prepare_keyframe_coords" not in vars(model.image_to_video)

    @pytest.mark.parametrize(("repo", "crf"), [("diffusers/LTX-2.3-Distilled-Diffusers", None), ("unsloth/LTX-2.5-FP8@" + _PIN, 0)])
    def test_an_image_the_backend_recompressed_is_not_recompressed_again(self, monkeypatch, tmp_path, repo, crf):
        monkeypatch.setattr(video_diffusers_module, "recompress", lambda image, crf: image)
        model = self._load(monkeypatch, repo=repo, image_crf=18 if crf == 0 else None)
        self._run(model, tmp_path)
        assert [c.crf for c in model.image_to_video.calls[0]["conditions"]] == [crf, crf]

    def test_every_pass_keeps_the_stock_scheduler_whatever_the_sampler(self, monkeypatch, tmp_path):
        seen: list[bool] = []
        real = video_diffusers_module._pass_scheduler

        def _spy(*args, **kwargs):
            seen.append(kwargs["conditioned"])
            return real(*args, **kwargs)

        monkeypatch.setattr(video_diffusers_module, "_pass_scheduler", _spy)
        _patch_upsampler_io(monkeypatch)
        model = self._load(monkeypatch, repo="unsloth/LTX-2.5-FP8@" + _PIN, sampler="euler_ancestral", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

        self._run(model, tmp_path, stage1_steps=8)

        assert seen == [True, True]

    def test_the_refine_pass_pins_both_images_too(self, monkeypatch, tmp_path):
        _patch_upsampler_io(monkeypatch)
        model = self._load(monkeypatch, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        self._run(model, tmp_path, stage1_steps=8)
        assert [[c.index for c in call["conditions"]] for call in model.image_to_video.calls] == [[0, -1], [0, -1]]
        assert model.image_to_video.keyframe_pixels == [48, 48]

    @pytest.mark.parametrize("loaded_for_end_image", [True, False])
    def test_a_clip_that_does_not_match_the_load_is_refused(self, monkeypatch, tmp_path, loaded_for_end_image):
        model = self._load(monkeypatch, end_image=loaded_for_end_image)
        first, last = self._images(tmp_path)
        with pytest.raises(RuntimeError, match="end_image="):
            DiffusersVideoBackend().image_to_video(model, first, "p", 64, 32, 49, 7, 8, str(tmp_path / "out.mp4"), **({} if loaded_for_end_image else {"end_image": last}))
        assert model.image_to_video.calls == []


class TestUpscaleReview:
    """The half-size render is decoded to a temporary video and shown to ``review`` before stage 2 starts."""

    @staticmethod
    def _recording_runtime():
        """Return a runtime whose encoder writes a file and notes each call, for the preview and the final export."""
        runtime = _make_runtime()
        calls: list[Path] = []

        def _encode(video, fps, output_path, audio=None, audio_sample_rate=None):
            Path(output_path).write_bytes(b"mp4")
            calls.append(Path(output_path))

        runtime.encode_video = _encode
        return runtime, calls

    def test_review_sees_the_decoded_half_size_render_before_stage_2(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        seen: list[dict] = []

        def review(path):
            seen.append({"path": path, "exists": path.is_file(), "calls_so_far": len(model.text_to_video.calls), "upsampled": len(model.upsample_pipeline.calls)})

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=review, **_text_to_video_args(tmp_path))

        (shown,) = seen
        assert shown["exists"] and shown["path"].suffix == ".mp4"
        assert shown["calls_so_far"] == 1  # only stage 1 had run
        assert shown["upsampled"] == 0
        assert len(model.text_to_video.calls) == 2  # stage 2 followed
        assert encoded[0] == shown["path"] and encoded[1] == tmp_path / "upscaled.mp4"

    def test_the_preview_file_is_deleted_when_review_returns(self, monkeypatch, tmp_path):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        shown: list[Path] = []

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=shown.append, **_text_to_video_args(tmp_path))

        assert not shown[0].exists() and not shown[0].parent.exists()

    def test_a_review_that_raises_skips_stage_2_and_the_export(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        shown: list[Path] = []

        class Discarded(Exception):
            pass

        def review(path):
            shown.append(path)
            raise Discarded

        with pytest.raises(Discarded):
            DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=review, **_text_to_video_args(tmp_path))

        assert len(model.text_to_video.calls) == 1  # stage 1 only
        assert model.upsample_pipeline.calls == []
        assert encoded == shown  # the preview was the only encode
        assert not shown[0].exists()
        assert not (tmp_path / "upscaled.mp4").exists()
        assert runtime.torch.cuda.empty_cache_calls == 1  # memory handed back even though the clip was discarded

    def test_image_to_video_reviews_too(self, monkeypatch, tmp_path):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, mode="i2v", runtime=runtime)
        image = tmp_path / "in.png"
        Image.new("RGB", (64, 64)).save(image)
        shown: list[Path] = []

        DiffusersVideoBackend().image_to_video(model=model, image_path=str(image), stage1_steps=4, review=shown.append, **_text_to_video_args(tmp_path))

        assert len(shown) == 1
        assert len(model.image_to_video.calls) == 2

    @pytest.mark.parametrize(("low_memory", "expects_offload"), [(True, True), (False, False)])
    def test_low_memory_sends_the_codecs_back_to_the_cpu_before_the_wait_for_the_decision(self, monkeypatch, tmp_path, low_memory, expects_offload):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=low_memory, runtime=runtime)
        events: list[str] = []
        vae = model.text_to_video.vae
        vae.to = lambda device: events.append(f"vae->{device}")

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=lambda path: events.append("review"), **_text_to_video_args(tmp_path))

        assert (events[:2] == ["vae->cpu", "review"]) is expects_offload
        assert ("review" in events) and (events.index("review") == (1 if expects_offload else 0))

    def test_without_review_nothing_is_decoded_before_the_final_export(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

        assert encoded == [tmp_path / "upscaled.mp4"]
