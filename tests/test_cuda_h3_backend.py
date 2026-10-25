"""Tests for the CUDA MiniMax H3 backend: stage order, releases, upscale, extend and file resolution.

Every model-loading building block is replaced by a recorder fake, so no weights, GPU or network are involved.
"""

from __future__ import annotations

import dataclasses
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from conftest import make_extend_source
from PIL import Image

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda.video.h3 import audio as h3_audio  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3 import backend as h3_backend  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3 import conditioning, decode, sampler, schedule, sources, stored_copy, text_encoder  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.backend import H3VideoBackend  # noqa: E402
from zvisiongenerator.core.video_backend import VideoBackend, VideoExtender  # noqa: E402
from zvisiongenerator.core.video_types import ExtendSegment  # noqa: E402
from zvisiongenerator.utils import hf_snapshot  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3Files, H3Preset, H3Schedule, HubFiles  # noqa: E402
from zvisiongenerator.utils.model_memory import MemoryBudget  # noqa: E402
from zvisiongenerator.utils.video_model_detect import VideoModelInfo  # noqa: E402

WIDTH, HEIGHT = 96, 64
FRAMES = 22  # the smallest valid clip, 17 * 1 + 5
OOM_MESSAGE = "MiniMax H3 ran out of GPU memory at {w}x{h}, {n} frames. Try a shorter clip or a smaller size."

PRESET = H3Preset(
    repo="MiniMaxAI/MiniMax-H3",
    revision="rev-main",
    repo_allow=("model_index.json", "transformer/*", "vae/*"),
    repo_ignore=("text_encoder/*",),
    text_encoder=HubFiles("te/repo", "rev-te", ("config.json", "model.safetensors.index.json", "model-1.safetensors", "model-2.safetensors")),
    text_encoder_layers=50,
    lora=HubFiles("lora/repo", "rev-lora", ("turbo.safetensors",)),
    lora_scale=0.0625,
    latent_upsampler=HubFiles("upsampler/repo", "rev-up", ("upsampler.safetensors",)),
    schedules={8: H3Schedule(8, 6.0, 3.0), 4: H3Schedule(4, 6.0, 3.0)},
)


class _Stop(Exception):
    """Raised by a review to end the generation."""


class Rig:
    """The recorder: an ordered event log, the arguments of each stage and the knobs the fakes read."""

    def __init__(self, tmp_path: Path) -> None:
        self.events: list[str] = []
        self.tmp_path = tmp_path
        self.real_encode = False
        self.row_sets: list[str] = []
        self.refine_forwards: list[int] = []
        self.calls: dict[str, list[Any]] = {}
        self.window_waveform: Any = None  # a callable (latent audio frames) -> (2, samples), else a quiet sine
        self.oom_in: str | None = None
        self.gpu_bytes = 10 * 2**30  # the card the backend asks about

    def note(self, name: str, **info: Any) -> None:
        self.events.append(name)
        self.calls.setdefault(name, []).append(info)

    def last(self, name: str) -> dict[str, Any]:
        return self.calls[name][-1]

    def maybe_oom(self, stage: str) -> None:
        if self.oom_in == stage:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def _frames_of(latent_frames: int) -> int:
    return (latent_frames - 2) // 5 * 17 + 5


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)
    real_encode = h3_backend.encode_video

    # --- load_model
    monkeypatch.setattr(h3_backend, "require_cuda", lambda *_a, **_k: rig.note("require_cuda"))
    monkeypatch.setattr(h3_backend, "configure_torch_runtime", lambda *_a, **_k: rig.note("configure_torch_runtime"))
    monkeypatch.setattr(h3_backend, "_models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(h3_backend, "_gpu_total_bytes", lambda device: rig.gpu_bytes)
    monkeypatch.setattr(h3_backend, "_check_system_memory", lambda loras: None)
    files = H3Files(tmp_path / "repo", tmp_path / "te", (tmp_path / "te" / "model-1.safetensors",), tmp_path / "turbo.safetensors")
    monkeypatch.setattr(h3_backend, "resolve_files", lambda ref, preset, *, upscale=False: (rig.note("resolve_files", ref=ref, upscale=upscale), files)[1])
    monkeypatch.setattr(h3_backend, "transformer_files", lambda repo_dir: ([repo_dir / "transformer" / "a.safetensors"], {"config": 1}))
    monkeypatch.setattr(h3_backend, "_all_fold_keys", lambda preset, repo_dir: [(0.0,), (0.5, 1.0)])
    monkeypatch.setattr(stored_copy, "ensure_copy", lambda *args: (rig.note("ensure_copy", args=args), tmp_path / "models" / "h3@q4")[1])

    # --- stages
    monkeypatch.setattr(h3_backend, "release_memory", lambda: rig.note("release"))
    monkeypatch.setattr(schedule, "build_schedulers", lambda sched, config_dir=None: (rig.note("schedulers", schedule=sched, config_dir=config_dir), ("video", "audio"))[1])

    def tokenize(prompt, tokenizer_dir, processor_dir, images):
        rig.note("presentation", prompt=prompt, tokenizer_dir=tokenizer_dir, processor_dir=processor_dir, images=list(images))
        return SimpleNamespace(token_tags=(1,) * 2 + (0,) * len(images) + (1,))

    monkeypatch.setattr(text_encoder, "tokenize_presentation", tokenize)
    monkeypatch.setattr(
        text_encoder, "load_text_encoder", lambda config_dir, shards, device, *, layers, vision: (rig.note("text_encoder.load", layers=layers, vision=vision, shards=shards), "encoder")[1]
    )
    monkeypatch.setattr(text_encoder, "encode", lambda encoder, presentation, device: (rig.note("text_encoder.encode"), torch.zeros(1, 3, 4))[1])
    monkeypatch.setattr(conditioning, "load_vae_encoder", lambda vae_dir, device: (rig.note("vae_encoder.load", vae_dir=vae_dir), "vae-encoder")[1])
    monkeypatch.setattr(conditioning, "encode_keyframe", lambda enc, image, w, h: (rig.note("keyframe", size=image.size, canvas=(w, h)), torch.zeros(1, 24, 1, h // 16, w // 16))[1])
    monkeypatch.setattr(conditioning, "encode_tail", lambda enc, frames: (rig.note("tail", shape=tuple(frames.shape)), torch.zeros(1, 24, 7, frames.shape[1] // 16, frames.shape[2] // 16))[1])
    monkeypatch.setattr(h3_audio, "load_audio_vae", lambda audio_dir, device: (rig.note("audio_vae.load", audio_dir=audio_dir), "audio-vae")[1])
    monkeypatch.setattr(conditioning, "encode_tail_audio", lambda vae, waveform: (rig.note("tail_audio", shape=tuple(waveform.shape)), torch.zeros(74, 32))[1])
    monkeypatch.setattr(
        h3_backend,
        "_fold_keys",
        lambda schedulers, row_set, start=0: (rig.row_sets.append(row_set), [("fold", row_set) if start == 0 else ("fold", row_set, start)])[1],
    )
    monkeypatch.setattr(stored_copy, "load_copy", lambda copy_dir, keys, device, *, loras=(), lora_names=(): (rig.note("copy.load", keys=keys, loras=loras), object())[1])

    def denoise(transformer, embeds, geo, schedulers, seed, device, *, conditioning=None, step_callback=None, phase="video", text_token_tags=None):
        rig.note("denoise", geo=geo, seed=seed, conditioning=conditioning, text_token_tags=text_token_tags, phase=phase, transformer=transformer, schedulers=schedulers)
        rig.maybe_oom("denoise")
        forwards = rig.last("schedulers")["schedule"].forwards
        for step in range(forwards):
            if step_callback is not None:
                step_callback({"phase": phase, "current_step": step + 1, "total_steps": forwards})
        return torch.zeros(1, 24, geo.latent_frames, geo.latent_height, geo.latent_width), torch.zeros(2, 32, geo.audio_latents)

    monkeypatch.setattr(sampler, "denoise", denoise)
    monkeypatch.setattr(decode, "load_video_vae", lambda vae_dir, device: (rig.note("video_vae.load", vae_dir=vae_dir), "video-vae")[1])

    def decode_video(vae, latents):
        rig.note("decode_video", shape=tuple(latents.shape))
        rig.maybe_oom("decode")
        count, height, width = _frames_of(latents.shape[2]), latents.shape[3] * 16, latents.shape[4] * 16
        return [np.full((height, width, 3), index % 256, dtype=np.uint8) for index in range(count)]

    monkeypatch.setattr(decode, "decode_video", decode_video)

    def decode_audio(vae, latents):
        rig.note("decode_audio", shape=tuple(latents.shape))
        if rig.window_waveform is not None:
            return rig.window_waveform(latents.shape[2])
        return torch.sin(torch.arange(latents.shape[2] * 800, dtype=torch.float32) / 50).repeat(2, 1) * 0.1

    monkeypatch.setattr(h3_audio, "decode_audio", decode_audio)

    def encode_video(frames, fps, output_path, audio=None, audio_sample_rate=None):
        rig.note("encode_video", frames=frames, fps=fps, output_path=output_path, audio=audio, audio_sample_rate=audio_sample_rate)
        if rig.real_encode:
            real_encode(frames, fps, output_path, audio=audio, audio_sample_rate=audio_sample_rate)
        else:
            Path(output_path).write_bytes(b"mp4")

    monkeypatch.setattr(h3_backend, "encode_video", encode_video)

    # --- upscale
    def upscale_latents(model, latents):
        rig.note("upsampler", shape=tuple(latents.shape))
        rig.maybe_oom("upsampler")
        return torch.zeros(1, 24, latents.shape[2], latents.shape[3] * 2, latents.shape[4] * 2)

    monkeypatch.setattr(h3_backend, "_upscale_latents", upscale_latents)

    def refine(transformer, embeds, geo, schedulers, latents, audio_rows, noise, *, start, device, conditioning=None, keyframe_seed=None, step_callback=None, phase="video", text_token_tags=None):
        rig.note(
            "refine",
            geo=geo,
            start=start,
            latents=latents,
            audio_rows=audio_rows,
            noise=noise,
            conditioning=conditioning,
            keyframe_seed=keyframe_seed,
            phase=phase,
            text_token_tags=text_token_tags,
            embeds=embeds,
            transformer=transformer,
        )
        rig.maybe_oom("refine")
        forwards = rig.last("schedulers")["schedule"].forwards - start
        rig.refine_forwards.append(forwards)
        for step in range(forwards):
            if step_callback is not None:
                step_callback({"phase": phase, "current_step": step + 1, "total_steps": forwards})
        return torch.full(latents.shape, float(len(rig.calls["refine"]) - 1))

    monkeypatch.setattr(sampler, "refine", refine)

    # --- extend
    monkeypatch.setattr(h3_backend, "read_source_tail", lambda source, frames: (rig.note("read_tail", frames=frames), np.zeros((frames, source.height, source.width, 3), np.uint8))[1])
    monkeypatch.setattr(
        h3_backend,
        "read_source_tail_audio",
        lambda source, sample_rate, frames: (rig.note("read_tail_audio", sample_rate=sample_rate, frames=frames), np.zeros((2, round(frames / 24 * sample_rate)), np.float32))[1],
    )
    monkeypatch.setattr(h3_backend, "write_segment_frames", lambda frames, command: rig.note("segment_frames", frames=list(frames), command=command))
    return rig


def _load(rig: Rig, **kwargs: Any) -> Any:
    kwargs.setdefault("preset", PRESET)
    model, _info = H3VideoBackend().load_model("MiniMaxAI/MiniMax-H3", **kwargs)
    model = dataclasses.replace(model, device=torch.device("cpu"))  # the stand-ins run anywhere; macOS CI has no CUDA
    rig.events.clear()
    rig.calls.clear()
    rig.row_sets.clear()
    rig.refine_forwards.clear()
    return model


def _image(path: Path, size: tuple[int, int] = (200, 100), color: tuple[int, int, int] = (200, 30, 30)) -> str:
    Image.new("RGB", size, color).save(path)
    return str(path)


def _t2v(model: Any, output: Path, **kwargs: Any) -> Any:
    arguments: dict[str, Any] = dict(prompt="waves", width=WIDTH, height=HEIGHT, num_frames=FRAMES, seed=5, steps=8, output_path=str(output))
    arguments.update(kwargs)
    return H3VideoBackend().text_to_video(model, **arguments)


# --- load_model ----------------------------------------------------------------------------------------------------


class TestLoadModel:
    @pytest.mark.parametrize("mode", ["t2v", "i2v", "extend"])
    def test_accepts_every_mode_and_returns_the_h3_info(self, rig, mode):
        model, info = H3VideoBackend().load_model("MiniMaxAI/MiniMax-H3", mode=mode, preset=PRESET)
        assert model.mode == mode
        assert isinstance(info, VideoModelInfo)
        assert (info.family, info.backend, info.supports_i2v, info.default_fps, info.resolution_alignment) == ("h3", "h3", True, 24, 32)

    def test_defaults_to_text_to_video(self, rig):
        assert H3VideoBackend().load_model("m", preset=PRESET)[0].mode == "t2v"

    def test_ensures_the_stored_copy_for_the_whole_key_union_and_the_turbo_scale(self, rig, tmp_path):
        model, _ = H3VideoBackend().load_model("MiniMaxAI/MiniMax-H3", preset=PRESET)
        models_dir, name, shards, config, lora_file, scale, keys, _device = rig.last("ensure_copy")["args"]
        assert models_dir == tmp_path / "models"
        assert name == "MiniMaxAI--MiniMax-H3"
        assert shards == [tmp_path / "repo" / "transformer" / "a.safetensors"]
        assert config == {"config": 1}
        assert lora_file == tmp_path / "turbo.safetensors"
        assert scale == 0.0625
        assert keys == [(0.0,), (0.5, 1.0)]
        assert model.copy_dir == tmp_path / "models" / "h3@q4"

    def test_the_stored_copy_is_named_after_the_model_it_was_picked_by(self, rig):
        H3VideoBackend().load_model("MiniMaxAI/MiniMax-H3@rev", preset=PRESET, model_name="h3")
        assert rig.last("ensure_copy")["args"][1] == "h3"
        H3VideoBackend().load_model("/data/my-tune", preset=PRESET, model_name="my-tune")
        assert rig.last("ensure_copy")["args"][1] == "my-tune"

    def test_checks_cuda_and_configures_the_runtime_before_resolving_files(self, rig):
        H3VideoBackend().load_model("m", preset=PRESET)
        assert rig.events == ["require_cuda", "configure_torch_runtime", "resolve_files", "ensure_copy"]

    def test_an_upscale_load_resolves_the_latent_upsampler_and_a_plain_load_does_not(self, rig):
        H3VideoBackend().load_model("m", preset=PRESET, upscale=True)
        assert rig.last("resolve_files")["upscale"] is True
        H3VideoBackend().load_model("m", preset=PRESET)
        assert rig.last("resolve_files")["upscale"] is False

    def test_an_upscale_needs_the_8_step_schedule_in_the_preset(self, rig):
        only_four = H3Preset(**{**PRESET.__dict__, "schedules": {4: H3Schedule(4, 6.0, 3.0)}})
        with pytest.raises(ValueError, match=r"video_model_presets\.h3\.diffusers\.schedules\.8"):
            H3VideoBackend().load_model("m", preset=only_four, upscale=True)
        assert rig.events == []
        assert H3VideoBackend().load_model("m", preset=only_four)[0].upscale is False

    def test_accepts_loras_and_upscale_and_ignores_low_memory_and_model_name(self, rig, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x")
        model, _ = H3VideoBackend().load_model("m", preset=PRESET, loras=[(str(lora), 0.7)], upscale=True, low_memory=True, model_name="h3")
        assert model.loras == ((lora, 0.7),)
        assert model.upscale is True

    @pytest.mark.parametrize("key", ["quantize", "transformer_source", "sampler", "image_crf", "repo_ignore", "unheard_of"])
    def test_refuses_options_it_does_not_take_naming_them(self, rig, key):
        with pytest.raises(ValueError, match=key):
            H3VideoBackend().load_model("m", preset=PRESET, **{key: 1})
        assert "ensure_copy" not in rig.calls

    def test_names_every_refused_option(self, rig):
        with pytest.raises(ValueError, match=r"quantize.*sampler"):
            H3VideoBackend().load_model("m", preset=PRESET, sampler="euler_ancestral", quantize=4)

    def test_needs_the_preset(self, rig):
        with pytest.raises(ValueError, match="preset"):
            H3VideoBackend().load_model("m")

    def test_rejects_an_unknown_mode(self, rig):
        with pytest.raises(ValueError, match="mode"):
            H3VideoBackend().load_model("m", preset=PRESET, mode="v2v")

    def test_a_missing_lora_file_fails_before_anything_is_loaded(self, rig, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.safetensors"):
            H3VideoBackend().load_model("m", preset=PRESET, loras=[(str(tmp_path / "nope.safetensors"), 1.0)])
        assert rig.events == []

    def test_the_backend_follows_the_protocols(self):
        backend = H3VideoBackend()
        assert backend.name == "h3"
        assert isinstance(backend, VideoBackend)
        assert callable(backend.extend_video)
        assert VideoExtender is not None


# --- text_to_video -------------------------------------------------------------------------------------------------


class TestTextToVideo:
    def test_runs_the_stages_in_order_releasing_each_before_the_next(self, rig, tmp_path):
        model = _load(rig)
        assert _t2v(model, tmp_path / "out.mp4") == tmp_path / "out.mp4"
        assert rig.events == [
            "schedulers",
            "presentation",
            "text_encoder.load",
            "text_encoder.encode",
            "release",
            "copy.load",
            "denoise",
            "release",
            "video_vae.load",
            "decode_video",
            "release",
            "audio_vae.load",
            "decode_audio",
            "release",
            "encode_video",
            "release",
        ]

    def test_text_only_prompt_loads_no_vision_tower_and_no_vae_encoder(self, rig, tmp_path):
        model = _load(rig)
        _t2v(model, tmp_path / "out.mp4")
        assert rig.last("text_encoder.load")["vision"] is False
        assert rig.last("presentation")["images"] == []
        assert "vae_encoder.load" not in rig.calls
        held = rig.last("denoise")["conditioning"]
        assert (held.keyframes, held.anchors, held.tail_latents, held.tail_audio) == ((), (), None, None)
        assert rig.row_sets == ["plain"]

    def test_passes_the_files_and_the_presentation_tags_on(self, rig, tmp_path):
        model = _load(rig)
        _t2v(model, tmp_path / "out.mp4", prompt="a quiet sea", seed=11)
        presentation = rig.last("presentation")
        assert presentation["prompt"] == "a quiet sea"
        assert presentation["tokenizer_dir"] == tmp_path / "repo" / "tokenizer"
        assert presentation["processor_dir"] == tmp_path / "repo" / "processor"
        assert rig.last("text_encoder.load")["layers"] == 50
        assert rig.last("video_vae.load")["vae_dir"] == tmp_path / "repo" / "vae"
        assert rig.last("audio_vae.load")["audio_dir"] == tmp_path / "repo" / "audio_vae"
        denoise = rig.last("denoise")
        assert denoise["text_token_tags"] == (1, 1, 1)
        assert denoise["seed"] == 11
        assert (denoise["geo"].width, denoise["geo"].height, denoise["geo"].num_frames) == (WIDTH, HEIGHT, FRAMES)

    @pytest.mark.parametrize("steps", [4, 8])
    def test_steps_select_the_schedule(self, rig, tmp_path, steps):
        model = _load(rig)
        _t2v(model, tmp_path / "out.mp4", steps=steps)
        chosen = rig.last("schedulers")
        assert chosen["schedule"] == PRESET.schedules[steps]
        assert chosen["config_dir"] == tmp_path / "repo"

    def test_loads_the_transformer_with_the_keys_of_the_row_set_and_the_loras(self, rig, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x")
        model = _load(rig, loras=[(str(lora), 0.5)])
        _t2v(model, tmp_path / "out.mp4")
        loaded = rig.last("copy.load")
        assert loaded["keys"] == [("fold", "plain")]
        assert loaded["loras"] == ((lora, 0.5),)

    @pytest.mark.parametrize(("setting", "expected_calls"), [(4, [4, 4, 4, 3]), (None, [15])])
    def test_the_feed_forwards_run_in_the_presets_chunks(self, rig, tmp_path, monkeypatch, setting, expected_calls):
        calls: list[int] = []

        class _Block:
            def __init__(self):
                self.ff = SimpleNamespace(forward=lambda hidden_states: (calls.append(hidden_states.shape[1]), hidden_states * 2)[1])

        transformer = SimpleNamespace(transformer_blocks=[_Block()])
        monkeypatch.setattr(stored_copy, "load_copy", lambda *_a, **_k: transformer)
        rig_denoise = sampler.denoise

        def denoise(tr, *args, **kwargs):
            tr.transformer_blocks[0].ff.forward(torch.ones(1, 15, 2))
            return rig_denoise(tr, *args, **kwargs)

        monkeypatch.setattr(sampler, "denoise", denoise)
        model = _load(rig, preset=dataclasses.replace(PRESET, ffn_chunk_tokens=setting))
        _t2v(model, tmp_path / "out.mp4")
        assert calls == expected_calls

    def test_unsupported_steps_fail_before_any_stage(self, rig, tmp_path):
        model = _load(rig)
        with pytest.raises(ValueError, match="4 or 8 steps, got 6"):
            _t2v(model, tmp_path / "out.mp4", steps=6)
        assert rig.events == []

    def test_a_frame_count_off_the_17n_plus_5_grid_is_refused(self, rig, tmp_path):
        model = _load(rig)
        with pytest.raises(ValueError, match="17"):
            _t2v(model, tmp_path / "out.mp4", num_frames=25)

    def test_reports_progress_under_the_video_phase(self, rig, tmp_path):
        model = _load(rig)
        events: list[dict[str, Any]] = []
        _t2v(model, tmp_path / "out.mp4", step_callback=events.append)
        assert events == [{"phase": "video", "current_step": step, "total_steps": 8} for step in range(1, 9)]

    def test_encode_video_gets_frames_at_24_fps_with_stereo_audio_at_32_khz(self, rig, tmp_path):
        model = _load(rig)
        _t2v(model, tmp_path / "out.mp4")
        call = rig.last("encode_video")
        assert isinstance(call["frames"], list) and len(call["frames"]) == FRAMES
        assert all(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.shape == (HEIGHT, WIDTH, 3) for frame in call["frames"])
        assert call["fps"] == 24
        assert call["output_path"] == str(tmp_path / "out.mp4")
        assert call["audio"].shape[0] == 2 and call["audio"].dtype == torch.float32
        assert call["audio_sample_rate"] == 32000 == h3_audio.AUDIO_SAMPLE_RATE

    def test_a_model_loaded_for_another_mode_is_refused(self, rig, tmp_path):
        model = _load(rig, mode="i2v")
        with pytest.raises(RuntimeError, match="'i2v'"):
            _t2v(model, tmp_path / "out.mp4")
        assert rig.events == []

    def test_releases_memory_when_a_stage_fails(self, rig, tmp_path, monkeypatch):
        model = _load(rig)

        def broken(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(sampler, "denoise", broken)
        with pytest.raises(RuntimeError, match="boom"):
            _t2v(model, tmp_path / "out.mp4")
        assert rig.events[-1] == "release"
        assert "encode_video" not in rig.calls

    def test_upscale_arguments_need_an_upscale_model(self, rig, tmp_path):
        model = _load(rig)
        with pytest.raises(RuntimeError, match="upscale=True"):
            _t2v(model, tmp_path / "out.mp4", stage1_steps=4)
        with pytest.raises(ValueError, match="review needs stage1_steps"):
            _t2v(model, tmp_path / "out.mp4", review=lambda _path: None)


# --- image_to_video ------------------------------------------------------------------------------------------------


class TestImageToVideo:
    def test_first_frame_only(self, rig, tmp_path):
        model = _load(rig, mode="i2v")
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "waves", WIDTH, HEIGHT, FRAMES, 3, 8, str(tmp_path / "out.mp4"))
        assert rig.events == [
            "schedulers",
            "presentation",
            "text_encoder.load",
            "text_encoder.encode",
            "release",
            "vae_encoder.load",
            "keyframe",
            "release",
            "copy.load",
            "denoise",
            "release",
            "video_vae.load",
            "decode_video",
            "release",
            "audio_vae.load",
            "decode_audio",
            "release",
            "encode_video",
            "release",
        ]
        assert [image.size for image in rig.last("presentation")["images"]] == [(WIDTH, HEIGHT)]  # fitted to the canvas
        assert rig.last("text_encoder.load")["vision"] is True
        held = rig.last("denoise")["conditioning"]
        assert (len(held.keyframes), held.anchors) == (1, ("first",))
        assert rig.last("denoise")["text_token_tags"] == (1, 1, 0, 1)
        assert rig.row_sets == ["keyframe"]
        assert rig.last("copy.load")["keys"] == [("fold", "keyframe")]

    def test_first_and_last_frame(self, rig, tmp_path):
        model = _load(rig, mode="i2v")
        first, last = _image(tmp_path / "a.png", color=(255, 0, 0)), _image(tmp_path / "b.png", size=(80, 160), color=(0, 0, 255))
        H3VideoBackend().image_to_video(model, first, "waves", WIDTH, HEIGHT, FRAMES, 3, 4, str(tmp_path / "out.mp4"), end_image=last)
        images = rig.last("presentation")["images"]
        assert [image.size for image in images] == [(WIDTH, HEIGHT), (WIDTH, HEIGHT)]
        assert images[0].getpixel((0, 0)) == (255, 0, 0) and images[1].getpixel((0, 0)) == (0, 0, 255)
        held = rig.last("denoise")["conditioning"]
        assert (len(held.keyframes), held.anchors) == (2, ("first", "last"))
        assert rig.calls["keyframe"] == [{"size": (WIDTH, HEIGHT), "canvas": (WIDTH, HEIGHT)}] * 2
        assert rig.last("denoise")["text_token_tags"] == (1, 1, 0, 0, 1)
        assert rig.row_sets == ["keyframe"]
        assert rig.last("schedulers")["schedule"] == PRESET.schedules[4]

    def test_a_text_to_video_model_is_refused(self, rig, tmp_path):
        model = _load(rig)
        with pytest.raises(RuntimeError, match="'t2v'"):
            H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", WIDTH, HEIGHT, FRAMES, 3, 8, str(tmp_path / "out.mp4"))


# --- upscale -------------------------------------------------------------------------------------------------------

_REAL_REFINE = sampler.refine  # the rig replaces these with recorders; some tests run the real code
_REAL_UPSCALE_LATENTS = h3_backend._upscale_latents


def _names(rig: Rig, *names: str) -> list[str]:
    """Return the events of *rig* that are one of *names*, in order."""
    return [event for event in rig.events if event in names]


class TestUpscale:
    def _run(self, rig: Rig, tmp_path: Path, **kwargs: Any) -> Any:
        model = _load(rig, upscale=True)
        arguments: dict[str, Any] = dict(width=128, height=64, num_frames=FRAMES, stage1_steps=4)
        arguments.update(kwargs)
        return _t2v(model, tmp_path / "out.mp4", **arguments)

    def test_stage_one_runs_at_half_size_then_the_latent_upscale_then_the_refine_then_one_full_size_decode(self, rig, tmp_path):
        self._run(rig, tmp_path)
        geo = rig.last("denoise")["geo"]
        assert (geo.width, geo.height) == (64, 32)
        assert PRESET.schedules[4] in [call["schedule"] for call in rig.calls["schedulers"]]
        assert _names(rig, "denoise", "decode_audio", "upsampler", "copy.load", "refine", "decode_video", "encode_video") == [
            "copy.load",
            "denoise",
            "decode_audio",
            "upsampler",
            "refine",
            "decode_video",
            "encode_video",
        ]
        assert rig.events.count("copy.load") == 1  # one load serves stage 1 and the refine
        assert rig.last("upsampler")["shape"] == (1, 24, 7, 2, 4)
        assert rig.last("decode_video")["shape"] == (1, 24, 7, 4, 8)  # the one decode is at the full size, 128x64
        final = rig.last("encode_video")
        assert len(rig.calls["encode_video"]) == 1
        assert all(frame.shape == (64, 128, 3) for frame in final["frames"]) and len(final["frames"]) == FRAMES
        assert final["audio"].shape[0] == 2 and final["audio_sample_rate"] == 32000
        assert rig.events[-1] == "release"

    def test_the_stage_one_sound_is_decoded_from_the_stage_one_audio_and_muxed(self, rig, tmp_path):
        self._run(rig, tmp_path)
        assert rig.last("decode_audio")["shape"] == (2, 32, 37)
        assert rig.last("encode_video")["audio"].shape == (2, 37 * 800)

    def test_each_stage_is_released_before_the_next(self, rig, tmp_path):
        self._run(rig, tmp_path)
        for stage in ("denoise", "decode_audio", "upsampler", "refine", "decode_video"):
            assert rig.events[rig.events.index(stage) + 1] == "release", stage

    def test_review_decodes_the_half_size_clip_and_looks_at_it_before_the_upscaler(self, rig, tmp_path):
        seen: dict[str, Any] = {}

        def review(path: Path) -> None:
            rig.note("review")
            seen["path"] = path
            seen["existed"] = path.is_file()
            seen["frames"] = [frame.shape for frame in rig.last("encode_video")["frames"]]

        self._run(rig, tmp_path, review=review)
        assert seen["existed"] is True
        assert seen["path"].suffix == ".mp4" and seen["path"].parent.name.startswith("ziv-review-")
        assert not seen["path"].exists()  # deleted afterwards
        assert set(seen["frames"]) == {(32, 64, 3)}
        assert [call["shape"] for call in rig.calls["decode_video"]] == [(1, 24, 7, 2, 4), (1, 24, 7, 4, 8)]  # half size for the review, then full size
        assert _names(rig, "decode_audio", "decode_video", "encode_video", "review", "upsampler", "refine") == [
            "decode_audio",
            "decode_video",
            "encode_video",
            "review",
            "upsampler",
            "refine",
            "decode_video",
            "encode_video",
        ]
        assert rig.events[rig.events.index("review") + 1] == "release"
        assert len(rig.calls["encode_video"]) == 2  # the preview and the final video
        assert rig.events.count("copy.load") == 2  # the half-size decode needs the memory, so the refine loads its own transformer

    def test_without_a_review_the_video_is_decoded_once(self, rig, tmp_path):
        self._run(rig, tmp_path)
        assert len(rig.calls["decode_video"]) == 1

    def test_a_review_that_raises_stops_before_the_upscaler(self, rig, tmp_path):
        def review(path: Path) -> None:
            raise _Stop("rejected")

        with pytest.raises(_Stop, match="rejected"):
            self._run(rig, tmp_path, review=review)
        assert "upsampler" not in rig.calls and "refine" not in rig.calls
        assert rig.events.count("copy.load") == 1  # stage 1 only
        assert len(rig.calls["decode_video"]) == 1
        assert len(rig.calls["encode_video"]) == 1  # only the preview
        assert not (tmp_path / "out.mp4").exists()
        assert rig.events[-1] == "release"

    def test_progress_runs_through_stage_one_then_the_refine_forwards(self, rig, tmp_path):
        events: list[dict[str, Any]] = []
        self._run(rig, tmp_path, step_callback=events.append)
        assert events[:4] == [{"phase": "video_upscale_stage_1", "current_step": step, "total_steps": 6} for step in range(1, 5)]
        assert events[4:] == [{"phase": "video_upscale_stage_2", "current_step": step, "total_steps": 6} for step in (5, 6)]

    def test_two_windows_make_a_total_of_eight_with_four_stage_one_steps(self, rig, tmp_path):
        events: list[dict[str, Any]] = []
        self._run(rig, tmp_path, step_callback=events.append, num_frames=124)
        assert len(rig.calls["refine"]) == 2
        assert {event["total_steps"] for event in events} == {8}
        assert [(event["phase"], event["current_step"]) for event in events[4:]] == [("video_upscale_stage_2", step) for step in (5, 6, 7, 8)]

    def test_eight_stage_one_steps_count_eight_forwards_before_the_refine(self, rig, tmp_path):
        events: list[dict[str, Any]] = []
        self._run(rig, tmp_path, step_callback=events.append, stage1_steps=8)
        assert [event["current_step"] for event in events] == list(range(1, 11))
        assert {event["total_steps"] for event in events} == {10}

    @pytest.mark.parametrize("stage1_steps", [4, 8])
    def test_the_refine_starts_at_step_six_of_the_eight_step_schedule_whatever_stage_one_ran(self, rig, tmp_path, stage1_steps):
        self._run(rig, tmp_path, stage1_steps=stage1_steps)
        assert PRESET.schedules[stage1_steps] in [call["schedule"] for call in rig.calls["schedulers"]]
        assert rig.last("schedulers")["schedule"] == PRESET.schedules[8]
        assert [call["start"] for call in rig.calls["refine"]] == [6]
        assert rig.refine_forwards == [2]

    def test_every_window_runs_exactly_two_forwards(self, rig, tmp_path):
        self._run(rig, tmp_path, num_frames=175)
        assert len(rig.calls["refine"]) == 3
        assert rig.refine_forwards == [2, 2, 2]
        assert {call["phase"] for call in rig.calls["refine"]} == {"video_upscale_stage_2"}

    def test_a_callback_that_raises_on_the_first_stage_two_event_ends_the_run_before_any_decode_or_write(self, rig, tmp_path):
        class Cancelled(Exception):
            pass

        def cancel(event: dict[str, Any]) -> None:
            if event["phase"] == "video_upscale_stage_2":
                raise Cancelled

        with pytest.raises(Cancelled):
            self._run(rig, tmp_path, step_callback=cancel)
        assert "decode_video" not in rig.calls and "encode_video" not in rig.calls
        assert not (tmp_path / "out.mp4").exists()
        assert rig.events[-1] == "release"

    def test_the_refine_loads_the_copy_once_with_the_user_loras(self, rig, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x")
        model = _load(rig, upscale=True, loras=[(str(lora), 0.7)])
        _t2v(model, tmp_path / "out.mp4", width=128, height=64, stage1_steps=4, num_frames=124)
        assert [call["loras"] for call in rig.calls["copy.load"]] == [((lora, 0.7),)]
        assert len(rig.calls["copy.load"]) == 1  # one load for stage 1 and every window
        assert len({id(call["transformer"]) for call in rig.calls["refine"]}) == 1
        assert rig.calls["refine"][0]["transformer"] is rig.last("denoise")["transformer"]

    def test_the_shared_transformer_loads_after_the_text_and_keyframe_encoders_have_let_go(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, FRAMES, 3, 8, str(tmp_path / "out.mp4"), stage1_steps=4)
        events = rig.events
        load = events.index("copy.load")
        assert events.count("copy.load") == 1
        assert load > max(events.index("text_encoder.encode"), events.index("vae_encoder.load"), events.index("keyframe"))
        assert events[load - 1] == "release"  # the last encoder was released before the load
        assert events[load + 1] == "denoise"

    def test_a_review_makes_the_refine_load_its_own_copy_with_the_user_loras(self, rig, tmp_path):
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x")
        model = _load(rig, upscale=True, loras=[(str(lora), 0.7)])
        _t2v(model, tmp_path / "out.mp4", width=128, height=64, stage1_steps=4, num_frames=124, review=lambda path: None)
        assert [call["loras"] for call in rig.calls["copy.load"]] == [((lora, 0.7),), ((lora, 0.7),)]
        assert len({id(call["transformer"]) for call in rig.calls["refine"]}) == 1
        assert rig.calls["refine"][0]["transformer"] is not rig.last("denoise")["transformer"]

    def test_a_card_that_cannot_hold_the_resident_transformer_loads_twice(self, rig, tmp_path):
        rig.gpu_bytes = 2**30
        self._run(rig, tmp_path)
        assert [call["keys"] for call in rig.calls["copy.load"]] == [[("fold", "plain")], [("fold", "plain", 6)]]
        assert rig.calls["refine"][0]["transformer"] is not rig.last("denoise")["transformer"]

    def test_the_decision_uses_the_cards_memory_the_clip_size_the_mode_and_the_loras(self, rig, tmp_path, monkeypatch):
        asked: list[tuple[Any, ...]] = []
        monkeypatch.setattr(h3_backend, "h3_upscale_keeps_transformer", lambda *args, **kwargs: (asked.append((args, kwargs)), False)[1])
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"12345")
        model = _load(rig, mode="i2v", upscale=True, loras=[(str(lora), 0.7)])
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, FRAMES, 3, 8, str(tmp_path / "out.mp4"), stage1_steps=4)
        assert asked == [((128, 64, FRAMES, rig.gpu_bytes), {"mode": "i2v", "loras_bytes": 5})]
        assert len(rig.calls["copy.load"]) == 2

    def test_the_shared_transformer_is_let_go_before_the_full_size_decode(self, rig, tmp_path, monkeypatch):
        import gc
        import weakref

        class _Transformer:
            pass

        monkeypatch.setattr(stored_copy, "load_copy", lambda copy_dir, keys, device, *, loras=(), lora_names=(): (rig.note("copy.load", keys=keys, loras=loras), _Transformer())[1])
        seen: dict[str, Any] = {}
        real_decode = decode.decode_video

        def decode_video(vae, latents):
            for stage in ("denoise", "refine"):
                for call in rig.calls[stage]:
                    call["transformer"] = None
            gc.collect()
            seen["alive"] = seen["ref"]() is not None
            return real_decode(vae, latents)

        real_refine = sampler.refine

        def refine(transformer, *args, **kwargs):
            seen["ref"] = weakref.ref(transformer)
            return real_refine(transformer, *args, **kwargs)

        monkeypatch.setattr(sampler, "refine", refine)
        monkeypatch.setattr(decode, "decode_video", decode_video)
        self._run(rig, tmp_path)
        assert seen["alive"] is False

    def test_a_failure_in_a_stage_between_the_passes_lets_go_of_the_transformer_and_releases_memory(self, rig, tmp_path):
        rig.oom_in = "upsampler"
        with pytest.raises(RuntimeError, match="ran out of GPU memory at 128x64"):
            self._run(rig, tmp_path)
        assert "refine" not in rig.calls and "decode_video" not in rig.calls
        assert rig.events[-1] == "release"

    def test_windows_get_their_own_slice_of_one_noise_draw_and_the_audio_at_proportional_offsets(self, rig, tmp_path):
        from zvisiongenerator.backends.cuda.video.h3 import refine as refine_module

        self._run(rig, tmp_path, num_frames=124)
        noise_video, noise_audio = refine_module.draw_refine_noise((1, 24, 37, 4, 8), (2, 32, 207), 5)
        first, second = rig.calls["refine"]
        assert first["latents"].shape == second["latents"].shape == (1, 24, 22, 4, 8)
        assert torch.equal(first["noise"][0], noise_video[:, :, 0:22]) and torch.equal(second["noise"][0], noise_video[:, :, 15:37])
        offset = round(15 / 37 * 207)
        assert torch.equal(second["noise"][1], refine_module.window_audio_rows(noise_audio, offset, 122))
        assert first["audio_rows"].shape == second["audio_rows"].shape == (244, 32)
        assert (first["geo"].num_frames, first["geo"].width, first["geo"].height) == (73, 128, 64)

    def test_the_windows_are_stitched_into_one_latent_for_the_decode(self, rig, tmp_path):
        self._run(rig, tmp_path, num_frames=124)
        assert rig.last("decode_video")["shape"] == (1, 24, 37, 4, 8)

    def test_stage_one_text_conditioning_is_reused_by_every_window(self, rig, tmp_path):
        self._run(rig, tmp_path, num_frames=124)
        assert len(rig.calls["presentation"]) == 1
        assert all(call["text_token_tags"] == rig.last("denoise")["text_token_tags"] for call in rig.calls["refine"])

    def test_an_upscale_model_needs_stage1_steps(self, rig, tmp_path):
        model = _load(rig, upscale=True)
        with pytest.raises(ValueError, match="stage1_steps must be provided"):
            _t2v(model, tmp_path / "out.mp4")

    def test_the_canvas_must_halve_to_a_multiple_of_32(self, rig, tmp_path):
        model = _load(rig, upscale=True)
        with pytest.raises(ValueError, match="divisible by 64"):
            _t2v(model, tmp_path / "out.mp4", width=96, height=64, stage1_steps=4)

    def test_image_to_video_encodes_the_keyframes_at_the_half_canvas_then_again_at_the_full_canvas(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, FRAMES, 3, 8, str(tmp_path / "out.mp4"), end_image=_image(tmp_path / "b.png"), stage1_steps=8)
        assert [image.size for image in rig.last("presentation")["images"]] == [(64, 32), (64, 32)]
        assert [call["canvas"] for call in rig.calls["keyframe"]] == [(64, 32), (64, 32), (128, 64), (128, 64)]
        assert [call["size"] for call in rig.calls["keyframe"][2:]] == [(128, 64), (128, 64)]
        names = _names(rig, "upsampler", "vae_encoder.load", "refine")
        assert names[-3:] == ["upsampler", "vae_encoder.load", "refine"]

    def test_a_single_window_holds_both_images(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, FRAMES, 3, 8, str(tmp_path / "out.mp4"), end_image=_image(tmp_path / "b.png"), stage1_steps=4)
        held = rig.last("refine")
        assert held["conditioning"].anchors == ("first", "last")
        assert all(frame.shape == (1, 24, 1, 4, 8) for frame in held["conditioning"].keyframes)
        assert held["keyframe_seed"] == 3 + 2

    def test_chunked_first_and_last_images_are_held_by_the_outer_windows_only(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, 175, 3, 8, str(tmp_path / "out.mp4"), end_image=_image(tmp_path / "b.png"), stage1_steps=4)
        assert [call["conditioning"].anchors for call in rig.calls["refine"]] == [("first",), (), ("last",)]
        assert [len(call["conditioning"].keyframes) for call in rig.calls["refine"]] == [1, 0, 1]

    def test_a_clip_without_images_holds_nothing_and_encodes_no_keyframe(self, rig, tmp_path):
        self._run(rig, tmp_path, num_frames=175)
        assert "keyframe" not in rig.calls
        assert all(call["conditioning"].keyframes == () for call in rig.calls["refine"])

    def test_the_refine_loads_the_union_of_the_fold_keys_of_the_row_sets_its_windows_use(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, 175, 3, 8, str(tmp_path / "out.mp4"), stage1_steps=4)
        assert len(rig.calls["copy.load"]) == 1
        # stage 1 keeps the whole-clip keys; window 0 holds the image and the others hold none
        assert set(rig.calls["copy.load"][0]["keys"]) == {("fold", "keyframe"), ("fold", "keyframe", 6), ("fold", "plain", 6)}

    def test_a_text_to_video_refine_loads_only_the_plain_keys(self, rig, tmp_path):
        self._run(rig, tmp_path, num_frames=175)
        assert rig.calls["copy.load"][-1]["keys"] == [("fold", "plain"), ("fold", "plain", 6)]

    def test_a_single_window_with_an_image_loads_only_the_keyframe_keys(self, rig, tmp_path):
        model = _load(rig, mode="i2v", upscale=True)
        H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", 128, 64, FRAMES, 3, 8, str(tmp_path / "out.mp4"), stage1_steps=4)
        assert rig.calls["copy.load"][-1]["keys"] == [("fold", "keyframe"), ("fold", "keyframe", 6)]


class TestUpscaleLatents:
    def _model(self, rig: Rig, tmp_path: Path, *, stats: dict[str, Any] | None, with_file: bool = True) -> Any:
        model = _load(rig, upscale=True)
        (model.files.repo_dir / "vae").mkdir(parents=True)
        if stats is not None:
            (model.files.repo_dir / "vae" / "config.json").write_text(json.dumps(stats))
        files = dataclasses.replace(model.files, latent_upsampler_file=tmp_path / "net.safetensors" if with_file else None)
        return dataclasses.replace(model, files=files)

    def test_it_loads_the_net_and_upscales_with_the_vae_statistics(self, rig, tmp_path, monkeypatch):
        from zvisiongenerator.backends.cuda.video.h3 import latent_upsampler

        model = self._model(rig, tmp_path, stats={"latents_mean": [0.5] * 24, "latents_std": [2.0] * 24})
        calls: dict[str, Any] = {}
        monkeypatch.setattr(latent_upsampler, "load_latent_upsampler", lambda path, device: (calls.__setitem__("load", (path, device)), "net")[1])
        monkeypatch.setattr(latent_upsampler, "upscale_latents", lambda net, latents, mean, std: (calls.__setitem__("up", (net, tuple(latents.shape), mean, std)), "result")[1])

        assert _REAL_UPSCALE_LATENTS(model, torch.zeros(1, 24, 7, 2, 4)) == "result"
        assert calls["load"] == (tmp_path / "net.safetensors", model.device)
        assert calls["up"] == ("net", (1, 24, 7, 2, 4), [0.5] * 24, [2.0] * 24)

    def test_a_vae_config_without_statistics_is_refused(self, rig, tmp_path):
        model = self._model(rig, tmp_path, stats={"scaling_factor": 1})
        with pytest.raises(RuntimeError, match="latents_mean"):
            _REAL_UPSCALE_LATENTS(model, torch.zeros(1, 24, 7, 2, 4))

    def test_a_model_loaded_without_the_net_file_is_refused(self, rig, tmp_path):
        model = self._model(rig, tmp_path, stats={"latents_mean": [0.0], "latents_std": [1.0]}, with_file=False)
        with pytest.raises(RuntimeError, match="latent upscaler"):
            _REAL_UPSCALE_LATENTS(model, torch.zeros(1, 24, 7, 2, 4))


class TestRefineStart:
    def test_it_is_the_last_two_forwards_of_the_eight_step_schedule(self):
        assert h3_backend._refine_start(PRESET) == 6

    def test_it_ignores_the_stage_one_schedules(self):
        only_eight = dataclasses.replace(PRESET, schedules={8: H3Schedule(8, 6.0, 3.0)})
        assert h3_backend._refine_start(only_eight) == 6


# --- extend --------------------------------------------------------------------------------------------------------

EXTEND = 17
WINDOW = 22 + EXTEND


def _window_waveform(latent_frames: int) -> torch.Tensor:
    """A window whose sound is silent under the carried 22 frames and a constant 0.5 afterwards."""
    samples = latent_frames * 800
    wave_ = torch.full((2, samples), 0.5)
    wave_[:, : round(22 / 24 * 32000)] = 0.0
    return wave_


class TestExtend:
    def _extend(self, rig: Rig, tmp_path: Path, *, source=None, model=None, **kwargs: Any) -> Any:
        arguments: dict[str, Any] = dict(extend_frames=EXTEND, seed=7, steps=8, segment_dir=tmp_path, audio_output=True)
        arguments.update(kwargs)
        return H3VideoBackend().extend_video(model or _load(rig, mode="extend"), source or make_extend_source(width=WIDTH, height=HEIGHT), "she waves", **arguments)

    def test_stages_in_order_with_the_tail_and_its_audio_encoded_first(self, rig, tmp_path):
        self._extend(rig, tmp_path)
        assert rig.events == [
            "read_tail",
            "read_tail_audio",
            "schedulers",
            "presentation",
            "text_encoder.load",
            "text_encoder.encode",
            "release",
            "vae_encoder.load",
            "tail",
            "release",
            "audio_vae.load",
            "tail_audio",
            "release",
            "copy.load",
            "denoise",
            "release",
            "video_vae.load",
            "decode_video",
            "release",
            "audio_vae.load",
            "decode_audio",
            "release",
            "segment_frames",
            "release",
        ]
        assert rig.row_sets == ["extend"]

    def test_reads_22_frames_and_their_audio_at_32_khz(self, rig, tmp_path):
        self._extend(rig, tmp_path)
        assert rig.last("read_tail") == {"frames": 22}
        assert rig.last("read_tail_audio") == {"sample_rate": 32000, "frames": 22}
        assert rig.last("tail")["shape"] == (22, HEIGHT, WIDTH, 3)
        assert rig.last("tail_audio")["shape"] == (2, round(22 / 24 * 32000))

    def test_the_window_is_the_22_tail_frames_plus_the_new_ones(self, rig, tmp_path):
        self._extend(rig, tmp_path)
        denoise = rig.last("denoise")
        assert denoise["geo"].num_frames == WINDOW
        assert (denoise["geo"].width, denoise["geo"].height) == (WIDTH, HEIGHT)
        held = denoise["conditioning"]
        assert held.keyframes == () and held.anchors == ()
        assert tuple(held.tail_latents.shape[:3]) == (1, 24, 7)
        assert tuple(held.tail_audio.shape) == (74, 32)
        assert rig.last("presentation")["images"] == [] and rig.last("text_encoder.load")["vision"] is False
        assert denoise["seed"] == 7
        assert rig.last("copy.load")["keys"] == [("fold", "extend")]

    def test_the_segment_has_only_the_new_frames(self, rig, tmp_path):
        segment = self._extend(rig, tmp_path)
        written = rig.last("segment_frames")
        # the fake decode fills frame i with the value i: the 22 carried frames (0..21) are gone
        assert [int(frame[0, 0, 0]) for frame in written["frames"]] == list(range(22, WINDOW))
        assert len(written["frames"]) == EXTEND
        assert str(tmp_path / "segment.mp4") in written["command"]
        assert isinstance(segment, ExtendSegment)
        assert segment.video_path == tmp_path / "segment.mp4"

    def test_the_audio_segment_starts_at_the_new_frames_and_has_the_exact_length(self, rig, tmp_path):
        rig.window_waveform = _window_waveform
        segment = self._extend(rig, tmp_path)
        assert segment.audio_path == tmp_path / "segment.wav"
        assert segment.lead_samples == 0
        with wave.open(str(segment.audio_path), "rb") as reader:
            assert (reader.getnchannels(), reader.getframerate(), reader.getnframes()) == (2, 48000, EXTEND * 2000)
            pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16).reshape(-1, 2).T / 32767
        # silence under the carried frames was dropped: the new part is the constant 0.5 from its first samples on
        assert np.abs(pcm[:, 300:-300] - 0.5).max() < 0.02
        assert np.abs(pcm[:, :40]).max() > 0.2

    def test_the_audio_is_cut_at_the_join_rate_by_the_tail_length(self, rig):
        waveform = _window_waveform(65)  # 52000 samples at 32 kHz: 39 frames
        assert waveform.shape[1] == 52000
        audio = h3_backend._new_audio(waveform, EXTEND)
        assert audio.shape == (2, EXTEND * 2000)
        assert audio.dtype == np.float32

    def test_a_short_window_audio_is_faded_and_padded_to_the_exact_length(self, rig):
        audio = h3_backend._new_audio(torch.full((2, 40000), 0.5), EXTEND)  # ends before the window does
        assert audio.shape == (2, EXTEND * 2000)
        assert audio[:, -1].max() == 0.0

    def test_audio_output_off_writes_the_video_only(self, rig, tmp_path):
        segment = self._extend(rig, tmp_path, audio_output=False)
        assert segment.audio_path is None and segment.lead_samples == 0
        assert not (tmp_path / "segment.wav").exists()

    def test_a_silent_source_reads_no_audio_and_writes_none(self, rig, tmp_path):
        segment = self._extend(rig, tmp_path, source=make_extend_source(width=WIDTH, height=HEIGHT, has_audio=False))
        assert "read_tail_audio" not in rig.calls and "tail_audio" not in rig.calls
        assert rig.last("denoise")["conditioning"].tail_audio is None
        assert rig.row_sets == ["keyframe"]  # without a held sound the forwards need the keys of a keyframe-only clip
        assert segment.audio_path is None

    def test_longer_extensions_make_longer_windows(self, rig, tmp_path):
        self._extend(rig, tmp_path, extend_frames=34)
        assert rig.last("denoise")["geo"].num_frames == 56
        assert len(rig.last("segment_frames")["frames"]) == 34

    @pytest.mark.parametrize("frames", [0, 8, 16, 18, 25, -17])
    def test_extend_frames_must_be_a_positive_multiple_of_17(self, rig, tmp_path, frames):
        with pytest.raises(ValueError, match="multiple of 17"):
            self._extend(rig, tmp_path, extend_frames=frames)
        assert rig.events == []

    @pytest.mark.parametrize("fps", [23.976, 25.0, 30.0, 60.0])
    def test_a_source_that_is_not_24_fps_is_refused(self, rig, tmp_path, fps):
        with pytest.raises(ValueError, match="24 fps"):
            self._extend(rig, tmp_path, source=make_extend_source(width=WIDTH, height=HEIGHT, fps=fps))
        assert rig.events == []

    @pytest.mark.parametrize("size", [(100, 64), (96, 70)])
    def test_a_source_that_is_not_32_aligned_is_refused(self, rig, tmp_path, size):
        with pytest.raises(ValueError, match="multiple of 32"):
            self._extend(rig, tmp_path, source=make_extend_source(width=size[0], height=size[1]))
        assert rig.events == []

    def test_a_model_not_loaded_for_extend_is_refused(self, rig, tmp_path):
        with pytest.raises(RuntimeError, match="'extend'"):
            self._extend(rig, tmp_path, model=_load(rig, mode="i2v"))

    def test_an_upscale_model_is_refused(self, rig, tmp_path):
        with pytest.raises(ValueError, match="upscale"):
            self._extend(rig, tmp_path, model=_load(rig, mode="extend", upscale=True))

    def test_a_window_that_decodes_to_the_wrong_length_is_an_error(self, rig, tmp_path, monkeypatch):
        monkeypatch.setattr(decode, "decode_video", lambda vae, latents: [np.zeros((HEIGHT, WIDTH, 3), np.uint8)] * 30)
        with pytest.raises(RuntimeError, match="30 frames, expected 39"):
            self._extend(rig, tmp_path)

    def test_progress_runs_under_the_video_phase(self, rig, tmp_path):
        events: list[dict[str, Any]] = []
        self._extend(rig, tmp_path, step_callback=events.append)
        assert [event["current_step"] for event in events] == list(range(1, 9))
        assert {event["phase"] for event in events} == {"video"}


# --- out of memory -------------------------------------------------------------------------------------------------


class TestOutOfMemory:
    @pytest.mark.parametrize("stage", ["denoise", "decode"])
    def test_text_to_video_reports_the_size_and_frames(self, rig, tmp_path, stage):
        model = _load(rig)
        rig.oom_in = stage
        with pytest.raises(RuntimeError) as raised:
            _t2v(model, tmp_path / "out.mp4")
        assert str(raised.value) == OOM_MESSAGE.format(w=WIDTH, h=HEIGHT, n=FRAMES)
        assert isinstance(raised.value.__cause__, torch.cuda.OutOfMemoryError)
        assert rig.events[-1] == "release"

    def test_image_to_video_reports_the_size_and_frames(self, rig, tmp_path):
        model = _load(rig, mode="i2v")
        rig.oom_in = "denoise"
        with pytest.raises(RuntimeError) as raised:
            H3VideoBackend().image_to_video(model, _image(tmp_path / "a.png"), "x", WIDTH, HEIGHT, FRAMES, 3, 8, str(tmp_path / "out.mp4"))
        assert str(raised.value) == OOM_MESSAGE.format(w=WIDTH, h=HEIGHT, n=FRAMES)

    def test_upscale_reports_the_target_size(self, rig, tmp_path):
        model = _load(rig, upscale=True)
        rig.oom_in = "denoise"
        with pytest.raises(RuntimeError) as raised:
            _t2v(model, tmp_path / "out.mp4", width=128, height=64, stage1_steps=4)
        assert str(raised.value) == OOM_MESSAGE.format(w=128, h=64, n=FRAMES)

    @pytest.mark.parametrize("stage", ["upsampler", "refine", "decode"])
    def test_upscale_reports_the_target_size_from_every_later_stage(self, rig, tmp_path, stage):
        model = _load(rig, upscale=True)
        rig.oom_in = stage
        with pytest.raises(RuntimeError) as raised:
            _t2v(model, tmp_path / "out.mp4", width=128, height=64, stage1_steps=4)
        assert str(raised.value) == OOM_MESSAGE.format(w=128, h=64, n=FRAMES)
        assert rig.events[-1] == "release"

    def test_extend_reports_the_window(self, rig, tmp_path):
        rig.oom_in = "denoise"
        with pytest.raises(RuntimeError) as raised:
            TestExtend()._extend(rig, tmp_path)
        assert str(raised.value) == OOM_MESSAGE.format(w=WIDTH, h=HEIGHT, n=WINDOW)
        assert rig.events[-1] == "release"

    def test_other_errors_are_not_rewritten(self, rig, tmp_path, monkeypatch):
        model = _load(rig)
        monkeypatch.setattr(decode, "decode_video", lambda vae, latents: (_ for _ in ()).throw(ValueError("bad latents")))
        with pytest.raises(ValueError, match="bad latents"):
            _t2v(model, tmp_path / "out.mp4")


# --- real encode ---------------------------------------------------------------------------------------------------


class TestRealEncode:
    def test_a_clip_encodes_to_an_mp4_with_video_and_stereo_audio(self, rig, tmp_path):
        av = pytest.importorskip("av")
        rig.real_encode = True
        model = _load(rig)
        output = tmp_path / "clip.mp4"
        _t2v(model, output)
        with av.open(str(output)) as container:
            video, audio = container.streams.video[0], container.streams.audio[0]
            assert (video.width, video.height) == (WIDTH, HEIGHT)
            assert float(video.average_rate) == 24.0
            assert sum(1 for _ in container.decode(video=0)) == FRAMES
            assert audio.rate == 32000 and audio.codec_context.channels == 2


# --- fold keys against the real schedulers -------------------------------------------------------------------------


class TestFoldKeys:
    def test_every_key_a_clip_asks_for_is_in_the_stored_copy(self, tmp_path):
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Scheduler

        for folder, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
            MiniMaxH3Scheduler(shift=shift).save_pretrained(tmp_path / folder)
        stored = set(h3_backend._all_fold_keys(PRESET, tmp_path))
        assert stored
        for steps, preset_schedule in PRESET.schedules.items():
            schedulers = schedule.build_schedulers(preset_schedule, tmp_path)
            for row_set in schedule.ROW_SETS:
                keys = h3_backend._fold_keys(schedulers, row_set)
                assert keys and set(keys) <= stored, (steps, row_set)


class _TimestepRecorder(torch.nn.Module):
    """A stand-in transformer that records the distinct row timesteps of each forward and returns a zero velocity."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: set[tuple[float, ...]] = set()

    def forward(self, hidden_states, audio_hidden_states, timestep, **_kwargs):
        self.keys.add(tuple(float(value) for value in timestep.tolist()))
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


def _clip_specs(forwards: int) -> dict[str, h3_backend._ClipSpec]:
    """Return one spec per kind of clip the backend makes, built as the backend builds them."""
    image = Image.new("RGB", (WIDTH, HEIGHT))
    common = {"prompt": "p", "width": WIDTH, "height": HEIGHT, "seed": 1, "forwards": forwards}
    tail = np.zeros((conditioning.TAIL_FRAMES, HEIGHT, WIDTH, 3), np.uint8)
    sound = np.zeros((2, 8), np.float32)
    window = conditioning.TAIL_FRAMES + 17
    return {
        "t2v": h3_backend._ClipSpec(num_frames=FRAMES, **common),
        "i2v": h3_backend._ClipSpec(num_frames=FRAMES, keyframes=(image,), anchors=("first",), **common),
        "first+last": h3_backend._ClipSpec(num_frames=FRAMES, keyframes=(image, image), anchors=("first", "last"), **common),
        "extend with audio": h3_backend._ClipSpec(num_frames=window, tail=tail, tail_audio=sound, **common),
        "extend without audio": h3_backend._ClipSpec(num_frames=window, tail=tail, **common),
    }


def _held_media(spec: h3_backend._ClipSpec, geo: Any) -> sampler.Conditioning:
    """Return the sampler conditioning a *spec* is rendered with, filled with zero latents of the right shapes."""
    cell = (geo.latent_height, geo.latent_width)
    return sampler.Conditioning(
        keyframes=tuple(torch.zeros(1, 24, 1, *cell) for _ in spec.keyframes),
        anchors=spec.anchors,
        tail_latents=torch.zeros(1, 24, 7, *cell) if spec.tail is not None else None,
        tail_audio=torch.zeros(2 * conditioning.TAIL_AUDIO_LATENTS, 32) if spec.tail_audio is not None else None,
    )


class TestRowSetCoversTheForwards:
    @pytest.mark.parametrize("forwards", [4, 8])
    @pytest.mark.parametrize("kind", ["t2v", "i2v", "first+last", "extend with audio", "extend without audio"])
    def test_the_timesteps_a_clip_runs_with_are_in_the_keys_its_row_set_loads(self, tmp_path, kind, forwards):
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Scheduler

        for folder, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
            MiniMaxH3Scheduler(shift=shift).save_pretrained(tmp_path / folder)
        spec = _clip_specs(forwards)[kind]
        schedulers = schedule.build_schedulers(PRESET.schedules[forwards], tmp_path)
        geo = h3_backend.geometry(spec.width, spec.height, spec.num_frames)
        recorder = _TimestepRecorder()
        sampler.denoise(recorder, torch.zeros(1, 3, 4), geo, schedulers, spec.seed, "cpu", conditioning=_held_media(spec, geo), text_token_tags=[0, 0, 0])
        loaded = set(h3_backend._fold_keys(schedulers, spec.row_set))
        assert len(recorder.keys) > 1
        assert recorder.keys <= loaded, (kind, forwards, spec.row_set, recorder.keys - loaded)


class TestRefineRowSetsCoverTheForwards:
    """The refine loads the union of the tail keys of the row sets its windows use; every window kind must find its keys in it."""

    @staticmethod
    def _schedulers(tmp_path: Path) -> Any:
        pytest.importorskip("diffusers")
        from diffusers import MiniMaxH3Scheduler

        for folder, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
            MiniMaxH3Scheduler(shift=shift).save_pretrained(tmp_path / folder)
        return schedule.build_schedulers(PRESET.schedules[8], tmp_path)

    def test_the_union_holds_both_tails_and_each_tail_is_two_keys_inside_the_stored_copy(self, tmp_path):
        schedulers = self._schedulers(tmp_path)
        stored = set(h3_backend._all_fold_keys(PRESET, tmp_path))
        plain = h3_backend._fold_keys(schedulers, "plain", 6)
        keyframe = h3_backend._fold_keys(schedulers, "keyframe", 6)
        assert len(plain) == len(keyframe) == 2
        assert set(plain) | set(keyframe) <= stored
        assert set(plain) != set(keyframe)

    @pytest.mark.parametrize("held", [(), ("first",), ("last",), ("first", "last")], ids=["none", "first", "last", "both"])
    def test_the_timesteps_of_every_window_kind_are_in_the_loaded_keys(self, tmp_path, held):
        schedulers = self._schedulers(tmp_path)
        loaded = set(h3_backend._fold_keys(schedulers, "plain", 6)) | set(h3_backend._fold_keys(schedulers, "keyframe", 6))
        geo = h3_backend.geometry(WIDTH, HEIGHT, FRAMES)
        cell = (geo.latent_height, geo.latent_width)
        conditioning_ = sampler.Conditioning(keyframes=tuple(torch.zeros(1, 24, 1, *cell) for _ in held), anchors=held)
        recorder = _TimestepRecorder()
        _REAL_REFINE(
            recorder,
            torch.zeros(1, 3, 4),
            geo,
            schedulers,
            torch.zeros(1, 24, geo.latent_frames, *cell),
            torch.zeros(geo.audio_rows, 32),
            (torch.zeros(1, 24, geo.latent_frames, *cell), torch.zeros(geo.audio_rows, 32)),
            start=h3_backend._refine_start(PRESET),
            device="cpu",
            conditioning=conditioning_,
            text_token_tags=[0, 0, 0],
        )
        assert len(recorder.keys) == 2
        assert recorder.keys <= loaded

    def test_one_row_set_alone_would_miss_a_key(self, tmp_path):
        schedulers = self._schedulers(tmp_path)
        plain = set(h3_backend._fold_keys(schedulers, "plain", 6))
        keyframe = set(h3_backend._fold_keys(schedulers, "keyframe", 6))
        assert keyframe - plain and plain - keyframe  # this is why the refine loads the union


# --- file resolution -----------------------------------------------------------------------------------------------


class _Downloader:
    """A ``snapshot_download`` stand-in that records its calls and lays out the files it was asked for."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        folder = self.root / kwargs["repo_id"].replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        for name in kwargs.get("allow_patterns") or ():
            if "*" not in name:
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(b"x")
        return str(folder)


class TestResolveFiles:
    @pytest.fixture(autouse=True)
    def _no_cache(self, monkeypatch):
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: [])

    def test_downloads_each_target_with_its_own_filters_and_revision(self, tmp_path):
        download = _Downloader(tmp_path)
        files = sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, snapshot_download=download)
        main, text, lora = download.calls
        assert main == {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": "rev-main", "allow_patterns": list(PRESET.repo_allow), "ignore_patterns": list(PRESET.repo_ignore)}
        assert (text["repo_id"], text["revision"], text["allow_patterns"]) == ("te/repo", "rev-te", list(PRESET.text_encoder.files))
        assert (lora["repo_id"], lora["revision"], lora["allow_patterns"]) == ("lora/repo", "rev-lora", ["turbo.safetensors"])
        assert files.repo_dir == tmp_path / "MiniMaxAI--MiniMax-H3"
        assert files.text_encoder_dir == tmp_path / "te--repo"
        assert [path.name for path in files.text_encoder_shards] == ["model-1.safetensors", "model-2.safetensors"]
        assert files.lora_file == tmp_path / "lora--repo" / "turbo.safetensors"

    def test_a_reference_with_its_own_revision_keeps_it(self, tmp_path):
        download = _Downloader(tmp_path)
        sources.resolve_files("MiniMaxAI/MiniMax-H3@abc123", PRESET, snapshot_download=download)
        assert download.calls[0]["revision"] == "abc123"

    def test_a_local_folder_is_used_as_it_is(self, tmp_path):
        folder = tmp_path / "h3"
        folder.mkdir()
        (folder / "model_index.json").write_text("{}")
        download = _Downloader(tmp_path / "hub")
        files = sources.resolve_files(str(folder), PRESET, snapshot_download=download)
        assert files.repo_dir == folder
        assert [call["repo_id"] for call in download.calls] == ["te/repo", "lora/repo"]

    def test_a_local_folder_without_a_model_index_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model_index.json"):
            sources.resolve_files(str(tmp_path), PRESET, snapshot_download=_Downloader(tmp_path / "hub"))

    def test_a_cached_snapshot_with_the_files_is_not_downloaded_again(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "cached"
        snapshot.mkdir()
        for name in PRESET.text_encoder.files + PRESET.lora.files:
            (snapshot / name).write_bytes(b"x")
        monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: [snapshot])
        download = _Downloader(tmp_path / "hub")
        files = sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, snapshot_download=download)
        assert [call["repo_id"] for call in download.calls] == ["MiniMaxAI/MiniMax-H3"]
        assert files.text_encoder_dir == snapshot and files.lora_file == snapshot / "turbo.safetensors"

    def test_an_upscale_resolves_the_latent_upsampler_with_the_files_of_its_plan_target(self, tmp_path):
        download = _Downloader(tmp_path)
        files = sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, upscale=True, snapshot_download=download)
        planned = download.calls[-1]
        assert (planned["repo_id"], planned["revision"], planned["allow_patterns"]) == ("upsampler/repo", "rev-up", ["upsampler.safetensors"])
        assert files.latent_upsampler_file == tmp_path / "upsampler--repo" / "upsampler.safetensors"

    def test_a_latent_upsampler_in_a_subfolder_of_its_repository_resolves_to_that_file(self, tmp_path):
        preset = H3Preset(**{**PRESET.__dict__, "latent_upsampler": HubFiles("upsampler/repo", "rev-up", ("net_v1/net_fp16.safetensors",))})
        download = _Downloader(tmp_path)
        files = sources.resolve_files("MiniMaxAI/MiniMax-H3", preset, upscale=True, snapshot_download=download)
        assert download.calls[-1]["allow_patterns"] == ["net_v1/net_fp16.safetensors"]
        assert files.latent_upsampler_file == tmp_path / "upsampler--repo" / "net_v1" / "net_fp16.safetensors"

    def test_without_an_upscale_the_latent_upsampler_is_not_requested(self, tmp_path):
        download = _Downloader(tmp_path)
        files = sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, snapshot_download=download)
        assert files.latent_upsampler_file is None
        assert "upsampler/repo" not in [call["repo_id"] for call in download.calls]

    def test_a_missing_latent_upsampler_file_is_reported(self, tmp_path):
        download = _Downloader(tmp_path)

        def without_upsampler(**kwargs: Any) -> str:
            folder = download(**kwargs)
            if kwargs["repo_id"] == "upsampler/repo":
                (Path(folder) / "upsampler.safetensors").unlink()
            return folder

        with pytest.raises(FileNotFoundError, match="Latent upsampler not found"):
            sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, upscale=True, snapshot_download=without_upsampler)

    def test_a_failed_download_names_the_model(self, tmp_path):
        def failing(**_kwargs: Any) -> str:
            raise OSError("offline")

        with pytest.raises(RuntimeError, match="MiniMaxAI/MiniMax-H3.*offline"):
            sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, snapshot_download=failing)

    def test_a_missing_lora_file_is_reported(self, tmp_path):
        download = _Downloader(tmp_path)

        def without_lora(**kwargs: Any) -> str:
            folder = download(**kwargs)
            if kwargs["repo_id"] == "lora/repo":
                (Path(folder) / "turbo.safetensors").unlink()
            return folder

        with pytest.raises(FileNotFoundError, match="Turbo LoRA"):
            sources.resolve_files("MiniMaxAI/MiniMax-H3", PRESET, snapshot_download=without_lora)

    def test_the_lora_must_be_one_file(self, tmp_path):
        preset = H3Preset(**{**PRESET.__dict__, "lora": HubFiles("lora/repo", "rev", ("a", "b"))})
        with pytest.raises(ValueError, match="one file"):
            sources.resolve_files("MiniMaxAI/MiniMax-H3", preset, snapshot_download=_Downloader(tmp_path))


class TestTransformerFiles:
    def test_lists_the_shards_the_index_names(self, tmp_path):
        folder = tmp_path / "transformer"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"layers": 2}))
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-2.safetensors", "b": "m-1.safetensors", "c": "m-2.safetensors"}}))
        shards, config = sources.transformer_files(tmp_path)
        assert [path.name for path in shards] == ["m-1.safetensors", "m-2.safetensors"]
        assert config == {"layers": 2}

    def test_falls_back_to_every_safetensors_file(self, tmp_path):
        folder = tmp_path / "transformer"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
        assert [path.name for path in sources.transformer_files(tmp_path)[0]] == ["diffusion_pytorch_model.safetensors"]

    def test_missing_files_are_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.json"):
            sources.transformer_files(tmp_path)
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="no transformer weights"):
            sources.transformer_files(tmp_path)


# --- the leaf moves ------------------------------------------------------------------------------------------------


class TestRuntimeLeaves:
    def test_the_ltx_backend_still_exposes_the_moved_helpers(self):
        from zvisiongenerator.backends.cuda.video import backend as ltx_backend
        from zvisiongenerator.backends.cuda.video import runtime

        assert ltx_backend._configure_torch_runtime is runtime.configure_torch_runtime
        assert ltx_backend._fit_image is runtime.fit_image

    def test_fit_image_covers_the_canvas_and_crops_the_centre(self):
        from zvisiongenerator.backends.cuda.video.runtime import fit_image

        image = Image.new("RGB", (200, 100), (0, 0, 0))
        image.paste((255, 0, 0), (75, 0, 125, 100))
        fitted = fit_image(image, 64, 64)
        assert fitted.size == (64, 64)
        assert fitted.getpixel((32, 32)) == (255, 0, 0) and fitted.getpixel((0, 32)) == (0, 0, 0)

    def test_the_keyframe_fit_is_the_same_function(self):
        image = Image.new("RGB", (200, 100), (10, 20, 30))
        image.paste((255, 0, 0), (75, 0, 125, 100))
        from zvisiongenerator.backends.cuda.video.runtime import fit_image

        assert conditioning.fit_keyframe(image, 64, 64).tobytes() == fit_image(image, 64, 64).tobytes()


class TestSystemMemoryCheck:
    """``load_model`` refuses a load that needs more system memory than the machine has, and warns above 80%."""

    @pytest.fixture
    def machine(self, monkeypatch):
        import zvisiongenerator.backends as backends

        def _set(total_gib: float, *, gpu_gib: float = 10, system_gib: float | None = None) -> None:
            monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=int(total_gib * 1024**3)))
            budget = MemoryBudget(int(gpu_gib * 1024**3), system_bytes=int((system_gib or total_gib) * 1024**3))
            monkeypatch.setattr(backends, "accelerator_memory_budget_or_none", lambda: budget)

        return _set

    def test_a_machine_with_less_than_the_base_is_refused(self, machine):
        machine(16)

        with pytest.raises(RuntimeError, match=r"needs about 18 GB of system memory; this machine has 16 GB"):
            h3_backend._check_system_memory(())

    def test_a_machine_above_80_percent_is_warned(self, machine):
        machine(20)

        with pytest.warns(UserWarning, match="leaving little for other apps") as seen:
            h3_backend._check_system_memory(())

        assert seen[0].filename == __file__

    def test_a_roomy_machine_loads_silently(self, machine, recwarn):
        machine(32)

        h3_backend._check_system_memory(())

        assert len(recwarn) == 0

    def test_a_lora_counts_toward_the_estimate(self, machine, tmp_path):
        lora = tmp_path / "big.safetensors"
        with lora.open("wb") as handle:
            handle.truncate(2 * 1024**3)
        machine(18)

        with pytest.raises(RuntimeError, match="system memory"):
            h3_backend._check_system_memory(((lora, 1.0),))
