"""Tests for the staged MiniMax H3 image path: the real fork functions and MLX run on tiny arrays, the components are fakes."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Callable

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from PIL import Image  # noqa: E402

from zvisiongenerator.backends.mac import h3_staged  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3Schedule  # noqa: E402

SCHEDULE = H3Schedule(4, 6.0, 3.0)
FRAMES = 22
SIZE = 32
LATENT_CHANNELS = 2
AUDIO_CHANNELS_LATENT = 3
SAMPLE_RATE = 16000
IMAGE_TOKEN = 9
# 22 frames: 7 latent frames, 37 audio latents. A 32 px canvas is a 4 x 4 latent grid, 4 rows of 8 values per frame.
LATENT_FRAMES = 7
AUDIO_LATENTS = 37
ROWS_PER_FRAME = 4


class _Component:
    """A weakref-able stand-in for one loaded model; it records which earlier components were still alive when it loaded."""

    def __init__(self, rig: SimpleNamespace, name: str) -> None:
        rig.events.append(f"load {name}")
        rig.alive_at_load[name] = [other for other, ref in rig.refs if ref() is not None]
        rig.refs.append((name, weakref.ref(self)))
        self.rig = rig


class _VideoVae(_Component):
    def _encode_clip(self, pixels: Any) -> Any:
        return mx.zeros((1, 1, SIZE // 8, SIZE // 8, 2 * LATENT_CHANNELS))

    def decode(self, latents: Any) -> Any:
        assert latents.shape == (1, LATENT_CHANNELS, LATENT_FRAMES, SIZE // 8, SIZE // 8)
        return mx.zeros((1, 3, FRAMES, SIZE, SIZE))


class _AudioVae(_Component):
    def decode(self, latents: Any) -> Any:
        assert latents.shape == (2, AUDIO_CHANNELS_LATENT, AUDIO_LATENTS)
        return mx.zeros((2, 1, 100))


class _Dit(_Component):
    """Returns zero velocities and checks the packed shapes it is handed."""

    def __call__(self, video: Any, audio: Any, embeds: Any, table: Any, indices: Any, tags: Any, positions: Any, video_indices: Any, audio_indices: Any, text_indices: Any, **kwargs: Any) -> Any:
        self.rig.dit_calls.append({"video": video.shape, "audio": audio.shape, "embeds_dtype": embeds.dtype, "kwargs": sorted(kwargs)})
        return mx.zeros_like(video), mx.zeros_like(audio)


class _VisionModel(_Component):
    def __init__(self, rig: SimpleNamespace, config: Any) -> None:
        super().__init__(rig, "vision")
        rig.vision_config = config

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        self.rig.sanitized = dict(weights)
        return weights

    def load_weights(self, items: list, strict: bool) -> None:
        self.rig.loaded_weights = (dict(items), strict)

    def __call__(self, pixels: Any, grid: Any) -> Any:
        self.rig.vision_input = (pixels.dtype, grid.dtype, grid.tolist())
        return mx.full((3, 4), 5, dtype=mx.bfloat16), [mx.full((3, 4), 6, dtype=mx.bfloat16), mx.full((3, 4), 7, dtype=mx.bfloat16)]


class _TextEncoder(_Component):
    image_token_id = IMAGE_TOKEN

    def __init__(self, rig: SimpleNamespace, model_dir: Path, **kwargs: Any) -> None:
        super().__init__(rig, "text encoder")
        rig.encoder_kwargs = kwargs
        self.language = SimpleNamespace(embed_tokens=lambda ids: mx.zeros((1, ids.shape[1], 4), dtype=mx.bfloat16))

    def build_request(self, prompt: str, images: list) -> tuple[Any, np.ndarray, tuple[None, np.ndarray]]:
        self.rig.request = (prompt, self._processor.image_processor)
        ids = [1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2, 3]
        return mx.array(np.array([ids], dtype=np.int32)), np.array([1, 0, 0, 0, 1, 1]), (None, np.array([[1, 2, 6]]))

    def _hidden_states(self, input_ids: Any, position_ids: Any, inputs_embeds: Any, visual_pos_masks: Any, deepstack_visual_embeds: list) -> Any:
        self.rig.hidden_inputs = (np.array(inputs_embeds.astype(mx.float32)), np.array(visual_pos_masks), [np.array(item.astype(mx.float32)) for item in deepstack_visual_embeds])
        return inputs_embeds + 1


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Fake every component loader of the fork and mlx-vlm; the fork's own packing, schedulers and keyframe and decode functions run for real."""
    import minimax_h3_mlx.adaln as adaln
    import minimax_h3_mlx.load as load
    import minimax_h3_mlx.streaming as streaming
    import minimax_h3_mlx.text_encoder as text_encoder
    from mlx_vlm.models.qwen3_vl import language, vision
    from mlx_vlm.models.qwen3_vl.config import VisionConfig

    rig = SimpleNamespace(events=[], refs=[], alive_at_load={}, rng=[], dit_calls=[], dit_loads=[], sanitized=None, tmp=tmp_path)
    fl2va, encoder, transformer = tmp_path / "FL2VA", tmp_path / "te", tmp_path / "int8"
    for folder in (fl2va / "processor", encoder, transformer):
        folder.mkdir(parents=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (encoder / "config.json").write_text(json.dumps({"vision_config": {"depth": 2}}), encoding="utf-8")
    mx.save_safetensors(str(encoder / "model-visual.safetensors"), {"model.visual.patch_embed.proj.weight": mx.zeros((2, 2)), "model.visual.merger.bias": mx.zeros((2,))})
    index = {
        "weight_map": {
            "model.visual.patch_embed.proj.weight": "model-visual.safetensors",
            "model.visual.merger.bias": "model-visual.safetensors",
            "model.language_model.norm.weight": "model-00001.safetensors",
        }
    }
    (encoder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    rig.dirs = SimpleNamespace(fl2va=fl2va, encoder=encoder, transformer=transformer)

    video_config = SimpleNamespace(spatial_compression_ratio=8, latent_channels=LATENT_CHANNELS, latents_mean=(0.0,) * LATENT_CHANNELS, latents_std=(1.0,) * LATENT_CHANNELS)
    audio_config = SimpleNamespace(latent_channels=AUDIO_CHANNELS_LATENT, sampling_rate=SAMPLE_RATE, latents_mean=(0.0,) * AUDIO_CHANNELS_LATENT, latents_std=(1.0,) * AUDIO_CHANNELS_LATENT)
    monkeypatch.setattr(load, "read_video_vae_config", lambda folder: video_config)
    monkeypatch.setattr(load, "read_audio_vae_config", lambda folder: audio_config)
    monkeypatch.setattr(load, "load_video_vae", lambda folder: _VideoVae(rig, "video vae"))
    monkeypatch.setattr(load, "load_audio_vae", lambda folder: _AudioVae(rig, "audio vae"))

    def load_streaming_dit(folder: Path, **kwargs: Any) -> tuple[_Dit, _Component]:
        rig.dit_loads.append(kwargs)
        return _Dit(rig, "dit"), _Component(rig, "block provider")

    monkeypatch.setattr(streaming, "load_streaming_dit", load_streaming_dit)
    monkeypatch.setattr(adaln.ModulationCache, "build_streaming", classmethod(lambda cls, dit, provider, table, dtype=None: _Component(rig, "modulation cache")))
    monkeypatch.setattr(text_encoder, "MiniMaxH3TextEncoder", lambda model_dir, **kwargs: _TextEncoder(rig, model_dir, **kwargs))
    monkeypatch.setattr(vision, "VisionModel", lambda config: _VisionModel(rig, config))
    monkeypatch.setattr(VisionConfig, "from_dict", classmethod(lambda cls, raw: raw))
    monkeypatch.setattr(language.LanguageModel, "get_rope_index", lambda encoder, ids, **kwargs: (mx.zeros((3, 1, ids.shape[1])), None))

    processor = lambda images, return_tensors: {"pixel_values": np.zeros((12, 6), dtype=np.float32), "image_grid_thw": np.array([[1, 2, 6]])}  # noqa: E731
    monkeypatch.setattr(h3_staged, "_image_processor", lambda folder: processor)
    rig.processor = processor

    real_seed, real_normal = mx.random.seed, mx.random.normal
    monkeypatch.setattr(mx.random, "seed", lambda value: (rig.rng.append(("seed", value)), real_seed(value))[1])
    monkeypatch.setattr(mx.random, "normal", lambda *args, **kwargs: (rig.rng.append(("draw",)), real_normal(*args, **kwargs))[1])
    monkeypatch.setattr(h3_staged, "release_memory", lambda: (gc.collect(), rig.events.append("release")))
    return rig


def _generate(
    rig: SimpleNamespace,
    *,
    turbo: Path | None = Path("turbo.safetensors"),
    callback: Callable[[int, int], None] | None = None,
    anchors: tuple[str, ...] = ("first",),
    images: int | None = None,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, int]:
    image = Image.new("RGB", (SIZE, SIZE), (200, 30, 30))
    return h3_staged.generate_conditioned(
        fl2va_dir=rig.dirs.fl2va,
        transformer_dir=rig.dirs.transformer,
        text_encoder_dir=rig.dirs.encoder,
        turbo_lora_path=turbo,
        turbo_lora_alpha=None if turbo is None else 8.0,
        turbo_lora_scale=1.0,
        schedule=SCHEDULE,
        prompt="a fox",
        width=SIZE,
        height=SIZE,
        frames=FRAMES,
        seed=seed,
        images=[image] * (len(anchors) if images is None else images),
        anchors=anchors,
        step_callback=callback,
        verbose=False,
    )


class TestPhases:
    def test_the_phases_run_in_order_and_release_after_each(self, rig):
        _generate(rig)

        assert rig.events == [
            "load video vae",
            "release",
            "load vision",
            "release",
            "load text encoder",
            "release",
            "load dit",
            "load block provider",
            "load modulation cache",
            "release",
            "load video vae",
            "release",
            "load audio vae",
            "release",
        ]

    def test_each_phases_components_are_collected_before_the_next_phase_loads(self, rig):
        _generate(rig)

        assert {name: alive for name, alive in rig.alive_at_load.items() if name != "video vae"} == {
            "vision": [],
            "text encoder": [],
            "dit": [],
            "block provider": ["dit"],  # the same phase
            "modulation cache": ["dit", "block provider"],
            "audio vae": [],
        }
        assert rig.alive_at_load["video vae"] == []
        assert all(ref() is None for _, ref in rig.refs)

    def test_returns_the_decoded_clip_and_the_audio_rate(self, rig):
        video, audio, rate = _generate(rig)

        assert (video.dtype, video.shape) == (np.uint8, (FRAMES, SIZE, SIZE, 3))
        assert (audio.dtype, audio.shape, rate) == (np.float32, (2, 100), SAMPLE_RATE)

    def test_a_failing_step_callback_cancels_and_loads_nothing_after_it(self, rig):
        def cancel(done: int, total: int) -> None:
            raise RuntimeError("cancelled")

        with pytest.raises(RuntimeError, match="cancelled"):
            _generate(rig, callback=cancel)
        assert "load audio vae" not in rig.events and len(rig.dit_calls) == 1

    def test_the_images_and_anchors_must_match(self, rig):
        with pytest.raises(ValueError, match="anchor"):
            _generate(rig, anchors=("first", "last"), images=1)
        assert rig.events == []

    def test_images_are_required(self, rig):
        with pytest.raises(ValueError, match="keyframe"):
            _generate(rig, anchors=(), images=0)


class TestSeeds:
    def test_the_keyframe_draw_runs_under_seed_42_and_every_request_draw_under_the_request_seed(self, rig):
        _generate(rig, seed=7)

        # One keyframe sample; then condition, video and audio noise.
        assert rig.rng == [("seed", 42), ("draw",), ("seed", 7), ("draw",), ("draw",), ("draw",)]

    def test_two_keyframes_draw_before_the_request_seed(self, rig):
        _generate(rig, anchors=("first", "last"), seed=3)

        assert rig.rng == [("seed", 42), ("draw",), ("draw",), ("seed", 3), ("draw",), ("draw",), ("draw",)]


class TestDenoise:
    def test_one_forward_per_step_with_the_conditioning_rows_in_the_video_input(self, rig):
        steps: list[tuple[int, int]] = []
        _generate(rig, callback=lambda done, total: steps.append((done, total)), anchors=("first", "last"))

        assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
        # Two conditioning frames plus the generated frames, 8 values a row; stereo audio rows.
        assert {call["video"] for call in rig.dit_calls} == {(1, (2 + LATENT_FRAMES) * ROWS_PER_FRAME, 8)}
        assert {call["audio"] for call in rig.dit_calls} == {(1, 2 * AUDIO_LATENTS, AUDIO_CHANNELS_LATENT)}
        assert {call["embeds_dtype"] for call in rig.dit_calls} == {mx.bfloat16}
        assert {tuple(call["kwargs"]) for call in rig.dit_calls} == {("block_provider", "modulation_cache")}

    def test_the_turbo_adapter_reaches_the_streaming_loader(self, rig):
        _generate(rig, turbo=Path("turbo.safetensors"))

        assert rig.dit_loads == [{"turbo_lora_path": Path("turbo.safetensors"), "turbo_lora_alpha": 8.0, "turbo_lora_scale": 1.0, "verbose": False}]

    def test_a_converted_folder_loads_the_transformer_with_no_adapter(self, rig):
        _generate(rig, turbo=None)

        assert rig.dit_loads[0]["turbo_lora_path"] is None and rig.dit_loads[0]["turbo_lora_alpha"] is None


class TestVision:
    def test_the_copys_vision_tensors_go_through_sanitize_with_their_prefix_stripped_and_load_strictly(self, rig):
        _generate(rig)

        assert sorted(rig.sanitized) == ["merger.bias", "patch_embed.proj.weight"]
        assert sorted(rig.loaded_weights[0]) == ["merger.bias", "patch_embed.proj.weight"]
        assert rig.loaded_weights[1] is True
        assert rig.vision_config == {"depth": 2}

    def test_the_tower_gets_bf16_pixels_and_int32_grids_from_the_pil_processor(self, rig):
        _generate(rig)

        assert rig.vision_input == (mx.bfloat16, mx.int32, [[1, 2, 6]])


class TestPromptRows:
    def test_the_vision_rows_replace_the_image_token_positions_only(self, rig):
        _generate(rig)

        embeds, mask, deepstack = rig.hidden_inputs
        assert embeds[0, :, 0].tolist() == [0, 5, 5, 5, 0, 0]
        assert mask[0].tolist() == [False, True, True, True, False, False]
        assert [float(item[0, 0]) for item in deepstack] == [6, 7]

    def test_the_encoder_is_built_for_the_resident_text_only_tower_and_uses_the_pil_processor(self, rig):
        _generate(rig)

        assert rig.encoder_kwargs == {"load_vision": False, "stream_layers": False, "tokenizer_dir": rig.dirs.fl2va / "tokenizer", "processor_dir": rig.dirs.fl2va / "processor"}
        assert rig.request == ("a fox", rig.processor)

    def test_a_vision_row_count_that_misses_the_image_positions_is_an_error(self, rig, monkeypatch):
        monkeypatch.setattr(h3_staged, "_vision_rows", lambda *args: (np.zeros((2, 4), dtype=np.float32), []))
        with pytest.raises(ValueError, match="3 image positions"):
            _generate(rig)


class TestConditioningRows:
    """The fork's denoise rules the staged loop mirrors: held anchors, their own timestep, only generated rows stepped."""

    @pytest.fixture
    def seen(self, rig, monkeypatch):
        calls: list[dict[str, Any]] = []

        def forward(
            dit: _Dit, video: Any, audio: Any, embeds: Any, table: Any, indices: Any, tags: Any, positions: Any, video_indices: Any, audio_indices: Any, text_indices: Any, **kwargs: Any
        ) -> Any:
            rows = np.array(table)[np.array(indices)]
            calls.append({"video": np.array(video[0].astype(mx.float32)), "audio": np.array(audio[0].astype(mx.float32)), "row_timesteps": rows, "video_indices": np.array(video_indices)})
            return mx.ones_like(video), mx.ones_like(audio)

        monkeypatch.setattr(_Dit, "__call__", forward)
        return calls

    def test_the_anchor_rows_are_held_unchanged_while_the_generated_rows_are_stepped(self, rig, seen):
        _generate(rig, anchors=("first", "last"))

        held = 2 * ROWS_PER_FRAME
        assert len(seen) == SCHEDULE.forwards
        for call in seen[1:]:
            np.testing.assert_array_equal(call["video"][:held], seen[0]["video"][:held])
        assert all(not np.array_equal(later["video"][held:], earlier["video"][held:]) for earlier, later in zip(seen, seen[1:], strict=False))
        assert all(not np.array_equal(later["audio"], earlier["audio"]) for earlier, later in zip(seen, seen[1:], strict=False))

    def test_the_anchor_rows_sit_at_the_noise_augmentation_timestep_and_the_generated_rows_at_the_schedule(self, rig, seen):
        from minimax_h3_mlx.packing import KEYFRAME_NOISE_AUG
        from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

        _generate(rig, anchors=("first",))

        scheduler = MiniMaxH3Scheduler(shift=SCHEDULE.video_shift)
        scheduler.set_timesteps(SCHEDULE.num_inference_steps)
        for call, timestep in zip(seen, scheduler.timesteps.tolist(), strict=True):
            video_rows = call["row_timesteps"][call["video_indices"]]
            assert set(video_rows[:ROWS_PER_FRAME].tolist()) == {np.float32(max(timestep, KEYFRAME_NOISE_AUG))}
            assert set(video_rows[ROWS_PER_FRAME:].tolist()) == {np.float32(timestep)}
