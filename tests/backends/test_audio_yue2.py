"""Tests for the YuE2 audio backend and its weight swap, with a fake kit pipeline and a tiny model; nothing real is loaded.

The swap's "GPU" is torch's ``meta`` device, so a tensor's device tells which weights are resident without any CUDA.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import zvisiongenerator.backends.audio.yue2 as yue2_module
from tests._audio_presets import audio_config
from tests._factories import load_packaged_config
from tests._torch_flags import current_flags, set_app_flags, set_kit_flags
from zvisiongenerator.backends.audio.yue2 import SAMPLE_RATE, YuE2Backend
from zvisiongenerator.backends.audio.yue2_swap import WeightSwap
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets

GPU = "gpu"
"""The device name the tests treat as the GPU; the fake tensors below carry it as a label."""


# ---- A tiny stand-in for YuE2ForCausalLM --------------------------------------------------------------------------
#
# Real torch will not assign a tensor of another device type to a parameter's ``.data`` without a GPU, so the model is a
# set of plain objects with the few members the swap touches. A tensor is just a device label and a name.


class _Tensor:
    pins: list[str] = []

    def __init__(self, device: str, name: str) -> None:
        self._device, self.name = device, name

    @property
    def device(self) -> SimpleNamespace:
        return SimpleNamespace(type=self._device)

    def cpu(self) -> _Tensor:
        return _Tensor("cpu", self.name)

    def pin_memory(self) -> _Tensor:
        self.pins.append(self.name)
        return _Tensor("cpu", self.name)

    def to(self, device: str, non_blocking: bool = False) -> _Tensor:
        return _Tensor(device, self.name)


class _Slot:
    """A parameter or a buffer: ``.data`` is replaced as the real swap replaces it."""

    def __init__(self, name: str) -> None:
        self.data = _Tensor("cpu", name)

    @property
    def device(self) -> SimpleNamespace:
        return self.data.device


class _Module:
    def __init__(self, name: str) -> None:
        self.slots = [_Slot(f"{name}.weight"), _Slot(f"{name}.bias")]

    def parameters(self):
        return iter(self.slots)


class _TinyModel:
    """``model.model.embed_tokens``, ``lm_head``, per-layer AR and NAR modules, norms, an embedder and a buffer."""

    def __init__(self) -> None:
        layers = [
            SimpleNamespace(
                input_layernorm=_Module(f"l{i}.ln"),
                self_attn=_Module(f"l{i}.self_attn"),
                mlp=_Module(f"l{i}.mlp"),
                nar_input_layernorm=_Module(f"l{i}.nar_ln"),
                nar_self_attn=_Module(f"l{i}.nar_self_attn"),
                nar_mlp=_Module(f"l{i}.nar_mlp"),
            )
            for i in range(2)
        ]
        self.model = SimpleNamespace(embed_tokens=_Module("embed"), layers=layers, norm=_Module("norm"))
        self.lm_head = _Module("head")
        self.vae2llm = _Module("vae2llm")
        self.inv_freq = _Slot("inv_freq")
        self.moved_to: list[str] = []

    def _modules(self) -> list[_Module]:
        layers = self.model.layers
        resident = [self.model.norm, self.vae2llm, *(layer.input_layernorm for layer in layers), *(layer.nar_input_layernorm for layer in layers)]
        return [self.model.embed_tokens, self.lm_head, *(m for layer in layers for m in (layer.self_attn, layer.mlp, layer.nar_self_attn, layer.nar_mlp)), *resident]

    def parameters(self):
        return (slot for module in self._modules() for slot in module.slots)

    def buffers(self):
        return iter([self.inv_freq])

    def to(self, device: str) -> _TinyModel:
        self.moved_to.append(device)
        return self


def _on_gpu(module: _Module) -> bool:
    return all(slot.device.type == GPU for slot in module.slots)


def _gpu_groups(model: _TinyModel) -> frozenset[str]:
    """Return the names of the weight groups that are on the GPU now."""
    layers = model.model.layers
    groups = {
        "embed": [model.model.embed_tokens],
        "head": [model.lm_head],
        "ar": [module for layer in layers for module in (layer.self_attn, layer.mlp)],
        "nar": [module for layer in layers for module in (layer.nar_self_attn, layer.nar_mlp)],
    }
    return frozenset(name for name, modules in groups.items() if all(_on_gpu(module) for module in modules))


def _resident_on_gpu(model: _TinyModel) -> bool:
    layers = model.model.layers
    resident = [model.model.norm, model.vae2llm, *(layer.input_layernorm for layer in layers), *(layer.nar_input_layernorm for layer in layers)]
    return all(_on_gpu(module) for module in resident) and model.inv_freq.device.type == GPU


def _pull_resident_to_cpu(model: _TinyModel) -> None:
    """Do what the kit's ``decode()`` does with ``self._model.to("cpu")`` to every tensor that is on the GPU."""
    for slot in [*model.parameters(), *model.buffers()]:
        if slot.device.type == GPU:
            slot.data = slot.data.to("cpu")


@pytest.fixture
def pinned(monkeypatch):
    """Make the CUDA cache calls work without a GPU; return the names of the tensors that were pinned."""
    _Tensor.pins = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return _Tensor.pins


# ---- WeightSwap ---------------------------------------------------------------------------------------------------


class TestWeightSwap:
    def test_the_groups_are_pinned_and_everything_else_moves_to_the_device(self, pinned):
        model = _TinyModel()

        WeightSwap(model, GPU)

        # embed + head + 2 layers x 4 modules, two tensors each; the norms, the embedder and the buffer are not pinned
        assert len(pinned) == 2 * (2 + 8)
        assert not any("ln" in name or "norm" in name or "vae2llm" in name or "inv_freq" in name for name in pinned)
        assert _gpu_groups(model) == frozenset()
        assert _resident_on_gpu(model)

    @pytest.mark.parametrize(
        ("phase", "groups"),
        [("ar", {"embed", "head", "ar"}), ("prefill", {"embed", "ar"}), ("ode", {"nar"}), ("none", set())],
    )
    def test_a_phase_puts_only_its_group_on_the_device(self, pinned, phase, groups):
        model = _TinyModel()
        swap = WeightSwap(model, GPU)

        swap.phase(phase)

        assert _gpu_groups(model) == groups
        assert _resident_on_gpu(model)

    def test_weights_survive_the_trip_to_the_cpu_and_back(self, pinned):
        model = _TinyModel()
        swap = WeightSwap(model, GPU)

        swap.phase("ode")
        swap.phase("ar")
        swap.phase("none")

        slot = model.model.layers[1].nar_mlp.slots[0]
        assert (slot.data.name, slot.device.type) == ("l1.nar_mlp.weight", "cpu")

    def test_restore_resident_brings_back_what_the_kits_decode_pulled_to_the_cpu(self, pinned):
        model = _TinyModel()
        swap = WeightSwap(model, GPU)
        swap.phase("none")
        _pull_resident_to_cpu(model)
        assert not _resident_on_gpu(model)

        swap.restore_resident()

        assert _resident_on_gpu(model)
        assert _gpu_groups(model) == frozenset()


# ---- A fake kit pipeline ------------------------------------------------------------------------------------------


class _FakePipe:
    """Record the stages a take runs, and what was on the GPU at each."""

    def __init__(self, *, truncated: bool = False, semantic_tokens: int = 30) -> None:
        self.model_dir = "/models/yue2"
        self.generation_config = SimpleNamespace(ode_steps=32, context=24576)
        self._model = None
        self.events: list[tuple[str, frozenset[str], bool]] = []
        self.calls: dict[str, dict] = {}
        self.truncated = truncated
        self.semantic_tokens = semantic_tokens
        self.closed = False
        self.cancelled_seen: list[bool] = []

    def _record(self, stage: str) -> None:
        model = self._model
        if isinstance(model, _TinyModel):
            self.events.append((stage, _gpu_groups(model), _resident_on_gpu(model)))
        else:
            self.events.append((stage, frozenset(), False))

    def plan(self, style, lyrics, *, cot, seed, cfg_scale, abc_sampling, cancelled, on_token):
        self._record("plan")
        self.calls["plan"] = {"style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "cfg_scale": cfg_scale, "abc_sampling": abc_sampling, "cancelled": cancelled}
        for token in range(25):
            on_token("abc", token)
        return SimpleNamespace(abc_ids=list(range(25)), prefix=[1, 2, 3], request=SimpleNamespace(seed=seed), truncated=False)

    def generate_semantic(self, plan, *, sampling, cancelled, on_token):
        self._record("semantic")
        self.calls["semantic"] = {"sampling": sampling, "cancelled": cancelled}
        if cancelled():
            raise InterruptedError("Cancelled during semantic")
        for token in range(self.semantic_tokens):
            on_token("semantic", token)
        return SimpleNamespace(plan=plan, tokens=list(range(self.semantic_tokens)), truncated=self.truncated)

    def decode(self, latents):
        self._record("decode")
        if isinstance(self._model, _TinyModel):
            _pull_resident_to_cpu(self._model)
        self.calls["decode"] = {"frames": len(latents)}
        return np.ones((960, 2), dtype=np.float32)

    def close(self):
        self.closed = True
        self._model = None


class _FakeNar:
    """Stand in for ``yue2.nar.CachedNAR``: records the GPU contents at the prefill and at the solve."""

    pipe: _FakePipe
    instances: list[_FakeNar]

    def __init__(self, model, chunk, attention, query_chunk_size):
        self.pipe._record("prefill")
        self.model, self.chunk, self.args = model, chunk, (attention, query_chunk_size)
        self.closed = False
        self.instances.append(self)

    def solve(self, steps, cancelled, on_progress=None):
        self.pipe._record("ode")
        for step in range(steps):
            if cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            if on_progress is not None:
                on_progress(step + 1, steps)
        return torch.zeros((len(self.chunk.noise), 64))

    def close(self):
        self.closed = True
        self.pipe.events.append(("nar-close", frozenset(), False))


@pytest.fixture
def kit(monkeypatch, pinned):
    """Wire the backend to a fake pipeline, a tiny model, a recording memory fraction and the fake NAR."""
    state = SimpleNamespace(pipe=_FakePipe(), model=_TinyModel(), fractions=[], build=[], nars=[], loads=[])
    monkeypatch.setattr(yue2_module, "snapshot_dir", lambda files: f"/models/{files.repo_id}")

    def build(model_dir, vae_dir, device, settings):
        state.build.append((model_dir, vae_dir, device, settings))
        return state.pipe

    monkeypatch.setattr(yue2_module, "_build_pipeline", build)

    def load_model(model_dir, *, pin):
        state.loads.append((model_dir, pin))
        return state.model

    monkeypatch.setattr(yue2_module, "load_model", load_model)
    monkeypatch.setattr(yue2_module, "WeightSwap", lambda model, device: WeightSwap(model, GPU))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, *args: state.fractions.append(fraction))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _FakeNar.pipe, _FakeNar.instances = state.pipe, state.nars
    monkeypatch.setattr("yue2.nar.CachedNAR", _FakeNar)
    return state


def _preset() -> AudioPreset:
    return parse_audio_presets(audio_config())[0]


def _request(**changes) -> AudioGenerationRequest:
    values = {"preset": "yue2", "kind": "music", "prompt": "Indie folk, warm female vocal", "lyrics": "[Verse]\nThe lamp is lit\n\n[Chorus]\nShine on", "seed": 7}
    return AudioGenerationRequest(**{**values, **changes})


def _loaded(platform: str = "cuda") -> YuE2Backend:
    backend = YuE2Backend()
    preset = _preset()
    backend.load(preset, preset.platforms[platform], platform)
    return backend


def _never_cancelled() -> bool:
    return False


# ---- load ---------------------------------------------------------------------------------------------------------


class TestLoad:
    def test_the_model_and_the_vae_folders_come_from_the_platform_entry(self, kit):
        _loaded()

        model_dir, vae_dir, device, settings = kit.build[0]
        assert model_dir == "/models/m-a-p/YuE2-3B"
        assert vae_dir == "/models/m-a-p/YuE2-Vae"
        assert device == "cuda"
        assert (settings["memory_budget_gib"], settings["vae_core_frames"], settings["ode_steps"], settings["weight_swap"]) == (24, 512, 32, True)

    def test_the_kits_pipeline_is_built_with_the_presets_values(self, monkeypatch):
        from_pretrained = MagicMock()
        monkeypatch.setattr("yue2.YuE2Pipeline.from_pretrained", from_pretrained)

        yue2_module._build_pipeline("/m", "/v", "mps", {"memory_budget_gib": 24, "vae_core_frames": 512, "ode_steps": 16})

        args, kwargs = from_pretrained.call_args
        assert args == ("/m",)
        assert {key: kwargs[key] for key in ("vae", "device", "progress", "memory_budget_gib", "vae_core_frames")} == {
            "vae": "/v",
            "device": "mps",
            "progress": False,
            "memory_budget_gib": 24,
            "vae_core_frames": 512,
        }
        assert kwargs["generation_config"].ode_steps == 16

    def test_the_memory_fraction_is_lifted_right_after_construction_on_cuda(self, kit):
        _loaded()

        assert kit.fractions == [1.0]

    def test_the_swap_hands_the_pipeline_its_model_and_its_loader(self, kit):
        _loaded()

        assert kit.pipe._model is kit.model
        assert kit.pipe._load_model(for_nar=True) is kit.model
        assert kit.pipe._load_model() is kit.model
        assert _gpu_groups(kit.model) == frozenset()
        assert _resident_on_gpu(kit.model)

    def test_the_swap_gets_the_model_streamed_in_pinned(self, kit):
        _loaded()

        assert kit.loads == [("/models/yue2", True)]

    def test_no_swap_is_built_with_weight_swap_off_and_the_model_goes_to_the_device(self, kit, monkeypatch):
        monkeypatch.setattr(yue2_module, "WeightSwap", MagicMock(side_effect=AssertionError("no swap expected")))

        backend = _loaded("mps")

        assert kit.loads == [("/models/yue2", False)]
        assert kit.pipe._model is kit.model
        assert kit.model.moved_to == ["mps"]
        assert backend._swap is None
        assert kit.fractions == []

    def test_without_the_swap_the_loader_brings_the_model_back_to_the_device_for_each_take(self, kit):
        _loaded("mps")

        assert kit.pipe._load_model(for_nar=True) is kit.model
        assert kit.model.moved_to == ["mps", "mps"]

    def test_the_kits_pipeline_is_built_with_the_size_check_in_place_of_the_hashing(self, monkeypatch):
        import yue2.pipeline as kit_pipeline

        seen = []
        original = kit_pipeline.model_identity
        monkeypatch.setattr("yue2.YuE2Pipeline.from_pretrained", lambda *args, **kwargs: seen.append(kit_pipeline.model_identity))

        yue2_module._build_pipeline("/m", "/v", "mps", {"memory_budget_gib": 24, "vae_core_frames": 512, "ode_steps": 16})

        assert seen and seen[0] is not original
        assert kit_pipeline.model_identity is original

    def test_a_failed_load_closes_the_pipeline_and_lifts_the_fraction(self, kit, monkeypatch):
        monkeypatch.setattr(yue2_module, "load_model", MagicMock(side_effect=OSError("disk gone")))
        backend = YuE2Backend()

        with pytest.raises(OSError, match="disk gone"):
            backend.load(_preset(), _preset().platforms["cuda"], "cuda")

        assert kit.pipe.closed
        assert kit.fractions[-1] == 1.0

    def test_close_gives_the_global_torch_flags_back(self, kit, monkeypatch, torch_flags_guard):
        set_app_flags()
        before = current_flags()
        build = yue2_module._build_pipeline

        def kit_build(*args):
            set_kit_flags()  # the kit's constructor does this
            return build(*args)

        monkeypatch.setattr(yue2_module, "_build_pipeline", kit_build)
        backend = _loaded()
        assert current_flags() != before

        backend.close()
        assert current_flags() == before
        backend.close()
        assert current_flags() == before

    def test_a_failed_load_gives_the_global_torch_flags_back(self, kit, monkeypatch, torch_flags_guard):
        set_app_flags()
        before = current_flags()

        def failing_build(*args):
            set_kit_flags()
            raise OSError("disk gone")

        monkeypatch.setattr(yue2_module, "_build_pipeline", failing_build)

        with pytest.raises(OSError, match="disk gone"):
            YuE2Backend().load(_preset(), _preset().platforms["cuda"], "cuda")

        assert current_flags() == before


# ---- generate -----------------------------------------------------------------------------------------------------


class TestGenerate:
    def test_a_song_plans_with_full_cot_and_the_lyrics_as_given(self, kit):
        backend = _loaded()

        backend.generate(_request(), MagicMock(), _never_cancelled)

        plan = kit.pipe.calls["plan"]
        assert (plan["style"], plan["lyrics"], plan["cot"], plan["seed"], plan["cfg_scale"]) == ("Indie folk, warm female vocal", "[Verse]\nThe lamp is lit\n\n[Chorus]\nShine on", "full", 7, 1.0)
        assert plan["abc_sampling"] == {"max_tokens": 4096}

    @pytest.mark.parametrize(
        ("lyrics", "expected_cot", "expected_lyrics"),
        [
            ("", "melody", "[Intro]\n\n[Instrumental]\n\n[Solo]\n\n[Instrumental]\n\n[Solo]\n\n[Bridge]\n\n[Instrumental]\n\n[Outro]"),
            ("[Intro] [Instrumental] [Outro]", "melody", "[Intro]\n\n[Instrumental]\n\n[Outro]"),
            ("[Verse]\nThe lamp is lit", "full", "[Verse]\nThe lamp is lit"),
        ],
        ids=["empty lyrics", "tags only", "a song"],
    )
    def test_the_shipped_default_reaches_the_pipeline_with_the_style_unchanged(self, kit, lyrics, expected_cot, expected_lyrics):
        """The packaged config (not the test fixture) decides the default section list; the style gets no 'no vocals' text added."""
        preset = next(item for item in parse_audio_presets(load_packaged_config()) if item.id == "yue2")
        backend = YuE2Backend()
        backend.load(preset, preset.platforms["cuda"], "cuda")

        backend.generate(_request(prompt="calm piano, slow", lyrics=lyrics), MagicMock(), _never_cancelled)

        plan = kit.pipe.calls["plan"]
        assert (plan["style"], plan["cot"], plan["lyrics"]) == ("calm piano, slow", expected_cot, expected_lyrics)

    def test_the_song_token_cap_is_25_per_second_of_the_limit(self, kit):
        backend = _loaded()

        backend.generate(_request(), MagicMock(), _never_cancelled)

        assert kit.pipe.calls["semantic"]["sampling"] == {"max_tokens": 9000}

    def test_the_clip_is_stereo_48_khz_float32(self, kit):
        backend = _loaded()

        clip = backend.generate(_request(), MagicMock(), _never_cancelled)

        assert clip.sample_rate == SAMPLE_RATE == 48000
        assert clip.samples.shape == (2, 960)
        assert clip.samples.dtype == np.float32
        assert clip.cut is False
        assert clip.samples.min() == 1.0

    def test_the_swap_runs_ar_then_prefill_then_ode_then_none_with_only_the_active_group_on_the_gpu(self, kit):
        backend = _loaded()

        backend.generate(_request(), MagicMock(), _never_cancelled)

        stages = [(stage, groups) for stage, groups, _ in kit.pipe.events if stage != "nar-close"]
        assert stages == [
            ("plan", {"embed", "head", "ar"}),
            ("semantic", {"embed", "head", "ar"}),
            ("prefill", {"embed", "ar"}),
            ("ode", {"nar"}),
            ("decode", set()),
        ]
        assert [stage for stage, _, _ in kit.pipe.events].index("nar-close") < [stage for stage, _, _ in kit.pipe.events].index("decode")

    def test_the_nar_is_built_with_the_kits_chunk_sdpa_and_the_pipelines_steps(self, kit):
        backend = _loaded()

        backend.generate(_request(), MagicMock(), _never_cancelled)

        (nar,) = kit.nars
        assert nar.args == ("sdpa", None)
        assert nar.model is kit.model
        assert nar.chunk.ar_tokens[:3] == [1, 2, 3]
        assert nar.chunk.ar_tokens[-1] == 151852  # MUSIC_END
        assert nar.chunk.noise.shape == (30, 64)
        assert nar.closed

    def test_the_noise_is_the_kits_seeded_cpu_draw(self, kit):
        backend = _loaded()
        generator = torch.Generator(device="cpu").manual_seed(7)
        expected = torch.randn((30, 64), dtype=torch.float32, device="cpu", generator=generator)

        backend.generate(_request(seed=7), MagicMock(), _never_cancelled)

        assert torch.equal(kit.nars[0].chunk.noise, expected)

    def test_without_the_swap_the_same_loop_runs_the_acoustic_part_on_the_pipelines_model(self, kit):
        backend = _loaded("mps")

        backend.generate(_request(), MagicMock(), _never_cancelled)

        assert [stage for stage, _, _ in kit.pipe.events if stage != "nar-close"] == ["plan", "semantic", "prefill", "ode", "decode"]
        assert kit.model.moved_to == ["mps", "mps"], "the model was placed at load and asked for again by the acoustic loop"
        (nar,) = kit.nars
        assert nar.model is kit.model
        assert nar.closed

    @pytest.mark.parametrize("platform", ["cuda", "mps"])
    def test_synthesis_progress_counts_every_ode_step_on_both_platforms(self, kit, platform):
        backend = _loaded(platform)
        on_progress = MagicMock()

        backend.generate(_request(), on_progress, _never_cancelled)

        synthesis = [call.args[1:3] for call in on_progress.call_args_list if call.args[0] == "Synthesis"]
        assert synthesis == [(step, 32) for step in range(33)]

    def test_two_takes_in_one_batch_the_second_starts_with_the_resident_tensors_back_on_the_gpu(self, kit):
        backend = _loaded()

        first = backend.generate(_request(seed=1), MagicMock(), _never_cancelled)
        assert not _resident_on_gpu(kit.model), "the kit's decode moved the model to the CPU"
        kit.pipe.events.clear()
        kit.nars.clear()
        second = backend.generate(_request(seed=2), MagicMock(), _never_cancelled)

        assert first.samples.shape == second.samples.shape
        assert kit.pipe.events[0][0] == "plan"
        assert all(resident for stage, _, resident in kit.pipe.events if stage != "nar-close"), "every stage of take two had the resident tensors on the GPU"
        assert kit.pipe.events[0][1] == {"embed", "head", "ar"}

    def test_a_take_that_hit_the_token_cap_is_faded_out_and_marked_cut(self, kit):
        kit.pipe.truncated = True
        backend = _loaded()

        clip = backend.generate(_request(), MagicMock(), _never_cancelled)

        assert clip.cut is True
        assert clip.samples.shape == (2, 960)
        assert clip.samples[:, 0].tolist() == [1.0, 1.0]
        assert clip.samples[:, -1].tolist() == [0.0, 0.0]
        fade_frames = 2 * SAMPLE_RATE
        assert fade_frames > 960, "the preset's 2 s fade is longer than this short test clip, so the whole clip fades"

    def test_progress_walks_plan_song_tokens_synthesis_decode(self, kit):
        backend = _loaded()
        on_progress = MagicMock()

        backend.generate(_request(), on_progress, _never_cancelled)

        calls = [call.args for call in on_progress.call_args_list]
        assert all(call[3:] == (1, 1) for call in calls)
        phases = [call[0] for call in calls]
        assert [phase for index, phase in enumerate(phases) if index == 0 or phases[index - 1] != phase] == ["Plan", "Song tokens", "Synthesis", "Decode"]
        assert calls[0] == ("Plan", 0, None, 1, 1)
        assert ("Plan", 25, None, 1, 1) in calls
        assert ("Song tokens", 30, None, 1, 1) in calls
        assert ("Synthesis", 32, 32, 1, 1) in calls
        assert calls[-1] == ("Decode", 1, 1, 1, 1)


# ---- stop ---------------------------------------------------------------------------------------------------------


class TestCancel:
    def test_the_runners_check_reaches_every_kit_stage(self, kit):
        calls = []
        backend = _loaded()

        backend.generate(_request(), MagicMock(), lambda: calls.append(1) or False)

        probe = kit.pipe.calls["plan"]["cancelled"]
        assert kit.pipe.calls["semantic"]["cancelled"] is probe
        assert probe() is False
        assert calls, "the probe asks the runner's check"

    def test_a_stop_in_the_song_tokens_raises_audio_cancelled(self, kit):
        backend = _loaded()

        with pytest.raises(AudioCancelled):
            backend.generate(_request(), MagicMock(), lambda: True)

    def test_a_stop_during_the_ode_raises_audio_cancelled_and_closes_the_engine(self, kit):
        backend = _loaded()
        asked = []

        def cancelled() -> bool:
            asked.append(1)
            return len(kit.nars) == 1  # false in the song tokens, true once the NAR exists

        with pytest.raises(AudioCancelled, match="acoustic flow matching"):
            backend.generate(_request(), MagicMock(), cancelled)

        assert kit.nars[0].closed

    @pytest.mark.parametrize("platform", ["cuda", "mps"])
    def test_a_stop_before_the_acoustic_part_raises_audio_cancelled_without_building_the_engine(self, kit, platform):
        backend = _loaded(platform)
        stops = iter([False, True])  # the song tokens ask once; the loop asks next

        with pytest.raises(AudioCancelled):
            backend.generate(_request(), MagicMock(), lambda: next(stops, True))

        assert kit.nars == []

    @pytest.mark.parametrize("platform", ["cuda", "mps"])
    def test_a_stop_after_the_acoustic_part_skips_the_decode(self, kit, platform):
        backend = _loaded(platform)

        def cancelled() -> bool:
            return bool(kit.nars) and kit.nars[0].closed

        with pytest.raises(AudioCancelled):
            backend.generate(_request(), MagicMock(), cancelled)

        assert "decode" not in kit.pipe.calls

    @pytest.mark.parametrize("platform", ["cuda", "mps"])
    def test_an_interrupted_error_the_user_did_not_ask_for_is_not_a_stop(self, kit, monkeypatch, platform):
        monkeypatch.setattr(_FakeNar, "solve", MagicMock(side_effect=InterruptedError("a system call was interrupted")))
        backend = _loaded(platform)

        with pytest.raises(InterruptedError, match="system call"):
            backend.generate(_request(), MagicMock(), _never_cancelled)

        assert kit.nars[0].closed


# ---- close --------------------------------------------------------------------------------------------------------


class TestClose:
    def test_close_closes_the_pipeline_drops_the_pins_and_lifts_the_fraction(self, kit):
        backend = _loaded()
        swap = backend._swap
        kit.fractions.clear()

        backend.close()

        assert kit.pipe.closed
        assert swap._pinned == {}
        assert kit.fractions == [1.0]
        assert backend._pipe is None and backend._swap is None

    def test_the_fraction_is_lifted_after_a_failed_take(self, kit, monkeypatch):
        monkeypatch.setattr(kit.pipe, "generate_semantic", MagicMock(side_effect=RuntimeError("CUDA out of memory")))
        backend = _loaded()
        kit.fractions.clear()

        with pytest.raises(RuntimeError, match="out of memory"):
            backend.generate(_request(), MagicMock(), _never_cancelled)
        backend.close()

        assert kit.fractions == [1.0]
        assert kit.pipe.closed

    def test_close_on_mps_leaves_the_memory_fraction_alone(self, kit):
        backend = _loaded("mps")

        backend.close()

        assert kit.fractions == []

    def test_nothing_keeps_the_pipeline_the_model_or_the_pinned_weights_alive_after_close(self, kit):
        # The pipeline holds the model through _model and the _load_model closure; the backend holds the swap. After close()
        # and after the test lets go of its own handles, a weak reference to each must be dead.
        backend = _loaded()
        backend.generate(_request(), MagicMock(), _never_cancelled)
        refs = {"pipe": weakref.ref(kit.pipe), "model": weakref.ref(kit.model), "swap": weakref.ref(backend._swap)}
        pinned = [weakref.ref(slot.data) for slot in kit.model.parameters()]

        backend.close()
        kit.pipe = kit.model = None
        _FakeNar.pipe = None
        _FakeNar.instances = []
        kit.nars.clear()
        gc.collect()

        assert {name: ref() for name, ref in refs.items()} == dict.fromkeys(refs)
        assert all(ref() is None for ref in pinned)
