"""Tests for the YuE2 audio backend and its weight swap, with a fake kit pipeline and a tiny model; nothing real is loaded.

The swap's "GPU" is torch's ``meta`` device, so a tensor's device tells which weights are resident without any CUDA.
"""

from __future__ import annotations

import dataclasses
import gc
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling, token_prefixes

import zvisiongenerator.backends.audio.yue2 as yue2_module
from tests._audio_presets import audio_config
from tests._factories import load_packaged_config
from tests._paths import FIXTURES
from zvisiongenerator.backends.audio.yue2 import SAMPLE_RATE, YuE2Backend
from zvisiongenerator.backends.audio.yue2_score import score_header, silence_vocal
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


def _score(name: str) -> str:
    return (FIXTURES / "yue2_scores" / f"{name}.abc").read_text(encoding="utf-8")


class _FakeTokenizer:
    """Turn text into one id per character and back, so a score survives the round trip."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token) for token in ids)


class _FakePipe:
    """Record the stages a take runs, and what was on the GPU at each."""

    def __init__(self, *, truncated: bool = False, semantic_tokens: int = 30) -> None:
        self.model_dir = "/models/yue2"
        self.generation_config = GenerationConfig(ode_steps=32)
        self.tokenizer = _FakeTokenizer()
        self._model = None
        self.events: list[tuple[str, frozenset[str], bool]] = []
        self.calls: dict[str, dict] = {}
        self.replans: list[SongRequest] = []
        self.generates: list[dict] = []
        self.truncated = truncated
        self.semantic_tokens = semantic_tokens
        self.closed = False
        self.cancelled_seen: list[bool] = []
        self.abc: str | None = None
        """The score an ordinary plan writes; the song fixture for a song and the instrumental fixture for an instrumental."""
        self.plan_truncated = False
        self.continuation = ""
        """What the model writes after a header given to ``_generate``."""
        self.continuation_truncated = False
        self.probe = ""
        """What the model writes in the short tempo probe, the start of an ordinary plan."""

    def _record(self, stage: str) -> None:
        model = self._model
        if isinstance(model, _TinyModel):
            self.events.append((stage, _gpu_groups(model), _resident_on_gpu(model)))
        else:
            self.events.append((stage, frozenset(), False))

    def plan(self, style=None, lyrics=None, *, cot=None, seed=None, cfg_scale=None, abc_sampling=None, cancelled=None, on_token=None, request=None):
        self._record("plan")
        if request is not None:
            self.replans.append(request)
            return SimpleNamespace(abc=request.abc, abc_ids=list(range(25)), prefix=[1, 2, 3], request=request, truncated=False)
        self.calls["plan"] = {"style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "cfg_scale": cfg_scale, "abc_sampling": abc_sampling, "cancelled": cancelled}
        for token in range(25):
            on_token("abc", token)
        abc = self.abc if self.abc is not None else _score("instrumental_planned" if cot == "melody" else "song_planned")
        return SimpleNamespace(abc=abc, abc_ids=list(range(25)), prefix=[1, 2, 3], request=SimpleNamespace(seed=seed), truncated=self.plan_truncated)

    def _generate(self, prefix, sampling, seed, phase, **kwargs):
        self._load_model()  # the kit's own _generate asks for the model here, which is where the CUDA swap's loader is
        self._record("header-plan")
        self.generates.append({"prefix": prefix, "sampling": sampling, "seed": seed, "phase": phase, **kwargs})
        if kwargs["cancelled"]():
            raise InterruptedError("Cancelled during the plan")
        if sampling.max_tokens == 64:  # the tempo probe: the start of an ordinary plan, not a header's continuation
            return self.tokenizer.encode(self.probe), {}, True
        for token in range(10):
            kwargs["on_token"]("abc", token)
        return self.tokenizer.encode(self.continuation), {}, self.continuation_truncated

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
    state = SimpleNamespace(pipe=_FakePipe(), model=_TinyModel(), fractions=[], build=[], nars=[], loads=[], load_cancelled=None)
    monkeypatch.setattr(yue2_module, "snapshot_dir", lambda files: f"/models/{files.repo_id}")

    def build(model_dir, vae_dir, device, settings):
        state.build.append((model_dir, vae_dir, device, settings))
        return state.pipe

    monkeypatch.setattr(yue2_module, "_build_pipeline", build)

    def load_model(model_dir, *, pin, cancelled=None):
        state.loads.append((model_dir, pin))
        state.load_cancelled = cancelled
        return state.model

    monkeypatch.setattr(yue2_module, "load_model", load_model)
    monkeypatch.setattr(yue2_module, "WeightSwap", lambda model, device: WeightSwap(model, GPU))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, *args: state.fractions.append(fraction))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    _FakeNar.pipe, _FakeNar.instances = state.pipe, state.nars
    monkeypatch.setattr("yue2.nar.CachedNAR", _FakeNar)
    # The MPS speed-ups need real weights; TestMpsSpeedups records them instead.
    monkeypatch.setattr(yue2_module, "folded_decode_attention", nullcontext)
    monkeypatch.setattr(yue2_module, "semantic_head", lambda model: nullcontext())
    return state


def _preset() -> AudioPreset:
    return parse_audio_presets(audio_config())[0]


def _request(**changes) -> AudioGenerationRequest:
    values = {"preset": "yue2", "kind": "music", "prompt": "Indie folk, warm female vocal", "lyrics": "[Verse]\nThe lamp is lit\n\n[Chorus]\nShine on", "seed": 7}
    return AudioGenerationRequest(**{**values, **changes})


def _loaded(platform: str = "cuda") -> YuE2Backend:
    backend = YuE2Backend()
    preset = _preset()
    backend.load(preset, platform)
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
        assert (settings["memory_budget_gib"], settings["vae_core_frames"], settings["ode_steps"]) == (24, 512, 32)

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

    def test_cuda_swaps_the_weights_by_default_and_a_preset_can_turn_it_off(self, kit):
        backend = _loaded()
        assert backend._swap is not None

        preset = _preset()
        off = dataclasses.replace(preset, platforms={"cuda": dataclasses.replace(preset.platforms["cuda"], settings={"weight_swap": False})})
        backend = YuE2Backend()
        backend.load(off, "cuda")

        assert backend._swap is None
        assert kit.loads[-1] == ("/models/yue2", False)

    def test_mps_loads_without_the_swap_by_default(self, kit):
        backend = _loaded("mps")

        assert backend._swap is None
        assert kit.loads == [("/models/yue2", False)]

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

    def test_the_stop_check_reaches_the_streamed_weight_load(self, kit):
        def check() -> bool:
            return False

        YuE2Backend().load(_preset(), "cuda", cancelled=check)

        assert kit.load_cancelled is check

    def test_a_stop_before_the_pipeline_is_built_builds_nothing(self, kit):
        with pytest.raises(AudioCancelled):
            YuE2Backend().load(_preset(), "cuda", cancelled=lambda: True)

        assert kit.build == [] and kit.loads == []

    def test_a_stop_after_the_pipeline_is_built_reads_no_weights_and_closes_it(self, kit, monkeypatch):
        stopped: list[bool] = []
        built = yue2_module._build_pipeline
        monkeypatch.setattr(yue2_module, "_build_pipeline", lambda *args: stopped.append(True) or built(*args))

        with pytest.raises(AudioCancelled):
            YuE2Backend().load(_preset(), "cuda", cancelled=lambda: bool(stopped))

        assert kit.loads == [] and kit.pipe.closed

    def test_a_stop_after_the_weights_are_read_pins_nothing(self, kit, monkeypatch):
        stopped: list[bool] = []
        original = yue2_module.load_model
        monkeypatch.setattr(yue2_module, "load_model", lambda *args, **kwargs: stopped.append(True) or original(*args, **kwargs))
        monkeypatch.setattr(yue2_module, "WeightSwap", MagicMock(side_effect=AssertionError("the weights were pinned after the stop")))

        with pytest.raises(AudioCancelled):
            YuE2Backend().load(_preset(), "cuda", cancelled=lambda: bool(stopped))

        assert kit.pipe.closed

    def test_a_stop_during_the_weight_load_closes_the_pipeline_and_lifts_the_fraction(self, kit, monkeypatch):
        monkeypatch.setattr(yue2_module, "load_model", MagicMock(side_effect=AudioCancelled))
        backend = YuE2Backend()

        with pytest.raises(AudioCancelled):
            backend.load(_preset(), "cuda", cancelled=lambda: False)

        assert kit.pipe.closed
        assert kit.fractions[-1] == 1.0
        assert backend._pipe is None and backend._swap is None

    def test_a_failed_load_closes_the_pipeline_and_lifts_the_fraction(self, kit, monkeypatch):
        monkeypatch.setattr(yue2_module, "load_model", MagicMock(side_effect=OSError("disk gone")))
        backend = YuE2Backend()

        with pytest.raises(OSError, match="disk gone"):
            backend.load(_preset(), "cuda")

        assert kit.pipe.closed
        assert kit.fractions[-1] == 1.0


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
        backend.load(preset, "cuda")

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


# ---- tempo, key and instrumentals ---------------------------------------------------------------------------------


def _written_after(header: str, name: str) -> str:
    """Return what the model wrote after *header* in the real score *name*."""
    score = _score(name)
    assert score.startswith(header)
    return score[len(header) :]


def _song_request(**changes) -> SongRequest:
    values = {"style": "Indie folk, warm female vocal", "lyrics": "[Verse]\nThe lamp is lit\n\n[Chorus]\nShine on", "cot": "full", "seed": 7, "cfg_scale": 1.0}
    return SongRequest(**{**values, **changes})


class TestScoreSteering:
    def test_a_take_without_tempo_key_or_instrumental_plans_as_before(self, kit):
        clip = _loaded().generate(_request(), MagicMock(), _never_cancelled)

        assert kit.pipe.generates == [] and kit.pipe.replans == []
        assert clip.notes == ()

    def test_a_tempo_is_written_into_the_header_the_model_continues(self, kit):
        header = score_header(90)
        kit.pipe.continuation = "K:Am\n" + _written_after(score_header(90, "Am"), "song_header_am")  # the model writes the key itself

        clip = _loaded().generate(_request(tempo=90), MagicMock(), _never_cancelled)

        _, generate = kit.pipe.generates
        tokenizer = kit.pipe.tokenizer
        assert generate["prefix"] == token_prefixes(_song_request(), tokenizer) + tokenizer.encode(header)
        assert "plan" not in kit.pipe.calls, "the header replaces the ordinary plan"
        (replan,) = kit.pipe.replans
        assert replan.abc == _score("song_header_am")
        assert clip.notes == ()

    def test_a_tempo_and_key_are_both_in_the_header(self, kit):
        header = score_header(90, "G")
        kit.pipe.continuation = _written_after(header, "song_header_g")

        _loaded().generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        _, generate = kit.pipe.generates
        assert generate["prefix"][-len(header) :] == kit.pipe.tokenizer.encode(header)
        assert kit.pipe.replans[0].abc == _score("song_header_g")

    def test_the_header_plan_uses_the_plans_sampling_seed_phase_and_stop(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        _loaded().generate(_request(tempo=90, key="G", seed=11), MagicMock(), _never_cancelled)

        _, generate = kit.pipe.generates
        assert generate["sampling"] == resolve_sampling({"max_tokens": 4096}, kit.pipe.generation_config.abc)
        assert (generate["seed"], generate["phase"]) == (11, "abc")
        assert callable(generate["cancelled"]) and generate["cancelled"]() is False

    def test_a_tempo_or_key_is_planned_in_the_meter_and_note_length_the_model_opens_with(self, kit):
        kit.pipe.probe = "X:1\nT:\nM:3/4\nL:1/16\nQ:1/4=74\n"
        header = score_header(76, "D", "3/4", "1/16")
        kit.pipe.continuation = "% verse\nV: Vocal\nz32|\nV: Ins\nA32|\n"

        clip = _loaded().generate(_request(tempo=76, key="D", seed=11), MagicMock(), _never_cancelled)

        probe, generate = kit.pipe.generates
        assert probe["sampling"].max_tokens == 64
        assert generate["prefix"][-len(header) :] == kit.pipe.tokenizer.encode(header)
        assert kit.pipe.replans[0].abc == header + kit.pipe.continuation
        assert "M:3/4\nL:1/16\nQ:1/4=76\n" in kit.pipe.replans[0].abc and clip.notes == ()

    def test_a_probe_with_no_meter_line_plans_in_4_4_and_1_32(self, kit):
        kit.pipe.probe = "X:1\nT:\n"
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        _loaded().generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        assert kit.pipe.replans[0].abc.startswith("X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\n")

    def test_the_users_tempo_wins_over_the_probes_and_the_probe_tempo_fills_in_for_auto(self, kit):
        kit.pipe.probe = "X:1\nT:\nM:6/8\nL:1/32\nQ:1/4=74\n"
        kit.pipe.continuation = "% verse\nV: Vocal\nz32|\nV: Ins\nA32|\n"

        _loaded().generate(_request(tempo=76, key="G"), MagicMock(), _never_cancelled)
        _loaded().generate(_request(key="G"), MagicMock(), _never_cancelled)

        user, auto = kit.pipe.replans
        assert user.abc.startswith("X:1\nT:\nM:6/8\nL:1/32\nQ:1/4=76\n")
        assert auto.abc.startswith("X:1\nT:\nM:6/8\nL:1/32\nQ:1/4=74\n")

    def test_the_header_plan_reports_plan_progress(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")
        on_progress = MagicMock()

        _loaded().generate(_request(tempo=90, key="G"), on_progress, _never_cancelled)

        assert ("Plan", 10, None, 1, 1) in [call.args for call in on_progress.call_args_list]

    def test_the_header_song_always_goes_back_to_the_kit_as_the_score_to_use(self, kit):
        """Even a song that needs no edit: the kit then plans from the written score, not from the model's token ids."""
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        _loaded().generate(_request(tempo=90, key="G", seed=11), MagicMock(), _never_cancelled)

        (replan,) = kit.pipe.replans
        assert (replan.style, replan.lyrics, replan.cot, replan.seed, replan.cfg_scale) == ("Indie folk, warm female vocal", "[Verse]\nThe lamp is lit\n\n[Chorus]\nShine on", "full", 11, 1.0)
        assert replan.abc == _score("song_header_g")

    def test_the_header_plan_runs_with_the_ar_weights_on_the_gpu(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        _loaded().generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        (_, groups, resident) = next(event for event in kit.pipe.events if event[0] == "header-plan")
        assert groups == {"embed", "head", "ar"} and resident

    def test_a_key_with_the_tempo_on_auto_takes_the_tempo_from_a_short_probe_then_plans_from_the_header(self, kit):
        kit.pipe.probe = _score("song_planned")[:300]  # 142 BPM
        header = score_header(142, "Am")
        kit.pipe.continuation = "% verse\nV: Vocal\nz32|\nV: Ins\nA32|\n"

        clip = _loaded().generate(_request(key="Am", seed=11), MagicMock(), _never_cancelled)

        probe, generate = kit.pipe.generates
        tokenizer = kit.pipe.tokenizer
        assert probe["prefix"] == token_prefixes(_song_request(), tokenizer)
        assert probe["sampling"] == resolve_sampling({"max_tokens": 64}, kit.pipe.generation_config.abc)
        assert (probe["seed"], probe["phase"]) == (11, "abc") and callable(probe["cancelled"])
        assert "on_token" not in probe, "the probe's tokens are not the Plan phase's progress"
        assert "plan" not in kit.pipe.calls, "no full plan before the header plan"
        assert generate["prefix"][-len(header) :] == tokenizer.encode(header)
        assert generate["sampling"].max_tokens == 4096
        assert kit.pipe.replans[0].abc == header + kit.pipe.continuation
        assert clip.notes == ()

    def test_the_probe_does_not_count_into_the_plan_progress(self, kit):
        kit.pipe.probe = "X:1\nL:1/32\nQ:1/4=90\n"
        kit.pipe.continuation = _written_after(score_header(90, "Am"), "song_header_am")
        on_progress = MagicMock()

        _loaded().generate(_request(key="Am"), on_progress, _never_cancelled)

        plan_counts = [call.args[1] for call in on_progress.call_args_list if call.args[0] == "Plan"]
        assert max(plan_counts) == 25 and 10 in plan_counts, "the header plan counts its own 10 tokens from zero"

    def test_a_key_with_no_complete_tempo_line_in_the_probe_falls_back_to_an_ordinary_plan_with_a_note(self, kit):
        kit.pipe.probe = "X:1\nL:1/8\nM:4/4\nK:Bb"

        clip = _loaded().generate(_request(key="Am"), MagicMock(), _never_cancelled)

        assert len(kit.pipe.generates) == 1 and kit.pipe.replans == []
        assert kit.pipe.calls["plan"]["cot"] == "full"
        assert clip.notes == ("YuE2 couldn't plan with this tempo and key; the take uses its own.",)

    def test_a_probe_cut_off_in_the_middle_of_the_tempo_is_not_misread(self, kit):
        kit.pipe.probe = "X:1\nL:1/8\nQ:1/4=9"  # the model went on to write 90

        clip = _loaded().generate(_request(key="Am"), MagicMock(), _never_cancelled)

        assert len(kit.pipe.generates) == 1 and kit.pipe.replans == []
        assert "plan" in kit.pipe.calls and len(clip.notes) == 1

    def test_a_tempo_line_that_ends_the_probe_with_its_newline_is_read(self, kit):
        kit.pipe.probe = "X:1\nL:1/8\nQ:1/4=90\nK:Am"
        kit.pipe.continuation = _written_after(score_header(90, "Am"), "song_header_am")

        clip = _loaded().generate(_request(key="Am"), MagicMock(), _never_cancelled)

        assert len(kit.pipe.generates) == 2 and clip.notes == ()

    @pytest.mark.parametrize("continuation", ["", "% only a comment\n"], ids=["empty", "no music"])
    def test_a_header_the_model_cannot_continue_falls_back_to_an_ordinary_plan_with_a_note(self, kit, continuation):
        kit.pipe.continuation = continuation

        clip = _loaded().generate(_request(tempo=90, key="Am"), MagicMock(), _never_cancelled)

        assert kit.pipe.calls["plan"]["cot"] == "full" and kit.pipe.replans == []
        assert clip.notes == ("YuE2 couldn't plan with this tempo and key; the take uses its own.",)

    def test_a_header_plan_that_hit_the_token_cap_is_used_as_written_like_the_kits_own_cut_off_plan(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")
        kit.pipe.continuation_truncated = True

        clip = _loaded().generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        (replan,) = kit.pipe.replans
        assert replan.abc == _score("song_header_g")
        assert "plan" not in kit.pipe.calls and clip.notes == ()

    def test_an_instrumental_gets_its_vocal_voice_silenced_and_the_kit_plans_from_that_score(self, kit):
        clip = _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        assert kit.pipe.calls["plan"]["cot"] == "melody"
        (replan,) = kit.pipe.replans
        assert replan.abc == silence_vocal(_score("instrumental_planned")) == _score("instrumental_silenced")
        assert replan.cot == "melody" and replan.seed == 7
        assert clip.notes == ()

    def test_a_song_keeps_its_voice(self, kit):
        _loaded().generate(_request(), MagicMock(), _never_cancelled)

        assert kit.pipe.replans == []

    def test_an_instrumental_that_is_already_silent_is_not_planned_again(self, kit):
        kit.pipe.abc = _score("instrumental_silenced")

        _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        assert kit.pipe.replans == []

    def test_an_instrumental_with_a_header_has_the_header_score_silenced_and_planned_once(self, kit):
        header = score_header(90, "G")
        kit.pipe.continuation = _written_after(header, "song_header_g")

        clip = _loaded().generate(_request(lyrics="", tempo=90, key="G"), MagicMock(), _never_cancelled)

        _, generate = kit.pipe.generates
        assert generate["prefix"][-len(header) :] == kit.pipe.tokenizer.encode(header)
        (replan,) = kit.pipe.replans
        assert replan.abc == silence_vocal(_score("song_header_g")) and replan.cot == "melody"
        assert clip.notes == ()

    def test_an_instrumental_score_that_cannot_be_silenced_is_used_as_planned_with_a_note(self, kit):
        kit.pipe.abc = "not a score"

        clip = _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        assert kit.pipe.replans == []
        assert clip.notes == ("YuE2 couldn't silence the voice in this instrumental; the take may have a voice in it.",)

    def test_an_instrumental_plan_that_was_cut_off_still_has_its_vocal_silenced(self, kit):
        kit.pipe.plan_truncated = True

        clip = _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        (replan,) = kit.pipe.replans
        assert replan.abc == _score("instrumental_silenced")
        assert clip.notes == ()

    def test_an_instrumental_plan_that_was_cut_off_before_it_has_a_vocal_line_is_used_as_planned_with_a_note(self, kit):
        kit.pipe.plan_truncated = True
        kit.pipe.abc = "X:1\nK:Bb\n"

        clip = _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        assert kit.pipe.replans == []
        assert len(clip.notes) == 1 and "may have a voice in it" in clip.notes[0]

    def test_both_notes_come_back_when_neither_step_could_be_done(self, kit):
        kit.pipe.abc = "X:1\nK:Bb\n"

        clip = _loaded().generate(_request(lyrics="", key="Am"), MagicMock(), _never_cancelled)

        assert [("tempo and key" in note, "may have a voice" in note) for note in clip.notes] == [(True, False), (False, True)]

    def test_the_notes_reach_a_cut_take_too(self, kit):
        kit.pipe.truncated = True
        kit.pipe.abc = "not a score"

        clip = _loaded().generate(_request(lyrics=""), MagicMock(), _never_cancelled)

        assert clip.cut and len(clip.notes) == 1

    def test_a_header_take_runs_the_same_stages_after_the_plan(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        clip = _loaded().generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        assert [stage for stage, _, _ in kit.pipe.events if stage not in {"nar-close", "header-plan"}] == ["plan", "semantic", "prefill", "ode", "decode"]
        assert clip.samples.shape == (2, 960)

    def test_a_stop_during_the_header_plan_raises_audio_cancelled(self, kit):
        kit.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        with pytest.raises(AudioCancelled, match="during the plan"):
            _loaded().generate(_request(tempo=90, key="G"), MagicMock(), lambda: True)

        assert "semantic" not in kit.pipe.calls


# ---- stop ---------------------------------------------------------------------------------------------------------


class TestMpsSpeedups:
    @pytest.fixture
    def speedups(self, kit, monkeypatch):
        """Record entering and leaving each speed-up among the pipeline's stages."""

        def recorder(name):
            @contextmanager
            def speedup(*args):
                kit.pipe.events.append((f"{name}+", frozenset(), args == (kit.pipe._model,) if args else False))
                yield
                kit.pipe.events.append((f"{name}-", frozenset(), False))

            return speedup

        monkeypatch.setattr(yue2_module, "folded_decode_attention", recorder("fold"))
        monkeypatch.setattr(yue2_module, "semantic_head", recorder("head"))
        return kit

    def test_on_mps_both_token_phases_fold_and_only_the_song_phase_narrows_the_head(self, speedups):
        _loaded("mps").generate(_request(), MagicMock(), _never_cancelled)

        stages = [stage for stage, _, _ in speedups.pipe.events if stage in {"fold+", "fold-", "head+", "head-", "plan", "semantic", "prefill", "decode"}]
        assert stages[:9] == ["fold+", "plan", "fold-", "fold+", "head+", "semantic", "head-", "fold-", "prefill"]
        head_on_the_pipeline_model = next(on_model for stage, _, on_model in speedups.pipe.events if stage == "head+")
        assert head_on_the_pipeline_model

    def test_on_mps_the_header_plan_runs_inside_the_folded_attention(self, speedups):
        speedups.pipe.continuation = _written_after(score_header(90, "G"), "song_header_g")

        _loaded("mps").generate(_request(tempo=90, key="G"), MagicMock(), _never_cancelled)

        stages = [stage for stage, _, _ in speedups.pipe.events if stage in {"fold+", "fold-", "header-plan", "plan", "semantic"}]
        assert stages[:6] == ["fold+", "header-plan", "header-plan", "plan", "fold-", "fold+"]
        assert "semantic" in stages

    def test_on_mps_the_acoustic_stage_attends_1024_queries_at_a_time(self, speedups):
        _loaded("mps").generate(_request(), MagicMock(), _never_cancelled)

        assert {nar.args for nar in speedups.nars} == {("sdpa", 1024)}

    def test_on_cuda_the_kit_runs_unchanged(self, speedups):
        _loaded().generate(_request(), MagicMock(), _never_cancelled)

        assert not {stage for stage, _, _ in speedups.pipe.events} & {"fold+", "head+"}
        assert {nar.args for nar in speedups.nars} == {("sdpa", None)}


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
