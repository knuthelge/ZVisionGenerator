"""Tests for the LTX-2.5 load options of the diffusers video backend: pinned revisions, absent components, the ancestral sampler and the image CRF."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from tests._cuda_helpers import (
    install_fake_weights,
    _DISTILLED_SIGMAS,
    _FakeScheduler,
    _FakeTorch,
    _PIN,
    _STAGE2_SIGMAS,
    _UPSAMPLER_SOURCE,
    _make_runtime,
    _patch_runtime,
    _patch_upsampler_io,
    _text_to_video_args,
    real_download_hub_file,
)

import zvisiongenerator.backends.cuda.video.backend as video_diffusers_module
import zvisiongenerator.backends.cuda.video.sampler as _sampler_module
from zvisiongenerator.backends.cuda.video.backend import (
    DiffusersVideoBackend,
    _absent_optional_components,
    _pass_scheduler,
    _repo_dir,
    _resolve_upsampler_weights,
    _stage1_sigmas,
)


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Fake the weights, placement and memory checks of ``load_model`` (see ``_cuda_helpers.install_fake_weights``)."""
    return install_fake_weights(monkeypatch)


# --- LTX-2.5: revisions, absent optional components, the ancestral sampler and the image CRF ---

_LTX25_REPO = "unsloth/LTX-2.5-FP8@" + _PIN
_LTX23_INDEX = {
    "_class_name": "LTX2Pipeline",
    "connectors": ["ltx2", "LTX2TextConnectors"],
    "text_encoder": ["transformers", "Gemma3ForConditionalGeneration"],
    "transformer": ["ltx2", "LTX2VideoTransformer3DModel"],
}
_LTX25_INDEX = {
    **_LTX23_INDEX,
    "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
    "processor": ["transformers", "Gemma4Processor"],
    "prompt_enhancer": ["ltx2", "LTX2PromptEnhancer"],
    "duration_head": ["ltx2", "LTX2DurationHead"],
    "diffusion_decoder": ["ltx2", "LTX2DiffusionDecoder"],
}
_ABSENT = {"processor": None, "prompt_enhancer": None, "duration_head": None}


def _write_index(folder: Path, index: dict[str, Any] | str) -> Path:
    (folder / "model_index.json").write_text(index if isinstance(index, str) else json.dumps(index), encoding="utf-8")
    return folder


@pytest.fixture
def snapshot_calls(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    import huggingface_hub

    calls: list[tuple[str, dict[str, Any]]] = []

    def _snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    return calls


class TestRepoRevisions:
    def test_a_pinned_repo_downloads_that_revision_without_the_extra_patterns(self, snapshot_calls):
        extra = ("LTX-2.5-*.safetensors", "samples/*")

        assert _repo_dir(_LTX25_REPO, with_transformer=True, extra_ignore=extra) == Path("/cache/snapshot")

        assert snapshot_calls == [("unsloth/LTX-2.5-FP8", {"revision": _PIN, "ignore_patterns": list(extra)})]

    def test_the_extra_patterns_follow_the_separate_transformer_ones(self, snapshot_calls):
        from zvisiongenerator.backends.cuda.download_plan import SEPARATE_TRANSFORMER_IGNORE

        _repo_dir(_LTX25_REPO, with_transformer=False, extra_ignore=("samples/*",))

        assert snapshot_calls[0][1]["ignore_patterns"] == [*SEPARATE_TRANSFORMER_IGNORE, "samples/*"]

    def test_an_unpinned_repo_passes_no_revision_and_keeps_the_2_3_patterns(self, snapshot_calls):
        from zvisiongenerator.backends.cuda.download_plan import SEPARATE_TRANSFORMER_IGNORE

        _repo_dir("owner/ltx", with_transformer=True)
        _repo_dir("owner/ltx", with_transformer=False)

        assert snapshot_calls == [
            ("owner/ltx", {"ignore_patterns": None}),
            ("owner/ltx", {"ignore_patterns": list(SEPARATE_TRANSFORMER_IGNORE)}),
        ]

    def test_a_local_folder_is_used_as_it_is(self, snapshot_calls, tmp_path):
        assert _repo_dir(str(tmp_path), with_transformer=True) == tmp_path
        assert snapshot_calls == []

    def test_a_failed_download_names_the_reference(self, monkeypatch):
        import huggingface_hub

        def _fail(*_args, **_kwargs):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)

        with pytest.raises(RuntimeError, match="offline") as excinfo:
            _repo_dir(_LTX25_REPO, with_transformer=True)
        assert _LTX25_REPO in str(excinfo.value)

    @pytest.mark.parametrize(("repo_id", "expected"), [("owner/up@" + _PIN, {"revision": _PIN}), ("owner/up", {})])
    def test_the_upsampler_download_takes_the_revision_of_its_reference(self, monkeypatch, repo_id, expected):
        import huggingface_hub

        monkeypatch.setattr(video_diffusers_module, "_download_hub_file", real_download_hub_file)
        calls: list[dict[str, str]] = []
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "/cache/up.safetensors")

        assert _resolve_upsampler_weights(repo_id, "up.safetensors") == "/cache/up.safetensors"

        assert calls == [{"repo_id": "owner/up", "filename": "up.safetensors", **expected}]


class TestRepoDirArguments:
    @pytest.fixture
    def repo_dir_calls(self, monkeypatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def _repo_dir_fake(model_path, **kwargs):
            calls.append(kwargs)
            return Path("/fake/repo")

        monkeypatch.setattr(video_diffusers_module, "_repo_dir", _repo_dir_fake)
        return calls

    def test_a_load_without_extra_patterns_passes_no_extra_ignore(self, monkeypatch, repo_dir_calls):
        _patch_runtime(monkeypatch, _make_runtime())

        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", repo_ignore=())

        assert repo_dir_calls == [{"with_transformer": True}, {"with_transformer": True}]

    def test_the_configured_patterns_reach_the_repo_download(self, monkeypatch, repo_dir_calls):
        _patch_runtime(monkeypatch, _make_runtime())

        DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="t2v", repo_ignore=["LTX-2.5-*.safetensors", "samples/*"])

        assert repo_dir_calls == [{"with_transformer": True, "extra_ignore": ("LTX-2.5-*.safetensors", "samples/*")}]


class TestAbsentOptionalComponents:
    def test_an_ltx_2_5_index_leaves_out_the_three_components_the_app_never_loads(self, tmp_path):
        assert _absent_optional_components(_write_index(tmp_path, _LTX25_INDEX)) == _ABSENT

    def test_an_ltx_2_3_index_lists_none(self, tmp_path):
        assert _absent_optional_components(_write_index(tmp_path, _LTX23_INDEX)) == {}

    def test_only_the_components_the_index_lists_are_left_out(self, tmp_path):
        index = {**_LTX23_INDEX, "processor": ["transformers", "Gemma4Processor"]}

        assert _absent_optional_components(_write_index(tmp_path, index)) == {"processor": None}

    @pytest.mark.parametrize("content", [None, "{not json"])
    def test_a_missing_or_unreadable_index_lists_none(self, tmp_path, content):
        if content is not None:
            _write_index(tmp_path, content)

        assert _absent_optional_components(tmp_path) == {}

    def test_the_pipeline_loads_a_2_5_snapshot_with_those_components_set_to_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(video_diffusers_module, "_repo_dir", lambda model_path, **_kwargs: _write_index(tmp_path, _LTX25_INDEX))
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="t2v")

        assert model.text_to_video.load_kwargs == {"transformer": None, "connectors": "connectors", "text_encoder": None, "torch_dtype": "bfloat16", **_ABSENT}

    @pytest.mark.parametrize("index", [_LTX23_INDEX, None])
    def test_a_2_3_load_passes_exactly_the_kwargs_it_always_did(self, monkeypatch, tmp_path, index):
        if index is not None:
            _write_index(tmp_path, index)
        monkeypatch.setattr(video_diffusers_module, "_repo_dir", lambda model_path, **_kwargs: tmp_path)
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

        assert model.text_to_video.load_kwargs == {"transformer": None, "connectors": "connectors", "text_encoder": None, "torch_dtype": "bfloat16"}


class TestLoadOptions:
    def test_an_unknown_sampler_is_refused_before_anything_loads(self, monkeypatch):
        monkeypatch.setattr(video_diffusers_module, "_load_runtime_dependencies", lambda: pytest.fail("the runtime loaded"))

        with pytest.raises(ValueError, match="sampler"):
            DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", sampler="bogus")

    def test_the_sampler_and_the_image_crf_are_kept_on_the_loaded_model(self, monkeypatch):
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", sampler="euler_ancestral", image_crf=18)
        plain, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v")

        assert (model.sampler, model.image_crf) == ("euler_ancestral", 18)
        assert (plain.sampler, plain.image_crf) == (None, None)


class _AncestralFactory:
    """Stands in for ``sampler.ancestral_scheduler``: records each call and returns a new scheduler object."""

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.made: list[SimpleNamespace] = []

    def __call__(self, generator, *, num_train_timesteps=1000):
        self.calls.append(SimpleNamespace(generator=generator, num_train_timesteps=num_train_timesteps))
        self.made.append(SimpleNamespace(name=f"ancestral-{len(self.made) + 1}"))
        return self.made[-1]


@pytest.fixture
def ancestral(monkeypatch) -> _AncestralFactory:
    factory = _AncestralFactory()
    monkeypatch.setattr(_sampler_module, "ancestral_scheduler", factory)
    return factory


def _stock_scheduler() -> _FakeScheduler:
    return _FakeScheduler(config=SimpleNamespace(num_train_timesteps=1000))


class TestAncestralSampler:
    @staticmethod
    def _load(monkeypatch, *, mode="t2v", sampler="euler_ancestral", upscale=False, repo="diffusers/LTX-2.3-Distilled-Diffusers"):
        _patch_runtime(monkeypatch, _make_runtime())
        options: dict[str, Any] = {"quantize": 4, "mode": mode, "low_memory": False, "sampler": sampler}
        if upscale:
            _patch_upsampler_io(monkeypatch)
            options.update(upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        model, _ = DiffusersVideoBackend().load_model(repo, **options)
        pipeline = model.text_to_video or model.image_to_video
        pipeline.scheduler = _stock_scheduler()
        return model, pipeline

    @staticmethod
    def _image(tmp_path) -> str:
        path = tmp_path / "input.png"
        Image.new("RGB", (8, 8), "blue").save(path)
        return str(path)

    def test_a_one_stage_t2v_pass_runs_on_a_fresh_scheduler_seeded_for_stage_1(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        stock = pipeline.scheduler

        DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path))

        (factory_call,) = ancestral.calls
        assert factory_call.generator.seed == 77 + 10_000
        assert factory_call.num_train_timesteps == 1000
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock

    def test_the_initial_noise_keeps_the_user_seed_and_the_sigmas_are_the_2_3_ones(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        events: list[dict[str, Any]] = []

        DiffusersVideoBackend().text_to_video(model=model, step_callback=events.append, **_text_to_video_args(tmp_path))

        (call,) = pipeline.calls
        assert call["generator"].seed == 77
        assert call["generator"] is not ancestral.calls[0].generator
        assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 8)
        assert call["num_inference_steps"] == 8
        assert [event["total_steps"] for event in events] == [8] * 8

    def test_a_one_stage_i2v_pass_runs_on_a_fresh_scheduler_seeded_for_stage_1(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, mode="i2v")
        stock = pipeline.scheduler

        DiffusersVideoBackend().image_to_video(model=model, image_path=self._image(tmp_path), **_text_to_video_args(tmp_path))

        assert [call.generator.seed for call in ancestral.calls] == [77 + 10_000]
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock

    def test_each_stage_of_a_two_stage_run_gets_its_own_scheduler_and_generator(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, upscale=True)
        stock = pipeline.scheduler
        monkeypatch.setattr(video_diffusers_module, "_refine_scheduler", lambda _pipeline: pytest.fail("the static-shift scheduler was used"))
        events: list[dict[str, Any]] = []

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, step_callback=events.append, **_text_to_video_args(tmp_path))

        stage1, stage2 = pipeline.calls
        assert [call.generator.seed for call in ancestral.calls] == [77 + 10_000, 77 + 20_000]
        assert ancestral.calls[0].generator is not ancestral.calls[1].generator
        assert ancestral.made[0] is not ancestral.made[1]
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock
        assert stage1["generator"] is stage2["generator"] and stage1["generator"].seed == 77
        assert stage1["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4) and stage2["sigmas"] == list(_STAGE2_SIGMAS)
        assert (stage1["num_inference_steps"], stage2["num_inference_steps"]) == (4, 3)
        assert {event["total_steps"] for event in events} == {7}

    def test_the_scheduler_scale_comes_from_the_pipelines_own_scheduler(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        pipeline.scheduler = _FakeScheduler(config=SimpleNamespace(num_train_timesteps=500))

        DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path))

        assert ancestral.calls[0].num_train_timesteps == 500

    def test_a_run_without_a_sampler_never_builds_the_ancestral_scheduler(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, sampler=None, upscale=True)
        stock = pipeline.scheduler

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

        assert ancestral.calls == []
        assert pipeline.scheduler is stock

    def test_a_conditioned_pass_keeps_the_stock_scheduler_even_for_the_refinement(self):
        model = SimpleNamespace(sampler="euler_ancestral", runtime=SimpleNamespace(torch=_FakeTorch()))
        pipeline = SimpleNamespace(scheduler=_stock_scheduler())
        stock = pipeline.scheduler

        with _pass_scheduler(model, pipeline, seed=1, stage=1, refine=False, conditioned=True):
            assert pipeline.scheduler is stock
        with _pass_scheduler(model, pipeline, seed=1, stage=2, refine=True, conditioned=True):
            assert pipeline.scheduler is not stock
            assert pipeline.scheduler.overrides == {"use_dynamic_shifting": False, "shift_terminal": None}
        assert pipeline.scheduler is stock


class TestImageCrf:
    @pytest.fixture
    def order(self, monkeypatch) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        real_fit = video_diffusers_module._fit_image

        def _recompress(image, crf):
            events.append(("recompress", (image.mode, image.size, crf)))
            return Image.new("RGB", image.size, "red")

        def _fit(image, width, height):
            events.append(("fit", (image.getpixel((0, 0)), width, height)))
            return real_fit(image, width, height)

        monkeypatch.setattr(video_diffusers_module, "recompress", _recompress)
        monkeypatch.setattr(video_diffusers_module, "_fit_image", _fit)
        return events

    @staticmethod
    def _run(monkeypatch, tmp_path, **options):
        _patch_runtime(monkeypatch, _make_runtime())
        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", low_memory=False, **options)
        input_path = tmp_path / "input.png"
        Image.new("RGBA", (8, 6), "blue").save(input_path)
        DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), **_text_to_video_args(tmp_path))
        return model.image_to_video.calls

    def test_the_image_is_recompressed_then_fitted_and_the_pipeline_skips_its_own_pass(self, monkeypatch, tmp_path, order):
        (call,) = self._run(monkeypatch, tmp_path, image_crf=18)

        assert order == [("recompress", ("RGB", (8, 6), 18)), ("fit", ((255, 0, 0), 512, 384))]
        assert call["image_crf"] == 0
        assert call["image"].size == (512, 384)

    def test_a_two_stage_run_passes_the_skip_to_both_stages(self, monkeypatch, tmp_path, order):
        _patch_runtime(monkeypatch, _make_runtime())
        _patch_upsampler_io(monkeypatch)
        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", low_memory=False, image_crf=18, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        input_path = tmp_path / "input.png"
        Image.new("RGB", (8, 6), "blue").save(input_path)

        DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), stage1_steps=4, **_text_to_video_args(tmp_path))

        assert [call["image_crf"] for call in model.image_to_video.calls] == [0, 0]
        assert [name for name, _ in order] == ["recompress", "fit"]

    def test_without_a_crf_nothing_is_recompressed_and_the_pipeline_gets_no_crf_kwarg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(video_diffusers_module, "recompress", lambda *_args: pytest.fail("the image was recompressed"))

        (call,) = self._run(monkeypatch, tmp_path)

        assert "image_crf" not in call
        assert call["image"].size == (512, 384)
