"""Tests for loading the shared diffusers LTX video backend: weights, placement, prompt encoding, memory checks, upsampler, LoRAs and converted folders."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import zvisiongenerator.backends.cuda.video.backend as video_diffusers_module
from zvisiongenerator.backends.cuda.video.backend import (
    DiffusersVideoBackend,
    _UPSAMPLER_CONFIG,
    _resolve_pipeline_classes,
    _resolve_upsampler_weights,
    _TextEncoderSource,
    _validate_diffusers_version,
)
from _cuda_helpers import (
    install_fake_weights,
    _FakeTextPipeline,
    _FakeTorch,
    _FakeUpsamplerModel,
    _FakeVideoCondition,
    _NoScalePipeline,
    _UPSAMPLER_SOURCE,
    _UPSAMPLER_STATE,
    _converted_folder,
    _load_upscale_model,
    _make_runtime,
    _patch_memory,
    _patch_runtime,
    _patch_upsampler_io,
    _text_to_video_args,
    real_check_system_memory,
    real_download_hub_file,
    real_encode_prompt,
    real_ltx_system_estimate,
    real_place_components,
    real_place_transformer,
    real_prepared_weights,
)


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Fake the weights, placement and memory checks of ``load_model`` (see ``_cuda_helpers.install_fake_weights``)."""
    return install_fake_weights(monkeypatch)


def test_load_model_rejects_missing_cuda(monkeypatch):
    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False)),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")


def test_load_model_builds_the_pipeline_without_a_text_encoder(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    pipeline = model.text_to_video
    assert pipeline.load_kwargs["text_encoder"] is None
    assert pipeline.load_kwargs["connectors"] == "connectors"
    assert pipeline.transformer == "transformer-1"  # attached after the pipeline is built
    assert model.text_encoder == _TextEncoderSource(repo_dir=Path("/fake/repo"), prepared=Path("/fake/cache/text-encoder"), torch_dtype="bfloat16")


@pytest.mark.parametrize("quantize", [4, 8])
def test_load_reads_the_prepared_copies_of_the_configured_checkpoint(monkeypatch, fake_weights, quantize):
    _patch_runtime(monkeypatch, _make_runtime())
    checkpoint = ("example-org/ltx", "ltx-distilled.safetensors")

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=quantize, transformer_source=checkpoint)

    assert fake_weights.prepared == [(Path("/fake/hub/ltx-distilled.safetensors"), quantize)]
    assert fake_weights.transformers == [{"prepared": Path("/fake/cache/transformer"), "bits": quantize}]
    assert model.text_encoder.prepared == Path("/fake/cache/text-encoder")
    assert fake_weights.placements == [("components", True), (quantize, True)]
    assert fake_weights.repo_dirs == [False]  # the repo's own transformer weights are not downloaded


def test_the_picked_name_names_the_prepared_copy(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, model_name="ltx-4", mode="t2v")

    assert fake_weights.prepared_names == ["ltx-4"]


class TestPreparedCopiesLiveWithTheAlias:
    def _prepare(self, monkeypatch, tmp_path, name, bits, writes):
        def _write(kind):
            def _writer(*args):
                folder = args[2] if kind == "transformer" else args[1]
                writes.append((name, kind))
                (folder / "weights.safetensors").write_text(f"{kind}-{name}")

            return _writer

        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.write_transformer", _write("transformer"))
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.write_text_encoder", _write("text_encoder"))
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.transformer_sources", lambda source: [source])
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.text_encoder_sources", lambda repo: [repo / "te.safetensors"])
        monkeypatch.setattr(video_diffusers_module, "_transformer_config", lambda repo_dir: {})
        # The NF4 storage format names the installed bitsandbytes version, which is not installed on macOS.
        monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.50.0")
        return real_prepared_weights(name, "org/repo", tmp_path, tmp_path / "ckpt.safetensors", bits, "cuda")

    def test_copies_are_the_alias_stored_copy_and_share_one_text_encoder(self, monkeypatch, tmp_path):
        (tmp_path / "ckpt.safetensors").write_text("checkpoint")
        (tmp_path / "te.safetensors").write_text("text encoder")
        writes: list[tuple[str, str]] = []

        transformer4, text_encoder4 = self._prepare(monkeypatch, tmp_path, "ltx-4", 4, writes)
        transformer8, text_encoder8 = self._prepare(monkeypatch, tmp_path, "ltx-8", 8, writes)

        assert (transformer4, text_encoder4) == (tmp_path / "models" / "ltx-4@q4" / "transformer", tmp_path / "models" / "ltx-4@q4" / "text_encoder")
        assert transformer8 == tmp_path / "models" / "ltx-8@q8" / "transformer"
        assert writes == [("ltx-4", "transformer"), ("ltx-4", "text_encoder"), ("ltx-8", "transformer")]
        assert (text_encoder8 / "weights.safetensors").stat().st_ino == (text_encoder4 / "weights.safetensors").stat().st_ino


def test_without_a_checkpoint_the_repos_own_transformer_is_prepared(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=8, mode="t2v", low_memory=False)

    assert fake_weights.prepared == [(Path("/fake/repo/transformer"), 8)]
    assert fake_weights.repo_dirs == [True]
    assert fake_weights.placements == [("components", False), (8, False)]


def test_load_model_checks_system_memory_before_loading_weights(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    def _refuse(estimate, quantize, *, converted):
        raise RuntimeError("needs more memory")

    monkeypatch.setattr(video_diffusers_module, "_check_system_memory", _refuse)

    with pytest.raises(RuntimeError, match="needs more memory"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=4)

    assert fake_weights.prepared == []
    assert fake_weights.transformers == []


@pytest.mark.parametrize("quantize", [None, 6])
def test_load_model_needs_q4_or_q8_and_names_the_models(monkeypatch, fake_weights, quantize):
    _patch_runtime(monkeypatch, _make_runtime())

    with pytest.raises(ValueError, match="quantize level") as excinfo:
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=quantize)

    for alias in ("ltx-4", "ltx-8", "ltx25-4"):
        assert alias in str(excinfo.value)
    assert fake_weights.repo_dirs == []  # refused before anything downloads


class TestTakingTurns:
    """When the text encoder and transformer do not fit in system memory together, they never are."""

    def _load(self, monkeypatch, fake_weights, **kwargs):
        fake_weights.fits_together = False
        _patch_runtime(monkeypatch, _make_runtime())
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=4, **kwargs)
        return model

    def test_the_transformer_waits_until_the_first_prompt_is_encoded(self, monkeypatch, fake_weights, tmp_path):
        model = self._load(monkeypatch, fake_weights)
        assert fake_weights.transformers == []

        DiffusersVideoBackend().text_to_video(model, prompt="a river", width=512, height=384, num_frames=9, seed=1, steps=8, output_path=str(tmp_path / "a.mp4"))

        assert fake_weights.events == ["release", "load text encoder", "release", "load transformer", "release"]
        assert model.text_to_video.calls[0]["prompt_embeds"] == "embeds:a river"

    def test_without_low_memory_the_gpu_held_transformer_never_takes_turns(self, monkeypatch, fake_weights):
        model = self._load(monkeypatch, fake_weights, low_memory=False)

        assert model.transformer.swap is False
        assert len(fake_weights.transformers) == 1  # attached at load

    def test_a_new_prompt_releases_the_transformer_before_the_text_encoder_loads(self, monkeypatch, fake_weights, tmp_path):
        model = self._load(monkeypatch, fake_weights)
        backend = DiffusersVideoBackend()
        args = {"width": 512, "height": 384, "num_frames": 9, "seed": 1, "steps": 8}
        backend.text_to_video(model, prompt="a river", output_path=str(tmp_path / "a.mp4"), **args)
        fake_weights.events.clear()

        backend.text_to_video(model, prompt="a river", output_path=str(tmp_path / "b.mp4"), **args)
        assert fake_weights.events == []  # a repeated prompt keeps both its embeddings and the transformer

        backend.text_to_video(model, prompt="a city", output_path=str(tmp_path / "c.mp4"), **args)
        # The transformer goes first, the text encoder comes and goes, then the transformer returns.
        assert fake_weights.events == ["release", "load text encoder", "release", "load transformer", "release"]
        assert model.text_to_video.encoded[-1]["text_encoder"] is not None

    def test_loras_are_unloaded_with_the_transformer_and_applied_again_after(self, monkeypatch, fake_weights, tmp_path):
        (lora,) = _write_loras(tmp_path, "style.safetensors")
        model = self._load(monkeypatch, fake_weights, loras=[(lora, 0.5)])
        pipeline = model.text_to_video
        unloads: list[bool] = []
        pipeline.unload_lora_weights = lambda: unloads.append(True)
        args = {"width": 512, "height": 384, "num_frames": 9, "seed": 1, "steps": 8}

        DiffusersVideoBackend().text_to_video(model, prompt="a river", output_path=str(tmp_path / "a.mp4"), **args)
        DiffusersVideoBackend().text_to_video(model, prompt="a city", output_path=str(tmp_path / "b.mp4"), **args)

        assert unloads == [True]
        assert pipeline.loaded_loras == [(lora, "lora_0"), (lora, "lora_0")]

    def test_a_missing_lora_still_fails_the_load(self, monkeypatch, fake_weights, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._load(monkeypatch, fake_weights, loras=[(str(tmp_path / "missing.safetensors"), 1.0)])


def _placement_pipeline():
    pipeline = MagicMock()
    pipeline.vae = MagicMock(spec=["to", "enable_tiling"])
    return pipeline


def _record_streaming(monkeypatch):
    streamed: list[tuple[str, object, object]] = []
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.group_offload", lambda module, device, kind=None: streamed.append(("group", module, kind)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.stream_blocks", lambda blocks, device: streamed.append(("blocks", blocks, None)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.move_outside", lambda model, blocks, device: streamed.append(("outside", model, None)))
    return streamed


def test_low_memory_streams_the_connectors_and_moves_the_vaes_to_the_gpu_only_when_used(monkeypatch, fake_weights):
    streamed = _record_streaming(monkeypatch)
    pipeline = _placement_pipeline()
    pipeline.vae = MagicMock(spec=["to", "enable_tiling", "encoder", "decoder"])

    real_place_components(pipeline, _FakeTorch(), low_memory=True)

    assert streamed == [("group", pipeline.connectors, "leaf_level")]
    pipeline.vae.to.assert_not_called()
    assert (pipeline.vae, [pipeline.vae.encoder, pipeline.vae.decoder]) in fake_weights.lazy
    assert any(component is pipeline.vocoder for component, _ in fake_weights.lazy)
    pipeline.vae.enable_tiling.assert_called_once()


def test_the_transformer_sends_the_vaes_back_before_it_runs_in_low_memory_mode(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=True)

    assert fake_weights.unload_before == [("transformer-1", [model.text_to_video.vae])]


def test_the_transformer_modulates_per_token_timesteps_by_runs(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v")

    assert fake_weights.run_wise == ["transformer-1"]


@pytest.mark.parametrize(("quantize", "streamed_by_blocks"), [(4, True), (8, False), (None, False)])
def test_low_memory_streams_the_transformer_block_by_block(monkeypatch, quantize, streamed_by_blocks):
    streamed = _record_streaming(monkeypatch)
    transformer = MagicMock()

    real_place_transformer(transformer, _FakeTorch(), quantize=quantize, low_memory=True)

    if streamed_by_blocks:
        assert streamed == [("blocks", transformer.transformer_blocks, None), ("outside", transformer, None)]
    else:
        assert streamed == [("group", transformer, "block_level")]
    transformer.to.assert_not_called()


def test_without_low_memory_everything_moves_to_the_gpu(monkeypatch):
    streamed = _record_streaming(monkeypatch)
    pipeline = _placement_pipeline()
    transformer = MagicMock()

    real_place_components(pipeline, _FakeTorch(), low_memory=False)
    real_place_transformer(transformer, _FakeTorch(), quantize=4, low_memory=False)

    assert streamed == []
    pipeline.connectors.to.assert_called_once_with("device:cuda")
    transformer.to.assert_called_once_with("device:cuda")


def test_the_vae_decodes_in_tiles_in_time_as_well_as_space():
    pipeline = MagicMock(spec=["connectors", "vae"])
    pipeline.vae = MagicMock(spec=["to", "enable_tiling", "use_framewise_decoding"])
    pipeline.vae.use_framewise_decoding = False

    real_place_components(pipeline, _FakeTorch(), low_memory=False)

    pipeline.vae.enable_tiling.assert_called_once()
    assert pipeline.vae.use_framewise_decoding is True


def test_every_block_feed_forward_and_rms_norm_is_chunked(monkeypatch, fake_weights):
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("diffusers.models.normalization")
    chunked: list[tuple[list[object], int]] = []
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.token_chunking.chunk_tokens", lambda modules, size: chunked.append((list(modules), size)))

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = torch.nn.Linear(4, 4)
            self.audio_ff = torch.nn.Linear(4, 4)
            self.norm3 = normalization.RMSNorm(4, eps=1e-6)
            self.attn1 = torch.nn.Module()
            self.attn1.norm_q = normalization.RMSNorm(4, eps=1e-6)
            self.other = torch.nn.LayerNorm(4)

    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([_Block()])
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.load_transformer", lambda config, **_: transformer)
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    block = transformer.transformer_blocks[0]
    ((modules, size),) = chunked
    assert modules == [block.ff, block.audio_ff, block.norm3, block.attn1.norm_q]
    assert size == video_diffusers_module._TOKEN_CHUNK


def _encoding_model(*, low_memory=True, swap=True, prepared=Path("/fake/cache/text-encoder")):
    """A loaded model whose text encoder and transformer take turns (*swap*) or fit together."""
    return SimpleNamespace(
        runtime=_make_runtime(),
        low_memory=low_memory,
        text_encoder=_TextEncoderSource(repo_dir=Path("/fake/repo"), prepared=prepared, torch_dtype="bfloat16"),
        transformer=SimpleNamespace(swap=swap),
        prompt_cache=None,
        resident_text_encoder=None,
    )


def test_a_prompt_is_encoded_with_a_text_encoder_loaded_for_it_and_released_after(fake_weights):
    model = _encoding_model()
    pipeline = _FakeTextPipeline()

    inputs = real_encode_prompt(model, pipeline, "a river")

    assert inputs == {"prompt_embeds": "embeds:a river", "prompt_attention_mask": "mask:a river"}
    assert pipeline.encoded[0]["do_classifier_free_guidance"] is False  # the distilled guidance needs no negative prompt
    (loaded,) = fake_weights.text_encoders
    assert loaded["prepared"] == Path("/fake/cache/text-encoder")
    assert pipeline.encoded[0]["text_encoder"] is loaded["encoder"]  # present while encoding
    assert pipeline.encoded[0]["dtype"] == "bfloat16"  # never the FP8 storage dtype
    assert pipeline.text_encoder is None
    assert fake_weights.released == 1
    assert fake_weights.offloaded == [loaded["encoder"]]


def test_a_repeated_prompt_reuses_its_embeddings_and_a_new_one_loads_the_text_encoder_again(fake_weights):
    model = _encoding_model()
    pipeline = _FakeTextPipeline()

    first = real_encode_prompt(model, pipeline, "a river")
    again = real_encode_prompt(model, pipeline, "a river")
    real_encode_prompt(model, pipeline, "a city")

    assert again is first
    assert len(fake_weights.text_encoders) == 2
    assert [call["prompt"] for call in pipeline.encoded] == ["a river", "a city"]


def test_a_text_encoder_that_fits_beside_the_transformer_stays_loaded_for_the_next_prompt(fake_weights):
    model = _encoding_model(swap=False)
    pipeline = _FakeTextPipeline()

    real_encode_prompt(model, pipeline, "a river")
    real_encode_prompt(model, pipeline, "a city")

    (loaded,) = fake_weights.text_encoders
    assert model.resident_text_encoder is loaded["encoder"]
    assert pipeline.encoded[1]["text_encoder"] is loaded["encoder"]
    assert pipeline.text_encoder is None
    assert fake_weights.released == 0


def test_without_low_memory_the_text_encoder_still_streams_beside_the_transformer(fake_weights):
    real_encode_prompt(_encoding_model(low_memory=False), _FakeTextPipeline(), "a river")

    (loaded,) = fake_weights.text_encoders
    loaded["encoder"].to.assert_not_called()
    assert fake_weights.offloaded == [loaded["encoder"]]


def test_the_text_encoder_is_released_when_encoding_fails(fake_weights):
    pipeline = _FakeTextPipeline()
    pipeline.encode_prompt = MagicMock(side_effect=RuntimeError("out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        real_encode_prompt(_encoding_model(), pipeline, "a river")

    assert pipeline.text_encoder is None
    assert fake_weights.released == 1


def test_without_a_text_encoder_source_the_pipeline_gets_the_prompt():
    model = SimpleNamespace(text_encoder=None, prompt_cache=None)

    assert real_encode_prompt(model, _FakeTextPipeline(), "a river") == {"prompt": "a river"}


def test_two_stage_upscale_encodes_the_prompt_once_for_both_stages(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=True)

    DiffusersVideoBackend().text_to_video(model, **_text_to_video_args(tmp_path), stage1_steps=8)

    pipeline = model.text_to_video
    assert len(pipeline.encoded) == 1
    assert [call["prompt_embeds"] for call in pipeline.calls] == ["embeds:a city", "embeds:a city"]


def _check_stock_memory(quantize):
    return real_check_system_memory(real_ltx_system_estimate(Path("/fake/repo"), Path("/fake/ckpt"), quantize), quantize, converted=False)


def test_a_load_larger_than_system_memory_is_refused_and_names_the_smaller_level(monkeypatch):
    _patch_memory(monkeypatch, total_gib=12)  # 10 G block weights at q8 and the runtime: 16 GB

    with pytest.raises(RuntimeError, match="ltx-4"):
        _check_stock_memory(8)


def test_a_tight_load_warns(monkeypatch):
    _patch_memory(monkeypatch, total_gib=19)  # 10 G block weights at q8 and the runtime: 16 GB, more than 80% of 19

    with pytest.warns(UserWarning, match="ltx-4"):
        _check_stock_memory(8)


@pytest.mark.parametrize(("total_gib", "together"), [(64, True), (24, False)])
def test_the_check_says_whether_text_encoder_and_transformer_fit_together(monkeypatch, total_gib, together):
    # 10 G block weights and a 10 G text encoder at q8: 10 GB each, 22.5 GB together with the runtime.
    _patch_memory(monkeypatch, total_gib=total_gib, text_encoder=10 * 1024**3)

    assert _check_stock_memory(8) is together


def test_resolve_pipeline_classes_requires_ltx2():
    ltx2 = SimpleNamespace(LTX2Pipeline=object, LTX2ImageToVideoPipeline=object, LTX2ConditionPipeline=object, LTX2LatentUpsamplePipeline=object)

    classes = _resolve_pipeline_classes(ltx2, _FakeUpsamplerModel, _FakeVideoCondition)

    assert classes.text_to_video is object
    assert classes.image_to_video is object
    assert classes.condition is object
    assert classes.video_condition is _FakeVideoCondition
    assert classes.latent_upsample_pipeline is object
    assert classes.latent_upsampler_model is _FakeUpsamplerModel
    partial = SimpleNamespace(**{name: object for name in vars(ltx2) if name != "LTX2LatentUpsamplePipeline"})
    with pytest.raises(RuntimeError, match="LTX-2"):
        _resolve_pipeline_classes(partial, _FakeUpsamplerModel, _FakeVideoCondition)


def test_upscale_load_downloads_upsampler_before_the_pipeline(monkeypatch):
    events: list[str] = []

    class _OrderedPipeline(_FakeTextPipeline):
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            events.append("from_pretrained")
            return super().from_pretrained(model_path, **kwargs)

    runtime = _make_runtime(text_pipeline=_OrderedPipeline)
    _patch_runtime(monkeypatch, runtime)
    downloads = _patch_upsampler_io(monkeypatch, events)

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert events == ["download", "from_pretrained"]
    assert downloads == [_UPSAMPLER_SOURCE]


def test_failed_upsampler_download_stops_before_the_pipeline_loads(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())

    def _fail(repo_id, filename):
        raise RuntimeError(f"Failed to download the LTX latent upsampler '{filename}' from '{repo_id}': offline")

    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._resolve_upsampler_weights", _fail)

    with pytest.raises(RuntimeError, match="latent upsampler"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert _FakeTextPipeline.loaded_instances == []


def test_upscale_load_builds_upsampler_with_x2_config_and_strict_weights(monkeypatch):
    _load_upscale_model(monkeypatch, low_memory=False)

    (upsampler,) = _FakeUpsamplerModel.instances
    assert upsampler.config == dict(_UPSAMPLER_CONFIG)
    assert upsampler.config["use_rational_resampler"] is False
    assert upsampler.state_dict_calls == [(_UPSAMPLER_STATE, True)]
    assert upsampler.eval_calls == 1


@pytest.mark.parametrize(("low_memory", "expected_to"), [(False, {"device": "cuda", "dtype": "bfloat16"}), (True, {"dtype": "bfloat16"})])
def test_upsampler_device_follows_low_memory(monkeypatch, low_memory, expected_to):
    model = _load_upscale_model(monkeypatch, low_memory=low_memory)

    assert model.upsample_pipeline.latent_upsampler.to_calls == [expected_to]


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
def test_upsample_pipeline_shares_the_generation_vae(monkeypatch, mode):
    model = _load_upscale_model(monkeypatch, low_memory=False, mode=mode)

    generation = model.text_to_video or model.image_to_video
    upsample = model.upsample_pipeline
    assert upsample.vae is generation.vae
    assert upsample.to_calls == []
    assert upsample.cpu_offload_enabled is False


def test_upscale_load_requires_upsampler_source(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())
    downloads = _patch_upsampler_io(monkeypatch)

    with pytest.raises(ValueError, match="latent_upsampler"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True)

    assert downloads == []
    assert _FakeTextPipeline.loaded_instances == []


def test_upsampler_weight_mismatch_is_reported(monkeypatch):
    _FakeUpsamplerModel.load_error = RuntimeError("Missing key(s) in state_dict")

    with pytest.raises(RuntimeError, match="latent_upsampler") as excinfo:
        _load_upscale_model(monkeypatch, low_memory=False)

    assert "Missing key(s)" in str(excinfo.value)


def test_non_upscale_load_never_touches_the_upsampler(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())
    downloads = _patch_upsampler_io(monkeypatch)

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", latent_upsampler=_UPSAMPLER_SOURCE)

    assert model.upsample_pipeline is None
    assert downloads == []
    assert _FakeUpsamplerModel.instances == []


def test_resolve_upsampler_weights_reports_download_failures(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", real_download_hub_file)

    def _fail(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)

    with pytest.raises(RuntimeError, match="offline") as excinfo:
        _resolve_upsampler_weights(*_UPSAMPLER_SOURCE)

    assert _UPSAMPLER_SOURCE[0] in str(excinfo.value)
    assert _UPSAMPLER_SOURCE[1] in str(excinfo.value)


def test_load_model_applies_lora_weights(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(lora_path), 0.75)])

    assert model.text_to_video.loaded_loras == [(str(lora_path), "lora_0")]
    assert model.text_to_video.adapter_calls == [(["lora_0"], [0.75])]


class _LoraNoMatchPipeline(_FakeTextPipeline):
    """A pipeline whose LoRA loads raise *error* for the paths in *unmatched*."""

    unmatched: tuple[str, ...] = ()
    error: type[Exception] = ValueError

    def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
        if path in self.unmatched:
            raise self.error("no module matches")
        super().load_lora_weights(path, adapter_name=adapter_name)


def _write_loras(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text("weights", encoding="utf-8")
        paths.append(str(path))
    return paths


def test_a_lora_matching_no_layer_is_skipped_with_a_warning(monkeypatch, tmp_path, fake_peft_no_match):
    first, second = _write_loras(tmp_path, "other-model.safetensors", "style.safetensors")
    _LoraNoMatchPipeline.unmatched = (first,)
    _LoraNoMatchPipeline.error = fake_peft_no_match
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.warns(UserWarning, match="other-model.safetensors: none of its LoRA tensors match"):
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(first, 1.0), (second, 0.5)])

    assert model.text_to_video.loaded_loras == [(second, "lora_1")]
    assert model.text_to_video.adapter_calls == [(["lora_1"], [0.5])]


@pytest.mark.parametrize("low_memory", [True, False])
def test_loading_loras_never_turns_on_model_cpu_offload(monkeypatch, tmp_path, fake_peft_no_match, low_memory):
    (path,) = _write_loras(tmp_path, "other-model.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = fake_peft_no_match
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.warns(UserWarning):
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=low_memory, loras=[(path, 1.0)])

    # The transformer streams block by block; whole-component CPU offload would hold every component in memory.
    assert model.text_to_video.cpu_offload_enabled is False


def test_other_lora_load_errors_still_fail_the_load(monkeypatch, tmp_path, fake_peft_no_match):
    (path,) = _write_loras(tmp_path, "bad.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = ValueError
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.raises(RuntimeError, match="Failed to load LoRA"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(path, 1.0)])


def test_load_model_rejects_missing_or_unsupported_lora(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(tmp_path / "missing.safetensors"), 1.0)])

    bad_lora = tmp_path / "bad.txt"
    bad_lora.write_text("weights", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported LoRA format"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(bad_lora), 1.0)])


def test_load_model_rejects_lora_scales_when_adapter_api_cannot_apply_them(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(text_pipeline=_NoScalePipeline)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit LoRA scales"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(lora_path), 0.5)])


def test_load_model_rejects_old_diffusers_versions(monkeypatch):
    with pytest.raises(RuntimeError, match=r"diffusers>=0\.41\.0"):
        _validate_diffusers_version("0.37.1")


def test_a_converted_folder_loads_its_own_weights_and_level(monkeypatch, fake_weights, tmp_path):
    _patch_runtime(monkeypatch, _make_runtime())
    folder = _converted_folder(tmp_path, bits=8)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.memory_estimate.converted_ltx_system_estimate", lambda model_dir, manifest: ("estimate", model_dir, manifest["bits"]))
    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", lambda *a: pytest.fail("a converted folder never reads the configured transformer"))

    # The alias table has no level for this name, and the configured transformer is ignored.
    model, _ = DiffusersVideoBackend().load_model(str(folder), mode="t2v", quantize=None, transformer_source=("org/ltx", "stock.safetensors"), model_name="myft")

    assert fake_weights.prepared == []  # no <name>@q<bits> copy is written
    assert fake_weights.transformers == [{"prepared": folder / "transformer", "bits": 8}]
    assert model.text_encoder.prepared == folder / "text_encoder"
    assert fake_weights.memory_checks == [(folder, 8)]


def test_text_encoder_copies_are_reused_from_converted_folders_and_aliases(tmp_path):
    from zvisiongenerator.backends.cuda.video.weights import prepared_text_encoders

    for name in ("ltx-4@q4", "myft", ".myft.1234.partial", "zit"):
        (tmp_path / name / "text_encoder").mkdir(parents=True)
    for name in ("ltx-4@q4", "myft", ".myft.1234.partial"):
        (tmp_path / name / "text_encoder" / "ziv-quant.json").write_text("{}")

    assert prepared_text_encoders(tmp_path) == [tmp_path / "ltx-4@q4" / "text_encoder", tmp_path / "myft" / "text_encoder"]
