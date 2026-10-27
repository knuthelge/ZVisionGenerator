"""Tests for the CUDA SeedVR2 upscaler, with the vendored core and the Hugging Face Hub mocked."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.backends.cuda import seedvr2 as module
from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, SEEDVR2_REPO


@pytest.fixture
def hub(monkeypatch):
    def _download(repo, name, *, local_dir):
        path = Path(local_dir) / name
        path.write_bytes(b"w")
        return str(path)

    download = MagicMock(side_effect=_download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(module, "_require_cuda", lambda: None)
    return download


def _model(model: str = "seedvr2-7b") -> module.TorchSeedVR2Model:
    (_, dit), (_, vae) = FP8_WEIGHTS.files[model]
    return module.TorchSeedVR2Model(model=model, dit_file=dit, vae_file=vae, model_dir="/models/.seedvr2/fp8")


class TestLoadModel:
    def test_implements_the_protocol_with_fp8_weights(self):
        upscaler = module.TorchSeedVR2Upscaler()

        assert isinstance(upscaler, SeedVR2Upscaler)
        assert upscaler.weights is FP8_WEIGHTS
        assert upscaler.weights.stores_copy is False

    @pytest.mark.parametrize("model", ["seedvr2-7b", "seedvr2-3b"])
    def test_downloads_the_tiers_transformer_and_vae_from_their_repos_into_the_folder(self, hub, tmp_path, model):
        loaded = module.TorchSeedVR2Upscaler().load_model(model, quantize=None, model_path=str(tmp_path))

        (dit_repo, dit), (vae_repo, vae) = FP8_WEIGHTS.files[model]
        assert [(c.args, c.kwargs) for c in hub.call_args_list] == [((dit_repo, dit), {"local_dir": str(tmp_path)}), ((vae_repo, vae), {"local_dir": str(tmp_path)})]
        assert vae_repo == SEEDVR2_REPO
        assert loaded == module.TorchSeedVR2Model(model=model, dit_file=dit, vae_file=vae, model_dir=str(tmp_path))

    def test_files_already_in_the_folder_are_not_downloaded_again(self, hub, tmp_path):
        for _repo, name in FP8_WEIGHTS.files["seedvr2-3b"]:
            (tmp_path / name).write_bytes(b"w")

        module.TorchSeedVR2Upscaler().load_model("seedvr2-3b", quantize=None, model_path=str(tmp_path))

        hub.assert_not_called()

    def test_unknown_model_raises_before_any_download(self, hub, tmp_path):
        with pytest.raises(ValueError, match="seedvr2-9b"):
            module.TorchSeedVR2Upscaler().load_model("seedvr2-9b", quantize=None, model_path=str(tmp_path))

        hub.assert_not_called()

    def test_without_cuda_it_fails_before_any_download(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        download = MagicMock()
        monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
        monkeypatch.setattr(module, "configure_allocator", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError):
            module.TorchSeedVR2Upscaler().load_model("seedvr2-7b", quantize=None, model_path=str(tmp_path))

        download.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_quantizing_while_loading_is_refused(self, hub, tmp_path):
        with pytest.raises(ValueError):
            module.TorchSeedVR2Upscaler().load_model("seedvr2-7b", quantize=8, model_path=str(tmp_path))

    def test_there_is_no_stored_copy_to_save(self, tmp_path):
        with pytest.raises(RuntimeError):
            module.TorchSeedVR2Upscaler().save_model(_model(), tmp_path)


class TestUpscaleImage:
    def test_the_core_gets_the_image_prepared_at_the_target_size(self, monkeypatch, tmp_path):
        source = tmp_path / "in.png"
        Image.new("RGBA", (100, 60), (10, 20, 30, 255)).save(source)
        run = MagicMock(return_value=Image.new("RGB", (200, 120)))
        monkeypatch.setattr(module, "_run", run)

        result = module.TorchSeedVR2Upscaler().upscale_image(_model(), str(source), width=200, height=120, seed=7, softness=0.75)

        prepared = run.call_args.args[1]
        assert run.call_args.args[0] == _model()
        assert run.call_args.kwargs == {"seed": 7}
        assert (prepared.mode, prepared.size) == ("RGB", (200, 120))
        assert result is run.return_value

    def test_an_exif_rotated_source_is_read_as_stored_like_the_stage_sizes_it(self, monkeypatch, tmp_path):
        source = tmp_path / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # rotated 90° clockwise: stored 100 × 60, shown 60 × 100
        stored = Image.new("RGB", (100, 60), (0, 0, 255))
        stored.paste((255, 0, 0), (0, 0, 50, 60))  # left half red as stored
        stored.save(source, exif=exif, quality=95)
        run = MagicMock(return_value=Image.new("RGB", (400, 240)))
        monkeypatch.setattr(module, "_run", run)

        module.TorchSeedVR2Upscaler().upscale_image(_model(), str(source), width=400, height=240, seed=1, softness=0.0)

        prepared = run.call_args.args[1]
        assert prepared.size == (400, 240)
        assert prepared.getpixel((20, 120))[0] > 200 and prepared.getpixel((380, 120))[2] > 200


class TestCoreSeed:
    @pytest.mark.parametrize("seed", [0, 2**32 - 1_000_001])
    def test_seeds_in_range_are_unchanged(self, seed):
        assert module._core_seed(seed) == seed

    @pytest.mark.parametrize("seed", [2**32 - 1_000_000, 2**40])
    def test_large_seeds_leave_room_for_the_vae_offset(self, seed):
        assert 0 <= module._core_seed(seed) + 1_000_000 <= 2**32 - 1

    def test_upscale_image_passes_the_folded_seed(self, monkeypatch, tmp_path):
        source = tmp_path / "in.png"
        Image.new("RGB", (10, 6)).save(source)
        run = MagicMock(return_value=Image.new("RGB", (20, 12)))
        monkeypatch.setattr(module, "_run", run)

        module.TorchSeedVR2Upscaler().upscale_image(_model(), str(source), width=20, height=12, seed=2**32 - 1, softness=0.0)

        assert run.call_args.kwargs == {"seed": 2**32 - 1 - (2**32 - 1_000_000)}


class TestRun:
    """The vendored core's call sequence and the memory settings the upscaler passes to it."""

    @pytest.fixture
    def core(self, monkeypatch):
        pytest.importorskip("torchvision")
        pytest.importorskip("seedvr2_torch")
        import torch
        from seedvr2_torch.src.core import generation_phases, generation_utils

        calls = MagicMock()
        ctx = {"dit_device": "cuda:0", "compute_dtype": torch.bfloat16}
        calls.setup_generation_context.return_value = ctx
        self.dit = torch.nn.Module()
        calls.prepare_runner.return_value = (SimpleNamespace(dit=self.dit), {"cache": True})
        calls.compute_generation_info.side_effect = lambda **kwargs: (kwargs["images"], {})
        for phase in ("encode_all_batches", "upscale_all_batches", "decode_all_batches"):
            getattr(calls, phase).side_effect = lambda runner, **kwargs: kwargs["ctx"]

        def postprocess(**kwargs):
            kwargs["ctx"]["final_video"] = torch.full((1, 4, 6, 3), 0.5)
            return kwargs["ctx"]

        calls.postprocess_all_batches.side_effect = postprocess
        for name in ("setup_generation_context", "prepare_runner", "compute_generation_info"):
            monkeypatch.setattr(generation_utils, name, getattr(calls, name))
        for name in ("encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches"):
            monkeypatch.setattr(generation_phases, name, getattr(calls, name))
        monkeypatch.setattr(module, "configure_allocator", calls.configure_allocator)
        return calls

    def _run(self) -> Image.Image:
        return module._run(_model("seedvr2-3b"), Image.new("RGB", (6, 4), (255, 0, 0)), seed=42)

    def test_phases_run_in_order_after_the_allocator_is_configured(self, core):
        self._run()

        assert [c[0] for c in core.mock_calls if not c[0].startswith("_")] == [
            "configure_allocator",
            "setup_generation_context",
            "prepare_runner",
            "compute_generation_info",
            "encode_all_batches",
            "upscale_all_batches",
            "decode_all_batches",
            "postprocess_all_batches",
        ]

    def test_the_memory_settings_stream_every_block_and_free_each_model_after_its_phase(self, core):
        self._run()

        context = core.setup_generation_context.call_args.kwargs
        runner = core.prepare_runner.call_args.kwargs
        assert (context["dit_offload_device"], context["vae_offload_device"], context["tensor_offload_device"]) == ("cpu", "cpu", "cpu")
        assert runner["block_swap_config"]["blocks_to_swap"] >= 36
        assert runner["block_swap_config"]["offload_device"] == "cpu"
        assert runner["dit_cache"] is False and runner["vae_cache"] is False
        assert runner["encode_tiled"] is True and runner["decode_tiled"] is True
        assert runner["encode_tile_size"] == runner["decode_tile_size"] == (1024, 1024)
        assert core.upscale_all_batches.call_args.kwargs["cache_model"] is False
        assert core.decode_all_batches.call_args.kwargs["cache_model"] is False
        assert (runner["dit_model"], runner["vae_model"], runner["model_dir"]) == (*(name for _repo, name in FP8_WEIGHTS.files["seedvr2-3b"]), "/models/.seedvr2/fp8")
        assert core.encode_all_batches.call_args.kwargs["color_correction"] == "lab"
        assert core.postprocess_all_batches.call_args.kwargs["color_correction"] == "lab"

    def test_the_image_is_upscaled_at_its_own_short_side_with_the_seed_and_returned_as_rgb(self, core):
        image = self._run()

        info = core.compute_generation_info.call_args.kwargs
        encode = core.encode_all_batches.call_args.kwargs
        assert info["resolution"] == encode["resolution"] == 4
        assert info["seed"] == encode["seed"] == core.upscale_all_batches.call_args.kwargs["seed"] == 42
        assert tuple(info["images"].shape) == (1, 4, 6, 3)
        assert float(info["images"][0, 0, 0, 0]) == 1.0
        assert (image.mode, image.size) == ("RGB", (6, 4))
        assert np.asarray(image)[0, 0].tolist() == [128, 128, 128]


class TestToUint8Frames:
    def test_values_are_clamped_scaled_and_rounded_into_one_frame_per_item(self):
        torch = pytest.importorskip("torch")
        pixels = torch.tensor([[[[-0.5, 0.0, 0.5], [1.0, 1.2, 0.2]]], [[[0.25, 0.75, 1.0], [0.0, 0.1, 0.9]]]])

        frames = module._to_uint8_frames(pixels)

        assert [frame.shape for frame in frames] == [(1, 2, 3)] * 2
        assert all(frame.dtype == np.uint8 for frame in frames)
        assert frames[0].tolist() == [[[0, 0, 128], [255, 255, 51]]]
        assert frames[1].tolist() == [[[64, 191, 255], [0, 26, 230]]]

    def test_a_half_precision_batch_is_converted_too(self):
        torch = pytest.importorskip("torch")

        (frame,) = module._to_uint8_frames(torch.full((1, 2, 2, 3), 0.5, dtype=torch.bfloat16))

        assert frame.tolist() == [[[128] * 3] * 2] * 2


class TestUpscaleVideo:
    """The video upscale streams chunks of two batches through the core on one runner and frees it at the end."""

    @pytest.fixture
    def core(self, monkeypatch):
        pytest.importorskip("torchvision")
        pytest.importorskip("seedvr2_torch")
        import torch
        from seedvr2_torch.src.core import generation_phases, generation_utils
        from seedvr2_torch.src.optimization import memory_manager

        calls = MagicMock()
        self.generation_contexts: list[dict] = []
        self.input_tensors: list = []
        self.text_embeds_seen: list = []
        calls.setup_generation_context.return_value = {"dit_device": "cuda:0", "compute_dtype": torch.bfloat16, "text_embeds": None}
        calls.prepare_runner.return_value = (SimpleNamespace(), {"cache": True})

        def generation_info(**kwargs):
            self.generation_contexts.append(dict(kwargs["ctx"]))
            self.input_tensors.append(kwargs["images"].clone())
            return kwargs["images"], {}

        def encode(runner, **kwargs):
            kwargs["ctx"].update(frames=kwargs["images"], batch_size=kwargs["batch_size"], true_target_dims=tuple(kwargs["images"].shape[1:3]))
            return kwargs["ctx"]

        def decode(runner, **kwargs):
            ctx = kwargs["ctx"]
            count = math.ceil(ctx["frames"].shape[0] / ctx["batch_size"])
            for current in range(1, count + 1):
                kwargs["progress_callback"](current, count, 5, "Phase 3: Decoding")
            return ctx

        def postprocess(**kwargs):
            kwargs["ctx"]["final_video"] = kwargs["ctx"]["frames"]
            return kwargs["ctx"]

        calls.compute_generation_info.side_effect = generation_info
        calls.encode_all_batches.side_effect = encode

        def upscale(runner, **kwargs):
            # The core loads the embeddings into the context it is given when the key is None, and frees them after the phase.
            self.text_embeds_seen.append(kwargs["ctx"].get("text_embeds"))
            kwargs["ctx"]["text_embeds"] = {"texts_pos": [torch.zeros(1)]}
            return kwargs["ctx"]

        calls.upscale_all_batches.side_effect = upscale
        calls.decode_all_batches.side_effect = decode
        calls.postprocess_all_batches.side_effect = postprocess
        for name in ("setup_generation_context", "prepare_runner", "compute_generation_info"):
            monkeypatch.setattr(generation_utils, name, getattr(calls, name))
        for name in ("encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches"):
            monkeypatch.setattr(generation_phases, name, getattr(calls, name))
        monkeypatch.setattr(memory_manager, "complete_cleanup", calls.complete_cleanup)
        monkeypatch.setattr(module, "configure_allocator", calls.configure_allocator)
        return calls

    def _frames(self, count: int) -> list[np.ndarray]:
        return [np.full((6, 8, 3), 10 * i, dtype=np.uint8) for i in range(count)]

    def _upscale(self, frames, **overrides):
        kwargs = {"width": 16, "height": 12, "total_frames": len(frames), "seed": 9, "softness": 0.0, "batch_size": 1}
        return module.TorchSeedVR2Upscaler().upscale_video(_model(), iter(frames), **{**kwargs, **overrides})

    @pytest.mark.parametrize(
        ("count", "batch_size", "chunks"),
        [(7, 1, [2, 2, 2, 1]), (11, 3, [6, 5])],
        ids=["batch 1", "batch 3"],
    )
    def test_a_chunk_is_two_batches_and_all_chunks_run_on_one_runner_with_the_models_kept(self, core, count, batch_size, chunks):
        result = list(self._upscale(self._frames(count), batch_size=batch_size))

        assert core.prepare_runner.call_count == 1
        assert [c.kwargs["images"].shape[0] for c in core.compute_generation_info.call_args_list] == chunks
        assert {c.kwargs["batch_size"] for c in core.encode_all_batches.call_args_list} == {batch_size}
        assert all(c.kwargs["cache_model"] is True for c in core.upscale_all_batches.call_args_list + core.decode_all_batches.call_args_list)
        assert all("cache_model" not in c.kwargs for c in core.encode_all_batches.call_args_list + core.postprocess_all_batches.call_args_list)
        assert [int(frame[0, 0, 0]) for frame in result] == [10 * i for i in range(count)]
        assert all(frame.dtype == np.uint8 and frame.shape == (12, 16, 3) for frame in result)

    def test_a_chunk_never_inherits_the_previous_chunks_target_size(self, core):
        list(self._upscale(self._frames(4)))

        assert len(self.generation_contexts) == 2
        assert all("true_target_dims" not in ctx for ctx in self.generation_contexts)

    def test_no_chunk_shares_text_embeddings_with_another(self, core):
        list(self._upscale(self._frames(7)))

        assert self.text_embeds_seen == [None] * 4

    def test_progress_counts_decoded_batches_across_chunks(self, core):
        progress = MagicMock()

        list(self._upscale(self._frames(7), progress=progress))

        assert [c.args for c in progress.call_args_list] == [(n, 7) for n in range(1, 8)]

    def test_the_total_grows_when_the_stream_is_longer_than_announced(self, core):
        progress = MagicMock()

        list(self._upscale(self._frames(7), total_frames=5, progress=progress))

        assert [c.args for c in progress.call_args_list] == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 6), (7, 7)]

    def _ends_normally(self, core):
        list(self._upscale(self._frames(3)))

    def _phase_raises(self, core):
        core.upscale_all_batches.side_effect = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            list(self._upscale(self._frames(3)))

    def _is_closed_midway(self, core):
        frames = self._upscale(self._frames(7))
        next(frames)
        frames.close()

    def _is_interrupted(self, core):
        class Stop(Exception):
            pass

        def encode(runner, **kwargs):
            kwargs["ctx"]["interrupt_fn"]()

        core.encode_all_batches.side_effect = encode
        with pytest.raises(Stop):
            list(self._upscale(self._frames(3), interrupt=MagicMock(side_effect=Stop)))

    @pytest.mark.parametrize("scenario", ["_ends_normally", "_phase_raises", "_is_closed_midway", "_is_interrupted"])
    def test_the_models_are_freed_once(self, core, scenario):
        getattr(self, scenario)(core)

        core.complete_cleanup.assert_called_once()
        assert core.complete_cleanup.call_args.kwargs == {"dit_cache": False, "vae_cache": False}

    def test_no_frames_yield_nothing_and_still_free_the_models(self, core):
        assert list(self._upscale([], total_frames=0)) == []

        core.compute_generation_info.assert_not_called()
        core.complete_cleanup.assert_called_once()

    @pytest.mark.parametrize(("batch_size", "read_for_first_chunk"), [(1, 2), (3, 6)])
    def test_frames_are_read_only_as_chunks_are_needed(self, core, batch_size, read_for_first_chunk):
        read = []

        def source():
            for frame in self._frames(13):
                read.append(1)
                yield frame

        frames = module.TorchSeedVR2Upscaler().upscale_video(_model(), source(), width=16, height=12, total_frames=13, seed=9, softness=0.0, batch_size=batch_size)
        next(frames)

        assert len(read) == read_for_first_chunk
        frames.close()

    def test_the_interrupt_reaches_the_core_context(self, core):
        interrupt = MagicMock()

        list(self._upscale(self._frames(2), interrupt=interrupt))

        assert self.generation_contexts[0]["interrupt_fn"] is interrupt
        for phase in ("encode_all_batches", "upscale_all_batches", "decode_all_batches"):
            assert getattr(core, phase).call_args.kwargs["ctx"]["interrupt_fn"] is interrupt

    def test_the_vae_always_uses_512_pixel_tiles(self, core):
        list(self._upscale([np.zeros((4, 4, 3), dtype=np.uint8)], width=1536, height=896))

        runner = core.prepare_runner.call_args.kwargs
        assert runner["encode_tile_size"] == runner["decode_tile_size"] == (512, 512)

    def test_the_seed_the_short_side_and_the_colour_correction_match_the_image_path(self, core):
        list(self._upscale(self._frames(2), seed=2**32 - 1))

        seed = 2**32 - 1 - (2**32 - 1_000_000)
        assert core.compute_generation_info.call_args.kwargs["seed"] == core.encode_all_batches.call_args.kwargs["seed"] == core.upscale_all_batches.call_args.kwargs["seed"] == seed
        assert core.compute_generation_info.call_args.kwargs["resolution"] == core.encode_all_batches.call_args.kwargs["resolution"] == 12
        assert core.encode_all_batches.call_args.kwargs["color_correction"] == "lab"
        assert core.postprocess_all_batches.call_args.kwargs["color_correction"] == "lab"

    def test_softness_pre_blurs_each_frame_at_the_target_size(self, core, monkeypatch):
        spy = MagicMock(wraps=module.seedvr2_input_image)
        monkeypatch.setattr(module, "seedvr2_input_image", spy)

        list(self._upscale(self._frames(2), softness=0.5))

        assert [c.args[1:] for c in spy.call_args_list] == [(16, 12, 0.5)] * 2

    def test_frames_reach_the_core_as_floats_at_the_target_size(self, core):
        list(self._upscale(self._frames(2)))

        tensor = self.input_tensors[0]
        assert tuple(tensor.shape) == (2, 12, 16, 3)
        assert tensor.dtype.is_floating_point
        assert [float(tensor[i, 0, 0, 0]) for i in range(2)] == pytest.approx([0.0, 10 / 255])

    def test_running_out_of_gpu_memory_becomes_a_readable_error_naming_the_output_size(self, core):
        import torch

        core.decode_all_batches.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="3840×2160"):
            list(self._upscale(self._frames(1), width=3840, height=2160))

        core.complete_cleanup.assert_called_once()

    def test_a_bad_batch_size_is_refused_before_anything_runs(self, core):
        with pytest.raises(ValueError, match="batch_size"):
            self._upscale(self._frames(2), batch_size=0)

        core.prepare_runner.assert_not_called()
