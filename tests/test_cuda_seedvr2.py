"""Tests for the CUDA SeedVR2 upscaler, with the vendored core and the Hugging Face Hub mocked."""

from __future__ import annotations

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
    @pytest.mark.parametrize("seed", [0, 42, 2_830_951_588, 2**32 - 1_000_001])
    def test_seeds_in_range_are_unchanged(self, seed):
        assert module._core_seed(seed) == seed

    @pytest.mark.parametrize("seed", [2**32 - 1_000_000, 4_294_000_000, 2**32 - 1, 2**40])
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
        calls.load_text_embeddings.return_value = {"texts_pos": []}
        calls.compute_generation_info.side_effect = lambda **kwargs: (kwargs["images"], {})
        for phase in ("encode_all_batches", "upscale_all_batches", "decode_all_batches"):
            getattr(calls, phase).side_effect = lambda runner, **kwargs: kwargs["ctx"]

        def postprocess(**kwargs):
            kwargs["ctx"]["final_video"] = torch.full((1, 4, 6, 3), 0.5)
            return kwargs["ctx"]

        calls.postprocess_all_batches.side_effect = postprocess
        for name in ("setup_generation_context", "prepare_runner", "load_text_embeddings", "compute_generation_info"):
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
            "load_text_embeddings",
            "compute_generation_info",
            "encode_all_batches",
            "upscale_all_batches",
            "decode_all_batches",
            "postprocess_all_batches",
        ]

    def test_every_block_streams_and_both_models_offload_to_system_memory(self, core):
        self._run()

        context = core.setup_generation_context.call_args.kwargs
        runner = core.prepare_runner.call_args.kwargs
        assert (context["dit_offload_device"], context["vae_offload_device"], context["tensor_offload_device"]) == ("cpu", "cpu", "cpu")
        assert runner["block_swap_config"]["blocks_to_swap"] >= 36
        assert runner["block_swap_config"]["offload_device"] == "cpu"
        assert runner["dit_cache"] is False and runner["vae_cache"] is False
        assert runner["encode_tiled"] is True and runner["decode_tiled"] is True

    def test_an_image_uses_the_core_tile_and_a_video_the_tile_it_is_given(self, core):
        import torch

        self._run()
        assert core.prepare_runner.call_args.kwargs["decode_tile_size"] == (1024, 1024)

        module._run_frames(_model("seedvr2-3b"), torch.zeros(5, 4, 6, 3), seed=1, resolution=4, batch_size=5, tile=(512, 512))

        runner = core.prepare_runner.call_args.kwargs
        assert runner["encode_tile_size"] == runner["decode_tile_size"] == (512, 512)

    def test_the_tiers_files_and_folder_are_passed(self, core):
        self._run()

        runner = core.prepare_runner.call_args.kwargs
        assert (runner["dit_model"], runner["vae_model"], runner["model_dir"]) == (*(name for _repo, name in FP8_WEIGHTS.files["seedvr2-3b"]), "/models/.seedvr2/fp8")

    def test_the_image_is_upscaled_at_its_own_short_side_with_the_seed(self, core):
        self._run()

        info = core.compute_generation_info.call_args.kwargs
        encode = core.encode_all_batches.call_args.kwargs
        assert info["resolution"] == encode["resolution"] == 4
        assert info["seed"] == encode["seed"] == core.upscale_all_batches.call_args.kwargs["seed"] == 42
        assert tuple(info["images"].shape) == (1, 4, 6, 3)
        assert float(info["images"][0, 0, 0, 0]) == 1.0

    def test_lab_colour_correction_matches_macos(self, core):
        self._run()

        assert core.encode_all_batches.call_args.kwargs["color_correction"] == "lab"
        assert core.postprocess_all_batches.call_args.kwargs["color_correction"] == "lab"

    def test_returns_the_final_frame_as_an_rgb_image(self, core):
        image = self._run()

        assert (image.mode, image.size) == ("RGB", (6, 4))
        assert np.asarray(image)[0, 0].tolist() == [128, 128, 128]


class TestUpscaleVideo:
    """The video upscale prepares every frame and hands the whole clip to the core in one run."""

    @pytest.fixture
    def run(self, monkeypatch):
        torch = pytest.importorskip("torch")

        def fake(model, frames, *, seed, resolution, batch_size, tile, progress_callback=None):
            if progress_callback is not None:
                for done in range(1, 4):
                    progress_callback(done, 3, 5, "Phase 2: Upscaling")
            height, width = frames.shape[1:3]
            return torch.full((frames.shape[0], height, width, 3), 0.5)

        run = MagicMock(side_effect=fake)
        monkeypatch.setattr(module, "_run_frames", run)
        return run

    def _frames(self, count: int) -> list[np.ndarray]:
        return [np.full((6, 8, 3), 10 * i, dtype=np.uint8) for i in range(count)]

    def _upscale(self, frames, **overrides):
        kwargs = {"width": 16, "height": 12, "seed": 9, "softness": 0.0}
        return module.TorchSeedVR2Upscaler().upscale_video(_model(), frames, **{**kwargs, **overrides})

    def test_the_whole_clip_goes_through_the_core_in_one_run(self, run):
        self._upscale(self._frames(12))

        assert run.call_count == 1
        assert run.call_args.args[1].shape[0] == 12

    def test_the_run_gets_the_folded_seed_the_target_short_side_and_the_batch_size(self, run):
        self._upscale(self._frames(11), seed=2**32 - 1, batch_size=9)

        assert run.call_args.kwargs["seed"] == 2**32 - 1 - (2**32 - 1_000_000)
        assert run.call_args.kwargs["resolution"] == 12
        assert run.call_args.kwargs["batch_size"] == 9

    @pytest.mark.parametrize(
        ("width", "height", "tile"),
        [(576, 320, (1024, 1024)), (768, 448, (1024, 1024)), (448, 768, (1024, 1024)), (1024, 576, (512, 512)), (1280, 704, (512, 512)), (1600, 896, (512, 512)), (896, 1600, (512, 512))],
    )
    def test_the_vae_tile_follows_the_frame_size(self, run, width, height, tile):
        self._upscale(self._frames(2), width=width, height=height)

        assert run.call_args.kwargs["tile"] == tile

    def test_the_batch_size_defaults_to_five(self, run):
        self._upscale(self._frames(7))

        assert run.call_args.kwargs["batch_size"] == 5

    def test_frames_reach_the_core_as_floats_at_the_target_size(self, run):
        self._upscale(self._frames(2))

        tensor = run.call_args.args[1]
        assert tuple(tensor.shape) == (2, 12, 16, 3)
        assert tensor.dtype.is_floating_point
        assert float(tensor.max()) == pytest.approx(10 / 255)

    def test_progress_is_forwarded_as_done_and_total(self, run):
        progress = MagicMock()

        self._upscale(self._frames(11), progress=progress)

        assert [c.args for c in progress.call_args_list] == [(1, 3), (2, 3), (3, 3)]

    def test_without_progress_the_core_gets_no_callback(self, run):
        self._upscale(self._frames(3))

        assert run.call_args.kwargs.get("progress_callback") is None

    def test_the_output_has_one_uint8_array_per_frame_at_the_target_size(self, run):
        result = self._upscale(self._frames(7))

        assert len(result) == 7
        assert all(frame.dtype == np.uint8 and frame.shape == (12, 16, 3) for frame in result)
        assert result[0][0, 0].tolist() == [128, 128, 128]

    def test_no_frames_and_a_bad_batch_size_are_refused_before_the_core_runs(self, run):
        with pytest.raises(ValueError):
            self._upscale([])
        with pytest.raises(ValueError):
            self._upscale(self._frames(2), batch_size=0)

        run.assert_not_called()

    def test_the_upscaler_has_the_optional_video_method(self):
        assert hasattr(module.TorchSeedVR2Upscaler(), "upscale_video")


class TestVideoCoreCalls:
    """The core's phases see the clip's batch size and the progress callback."""

    @pytest.fixture
    def core(self, monkeypatch):
        pytest.importorskip("torchvision")
        pytest.importorskip("seedvr2_torch")
        import torch
        from seedvr2_torch.src.core import generation_phases, generation_utils

        calls = MagicMock()
        calls.setup_generation_context.return_value = {"dit_device": "cuda:0", "compute_dtype": torch.bfloat16}
        calls.prepare_runner.return_value = (SimpleNamespace(), {"cache": True})
        calls.compute_generation_info.side_effect = lambda **kwargs: (kwargs["images"], {})
        for phase in ("encode_all_batches", "upscale_all_batches", "decode_all_batches"):
            getattr(calls, phase).side_effect = lambda runner, **kwargs: kwargs["ctx"]

        def postprocess(**kwargs):
            kwargs["ctx"]["final_video"] = torch.full((7, 12, 16, 3), 0.25)
            return kwargs["ctx"]

        calls.postprocess_all_batches.side_effect = postprocess
        for name in ("setup_generation_context", "prepare_runner", "load_text_embeddings", "compute_generation_info"):
            monkeypatch.setattr(generation_utils, name, getattr(calls, name))
        for name in ("encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches"):
            monkeypatch.setattr(generation_phases, name, getattr(calls, name))
        monkeypatch.setattr(module, "configure_allocator", calls.configure_allocator)
        return calls

    def _upscale(self, **overrides):
        frames = [np.zeros((6, 8, 3), dtype=np.uint8)] * 7
        kwargs = {"width": 16, "height": 12, "seed": 3, "softness": 0.0}
        return module.TorchSeedVR2Upscaler().upscale_video(_model("seedvr2-3b"), frames, **{**kwargs, **overrides})

    def test_every_batching_phase_gets_the_batch_size_and_the_models_are_built_once(self, core):
        self._upscale(batch_size=5)

        assert core.prepare_runner.call_count == 1
        assert core.compute_generation_info.call_args.kwargs["batch_size"] == 5
        assert core.encode_all_batches.call_args.kwargs["batch_size"] == 5
        assert core.postprocess_all_batches.call_args.kwargs["batch_size"] == 5
        assert tuple(core.compute_generation_info.call_args.kwargs["images"].shape) == (7, 12, 16, 3)

    def test_the_seed_and_colour_correction_match_the_image_path(self, core):
        self._upscale()

        assert core.compute_generation_info.call_args.kwargs["seed"] == core.encode_all_batches.call_args.kwargs["seed"] == 3
        assert core.upscale_all_batches.call_args.kwargs["seed"] == 3
        assert core.encode_all_batches.call_args.kwargs["color_correction"] == "lab"
        assert core.postprocess_all_batches.call_args.kwargs["color_correction"] == "lab"

    def test_the_upscale_phase_gets_the_progress_adapter(self, core):
        progress = MagicMock()
        self._upscale(progress=progress)

        core.upscale_all_batches.call_args.kwargs["progress_callback"](2, 3, 5, "Phase 2: Upscaling")

        progress.assert_called_once_with(2, 3)

    def test_the_result_is_one_uint8_array_per_frame(self, core):
        result = self._upscale()

        assert len(result) == 7 and result[0].shape == (12, 16, 3) and result[0].dtype == np.uint8
