"""Tests for the background removal chain, with the model loaders mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from zvisiongenerator.backends.matte import birefnet as birefnet_loader
from zvisiongenerator.backends.matte import chain as chain_module
from zvisiongenerator.backends.matte.chain import TorchMatteChain
from zvisiongenerator.utils.matte_models import BIREFNET, VITMATTE


class _FakeBirefnet:
    """Return a logit map that is strongly foreground inside a centred box."""

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        logits = torch.full((1, 1, 1024, 1024), -10.0)
        logits[:, :, 256:768, 256:768] = 10.0
        return [torch.zeros(1), logits]


class _Inputs(dict):
    def to(self, device: str) -> _Inputs:
        return self


class _FakeVitmatte:
    """Report a half-opaque alpha for every pixel; the chain keeps it only inside the trimap's unknown band."""

    def __call__(self, **inputs: object) -> SimpleNamespace:
        height, width = inputs["trimaps"].shape
        return SimpleNamespace(alphas=torch.full((1, 1, height + 5, width + 5), 0.5))


def _processor(*, images: np.ndarray, trimaps: np.ndarray, do_rescale: bool, return_tensors: str) -> _Inputs:
    return _Inputs(images=images, trimaps=trimaps)


@pytest.fixture
def loaders(monkeypatch):
    """Patch the snapshot lookup and both loaders; return the mocks."""
    snapshot = MagicMock(side_effect=lambda files, download: Path(f"/snapshots/{files.repo_id}"))
    load_birefnet = MagicMock(return_value=_FakeBirefnet())
    load_vitmatte = MagicMock(return_value=(_processor, _FakeVitmatte()))
    monkeypatch.setattr(chain_module, "snapshot_with", snapshot)
    monkeypatch.setattr(chain_module, "load_birefnet", load_birefnet)
    monkeypatch.setattr(chain_module, "load_vitmatte", load_vitmatte)
    return SimpleNamespace(snapshot=snapshot, birefnet=load_birefnet, vitmatte=load_vitmatte)


@pytest.fixture
def chain(loaders):
    return TorchMatteChain("cpu", download=MagicMock(), release_memory=MagicMock(), after_cut_out=MagicMock())


def _picture(width: int = 64, height: int = 48) -> Image.Image:
    return Image.new("RGB", (width, height), (200, 40, 40))


class TestLoading:
    def test_the_models_load_on_the_first_cut_out_only(self, chain, loaders):
        assert chain.loaded is False
        loaders.birefnet.assert_not_called()

        chain.cut_out(_picture())
        chain.cut_out(_picture())

        assert chain.loaded is True
        loaders.birefnet.assert_called_once()
        loaders.vitmatte.assert_called_once()

    def test_snapshots_come_from_the_pinned_files(self, chain, loaders):
        chain.cut_out(_picture())

        assert [call.args[0] for call in loaders.snapshot.call_args_list] == [BIREFNET, VITMATTE]
        assert loaders.birefnet.call_args.args == (Path(f"/snapshots/{BIREFNET.repo_id}"), "cpu", torch.float16)
        assert loaders.vitmatte.call_args.args == (Path(f"/snapshots/{VITMATTE.repo_id}"), "cpu")

    def test_a_failed_matting_load_leaves_the_chain_unloaded(self, chain, loaders):
        loaders.vitmatte.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            chain.cut_out(_picture())

        assert chain.loaded is False


class TestCutOut:
    @pytest.mark.parametrize("size", [(64, 48), (1500, 900)])
    def test_returns_rgba_of_the_input_size(self, chain, size):
        result = chain.cut_out(_picture(*size))

        assert result.mode == "RGBA"
        assert result.size == size

    def test_the_centre_is_kept_and_the_corner_is_transparent(self, chain):
        alpha = np.asarray(chain.cut_out(_picture(256, 256)))[..., 3]

        assert alpha[128, 128] == 255
        assert alpha[2, 2] == 0

    def test_after_cut_out_runs_once_per_cut_out_even_when_it_fails(self, loaders):
        after = MagicMock()
        chain = TorchMatteChain("cpu", download=MagicMock(), release_memory=MagicMock(), after_cut_out=after)

        chain.cut_out(_picture())
        chain.cut_out(_picture())
        assert after.call_count == 2

        loaders.vitmatte.return_value = (_processor, MagicMock(side_effect=RuntimeError("boom")))
        failing = TorchMatteChain("cpu", download=MagicMock(), release_memory=MagicMock(), after_cut_out=after)
        with pytest.raises(RuntimeError, match="boom"):
            failing.cut_out(_picture())
        assert after.call_count == 3


class TestClose:
    def test_drops_the_models_and_releases_memory(self, loaders):
        release = MagicMock()
        chain = TorchMatteChain("cpu", download=MagicMock(), release_memory=release, after_cut_out=MagicMock())
        chain.cut_out(_picture())

        chain.close()

        assert chain.loaded is False
        release.assert_called_once_with()

    def test_a_later_cut_out_loads_again(self, chain, loaders):
        chain.cut_out(_picture())
        chain.close()

        chain.cut_out(_picture())

        assert loaders.birefnet.call_count == 2
        assert chain.loaded is True


class TestLoadBirefnet:
    def test_builds_from_the_vendored_classes_and_loads_strictly(self, monkeypatch, tmp_path):
        model = MagicMock(name="model")
        model_class = MagicMock(return_value=model)
        config_class = MagicMock(name="config_class")
        monkeypatch.setitem(sys.modules, "birefnet.birefnet", SimpleNamespace(BiRefNet=model_class))
        monkeypatch.setitem(sys.modules, "birefnet.BiRefNet_config", SimpleNamespace(BiRefNetConfig=config_class))
        state = {"weight": object()}
        loaded = MagicMock(return_value=state)
        monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=loaded))
        replace = MagicMock(return_value=20)
        monkeypatch.setattr("zvisiongenerator.backends.matte.deform_conv.replace_deformable_convs", replace)

        result = birefnet_loader.load_birefnet(tmp_path, "cpu", torch.float16)

        config_class.assert_called_once_with(bb_pretrained=False)
        model_class.assert_called_once_with(config=config_class.return_value)
        assert "trust_remote_code" not in model_class.call_args.kwargs
        replace.assert_called_once()
        assert replace.call_args.args[0] is model
        loaded.assert_called_once_with(tmp_path / "model.safetensors")
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.to.assert_called_once_with("cpu", torch.float16)
        assert result is model.to.return_value.eval.return_value

    def test_warns_when_nothing_was_swapped(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "birefnet.birefnet", SimpleNamespace(BiRefNet=MagicMock()))
        monkeypatch.setitem(sys.modules, "birefnet.BiRefNet_config", SimpleNamespace(BiRefNetConfig=MagicMock()))
        monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=MagicMock()))
        monkeypatch.setattr("zvisiongenerator.backends.matte.deform_conv.replace_deformable_convs", MagicMock(return_value=0))

        with pytest.warns(UserWarning, match="deformable"):
            birefnet_loader.load_birefnet(tmp_path, "cpu", torch.float16)


class TestCreateMatteChain:
    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_cuda_platforms_use_cuda_and_a_no_op_after_each_cut_out(self, monkeypatch, platform):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", platform)

        chain = backends.create_matte_chain()

        assert chain._device == "cuda"
        assert chain._after_cut_out() is None
        assert chain._after_cut_out is not backends.release_accelerator_memory
        assert chain._release_memory is backends.release_accelerator_memory

    def test_mac_uses_mps_and_releases_the_mac_caches_after_each_cut_out(self, monkeypatch):
        import zvisiongenerator.backends as backends
        from zvisiongenerator.backends.mac.memory import release_memory

        monkeypatch.setattr(backends.sys, "platform", "darwin")

        chain = backends.create_matte_chain()

        assert chain._device == "mps"
        assert chain._after_cut_out is release_memory

    def test_other_platforms_raise(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "freebsd")

        with pytest.raises(RuntimeError, match="freebsd"):
            backends.create_matte_chain()
