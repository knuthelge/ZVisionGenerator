"""Pin the mflux FLUX.2 Klein internals that ``backends/mac/klein_references.py`` relies on.

Signature pins and behaviour on stubs: no weights are loaded. A failure here after an mflux bump means upstream
changed an attribute, signature or call the references view or its start-image hook depends on.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.lora.mapping.lora_loader import LoRALoader  # noqa: E402
from mflux.models.common.tokenizer import TokenizerLoader  # noqa: E402
from mflux.models.common.vae.vae_util import VAEUtil  # noqa: E402
from mflux.models.common.weights.loading.weight_applier import WeightApplier  # noqa: E402
from mflux.models.common.weights.loading.weight_loader import WeightLoader  # noqa: E402
from mflux.models.flux2 import Flux2Initializer, Flux2Klein  # noqa: E402
from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402
from mlx import nn  # noqa: E402
from PIL import Image  # noqa: E402

from zvisiongenerator.backends.mac.klein_references import DEFAULT_SCHEDULER, SHARED_ATTRIBUTES  # noqa: E402


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


class _AssignmentRecorder(nn.Module):
    """A module that remembers the name of every attribute assigned to it."""

    def __init__(self) -> None:
        object.__setattr__(self, "assigned", set())
        super().__init__()
        self.assigned.clear()  # nn.Module's own bookkeeping

    def __setattr__(self, name, value):
        self.assigned.add(name)
        super().__setattr__(name, value)


def _initializer_attributes() -> set[str]:
    """Return the model attributes ``Flux2Initializer.init`` assigns, run with the weight loading stubbed out."""
    recorder = _AssignmentRecorder()
    with (
        patch.object(WeightLoader, "load", return_value=MagicMock()),
        patch.object(TokenizerLoader, "load_all", return_value={}),
        patch.object(WeightApplier, "apply_and_quantize", return_value=None),
        patch.object(LoRALoader, "load_and_apply_lora", return_value=([], [])),
    ):
        Flux2Initializer.init(model=recorder, model_config=ModelConfig.flux2_klein_4b(), quantize=None, model_path="unused")
    return recorder.assigned


class TestViewContract:
    def test_the_view_shares_exactly_what_the_initializer_sets(self):
        assert set(SHARED_ATTRIBUTES) == _initializer_attributes()

    def test_the_edit_class_is_built_by_the_same_initializer(self):
        with patch.object(Flux2Initializer, "init") as init:
            edit = Flux2KleinEdit()

        init.assert_called_once()
        assert init.call_args.kwargs["model"] is edit


class _Reached(Exception):
    """Raised by a spy to stop ``generate_image`` at the call under test."""


def _stub_vae() -> SimpleNamespace:
    """Return a VAE stand-in that records its ``encode`` inputs and yields 32-channel latents at an eighth of the size."""
    encoded: list[mx.array] = []

    def encode(image):
        encoded.append(image)
        return mx.ones((1, 32, image.shape[-2] // 8, image.shape[-1] // 8))

    bn = SimpleNamespace(running_mean=mx.zeros(128), running_var=mx.ones(128), eps=1e-5)
    return SimpleNamespace(encode=encode, encoded=encoded, bn=bn)


class TestGenerateContract:
    def test_edit_generate_image_arguments(self):
        assert _params(Flux2KleinEdit.generate_image)[1:] == [
            "seed",
            "prompt",
            "num_inference_steps",
            "height",
            "width",
            "guidance",
            "image_paths",
            "image_strength",
            "scheduler",
            "use_kv_cache",
        ]

    def test_edit_default_scheduler_is_the_one_the_hook_configures(self):
        assert inspect.signature(Flux2KleinEdit.generate_image).parameters["scheduler"].default == DEFAULT_SCHEDULER


class TestStartImageHookContract:
    def test_edit_builds_its_latents_through_the_helper_the_hook_replaces(self):
        def spy(**kwargs):
            raise _Reached(kwargs)

        shell = SimpleNamespace(model_config=ModelConfig.flux2_klein_4b(), _encode_prompt_pair=lambda **kwargs: (None, None, None, None))

        with patch.object(_Flux2KleinEditHelpers, "prepare_generation_latents", staticmethod(spy)):
            with pytest.raises(_Reached) as reached:
                Flux2KleinEdit.generate_image(shell, seed=3, prompt="p", num_inference_steps=1, height=64, width=48)

        assert reached.value.args[0] == {"seed": 3, "height": 64, "width": 48}

    def test_helper_and_replacement_share_one_keyword_signature(self):
        assert _params(_Flux2KleinEditHelpers.prepare_generation_latents) == ["seed", "height", "width"]

    def test_klein_builds_img2img_latents_from_a_config(self):
        assert _params(Flux2Klein._prepare_img2img_latents)[1:] == ["seed", "config"]


class TestReferenceEncodingContract:
    def test_vae_util_encodes_through_the_vae_encode_method_and_flattens_a_single_frame(self):
        vae = SimpleNamespace(encode=lambda image: mx.ones((1, 32, 1, 4, 4)))

        encoded = VAEUtil.encode(vae=vae, image=mx.zeros((1, 3, 32, 32)))

        assert encoded.shape == (1, 32, 4, 4)

    def test_each_reference_is_encoded_once_with_its_own_time_coordinate(self, tmp_path):
        vae = _stub_vae()
        paths = []
        for index in range(3):
            path = tmp_path / f"ref{index}.png"
            Image.new("RGB", (64, 48)).save(path)
            paths.append(path)

        _, ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(vae=vae, tiling_config=None, image_paths=paths)

        assert len(vae.encoded) == 3
        assert sorted({int(t) for t in ids[0, :, 0].tolist()}) == [10, 20, 30]

    def test_the_start_image_is_encoded_once_through_the_loaded_models_vae(self, tmp_path):
        start = tmp_path / "start.png"
        Image.new("RGB", (64, 64)).save(start)
        model = Flux2Klein.__new__(Flux2Klein)
        nn.Module.__init__(model)
        model.vae = _stub_vae()
        model.tiling_config = None
        config = Config(
            model_config=ModelConfig.flux2_klein_4b(),
            num_inference_steps=4,
            height=64,
            width=64,
            guidance=1.0,
            image_path=start,
            image_strength=0.5,
            scheduler=DEFAULT_SCHEDULER,
        )

        Flux2Klein._prepare_img2img_latents(model, seed=1, config=config)

        assert len(model.vae.encoded) == 1

    def test_reference_conditioning_arguments(self):
        assert _params(_Flux2KleinEditHelpers.prepare_reference_image_conditioning) == ["vae", "tiling_config", "image_paths", "batch_size"]
