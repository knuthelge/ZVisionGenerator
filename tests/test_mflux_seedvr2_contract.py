"""Pin the mflux SeedVR2 internals that ``backends/mac/seedvr2.py`` and the stored 8-bit copy rely on.

Introspection only: no model is constructed and no weights are loaded. A failure here after an mflux bump
means upstream changed a name, signature or file layout the SeedVR2 upscale depends on.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition  # noqa: E402
from mflux.models.seedvr2 import SeedVR2  # noqa: E402
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition7B, SeedVR2WeightDefinition3B  # noqa: E402
from mflux.utils.generated_image import GeneratedImage  # noqa: E402

from zvisiongenerator.utils.seedvr2 import MLX_WEIGHTS, SEEDVR2_REPO  # noqa: E402

_DEFINITIONS = {"seedvr2-3b": SeedVR2WeightDefinition3B, "seedvr2-7b": SeedVR2WeightDefinition7B}


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def _weight_files(definition, component: str) -> list[str]:
    return next(item.weight_files for item in definition.get_components() if item.name == component)


class TestSeedVR2Contract:
    def test_constructor_arguments(self):
        assert _params(SeedVR2.__init__)[1:] == ["quantize", "model_path", "model_config"]

    def test_generate_image_arguments(self):
        assert _params(SeedVR2.generate_image)[1:] == ["seed", "image_path", "resolution", "softness"]

    def test_generated_image_carries_the_image(self):
        assert "image" in _params(GeneratedImage.__init__)

    def test_save_model_takes_a_base_path(self):
        assert _params(SeedVR2.save_model)[1:] == ["base_path"]

    def test_model_configs_exist_for_both_variants(self):
        assert callable(ModelConfig.seedvr2_3b)
        assert callable(ModelConfig.seedvr2_7b)

    def test_model_precision_is_a_settable_class_attribute(self):
        assert hasattr(ModelConfig, "precision")


class TestWeightLayoutContract:
    @pytest.mark.parametrize("model", sorted(_DEFINITIONS))
    def test_definition_reads_the_files_the_loader_deletes(self, model):
        (transformer_repo, transformer), (vae_repo, vae) = MLX_WEIGHTS.files[model]
        definition = _DEFINITIONS[model]

        assert _weight_files(definition, "transformer") == [transformer]
        assert _weight_files(definition, "vae") == [vae]
        assert transformer_repo == vae_repo == SEEDVR2_REPO  # the repo mflux downloads from

    @pytest.mark.parametrize("model", sorted(_DEFINITIONS))
    def test_saved_copy_carries_its_own_vae(self, model):
        subdirs = ComponentDefinition.save_subdirs(_DEFINITIONS[model].get_components())

        assert subdirs == {"transformer": "transformer", "vae": "vae"}
