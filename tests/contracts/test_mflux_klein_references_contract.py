"""Pin the mflux FLUX.2 Klein internals that ``backends/mac/klein_references.py`` relies on.

Introspection only: no model is constructed and no weights are loaded. A failure here after an mflux bump means
upstream changed an attribute, signature or call the references view or its start-image hook depends on.
"""

from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

from mflux.models.flux2 import Flux2Initializer, Flux2Klein  # noqa: E402
from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402

from zvisiongenerator.backends.mac.klein_references import DEFAULT_SCHEDULER, SHARED_ATTRIBUTES  # noqa: E402


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def _initializer_attributes() -> set[str]:
    """Return the model attributes ``Flux2Initializer`` assigns (``model.name = ...``, including tuple targets)."""
    names: set[str] = set()
    for line in inspect.getsource(Flux2Initializer).splitlines():
        target, equals, _ = line.partition("=")
        if equals and re.fullmatch(r"\s*model\.[\w.]+(,\s*model\.[\w.]+)*\s*", target):
            names.update(re.findall(r"model\.(\w+)", target))
    return names


class TestViewContract:
    def test_the_view_shares_exactly_what_the_initializer_sets(self):
        assert set(SHARED_ATTRIBUTES) == _initializer_attributes()

    def test_the_edit_class_is_built_by_the_same_initializer(self):
        assert "Flux2Initializer.init(" in inspect.getsource(Flux2KleinEdit.__init__)


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
        assert "_Flux2KleinEditHelpers.prepare_generation_latents(" in inspect.getsource(Flux2KleinEdit.generate_image)

    def test_helper_and_replacement_share_one_keyword_signature(self):
        assert _params(_Flux2KleinEditHelpers.prepare_generation_latents) == ["seed", "height", "width"]

    def test_klein_builds_img2img_latents_from_a_config(self):
        assert _params(Flux2Klein._prepare_img2img_latents)[1:] == ["seed", "config"]
