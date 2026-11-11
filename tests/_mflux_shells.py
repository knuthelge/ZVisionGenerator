"""A hand-built mflux FLUX.2 Klein with stub components, so mflux's real denoising loops run without weights.

The shell is the real model class with a stub transformer that predicts a constant velocity, a stub VAE that encodes to
known arrays and decodes to zeros, and a prompt encoder that returns empty embeddings. Import only after
``pytest.importorskip("mflux")``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2 import Flux2Klein
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mlx import nn

# Klein's stub VAE batch-norm statistics: not 0 and 1, so a missing normalisation shows.
BN_MEAN = 0.25
BN_VAR = 4.0
BN_EPS = 0.0


@dataclass(frozen=True)
class Shell:
    """A model ready for ``generate_image``, and the raw array its stub VAE returns from ``encode``."""

    model: nn.Module
    encoded: mx.array


def _base(model: nn.Module, model_config: ModelConfig) -> nn.Module:
    nn.Module.__init__(model)
    model.prompt_cache = {}
    model.model_config = model_config
    model.callbacks = CallbackRegistry()
    model.tiling_config = None
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None
    model.compute_precision = ComputePrecision(None)
    return model


def make_shell(family: str, height: int, width: int, velocity: float = 0.0, key: int = 7) -> Shell:
    """Return a shell of *family* (``flux2_klein``) whose transformer predicts *velocity* everywhere and whose VAE encodes to a seeded normal array.

    ``decode`` returns zeros, so the loops finish and return an image of ``height`` x ``width``.
    """
    if family != "flux2_klein":
        raise ValueError(f"no shell for family {family!r}")
    model = _base(Flux2Klein.__new__(Flux2Klein), ModelConfig.flux2_klein_4b())
    model.small_text_encoder = False
    model.small_decoder = False
    model.tokenizers = {"qwen3": object()}
    model.text_encoder = None
    model.transformer = lambda *, hidden_states, **_: mx.full(hidden_states.shape, velocity)
    encoded = mx.random.normal((1, 32, height // 8, width // 8), key=mx.random.key(key))
    bn = SimpleNamespace(running_mean=mx.full((128,), BN_MEAN), running_var=mx.full((128,), BN_VAR), eps=BN_EPS)
    model.vae = SimpleNamespace(
        encode=lambda image: encoded,
        bn=bn,
        decode_packed_latents=lambda packed, tiling_config=None: mx.zeros((1, 3, height, width)),
    )
    return Shell(model, encoded)


@contextmanager
def stub_prompt_encoders() -> Iterator[None]:
    """Patch Klein's prompt encoder to return empty embeddings."""
    with patch.object(Flux2PromptEncoder, "encode_prompt", staticmethod(lambda **_: (mx.zeros((1, 8, 16)), mx.zeros((1, 8, 4), dtype=mx.int32)))):
        yield
