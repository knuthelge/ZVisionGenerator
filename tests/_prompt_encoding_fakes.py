"""Fake mflux models, text encoders and tokenizers for the macOS prompt-encoding tests.

The fakes run mflux's real prompt-encoder code (and its real ``_encode_prompt_pair`` / ``_encode_prompts`` methods) against
tiny MLX arrays, so no weights are loaded. Import this module only after ``pytest.importorskip("mlx")`` and
``pytest.importorskip("mflux")``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.krea2 import Krea2
from mflux.models.z_image.variants.z_image import ZImage
from mlx import nn

FEATURES = 4


class CallLog:
    """Record the order of calls made to a fake encoder (a plain object, so it stays out of the module tree)."""

    def __init__(self) -> None:
        self.calls: list[Any] = []


class FakeTokenizer:
    """Tokenise a string to one id per character."""

    max_length = 512

    def tokenize(self, prompt: str, max_length: int | None = None, keep_empty: bool = False) -> Any:
        ids = mx.array([[ord(char) for char in prompt] or [0]], dtype=mx.int32)
        return SimpleNamespace(input_ids=ids, attention_mask=mx.ones_like(ids))

    def tokenize_one(self, prompt: str) -> np.ndarray:
        return np.array([ord(char) for char in prompt] or [0], dtype=np.int64)


class FakeEncoder(nn.Module):
    """A text encoder with one parameter whose outputs depend on the token ids alone."""

    def __init__(self, log: CallLog | None = None, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.weight = mx.ones((FEATURES,))
        self.log = log or CallLog()
        self.fail_on_call = fail_on_call

    def _embed(self, name: str, input_ids: mx.array) -> mx.array:
        self.log.calls.append(name)
        if self.fail_on_call is not None and len(self.log.calls) == self.fail_on_call:
            raise ValueError("encoder failed")
        return input_ids.astype(mx.float32)[..., None] * self.weight

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self._embed("__call__", input_ids)

    def get_prompt_embeds(self, input_ids: mx.array, attention_mask: mx.array | None = None, *args: Any, **kwargs: Any) -> mx.array:
        return self._embed("get_prompt_embeds", input_ids)


class _Model(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer_name: str) -> None:
        super().__init__()
        self.text_encoder = encoder
        self.tokenizers = {tokenizer_name: FakeTokenizer()}
        self.prompt_cache = {}
        # What Flux2Initializer sets, for Flux2Klein.reference_view-style sharing.
        self.model_config = "config"
        self.callbacks = "callbacks"
        self.tiling_config = None
        self.vae = "vae"
        self.transformer = "transformer"
        self.bits = None
        self.small_text_encoder = False
        self.small_decoder = False
        self.lora_paths = None
        self.lora_scales = None
        self.compute_precision = "precision"


class FakeKlein(_Model):
    """A model with mflux's real ``Flux2Klein._encode_prompt_pair``."""

    _encode_prompt_pair = Flux2Klein._encode_prompt_pair

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder, "qwen3")


class FakeKleinEdit(_Model):
    """A model with mflux's real ``Flux2KleinEdit._encode_prompt_pair``."""

    _encode_prompt_pair = Flux2KleinEdit._encode_prompt_pair

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder, "qwen3")


class FakeZImage(_Model):
    """A model with mflux's real ``ZImage._encode_prompts``."""

    _encode_prompts = ZImage._encode_prompts

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder, "z_image")


class FakeKrea(_Model):
    """A model with mflux's real ``Krea2._encode_prompts``."""

    _encode_prompts = Krea2._encode_prompts

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder, "qwen3vl")


class FakeIdeogram(_Model):
    """A model shaped like ``Ideogram4`` for the prompt-encoding steps."""

    def __init__(self, encoder: nn.Module | None) -> None:
        super().__init__(encoder, "ideogram4")  # type: ignore[arg-type]
