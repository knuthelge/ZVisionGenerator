"""Pin the mlx-audio names the Higgs TTS 3 MLX backend calls, so an mlx-audio upgrade that moves them fails here.

The backend runs mlx-audio's ``generate`` loop itself, for progress and Stop on every frame, so it reaches into the
model's parts and the sampler; this lists every one of them with the arguments it passes, and checks that the loop samples
the same rows as mlx-audio's own ``generate``.
"""

from __future__ import annotations

import inspect
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mlx-audio is macOS-only")
pytest.importorskip("mlx_audio")

from mlx_audio.lm.models import cache  # noqa: E402
from mlx_audio.tts import utils  # noqa: E402
from mlx_audio.tts.models.higgs_audio_v3 import config, generation, model, prompt  # noqa: E402


def _parameters(function) -> list[str]:
    return [name for name in inspect.signature(function).parameters if name != "self"]


def test_the_loader_and_the_prompt_cache():
    assert _parameters(utils.load)[0] == "model_path"
    assert _parameters(cache.make_prompt_cache)[0] == "model"


def test_the_sampler_step_and_its_state():
    assert {"temperature", "top_p", "top_k", "boc_id", "eoc_id"} <= set(_parameters(generation.step))
    state = generation.HiggsSamplerState(num_codebooks=8)
    assert state.generation_done is False


def test_the_reference_holds_codes_and_a_transcript():
    reference = prompt.ReferenceCodes(codes="codes", text="What it says.")
    assert (reference.codes, reference.text) == ("codes", "What it says.")


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("_build_prompt_embeddings", ["text", "references"]),
        ("_audio_logits", ["hidden"]),
        ("_embed_audio_codes", ["codes"]),
        ("_decode_audio", ["delayed_rows"]),
        ("encode_reference_audio", ["audio"]),
    ],
)
def test_the_model_parts_the_loop_calls(name, parameters):
    assert _parameters(getattr(model.Model, name)) == parameters


def test_the_backbone_takes_input_embeddings_and_a_cache():
    from mlx_audio.lm.models.qwen3 import Qwen3Model

    assert {"cache", "input_embeddings"} <= set(_parameters(Qwen3Model.__call__))


def test_the_config_names_the_codebooks_and_the_stream_markers():
    settings = config.HiggsAudioV3Config()
    assert (settings.audio_num_codebooks, settings.audio_boc_token_id, settings.audio_eoc_token_id, settings.sample_rate) == (8, 1024, 1025, 24000)


class _NoisyModel:
    """A stand-in for the model with mlx-audio's own ``generate``: scripted noisy logits, so sampling and its seed matter, and an end after a dozen frames."""

    config = config.HiggsAudioV3Config(audio_num_codebooks=4)
    layers: list = []
    sample_rate = 24000
    generate = model.Model.generate
    _resolve_generation_limit = model.Model._resolve_generation_limit
    _normalize_references = model.Model._normalize_references
    _apply_fades = model.Model._apply_fades

    def __init__(self) -> None:
        import numpy as np

        size = self.config.audio_codebook_size
        self.script = np.random.default_rng(0).normal(scale=2.0, size=(40, 1, 4, size)).astype(np.float32)
        self.script[12:, 0, 0, self.config.audio_eoc_token_id] = 50.0
        self.calls = 0
        self.decoded: list = []

    def _build_prompt_embeddings(self, text, references):
        import mlx.core as mx

        self.calls = 0
        return mx.zeros((1, 3, 4)), 3

    def backbone(self, ids, cache=None, input_embeddings=None):
        import mlx.core as mx

        return mx.zeros((1, input_embeddings.shape[1], 4))

    def _audio_logits(self, hidden):
        import mlx.core as mx

        self.calls += 1
        return mx.array(self.script[self.calls - 1])

    def _embed_audio_codes(self, codes):
        import mlx.core as mx

        return mx.zeros((1, 4))

    def _decode_audio(self, rows):
        import mlx.core as mx

        self.decoded.append([row.tolist() for row in rows])
        return mx.zeros((len(rows) * 960,))


def test_the_backends_frame_loop_samples_the_rows_mlx_audios_generate_decodes():
    from zvisiongenerator.backends.audio.higgs3_mlx import _generate_rows

    noisy = _NoisyModel()
    ours = {}
    for seed in (1, 2):
        list(noisy.generate("Hello there.", seed=seed, temperature=0.7, top_p=0.95, max_new_tokens=30))
        rows, cut = _generate_rows(noisy, "Hello there.", None, seed=seed, max_new_tokens=30, temperature=0.7, top_p=0.95, on_frame=lambda: None)
        ours[seed] = [row.tolist() for row in rows]
        assert cut is False

    assert [ours[1], ours[2]] == noisy.decoded
    assert ours[1] != ours[2]
