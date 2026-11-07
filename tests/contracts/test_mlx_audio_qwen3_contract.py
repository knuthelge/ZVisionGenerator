"""Pin the mlx-audio names the Qwen3-TTS MLX backend calls, so an mlx-audio upgrade that moves them fails here.

The backend loads a checkpoint with ``mlx_audio.tts.utils.load`` and reads each part with ``Model.generate``; this lists the
parameters it passes and the result fields it reads, in the locked wheel. Runs on macOS only, where mlx-audio is installed.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mlx-audio is macOS-only")
pytest.importorskip("mlx_audio")

from mlx_audio.tts import utils  # noqa: E402
from mlx_audio.tts.models.base import GenerationResult  # noqa: E402
from mlx_audio.tts.models.qwen3_tts import qwen3_tts  # noqa: E402


def test_the_loader_takes_a_folder():
    assert next(iter(inspect.signature(utils.load).parameters)) == "model_path"


def test_generate_takes_every_argument_the_backend_passes():
    parameters = inspect.signature(qwen3_tts.Model.generate).parameters

    assert {"text", "voice", "instruct", "lang_code", "ref_audio", "ref_text", "split_pattern", "max_tokens"} <= set(parameters)


def test_generate_is_a_generator_of_results():
    assert inspect.isgeneratorfunction(qwen3_tts.Model.generate)


def test_a_result_holds_the_audio_and_the_token_count():
    assert {"audio", "token_count"} <= {field.name for field in dataclasses.fields(GenerationResult)}


def test_the_global_random_key_can_be_seeded():
    import mlx.core as mx

    assert callable(mx.random.seed)


def _routed_calls(model_type: str, **arguments) -> list[tuple[str, dict]]:
    """Call the real ``Model.generate`` on a stand-in model of *model_type* and return which branch it took, with that branch's arguments."""
    taken: list[tuple[str, dict]] = []

    def branch(name: str):
        def record(**kwargs):
            taken.append((name, kwargs))
            return iter(())

        return record

    model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type=model_type),
        sample_rate=24000,
        generate_config=None,
        speech_tokenizer=SimpleNamespace(has_encoder=True),
        supported_speakers=["ryan"],
        generate_custom_voice=branch("custom_voice"),
        _generate_icl=branch("icl"),
    )
    model._resolve_generation_options = lambda *options: qwen3_tts.Model._resolve_generation_options(model, *options)
    list(qwen3_tts.Model.generate(model, **arguments))
    return taken


def test_a_customvoice_checkpoint_reads_the_voice_in_one_piece_whatever_the_split_pattern():
    ((branch, kwargs),) = _routed_calls("custom_voice", text="Hello.", voice="ryan", instruct=None, lang_code="auto", split_pattern="", max_tokens=7)

    assert branch == "custom_voice"
    assert (kwargs["speaker"], kwargs["language"], kwargs["instruct"], kwargs["max_tokens"]) == ("ryan", "auto", None, 7)


def test_a_base_checkpoint_with_a_recording_follows_it_and_takes_no_voice():
    import mlx.core as mx

    wave = mx.zeros((24000,))

    ((branch, kwargs),) = _routed_calls("base", text="Hello.", voice=None, ref_audio=wave, ref_text="Hi there.", lang_code="auto", split_pattern="", max_tokens=7)

    assert branch == "icl"
    assert (kwargs["ref_text"], kwargs["language"], kwargs["max_tokens"]) == ("Hi there.", "auto", 7) and kwargs["ref_audio"] is wave


def test_the_loader_converts_the_official_convolution_layout_and_keeps_the_mlx_one():
    """The official ``Qwen/…`` snapshots hold PyTorch ``[out, in, kernel]`` convolutions; the MLX repos hold ``[out, kernel, in]``."""
    import mlx.core as mx

    official = {
        "speaker_encoder.blocks.0.conv.weight": mx.zeros((512, 128, 5)),
        "speaker_encoder.blocks.1.se_block.conv1.weight": mx.zeros((128, 512, 1)),
        "speaker_encoder.asp.tdnn.conv.weight": mx.zeros((128, 4608, 1)),
        "talker.model.layers.0.mlp.gate_proj.weight": mx.zeros((8, 4)),
    }

    converted = qwen3_tts.Model.sanitize(official)
    again = qwen3_tts.Model.sanitize(converted)

    assert {key: tuple(value.shape) for key, value in converted.items()} == {
        "speaker_encoder.blocks.0.conv.weight": (512, 5, 128),
        "speaker_encoder.blocks.1.se_block.conv1.weight": (128, 1, 512),
        "speaker_encoder.asp.tdnn.conv.weight": (128, 1, 4608),
        "talker.model.layers.0.mlp.gate_proj.weight": (8, 4),
    }
    assert {key: tuple(value.shape) for key, value in again.items()} == {key: tuple(value.shape) for key, value in converted.items()}
