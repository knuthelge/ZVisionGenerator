"""Pin the mflux and MLX internals that ``backends/mac/prompt_encoding.py`` relies on.

Introspection and behaviour on fakes only: no model is constructed and no weights are loaded. A failure here after an
mflux or MLX bump means upstream changed a signature, an encoder call or a cache the prompt pre-encoding depends on.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

import mlx.core as mx  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.flux2 import Flux2Initializer, Flux2Klein  # noqa: E402
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder  # noqa: E402
from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: E402
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer  # noqa: E402
from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator  # noqa: E402
from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder  # noqa: E402
from mflux.models.krea2 import Krea2  # noqa: E402
from mflux.models.krea2.krea2_initializer import Krea2Initializer  # noqa: E402
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder  # noqa: E402
from mflux.models.z_image.variants.z_image import ZImage  # noqa: E402
from mlx import nn  # noqa: E402

from tests._prompt_encoding_fakes import FakeEncoder, FakeIdeogram, FakeKlein, FakeKleinEdit, FakeKrea, FakeZImage  # noqa: E402
from zvisiongenerator.backends.mac.prompt_encoding import _ENCODE_TEXT  # noqa: E402


def _keyword_only(fn) -> list[str]:
    return [name for name, parameter in inspect.signature(fn).parameters.items() if parameter.kind is inspect.Parameter.KEYWORD_ONLY]


@pytest.mark.parametrize("encode", [Flux2Klein._encode_prompt_pair, Flux2KleinEdit._encode_prompt_pair, ZImage._encode_prompts, Krea2._encode_prompts])
def test_the_model_encode_methods_take_the_keywords_prompt_encoding_passes(encode):
    assert _keyword_only(encode) == ["prompt", "negative_prompt", "guidance"]


@pytest.mark.parametrize(("family", "model_class"), [("flux2_klein", Flux2Klein), ("zimage", ZImage), ("krea2", Krea2)])
def test_the_encode_method_table_names_methods_the_models_have(family, model_class):
    assert callable(getattr(model_class, _ENCODE_TEXT[family]))


class _Reached(Exception):
    """Raised by a spy to stop ``generate_image`` at the point it asks the model to encode."""


@pytest.mark.parametrize(
    ("model_class", "method", "model_config", "before_encoding"),
    [
        (Flux2Klein, "_encode_prompt_pair", ModelConfig.flux2_klein_4b(), {}),
        (Flux2KleinEdit, "_encode_prompt_pair", ModelConfig.flux2_klein_4b(), {}),
        (ZImage, "_encode_prompts", ModelConfig.z_image_turbo(), {"vae": None, "tiling_config": None}),
        (Krea2, "_encode_prompts", ModelConfig.krea2(), {"_prepare_latents": lambda **kwargs: mx.zeros((1,))}),
    ],
)
def test_generate_image_encodes_through_the_model_method(model_class, method, model_config, before_encoding):
    def spy(**kwargs):
        raise _Reached(kwargs)

    shell = SimpleNamespace(model_config=model_config, small_decoder=False, **before_encoding, **{method: spy})

    with pytest.raises(_Reached) as reached:
        model_class.generate_image(shell, seed=1, prompt="a cat", num_inference_steps=1, height=64, width=64)

    assert reached.value.args[0]["prompt"] == "a cat"


def _described(value):
    return tuple(value.shape) if isinstance(value, mx.array) else value


class _RecordingEncoder(FakeEncoder):
    """Record each call as ``(method, positional argument shapes, keyword arguments with arrays as shapes)``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, tuple[int, ...], dict[str, object]]] = []

    def _record(self, method, args, kwargs) -> None:
        self.seen.append((method, tuple(_described(a) for a in args), {name: _described(value) for name, value in kwargs.items()}))

    def __call__(self, *args, **kwargs):
        self._record("__call__", args, kwargs)
        return super().__call__(*args, **kwargs)

    def get_prompt_embeds(self, *args, **kwargs):
        self._record("get_prompt_embeds", args, kwargs)
        return super().get_prompt_embeds(*args, **kwargs)


def _kwargs_with_arrays_named(seen) -> list[str]:
    return sorted(seen[0][2])


def test_klein_txt2img_and_edit_call_the_encoder_identically_with_keyword_arguments():
    txt2img, edit = _RecordingEncoder(), _RecordingEncoder()

    FakeKlein(txt2img)._encode_prompt_pair(prompt="a cat", negative_prompt=None, guidance=1.0)
    FakeKleinEdit(edit)._encode_prompt_pair(prompt="a cat", negative_prompt=None, guidance=1.0)

    assert txt2img.seen == edit.seen
    assert len(txt2img.seen) == 1
    assert txt2img.seen[0][0] == "get_prompt_embeds"
    assert _kwargs_with_arrays_named(txt2img.seen) == ["attention_mask", "hidden_state_layers", "input_ids"]
    assert txt2img.seen[0][2]["hidden_state_layers"] == (9, 18, 27)


def test_klein_encodes_the_negative_only_above_guidance_one_and_only_when_given():
    encoder = _RecordingEncoder()
    model = FakeKlein(encoder)

    model._encode_prompt_pair(prompt="a", negative_prompt=" ", guidance=1.0)
    model._encode_prompt_pair(prompt="a", negative_prompt=None, guidance=2.0)
    assert len(encoder.seen) == 2

    model._encode_prompt_pair(prompt="a", negative_prompt=" ", guidance=2.0)
    assert len(encoder.seen) == 4


def test_zimage_calls_the_encoder_with_two_positional_arrays_and_encodes_a_blank_negative_above_guidance_one():
    encoder = _RecordingEncoder()
    model = FakeZImage(encoder)

    model._encode_prompts(prompt="a", negative_prompt=None, guidance=1.0)
    assert [(name, len(shapes), kwargs) for name, shapes, kwargs in encoder.seen] == [("__call__", 2, {})]

    model._encode_prompts(prompt="a", negative_prompt="   ", guidance=2.0)
    assert len(encoder.seen) == 3


def test_krea_calls_the_encoder_with_two_positional_arrays_and_caches_by_prompt_negative_and_guidance():
    encoder = _RecordingEncoder()
    model = FakeKrea(encoder)

    model._encode_prompts(prompt="a", negative_prompt=None, guidance=1.0)
    assert [(name, len(shapes), kwargs) for name, shapes, kwargs in encoder.seen] == [("get_prompt_embeds", 2, {})]
    assert list(model.prompt_cache) == [("a", None, 1.0)]

    model._encode_prompts(prompt="a", negative_prompt=None, guidance=0.0)
    assert len(encoder.seen) == 3
    assert set(model.prompt_cache) == {("a", None, 1.0), ("a", None, 0.0)}
    assert callable(Krea2PromptEncoder.encode_prompt_pair)


def test_flux2_prompt_encoder_asks_a_stand_in_for_prompt_embeds_like_the_full_encoder():
    encoder = _RecordingEncoder()
    tokenizer = FakeKlein(encoder).tokenizers["qwen3"]

    Flux2PromptEncoder.encode_prompt(prompt="a cat", tokenizer=tokenizer, text_encoder=encoder, num_images_per_prompt=1, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))

    assert [method for method, _args, _kwargs in encoder.seen] == ["get_prompt_embeds"]
    assert inspect.signature(Flux2Initializer.init).parameters["small_text_encoder"].default is False


@pytest.mark.parametrize(
    ("initializer", "arguments"),
    [
        (Flux2Initializer, (ModelConfig.flux2_klein_4b(),)),
        (Krea2Initializer, (ModelConfig.krea2(),)),
        (Ideogram4Initializer, (ModelConfig.ideogram4_fp8(), "model_path")),
    ],
)
def test_initializers_start_the_model_with_an_empty_prompt_cache_dict(initializer, arguments):
    model = SimpleNamespace()

    initializer._init_config(model, *arguments)

    assert model.prompt_cache == {}
    assert isinstance(model.prompt_cache, dict)


def test_ideogram_steps_and_signatures():
    assert _keyword_only(Ideogram4PromptEncoder.resolve_prompt) == ["strict_caption_validation", "warn_on_caption_issues"]
    assert list(inspect.signature(Ideogram4PromptEncoder.build_inputs).parameters) == ["tokenizer", "prompts", "height", "width"]
    assert _keyword_only(Ideogram4PromptEncoder.encode_prompt) == ["prompt", "width", "height", "inputs", "text_encoder", "prompt_cache"]
    assert _keyword_only(Ideogram4LatentCreator.validate_dimensions) == ["width", "height"]


def test_ideogram_prompt_cache_is_keyed_by_caption_and_rounded_size_and_a_miss_without_an_encoder_raises():
    model = FakeIdeogram(FakeEncoder())
    caption = Ideogram4PromptEncoder.resolve_prompt("a cat", strict_caption_validation=False, warn_on_caption_issues=False)
    inputs = Ideogram4PromptEncoder.build_inputs(model.tokenizers["ideogram4"], [caption], height=32, width=48)

    features = Ideogram4PromptEncoder.encode_prompt(prompt=caption, width=48, height=32, inputs=inputs, text_encoder=model.text_encoder, prompt_cache=model.prompt_cache)

    assert list(model.prompt_cache) == [(caption, 48, 32)]
    assert Ideogram4PromptEncoder.encode_prompt(prompt=caption, width=48, height=32, inputs={}, text_encoder=None, prompt_cache=model.prompt_cache) is features
    with pytest.raises(RuntimeError):
        Ideogram4PromptEncoder.encode_prompt(prompt="other", width=48, height=32, inputs={}, text_encoder=None, prompt_cache=model.prompt_cache)


def test_ideogram_encode_prompt_evaluates_its_embeddings_before_caching_them(monkeypatch):
    log: list[str] = []

    class _LoggedCache(dict):
        def __setitem__(self, key, value):
            log.append("cache")
            super().__setitem__(key, value)

    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *arrays: (log.append("eval"), real_eval(*arrays))[1])
    model = FakeIdeogram(FakeEncoder())
    caption = Ideogram4PromptEncoder.resolve_prompt("a cat", strict_caption_validation=False, warn_on_caption_issues=False)
    inputs = Ideogram4PromptEncoder.build_inputs(model.tokenizers["ideogram4"], [caption], height=32, width=48)

    Ideogram4PromptEncoder.encode_prompt(prompt=caption, width=48, height=32, inputs=inputs, text_encoder=model.text_encoder, prompt_cache=_LoggedCache())

    assert log == ["eval", "cache"]


def test_ideogram_dimension_errors_are_value_errors_and_config_rounds_down_to_sixteen():
    for width, height in ((100, 1024), (1024, 4096), (1030, 1024)):
        with pytest.raises(ValueError):
            Ideogram4LatentCreator.validate_dimensions(width=width, height=height)

    config = Config(model_config=ModelConfig.ideogram4_fp8(), width=1030, height=1025, scheduler="linear", num_inference_steps=1, guidance=1.0)

    assert (config.width, config.height) == (1024, 1024)


class _Holder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = FakeEncoder()


@pytest.mark.parametrize("replacement", [None, object()])
def test_assigning_a_non_container_to_a_module_attribute_removes_it_from_the_module_tree(replacement):
    holder = _Holder()
    encoder = holder.text_encoder
    assert "text_encoder" in holder.parameters()

    holder.text_encoder = replacement

    assert "text_encoder" not in holder.parameters()
    assert holder.text_encoder is replacement

    holder.text_encoder = encoder
    assert "text_encoder" in holder.parameters()
    assert holder.text_encoder is encoder
