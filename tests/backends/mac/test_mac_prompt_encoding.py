"""Tests for encoding a job's prompts ahead on macOS and freeing the text encoder."""

from __future__ import annotations

import gc
import sys
import weakref

import pytest

from zvisiongenerator.backends.mac import prompt_encoding as pe
from zvisiongenerator.core import JobPrompt


def _prompt(text: str, negative: str | None = None) -> JobPrompt:
    return JobPrompt(text, negative)


def _prepare(model, family, prompts, *, width=64, height=64, cancelled=None) -> None:
    pe.prepare_prompts(model, family, prompts, width=width, height=height, cancelled=cancelled)


@pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")
class TestMac:
    @pytest.fixture(autouse=True)
    def _mlx(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mflux")

    @pytest.fixture
    def fakes(self):
        import tests._prompt_encoding_fakes as fakes

        return fakes

    @pytest.fixture
    def events(self, monkeypatch):
        """Record the order of evaluate and free steps."""
        log: list[str] = []
        real_evaluate = pe.evaluate
        monkeypatch.setattr(pe, "evaluate", lambda outputs: (log.append("eval"), real_evaluate(outputs))[1])
        monkeypatch.setattr(pe, "release_memory", lambda: log.append("free"))
        return log

    @staticmethod
    def _parameter_names(model) -> list[str]:
        from mlx.utils import tree_flatten

        return [name for name, _ in tree_flatten(model.parameters())]

    # -- the stand-in -------------------------------------------------------------------------------------------------

    def test_replay_records_while_attached_and_replays_by_input_content_after_release(self, fakes):
        import mlx.core as mx

        encoder = fakes.FakeEncoder()
        replay = pe._TextEncoderReplay(encoder)
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        first = replay(ids, mx.ones_like(ids))
        replay.release()
        again = replay(mx.array([[1, 2, 3]], dtype=mx.int32), mx.ones_like(ids))

        assert again is first
        assert encoder.log.calls == ["__call__"]

    def test_replay_misses_after_release_raise_a_runtime_error(self, fakes):
        import mlx.core as mx

        replay = pe._TextEncoderReplay(fakes.FakeEncoder())
        replay.release()
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        with pytest.raises(RuntimeError):
            replay.get_prompt_embeds(input_ids=ids, attention_mask=mx.ones_like(ids), hidden_state_layers=(9, 18, 27))

    def test_replay_separates_calls_that_differ_in_method_dtype_shape_or_keyword(self, fakes):
        import mlx.core as mx

        encoder = fakes.FakeEncoder()
        replay = pe._TextEncoderReplay(encoder)
        ids = mx.array([[1, 2]], dtype=mx.int32)
        replay(ids)
        replay.get_prompt_embeds(ids)
        replay.get_prompt_embeds(ids, hidden_state_layers=(1,))
        replay(ids.astype(mx.int64))
        replay(mx.array([[1], [2]], dtype=mx.int32))
        replay.release()

        assert len(encoder.log.calls) == 5
        with pytest.raises(RuntimeError):
            replay(mx.array([[2, 1]], dtype=mx.int32))

    # -- the families that replay -------------------------------------------------------------------------------------

    def test_klein_serves_every_planned_text_after_the_encoder_is_freed(self, fakes, events):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeKlein(encoder)
        reference = fakes.FakeKlein(fakes.FakeEncoder())
        expected = reference._encode_prompt_pair(prompt="a cat", negative_prompt=" ", guidance=2.0)
        assert "text_encoder.weight" in self._parameter_names(model)

        _prepare(model, "flux2_klein", [_prompt("a cat"), _prompt("a dog"), _prompt("a cat")])

        assert encoder.log.calls == ["get_prompt_embeds"] * 3
        assert self._parameter_names(model) == []
        got = model._encode_prompt_pair(prompt="a cat", negative_prompt=" ", guidance=2.0)
        assert [bool((g == e).all().item()) for g, e in zip(got[::2], expected[::2], strict=True)] == [True, True]
        assert encoder.log.calls == ["get_prompt_embeds"] * 3

    @staticmethod
    def _encoded_texts(model, method: str) -> list[str]:
        """Record the texts *model* is asked to encode through *method* while still encoding them."""
        texts: list[str] = []
        real = getattr(model, method)

        def spy(*, prompt, **kwargs):
            texts.append(prompt)
            return real(prompt=prompt, **kwargs)

        setattr(model, method, spy)
        return texts

    def test_klein_encodes_each_prompt_and_the_blank_once_in_first_seen_order(self, fakes, events):
        model = fakes.FakeKlein(fakes.FakeEncoder())
        texts = self._encoded_texts(model, "_encode_prompt_pair")

        _prepare(model, "flux2_klein", [_prompt("b"), _prompt("a"), _prompt("b", "ignored")])

        assert texts == ["b", "a", " "]

    @pytest.mark.parametrize(("family", "make"), [("zimage", "FakeZImage"), ("krea2", "FakeKrea")])
    def test_negatives_are_encoded_as_written_and_blank_ones_are_left_to_the_blank(self, fakes, events, family, make):
        model = getattr(fakes, make)(fakes.FakeEncoder())
        texts = self._encoded_texts(model, "_encode_prompts")
        prompts = [_prompt("a", " bad "), _prompt("b", "   "), _prompt("c", ""), _prompt("d", None), _prompt("e", "a")]

        _prepare(model, family, prompts)

        assert texts == ["a", "b", "c", "d", "e", " bad ", " "]

    def test_klein_references_view_reads_the_same_replay(self, fakes, events):
        from zvisiongenerator.backends.mac.klein_references import reference_view

        model = fakes.FakeKlein(fakes.FakeEncoder())
        _prepare(model, "flux2_klein", [_prompt("a cat")])
        view = reference_view(model)

        embeds, ids, negative, negative_ids = view._encode_prompt_pair(prompt="a cat", negative_prompt=" ", guidance=1.5)

        assert embeds.shape == (1, 5, fakes.FEATURES)
        assert negative.shape == (1, 1, fakes.FEATURES)
        with pytest.raises(RuntimeError):
            view._encode_prompt_pair(prompt="other", negative_prompt=None, guidance=1.0)

    def test_zimage_serves_prompts_negatives_and_the_blank_negative(self, fakes, events):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeZImage(encoder)

        _prepare(model, "zimage", [_prompt("a cat", "blurry"), _prompt("a dog", "  ")])

        assert encoder.log.calls == ["__call__"] * 4
        model._encode_prompts(prompt="a cat", negative_prompt="blurry", guidance=2.0)
        model._encode_prompts(prompt="a dog", negative_prompt="  ", guidance=2.0)
        model._encode_prompts(prompt="a dog", negative_prompt=None, guidance=0.0)
        assert encoder.log.calls == ["__call__"] * 4
        with pytest.raises(RuntimeError):
            model._encode_prompts(prompt="a cat", negative_prompt="other", guidance=2.0)

    def test_krea_misses_in_mflux_cache_fall_through_to_the_replay(self, fakes, events):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeKrea(encoder)
        direct = fakes.FakeKrea(fakes.FakeEncoder())._encode_prompts(prompt="a cat", negative_prompt="blurry", guidance=3.0)

        _prepare(model, "krea2", [_prompt("a cat", "blurry")])
        calls_after_prepare = list(encoder.log.calls)
        embeds, negative = model._encode_prompts(prompt="a cat", negative_prompt="blurry", guidance=3.0)

        assert encoder.log.calls == calls_after_prepare
        assert bool((embeds == direct[0]).all().item()) and bool((negative == direct[1]).all().item())

    def test_each_text_is_evaluated_after_it_is_encoded_and_before_the_encoder_is_released(self, fakes, monkeypatch):
        log: list[str] = []
        encoder = fakes.FakeEncoder(log=fakes.CallLog())
        model = fakes.FakeKlein(encoder)
        monkeypatch.setattr(pe, "evaluate", lambda output: (log.append(f"eval:{len(encoder.log.calls)}"), output)[1])

        def free() -> None:
            with pytest.raises(RuntimeError):
                model.text_encoder.get_prompt_embeds("not recorded")
            log.append("free:released")

        monkeypatch.setattr(pe, "release_memory", free)

        _prepare(model, "flux2_klein", [_prompt("a"), _prompt("b")])

        assert log == ["eval:1", "eval:2", "eval:3", "free:released"]

    @pytest.mark.parametrize(("family", "make"), [("zimage", "FakeZImage"), ("flux2_klein", "FakeKlein"), ("krea2", "FakeKrea"), ("ideogram4", "FakeIdeogram")])
    def test_the_real_encoder_is_unreachable_when_memory_is_returned(self, fakes, monkeypatch, family, make):
        encoder = fakes.FakeEncoder()
        model = getattr(fakes, make)(encoder)
        ref = weakref.ref(encoder)
        del encoder
        reachable_when_freed: list[bool] = []

        def free() -> None:
            gc.collect()
            reachable_when_freed.append(ref() is not None)

        monkeypatch.setattr(pe, "release_memory", free)

        _prepare(model, family, [_prompt("a")], width=256, height=256)

        assert reachable_when_freed == [False]

    # -- Ideogram 4 ---------------------------------------------------------------------------------------------------

    def test_ideogram_fills_its_prompt_cache_and_drops_the_encoder(self, fakes, events):
        from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder

        encoder = fakes.FakeEncoder()
        model = fakes.FakeIdeogram(encoder)
        log = encoder.log
        ref = weakref.ref(encoder)
        del encoder

        _prepare(model, "ideogram4", [_prompt("a cat"), _prompt("a dog"), _prompt("a cat", "n")], width=1024, height=768)
        gc.collect()

        assert model.text_encoder is None
        assert ref() is None
        assert sorted((width, height) for _caption, width, height in model.prompt_cache) == [(1024, 768)] * 2
        assert len(log.calls) == 2  # one encode per distinct prompt
        (caption, width, height) = next(iter(model.prompt_cache))
        assert (
            Ideogram4PromptEncoder.encode_prompt(prompt=caption, width=width, height=height, inputs={}, text_encoder=None, prompt_cache=model.prompt_cache)
            is model.prompt_cache[(caption, width, height)]
        )
        with pytest.raises(RuntimeError):
            Ideogram4PromptEncoder.encode_prompt(prompt="other", width=64, height=64, inputs={}, text_encoder=None, prompt_cache=model.prompt_cache)

    def test_ideogram_skips_an_input_mflux_rejects_and_keeps_going(self, fakes, events, monkeypatch):
        from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder

        real_resolve = Ideogram4PromptEncoder.resolve_prompt

        def resolve(prompt, **kwargs):
            if "bad" in str(prompt):
                raise ValueError("prompt must not be empty")
            return real_resolve(prompt, **kwargs)

        monkeypatch.setattr(Ideogram4PromptEncoder, "resolve_prompt", staticmethod(resolve))
        model = fakes.FakeIdeogram(fakes.FakeEncoder())

        _prepare(model, "ideogram4", [_prompt("bad"), _prompt("ok")], width=1024, height=1024)

        assert model.text_encoder is None
        assert [key[1:] for key in model.prompt_cache] == [(1024, 1024)]

    def test_ideogram_skips_a_size_mflux_rejects(self, fakes, events):
        model = fakes.FakeIdeogram(fakes.FakeEncoder())

        _prepare(model, "ideogram4", [_prompt("a cat")], width=100, height=1024)

        assert model.text_encoder is None
        assert model.prompt_cache == {}

    # -- Quit, errors and no-ops --------------------------------------------------------------------------------------

    @pytest.mark.parametrize("cancel_after", [0, 1])
    def test_quit_restores_the_encoder(self, fakes, events, cancel_after):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeKrea(encoder)
        polls = iter(range(100))

        _prepare(model, "krea2", [_prompt("a"), _prompt("b")], cancelled=lambda: next(polls) >= cancel_after)

        assert model.text_encoder is encoder
        assert "text_encoder.weight" in self._parameter_names(model)
        assert "free" not in events

    def test_quit_during_ideogram_restores_the_encoder(self, fakes, events):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeIdeogram(encoder)
        polls = iter(range(100))

        _prepare(model, "ideogram4", [_prompt("a"), _prompt("b")], width=256, height=256, cancelled=lambda: next(polls) >= 1)

        assert model.text_encoder is encoder
        assert "free" not in events

    def test_an_unexpected_error_restores_the_encoder_and_warns(self, fakes, events):
        encoder = fakes.FakeEncoder(fail_on_call=2)
        model = fakes.FakeKrea(encoder)

        with pytest.warns(UserWarning) as caught:
            _prepare(model, "krea2", [_prompt("a"), _prompt("b")])

        assert "encoder failed" in str(caught[0].message)  # the cause is reported
        assert model.text_encoder is encoder
        assert "free" not in events

    @pytest.mark.parametrize(("family", "prompts"), [("moss", [_prompt("a")]), ("zimage", [])])
    def test_unknown_families_and_empty_prompts_leave_the_model_alone(self, fakes, events, family, prompts):
        encoder = fakes.FakeEncoder()
        model = fakes.FakeZImage(encoder)

        _prepare(model, family, prompts)

        assert model.text_encoder is encoder
        assert encoder.log.calls == []
        assert events == []
