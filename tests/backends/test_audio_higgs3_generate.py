"""Tests for the Higgs TTS 3 generation loop: stop, cut, empty takes, ``on_frame`` and a tiny random backbone."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from tests.backends._helpers import TINY, tiny_random_model
from zvisiongenerator.backends.audio import higgs3_generate
from zvisiongenerator.backends.audio.higgs3_codes import delay_codes
from zvisiongenerator.backends.audio.higgs3_generate import SpeechCodes, generate_speech_codes
from zvisiongenerator.backends.audio.higgs3_prompt import AUDIO_SLOT, SpecialIds, build_prompt_ids

TTS, REF_TEXT, REF_AUDIO, TEXT, AUDIO = 500, 501, 502, 503, 504
IDS = SpecialIds(tts=TTS, ref_text=REF_TEXT, ref_audio=REF_AUDIO, text=TEXT, audio=AUDIO)
N, V = 4, 10
BOC, EOC = V - 2, V - 1


class FakeTokenizer:
    """Encodes a string to its character codes."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) for char in text]


class TinyTokenizer:
    """Encodes a string to ids inside the tiny model's text vocabulary."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) % 50 for char in text]


TINY_IDS = SpecialIds(tts=60, ref_text=61, ref_audio=62, text=63, audio=59)


class ScriptedHead:
    """A head whose logits make the sampler (at temperature 0) pick the codes ``script(call_index)`` returns."""

    num_codebooks = N
    codebook_size = V

    def __init__(self, script: Callable[[int], list[int]]) -> None:
        self.script = script
        self.calls = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(1, N, V)
        for codebook, code in enumerate(self.script(self.calls)):
            logits[0, codebook, code] = 5.0
        self.calls += 1
        return logits


class ScriptedModel:
    """A model whose head follows a script, and which records what the loop asks of it."""

    def __init__(self, script: Callable[[int], list[int]]) -> None:
        self.audio_head = ScriptedHead(script)
        self.prompts: list[list[int]] = []
        self.rows: list[torch.Tensor | None] = []
        self.steps: list[torch.Tensor] = []

    def embed_prompt(self, prompt_ids: list[int], reference_rows: torch.Tensor | None) -> torch.Tensor:
        self.prompts.append(prompt_ids)
        self.rows.append(reference_rows)
        return torch.zeros(1, len(prompt_ids), 4)

    def prefill(self, embeds: torch.Tensor) -> tuple[torch.Tensor, str]:
        return torch.zeros(1, 4), "cache"

    def step(self, codes: torch.Tensor, cache: str) -> torch.Tensor:
        assert cache == "cache"
        self.steps.append(codes.clone())
        return torch.zeros(1, 4)


def _normal(index: int) -> list[int]:
    """Return frame *index* of a take that never ends: codes below the stream markers."""
    return [(index + codebook) % 7 for codebook in range(N)]


def _ending_at(k: int) -> Callable[[int], list[int]]:
    """Return a script whose codebook 0 is EOC at iteration *k*."""
    return lambda index: [EOC, *_normal(index)[1:]] if index == k else _normal(index)


def _run(model: object, *, cap: int = 100, on_frame: Callable[[], None] | None = None, **overrides: object) -> SpeechCodes:
    arguments = {"reference_codes": None, "reference_text": None, "max_new_tokens": cap, "temperature": 0.0, "top_p": 1.0, "on_frame": on_frame or (lambda: None)}
    arguments.update(overrides)
    return generate_speech_codes(model, FakeTokenizer(), IDS, "hi", **arguments)  # type: ignore[arg-type]


class TestStop:
    @pytest.mark.parametrize("k", [N, N + 1, 9, 20])
    def test_an_eoc_in_codebook_zero_ends_the_take_after_the_countdown(self, k: int) -> None:
        model = ScriptedModel(_ending_at(k))
        frames = []

        result = _run(model, on_frame=lambda: frames.append(1))

        assert result.cut is False
        assert result.codes.shape == (k - 1, N)
        assert len(frames) == k + N - 1
        assert model.audio_head.calls == k + N - 1
        assert len(model.steps) == k + N - 2  # the iteration that ends the take is not kept

    def test_the_kept_codes_are_the_sampled_ones_after_the_ramp_forcing(self) -> None:
        model = ScriptedModel(_ending_at(9))

        result = _run(model)

        assert result.codes.dtype == torch.long
        assert result.codes.device.type == "cpu"
        # Codebook 0 is never delayed: its frames are the first T sampled rows.
        assert result.codes[:, 0].tolist() == [_normal(i)[0] for i in range(8)]
        # The ramp wrote BOC into the later codebooks of the first rows; the delay pattern puts them before the data, so none survives.
        assert not (result.codes == BOC).any()
        assert model.steps[0].tolist() == [_normal(0)[0], BOC, BOC, BOC]

    def test_an_eoc_during_the_ramp_is_ignored(self) -> None:
        model = ScriptedModel(_ending_at(1))

        result = _run(model, cap=30)

        assert result.cut is True
        assert result.codes.shape[0] == 30 - N + 1


class TestCut:
    def test_running_out_of_iterations_keeps_every_row_and_is_cut(self) -> None:
        model = ScriptedModel(_normal)
        frames = []

        result = _run(model, cap=12, on_frame=lambda: frames.append(1))

        assert result.cut is True
        assert result.codes.shape == (12 - N + 1, N)
        assert len(frames) == 12
        assert len(model.steps) == 12

    def test_a_cap_below_the_codebook_count_gives_no_sound(self) -> None:
        result = _run(ScriptedModel(_normal), cap=N - 1)

        assert result.codes.shape == (0, N)
        assert result.cut is True

    def test_a_take_that_finishes_on_the_last_allowed_iteration_is_not_cut(self) -> None:
        k = 6
        done_at = k + N - 2  # iteration index at which the countdown reaches zero

        result = _run(ScriptedModel(_ending_at(k)), cap=done_at + 1)

        assert result.cut is False
        assert result.codes.shape == (k - 1, N)


class TestOnFrame:
    def test_an_exception_propagates_and_no_further_step_runs(self) -> None:
        model = ScriptedModel(_normal)
        calls = []

        def stop() -> None:
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt("stop")

        with pytest.raises(KeyboardInterrupt):
            _run(model, on_frame=stop)

        assert len(calls) == 3
        assert len(model.steps) == 2

    def test_it_runs_after_the_head_and_before_the_sample(self) -> None:
        model = ScriptedModel(_normal)
        seen = []

        _run(model, cap=3, on_frame=lambda: seen.append((model.audio_head.calls, len(model.steps))))

        assert seen == [(1, 0), (2, 1), (3, 2)]

    def test_it_runs_with_gradients_off(self) -> None:
        enabled = []

        _run(ScriptedModel(_normal), cap=2, on_frame=lambda: enabled.append(torch.is_grad_enabled()))

        assert enabled == [False, False]


class TestPrompt:
    def test_no_reference_passes_no_rows(self) -> None:
        model = ScriptedModel(_normal)

        _run(model, cap=1)

        assert model.rows == [None]
        assert model.prompts == [[TTS, TEXT, ord("h"), ord("i"), AUDIO]]

    def test_the_reference_is_passed_delayed_with_one_slot_per_row(self) -> None:
        reference = torch.randint(0, BOC, (5, N))
        model = ScriptedModel(_normal)

        _run(model, cap=1, reference_codes=reference, reference_text=None)

        (rows,) = model.rows
        assert torch.equal(rows, delay_codes(reference, BOC, EOC))
        assert rows.shape == (5 + N - 1, N)
        assert model.prompts[0].count(AUDIO_SLOT) == 5 + N - 1
        assert REF_TEXT not in model.prompts[0]

    def test_the_transcript_adds_ref_text_only_with_a_reference(self) -> None:
        reference = torch.randint(0, BOC, (2, N))
        with_reference = ScriptedModel(_normal)
        without = ScriptedModel(_normal)

        _run(with_reference, cap=1, reference_codes=reference, reference_text="yo")
        _run(without, cap=1, reference_text="yo")

        assert with_reference.prompts[0][:4] == [TTS, REF_TEXT, ord("y"), ord("o")]
        assert REF_TEXT not in without.prompts[0]


class TestSamplingArguments:
    def test_temperature_and_top_p_reach_the_sampler(self) -> None:
        seen = []

        def spy(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
            seen.append((tuple(logits.shape), logits.dtype, temperature, top_p))
            return logits.argmax(dim=-1)

        with patch.object(higgs3_generate, "sample_codebooks", spy):
            _run(ScriptedModel(_normal), cap=2, temperature=0.7, top_p=0.95)

        assert seen == [((N, V), torch.float32, 0.7, 0.95)] * 2


class TestTinyBackbone:
    """A random 2-layer backbone on the CPU in fp32: the cache and the positions must agree with a full forward."""

    def _run_recording(self, seed: int = 0, cap: int = 12) -> tuple[SpeechCodes, list[torch.Tensor], list[torch.Tensor], object]:
        model = tiny_random_model(seed)
        logits: list[torch.Tensor] = []
        fed: list[torch.Tensor] = []
        model.audio_head.register_forward_hook(lambda _m, _i, output: logits.append(output.clone()))
        real_step = model.step

        def recording_step(codes: torch.Tensor, cache: object) -> torch.Tensor:
            fed.append(codes.clone())
            return real_step(codes, cache)

        model.step = recording_step  # type: ignore[method-assign]
        result = generate_speech_codes(
            model,
            TinyTokenizer(),
            TINY_IDS,
            "hi",
            reference_codes=None,
            reference_text=None,
            max_new_tokens=cap,
            temperature=0.0,
            top_p=1.0,
            on_frame=lambda: None,
        )
        return result, logits, fed, model

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_last_logits_equal_a_forward_without_the_cache(self, seed: int) -> None:
        _, logits, fed, model = self._run_recording(seed)
        prompt = build_prompt_ids(TinyTokenizer(), TINY_IDS, "hi", reference_rows=0, reference_text=None)  # type: ignore[arg-type]
        last = len(logits) - 1  # the last head call read the hidden state after `last` cached steps

        with torch.no_grad():
            sequence = torch.cat([model.embed_prompt(prompt, None), model.audio_embedding(torch.stack(fed[:last])).unsqueeze(0)], dim=1)
            hidden = model.backbone(inputs_embeds=sequence).last_hidden_state[:, -1, :]
            expected = model.audio_head(hidden)

        assert last >= TINY.num_codebooks
        assert torch.allclose(logits[last], expected, atol=1e-5)
