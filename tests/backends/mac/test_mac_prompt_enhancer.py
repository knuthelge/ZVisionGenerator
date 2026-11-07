"""Tests for the presence penalty on the mlx-lm prompt enhancer (run without MLX, on a numpy stand-in)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from zvisiongenerator.backends.mac import prompt_enhancer

_PROMPT = [5, 6, 7]


class _Counts(np.ndarray):
    """A numpy array with MLX's ``.at[index].add(value)``, which accumulates over repeated indices."""

    @property
    def at(self):
        array = self

        class _At:
            def __getitem__(self, index):
                class _Op:
                    @staticmethod
                    def add(value):
                        out = np.array(array, copy=True).view(_Counts)
                        np.add.at(out, index, value)
                        return out

                return _Op()

        return _At()


@pytest.fixture
def fake_mlx(monkeypatch):
    """Install stand-ins for ``mlx.core`` and ``mlx_lm``; ``stream_generate`` records its keyword arguments."""
    mx = types.SimpleNamespace(
        random=types.SimpleNamespace(seed=MagicMock()),
        zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype).view(_Counts),
        minimum=np.minimum,
    )
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = MagicMock(return_value=iter([types.SimpleNamespace(text="hi")]))
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = MagicMock(return_value="sampler")
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr("zvisiongenerator.backends.mac.memory.release_memory", MagicMock())
    return mlx_lm


def _penalised(processor, generated: list[int], scores: np.ndarray) -> np.ndarray:
    """Run *processor* the way mlx-lm does: once on the prompt, then once after each generated token."""
    processor(np.array(_PROMPT), scores.copy())
    return processor(np.array([*_PROMPT, *generated]), scores.copy())


class TestPresencePenalty:
    def test_generate_passes_the_penalty_to_mlx_lm(self, fake_mlx):
        enhancer = object.__new__(prompt_enhancer.MlxPromptEnhancer)
        enhancer._model = "model"
        enhancer._tokenizer = MagicMock()
        list(enhancer.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.7))
        (processor,) = fake_mlx.stream_generate.call_args.kwargs["logits_processors"]
        out = _penalised(processor, [2], np.zeros((1, 8)))
        assert out[0].tolist() == [0, 0, -1.5, 0, 0, 0, 0, 0]

    def test_only_generated_tokens_are_penalised_once_each(self, fake_mlx):
        scores = np.arange(8, dtype=np.float32)[None, :]
        # The prompt holds 5, 6 and 7; the output so far holds 1 three times and 2 once.
        out = _penalised(prompt_enhancer._presence_penalty_processor(1.5), [1, 2, 1, 1], scores)
        expected = scores.copy()
        expected[0, 1] -= 1.5
        expected[0, 2] -= 1.5
        assert np.array_equal(out, expected)

    def test_nothing_is_penalised_before_the_first_token(self, fake_mlx):
        processor = prompt_enhancer._presence_penalty_processor(1.5)
        scores = np.arange(8, dtype=np.float32)[None, :]
        assert np.array_equal(processor(np.array(_PROMPT), scores.copy()), scores)

    def test_each_request_gets_its_own_processor(self, fake_mlx):
        # A processor remembers where its prompt ended, so a second request must not reuse the first one's.
        enhancer = object.__new__(prompt_enhancer.MlxPromptEnhancer)
        enhancer._model = "model"
        enhancer._tokenizer = MagicMock()
        for _ in range(2):
            fake_mlx.stream_generate.return_value = iter([])
            list(enhancer.generate([], seed=1, max_tokens=1, temperature=0.7))
        first, second = (call.kwargs["logits_processors"][0] for call in fake_mlx.stream_generate.call_args_list)
        assert first is not second
