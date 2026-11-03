"""Tests for the Higgs TTS 3 code rules: the delay pattern, the sampler and the frame state machine."""

from __future__ import annotations


from typing import Any

import pytest
import torch
from torch.overrides import TorchFunctionMode

from zvisiongenerator.backends.audio.higgs3_codes import FrameState, advance_frame, delay_codes, sample_codebooks, undelay_codes

BOC, EOC = 90, 91


class TestDelay:
    def test_the_layout_for_two_frames_and_three_codebooks(self) -> None:
        codes = torch.tensor([[1, 2, 3], [4, 5, 6]])

        delayed = delay_codes(codes, BOC, EOC)

        assert delayed.tolist() == [
            [1, BOC, BOC],
            [4, 2, BOC],
            [EOC, 5, 3],
            [EOC, EOC, 6],
        ]

    def test_undelay_restores_the_raw_codes(self) -> None:
        codes = torch.randint(0, 80, (7, 4))

        assert torch.equal(undelay_codes(delay_codes(codes, BOC, EOC)), codes)

    @pytest.mark.parametrize("rows", [0, 1, 2])
    def test_a_stream_shorter_than_the_codebook_count_gives_no_frames(self, rows: int) -> None:
        assert undelay_codes(torch.zeros(rows, 3, dtype=torch.long)).shape == (0, 3)

    def test_undelay_takes_each_codebook_from_its_own_offset(self) -> None:
        rows = torch.arange(15).reshape(5, 3)

        assert undelay_codes(rows).tolist() == [[0, 4, 8], [3, 7, 11], [6, 10, 14]]

    def test_a_tensor_that_is_not_two_dimensional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            delay_codes(torch.zeros(3), BOC, EOC)
        with pytest.raises(ValueError, match="2-D"):
            undelay_codes(torch.zeros(3))


class TestSampling:
    def test_top_p_keeps_exactly_the_nucleus_and_always_the_top_code(self) -> None:
        # Probabilities 0.5, 0.3, 0.15, 0.05: at top_p 0.6 the nucleus is the first two codes.
        logits = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))
        seen = {int(sample_codebooks(logits, temperature=1.0, top_p=0.6).item()) for _ in range(300)}

        assert seen == {1, 3}

    def test_a_tiny_top_p_still_keeps_the_top_code(self) -> None:
        logits = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))

        assert {int(sample_codebooks(logits, temperature=1.0, top_p=1e-6).item()) for _ in range(50)} == {1}

    def test_top_p_of_one_keeps_every_code(self) -> None:
        logits = torch.zeros(1, 4)

        assert {int(sample_codebooks(logits, temperature=1.0, top_p=1.0).item()) for _ in range(300)} == {0, 1, 2, 3}

    def test_a_fixed_seed_gives_the_same_codes(self) -> None:
        logits = torch.randn(4, 20)

        torch.manual_seed(11)
        first = sample_codebooks(logits, temperature=0.7, top_p=0.95)
        torch.manual_seed(11)
        second = sample_codebooks(logits, temperature=0.7, top_p=0.95)

        assert torch.equal(first, second)
        assert first.dtype == torch.long
        assert first.shape == (4,)


class _OpLog(TorchFunctionMode):
    """Records the name of each torch call made inside the ``with`` block."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def __torch_function__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        self.calls.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))

    def names(self) -> list[str]:
        return self.calls


def _reference_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Return the sample the Higgs TTS 3 rule prescribes (greedy; else divide, nucleus cut, one multinomial over [N, V]), written out step by step."""
    if temperature <= 1e-5:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
        logits = torch.where(mask, float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1).to(torch.long)


class TestSamplerBitContract:
    """The order and form of the operations decide the bits of a seeded take, so they are pinned."""

    def _log(self, logits: torch.Tensor, temperature: float, top_p: float) -> _OpLog:
        log = _OpLog()
        with log:
            sample_codebooks(logits, temperature=temperature, top_p=top_p)
        return log

    @pytest.mark.parametrize("temperature", [1e-5, 1e-6, 0.0])
    def test_a_temperature_up_to_the_threshold_is_greedy(self, temperature: float) -> None:
        logits = torch.randn(4, 20)
        torch.manual_seed(2)
        before = torch.get_rng_state()

        codes = sample_codebooks(logits, temperature=temperature, top_p=0.9)

        assert torch.equal(codes, logits.argmax(dim=-1))
        assert torch.equal(torch.get_rng_state(), before)

    def test_just_above_the_threshold_it_samples(self) -> None:
        names = self._log(torch.randn(4, 20), 2e-5, 1.0).names()

        assert "multinomial" in names

    @pytest.mark.parametrize(("temperature", "top_p"), [(0.7, 0.95), (0.8, 0.5), (1.0, 0.9), (0.3, 1.0), (1.3, 0.99)])
    def test_a_seeded_draw_equals_the_reference_rule_on_a_full_size_input(self, temperature: float, top_p: float) -> None:
        logits = torch.randn(8, 1026) * 3
        for seed in range(25):
            torch.manual_seed(seed)
            ours = sample_codebooks(logits, temperature=temperature, top_p=top_p)
            torch.manual_seed(seed)
            expected = _reference_sample(logits, temperature, top_p)
            assert torch.equal(ours, expected), seed

    def test_a_cumulative_mass_exactly_at_top_p_does_not_cut_the_next_code(self) -> None:
        # Two equal codes: the mass up to the first is exactly 0.5 = top_p. That does not exceed it, so the second code is still in the nucleus.
        logits = torch.zeros(1, 2)
        seen = {int(sample_codebooks(logits, temperature=1.0, top_p=0.5).item()) for _ in range(100)}

        assert len(seen) == 2

    def test_the_nucleus_follows_the_probability_order_not_the_code_order(self) -> None:
        # Probabilities 0.05, 0.10, 0.60, 0.25: at top_p 0.7 the nucleus is codes 2 and 3.
        logits = torch.log(torch.tensor([[0.05, 0.10, 0.60, 0.25]]))
        seen = {int(sample_codebooks(logits, temperature=1.0, top_p=0.7).item()) for _ in range(400)}

        assert seen == {2, 3}


class TestFrameState:
    N = 4

    def _frame(self, first: int = 5) -> torch.Tensor:
        return torch.full((self.N,), first, dtype=torch.long)

    def test_the_ramp_overwrites_the_later_codebooks_with_boc(self) -> None:
        state = FrameState(self.N)
        seen = []
        for _ in range(self.N):
            codes = self._frame()
            advance_frame(state, codes, boc=BOC, eoc=EOC)
            seen.append(codes.tolist())

        assert seen == [[5, BOC, BOC, BOC], [5, 5, BOC, BOC], [5, 5, 5, BOC], [5, 5, 5, 5]]
        assert (state.delay, state.countdown, state.done) == (self.N, None, False)

    def test_an_eoc_in_codebook_zero_during_the_ramp_is_ignored(self) -> None:
        state = FrameState(self.N)
        for _ in range(self.N):
            advance_frame(state, self._frame(EOC), boc=BOC, eoc=EOC)

        assert (state.countdown, state.done) == (None, False)

    def test_after_the_ramp_an_eoc_starts_a_countdown_and_the_last_frame_ends_the_take(self) -> None:
        state = FrameState(self.N, delay=self.N)

        advance_frame(state, self._frame(7), boc=BOC, eoc=EOC)
        assert (state.countdown, state.done) == (None, False)

        advance_frame(state, self._frame(EOC), boc=BOC, eoc=EOC)
        assert (state.countdown, state.done) == (self.N - 2, False)

        advance_frame(state, self._frame(7), boc=BOC, eoc=EOC)
        assert (state.countdown, state.done) == (1, False)

        advance_frame(state, self._frame(7), boc=BOC, eoc=EOC)
        assert (state.countdown, state.done) == (0, True)

    @pytest.mark.parametrize("codebooks", [1, 2])
    def test_two_codebooks_or_fewer_end_at_once(self, codebooks: int) -> None:
        state = FrameState(codebooks, delay=codebooks)

        advance_frame(state, torch.full((codebooks,), EOC, dtype=torch.long), boc=BOC, eoc=EOC)

        assert state.done
