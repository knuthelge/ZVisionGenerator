"""Tests for resolve_prompt_stage() — flat and nested {option|option} support."""

from __future__ import annotations

from unittest.mock import MagicMock

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.workflows.image_stages import resolve_prompt_stage


def _make_request(prompt: str, seed: int = 0) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        backend=MagicMock(),
        model=MagicMock(),
        prompt=prompt,
        seed=seed,
    )


def _resolve(prompt: str, seed: int = 0) -> str:
    req = _make_request(prompt, seed)
    arts = ImageWorkingArtifacts()
    outcome = resolve_prompt_stage(req, arts)
    assert outcome is StageOutcome.success
    return arts.resolved_prompt


class TestResolvePromptStage:
    def test_no_braces_passthrough(self):
        assert _resolve("a beautiful sunset") == "a beautiful sunset"

    def test_flat_single_group(self):
        result = _resolve("{a|b|c}", seed=42)
        assert result in {"a", "b", "c"}

    def test_multiple_independent_groups(self):
        result = _resolve("{a|b} and {c|d}", seed=7)
        parts = result.split(" and ")
        assert len(parts) == 2
        assert parts[0] in {"a", "b"}
        assert parts[1] in {"c", "d"}

    def test_nested_one_level(self):
        # {Nikon Z9 {50mm|35mm}|Canon EOS 5D}
        # Inner resolves first: {50mm|35mm} -> one of them
        # Then outer: {Nikon Z9 50mm|Canon EOS 5D} or {Nikon Z9 35mm|Canon EOS 5D}
        valid = {"Nikon Z9 50mm", "Nikon Z9 35mm", "Canon EOS 5D"}
        result = _resolve("{Nikon Z9 {50mm|35mm}|Canon EOS 5D}", seed=0)
        assert result in valid

    def test_deeply_nested(self):
        # {a {b {c|d}|e}|f}
        # Innermost: {c|d} -> c or d
        # Then: {b c|e} or {b d|e} -> "b c", "b d", or "e"
        # Then: {a b c|f}, {a b d|f}, or {a e|f}
        # Final: "a b c", "a b d", "a e", or "f"
        valid = {"a b c", "a b d", "a e", "f"}
        result = _resolve("{a {b {c|d}|e}|f}", seed=0)
        assert result in valid

    def test_option_resolution_not_seed_locked(self):
        prompt = "{a|b}"
        results = {_resolve(prompt, seed=123) for _ in range(20)}
        # With true randomness, 20 runs of {a|b} should not all produce the same value
        assert len(results) > 1, "option resolution should not be deterministic based on seed"

    def test_single_option_group(self):
        assert _resolve("{only}") == "only"

    def test_adjacent_nested_groups(self):
        # {a {x|y}|b} {c|d}
        valid_first = {"a x", "a y", "b"}
        valid_second = {"c", "d"}
        result = _resolve("{a {x|y}|b} {c|d}", seed=5)
        # The first group result may contain a space, so split on the last space-separated token
        # Actually let's just check the full result is in the cross product
        valid = {f"{f} {s}" for f in valid_first for s in valid_second}
        assert result in valid

    def test_empty_braces(self):
        # {} has no content matching [^{}]+, so it passes through unchanged
        assert _resolve("hello {} world") == "hello {} world"

    def test_pipe_outside_braces(self):
        assert _resolve("a|b") == "a|b"
