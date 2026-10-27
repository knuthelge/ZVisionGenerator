"""Tests for zvisiongenerator.utils.console — _fmt_time and format_generation_info."""

from __future__ import annotations

import pytest

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.utils.console import _fmt_time, format_generation_info


# ── _fmt_time ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "–"), (0, "0s"), (45, "45s"), (60, "1m 00s"), (125, "2m 05s"), (3661, "1h 01m 01s")],
)
def test_fmt_time(seconds, expected):
    assert _fmt_time(seconds) == expected


# ── format_generation_info ──────────────────────────────────────────────────


def _make_request(**overrides):
    """Build a minimal GenerationRequest for testing."""
    defaults = dict(
        backend=None,
        model=None,
        prompt="test prompt",
        model_name="org/test-model",
        model_family="zimage",
        steps=10,
        guidance=0.5,
        width=1024,
        height=768,
    )
    defaults.update(overrides)
    return ImageGenerationRequest(**defaults)


def _make_artifacts(**overrides):
    return ImageWorkingArtifacts(**overrides)


def _call_format(request=None, artifacts=None, **kwargs):
    defaults = dict(
        run_number=0,
        total_runs=1,
        ran_iterations=1,
        total_iterations=5,
        set_name="default",
        prompt_idx=0,
        total_prompts=1,
    )
    defaults.update(kwargs)
    return format_generation_info(
        request or _make_request(),
        artifacts or _make_artifacts(),
        **defaults,
    )


def _content_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip() and set(line) != {"–"}]


def _settings_line(output: str) -> str:
    return _content_lines(output)[2]


def _model_line(output: str) -> str:
    return _content_lines(output)[3]


def _timing_parts(output: str) -> list[str]:
    return _content_lines(output)[4].split(" | ")


def _timing_values(output: str) -> list[str]:
    return [part.split(": ", 1)[1] for part in _timing_parts(output)]


class TestFormatGenerationInfo:
    def test_includes_model_family(self):
        output = _call_format()
        assert "(zimage)" in output

    def test_includes_dimensions_without_preset(self):
        output = _call_format()
        settings_segments = _settings_line(output).split(". ", 1)[0].split(", ")
        assert settings_segments[-1] == "1024\u00d7768"
        assert not any("2:3" in segment for segment in settings_segments)
        assert len(settings_segments) == 3

    def test_includes_dimensions_with_preset(self):
        req = _make_request(ratio="2:3", size="m")
        output = _call_format(request=req)
        settings_segments = _settings_line(output).split(". ", 1)[0].split(", ")
        assert any(segment.endswith("2:3") for segment in settings_segments)
        assert any(segment.endswith("m") for segment in settings_segments)
        assert settings_segments[-1] == "1024\u00d7768"

    def test_upscale_dimensions_multiplied(self):
        req = _make_request(upscale_factor=2, width=512, height=512)
        output = _call_format(request=req)
        settings_segments = _settings_line(output).split(". ", 1)[0].split(", ")
        assert settings_segments[-1] == "1024\u00d71024"

    def test_lora_info_included(self):
        req = _make_request(
            lora_paths=["style.safetensors", "detail.safetensors"],
            lora_weights=[0.8, 0.5],
        )
        output = _call_format(request=req)
        model_line = _model_line(output)
        assert "style (0.8)" in model_line
        assert "detail (0.5)" in model_line

    def test_cross_platform_model_and_lora_display_names(self):
        req = _make_request(
            model_name=r"C:\models\model.fp16.SAFETENSORS",
            lora_paths=["owner/style.v1.safetensors", "C:/loras/detail.CKPT"],
            lora_weights=[0.8, 0.5],
        )
        output = _call_format(request=req)

        assert "Model: model.fp16 (zimage)" in output
        assert "style.v1 (0.8)" in output
        assert "detail (0.5)" in output

    def test_timing_with_avg(self):
        output = _call_format(elapsed_secs=120, avg_secs=30, eta_secs=90)
        timing_values = _timing_values(output)
        assert timing_values == ["2m 00s", "30s/img", "~1m 30s"]

    def test_run_number_display(self):
        output = _call_format(run_number=2, total_runs=5)
        assert _content_lines(output)[0].endswith("3/5.")

    def test_prompt_set_info(self):
        output = _call_format(set_name="landscapes", prompt_idx=2, total_prompts=10)
        prompt_line = _content_lines(output)[1]
        assert "landscapes" in prompt_line
        assert "3/10" in prompt_line

    def test_unknown_model_family_hides_type(self):
        req = _make_request(model_family="unknown")
        output = _call_format(request=req)
        # Should show model name without family in parentheses
        assert "test-model" in output
        assert "(unknown)" not in output
