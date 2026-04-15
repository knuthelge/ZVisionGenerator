"""Tests for zvisiongenerator.utils.console — _fmt_time and format_generation_info."""

from __future__ import annotations

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.utils.console import _fmt_time, format_generation_info


# ── _fmt_time ───────────────────────────────────────────────────────────────


class TestFmtTime:
    def test_none_returns_dash(self):
        assert _fmt_time(None) == "–"

    def test_seconds_only(self):
        assert _fmt_time(45) == "45s"

    def test_minutes_and_seconds(self):
        assert _fmt_time(125) == "2m 05s"

    def test_hours_minutes_seconds(self):
        assert _fmt_time(3661) == "1h 01m 01s"

    def test_zero_seconds(self):
        assert _fmt_time(0) == "0s"

    def test_exactly_one_minute(self):
        assert _fmt_time(60) == "1m 00s"


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


class TestFormatGenerationInfo:
    def test_includes_model_name(self):
        output = _call_format()
        assert "test-model" in output

    def test_includes_model_family(self):
        output = _call_format()
        assert "zimage" in output

    def test_includes_steps_and_guidance(self):
        output = _call_format()
        assert "Steps: 10" in output
        assert "Guidance: 0.5" in output

    def test_includes_dimensions_without_preset(self):
        output = _call_format()
        assert "1024\u00d7768" in output
        assert "Ratio:" not in output
        assert "Size:" not in output

    def test_includes_dimensions_with_preset(self):
        req = _make_request(ratio="2:3", size="m")
        output = _call_format(request=req)
        assert "Ratio: 2:3" in output
        assert "Size: m" in output
        assert "1024\u00d7768" in output

    def test_upscale_dimensions_multiplied(self):
        req = _make_request(upscale_factor=2, width=512, height=512)
        output = _call_format(request=req)
        assert "1024\u00d71024" in output

    def test_upscale_enabled_text(self):
        req = _make_request(upscale_factor=2)
        output = _call_format(request=req)
        assert "Upscaling enabled" in output

    def test_upscale_disabled_text(self):
        output = _call_format()
        assert "Upscaling disabled" in output

    def test_lora_info_included(self):
        req = _make_request(
            lora_paths=["style.safetensors", "detail.safetensors"],
            lora_weights=[0.8, 0.5],
        )
        output = _call_format(request=req)
        assert "style" in output
        assert "0.8" in output

    def test_lora_disabled_when_none(self):
        output = _call_format()
        assert "LoRA: disabled" in output

    def test_timing_with_avg(self):
        output = _call_format(elapsed_secs=120, avg_secs=30, eta_secs=90)
        assert "Elapsed: 2m 00s" in output
        assert "Avg: 30s/img" in output
        assert "ETA:" in output

    def test_timing_without_avg(self):
        output = _call_format(elapsed_secs=10)
        assert "ETA: calculating..." in output

    def test_run_number_display(self):
        output = _call_format(run_number=2, total_runs=5)
        assert "run nr 3/5" in output

    def test_prompt_set_info(self):
        output = _call_format(set_name="landscapes", prompt_idx=2, total_prompts=10)
        assert "landscapes" in output
        assert "prompt 3/10" in output

    def test_unknown_model_family_hides_type(self):
        req = _make_request(model_family="unknown")
        output = _call_format(request=req)
        # Should show model name without family in parentheses
        assert "test-model" in output
        assert "(unknown)" not in output
