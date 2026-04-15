"""Golden test for generate_filename()."""

from __future__ import annotations

from zvisiongenerator.utils.filename import generate_filename


def test_golden_basic():
    """Known inputs produce a deterministic filename (except timestamp)."""
    result = generate_filename(
        set_name="cats",
        width=1440,
        height=768,
        seed=12345,
        steps=10,
        guidance=0.5,
        scheduler="beta",
        model="models/my-model",
        lora_paths=None,
        lora_weights=None,
    )
    # Timestamp varies, so check the non-timestamp parts
    assert "_1440x768_" in result
    assert "_my-model_" in result
    assert "_beta_" in result
    assert "_steps10_" in result
    assert "_cfg0.5_" in result
    assert "_seed12345" in result
    assert result.startswith("cats_")


def test_golden_with_lora():
    """Filename includes LoRA names and weights."""
    result = generate_filename(
        set_name="portrait",
        width=1920,
        height=1024,
        seed=99999,
        steps=20,
        guidance=3.5,
        scheduler=None,
        model="models/myModel",
        lora_paths=["/path/to/detail.safetensors", "/path/to/style.safetensors"],
        lora_weights=[0.8, 0.5],
    )
    assert "_1920x1024_" in result
    assert "_detail_80_" in result
    assert "_style_50_" in result
    assert "_steps20_" in result
    assert "_seed99999" in result


def test_golden_no_scheduler():
    """When scheduler is None, no scheduler segment appears."""
    result = generate_filename(
        set_name="test",
        width=640,
        height=320,
        seed=1,
        steps=4,
        guidance=1.0,
        scheduler=None,
        model="models/klein4b",
        lora_paths=None,
        lora_weights=None,
    )
    assert "_beta" not in result
    assert "_steps4_" in result


def test_golden_no_guidance():
    """When guidance is None, no cfg segment appears."""
    result = generate_filename(
        set_name="test",
        width=640,
        height=320,
        seed=1,
        steps=4,
        guidance=None,
        scheduler=None,
        model="models/klein4b",
        lora_paths=None,
        lora_weights=None,
    )
    assert "_cfg" not in result


def test_golden_with_num_frames():
    """When num_frames is set, a frames segment appears after dimensions."""
    result = generate_filename(
        set_name="video",
        width=704,
        height=480,
        seed=42,
        steps=30,
        guidance=3.0,
        model="models/ltx-video",
        num_frames=49,
    )
    assert "_704x480_49f_" in result
    assert "_steps30_" in result
    assert "_cfg3.0_" in result
    assert "_seed42" in result
    assert result.startswith("video_")
