"""Tests for dynamic workflow composition via build_workflow()."""

from __future__ import annotations

import os
from argparse import Namespace
from unittest.mock import MagicMock

from PIL import Image

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.workflows import build_workflow
from zvisiongenerator.workflows.image_stages import (
    resolve_prompt_stage,
    suppress_negative_stage,
    load_reference_stage,
    text_to_image_stage,
    upscale_stage,
    contrast_stage,
    saturation_stage,
    sharpen_stage,
    save_image_stage,
)


# ---------------------------------------------------------------------------
# Helper: mock backend
# ---------------------------------------------------------------------------


def _make_mock_backend(name="mock"):
    mock = MagicMock()
    mock.name = name
    mock.text_to_image.return_value = Image.new("RGB", (64, 64), color="red")
    mock.image_to_image.return_value = Image.new("RGB", (128, 128), color="blue")
    return mock


def _default_args(**overrides):
    """Minimal Namespace matching what CLI produces."""
    defaults = dict(
        sharpen=True,
        contrast=False,
        saturation=False,
        upscale=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# Dynamic composition tests
# ---------------------------------------------------------------------------


class TestBuildWorkflow:
    """Verify build_workflow() assembles the correct stage list."""

    def test_default_args_include_sharpen(self):
        """Default: sharpen on, contrast/saturation off, no upscale."""
        wf = build_workflow(_default_args())
        assert wf.stages == [
            resolve_prompt_stage,
            suppress_negative_stage,
            load_reference_stage,
            text_to_image_stage,
            sharpen_stage,
            save_image_stage,
        ]

    def test_all_disabled_generates_only(self):
        """Equivalent to old 'generate_only': no post-processing."""
        wf = build_workflow(_default_args(sharpen=False, contrast=False, saturation=False, upscale=None))
        assert wf.stages == [
            resolve_prompt_stage,
            suppress_negative_stage,
            load_reference_stage,
            text_to_image_stage,
            save_image_stage,
        ]

    def test_all_enabled_with_upscale(self):
        """All stages on: upscale -> contrast -> saturation -> sharpen -> save."""
        wf = build_workflow(_default_args(sharpen=True, contrast=True, saturation=True, upscale=2))
        assert wf.stages == [
            resolve_prompt_stage,
            suppress_negative_stage,
            load_reference_stage,
            text_to_image_stage,
            upscale_stage,
            contrast_stage,
            saturation_stage,
            sharpen_stage,
            save_image_stage,
        ]

    def test_zero_amount_is_not_false(self):
        """0.0 is not False -- stages should still be included."""
        wf = build_workflow(_default_args(sharpen=0.0, contrast=0.0, saturation=0.0, upscale=None))
        assert contrast_stage in wf.stages
        assert sharpen_stage in wf.stages
        assert saturation_stage in wf.stages

    def test_stage_ordering_contrast_before_saturation_before_sharpen(self):
        """Fixed order: contrast -> saturation -> sharpen."""
        wf = build_workflow(_default_args(sharpen=True, contrast=True, saturation=True))
        stage_names = wf.stages
        c_idx = stage_names.index(contrast_stage)
        s_idx = stage_names.index(saturation_stage)
        sh_idx = stage_names.index(sharpen_stage)
        assert c_idx < s_idx < sh_idx

    def test_workflow_name_is_dynamic(self):
        wf = build_workflow(_default_args())
        assert wf.name == "dynamic"

    def test_upscale_without_post_processing(self):
        """Upscale on, all post-processing off."""
        wf = build_workflow(_default_args(sharpen=False, contrast=False, saturation=False, upscale=4))
        assert wf.stages == [
            resolve_prompt_stage,
            suppress_negative_stage,
            load_reference_stage,
            text_to_image_stage,
            upscale_stage,
            save_image_stage,
        ]


# ---------------------------------------------------------------------------
# E2E mock workflow: verify stage order and correct backend method calls
# ---------------------------------------------------------------------------


class TestE2EWorkflow:
    """Mock backend -> ImageGenerationRequest -> run workflow -> verify."""

    def test_default_workflow_calls_text_to_image(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args())

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a cat",
            seed=42,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        outcome = workflow.run(request, artifacts)

        assert outcome is StageOutcome.success
        mock_backend.text_to_image.assert_called_once()
        call_kwargs = mock_backend.text_to_image.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["width"] == 64
        assert call_kwargs["height"] == 64
        assert call_kwargs["steps"] == 10

    def test_default_workflow_saves_image(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args())

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a cat",
            seed=42,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        workflow.run(request, artifacts)

        assert artifacts.filepath is not None
        assert os.path.isfile(artifacts.filepath)

    def test_default_workflow_with_upscale(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args(upscale=2))

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a cat",
            seed=42,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            upscale_factor=2,
            upscale_denoise=0.3,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        outcome = workflow.run(request, artifacts)

        assert outcome is StageOutcome.success
        mock_backend.text_to_image.assert_called_once()
        mock_backend.image_to_image.assert_called_once()
        assert artifacts.was_upscaled is True

    def test_workflow_suppresses_negative_prompt(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args())

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="flux2_klein",
            supports_negative_prompt=False,
            prompt="a cat",
            negative_prompt="blurry",
            seed=42,
            width=64,
            height=64,
            steps=4,
            guidance=1.0,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        workflow.run(request, artifacts)

        call_kwargs = mock_backend.text_to_image.call_args.kwargs
        assert call_kwargs["negative_prompt"] is None

    def test_all_disabled_does_not_sharpen(self, tmp_path):
        """All post-processing off: no upscale, no sharpening."""
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args(sharpen=False))

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="test",
            seed=1,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            sharpen=False,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        outcome = workflow.run(request, artifacts)

        assert outcome is StageOutcome.success
        # No upscale stage → image_to_image never called
        mock_backend.image_to_image.assert_not_called()
        # Sharpen stage not in workflow
        assert sharpen_stage not in workflow.stages

    def test_all_disabled_still_generates_and_saves(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args(sharpen=False))

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a cat",
            seed=42,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            sharpen=False,
            output_dir=str(tmp_path),
            filename_base="test",
        )
        artifacts = ImageWorkingArtifacts(filename="test")
        outcome = workflow.run(request, artifacts)

        assert outcome is StageOutcome.success
        mock_backend.text_to_image.assert_called_once()
        assert artifacts.filepath is not None
        assert os.path.isfile(artifacts.filepath)


# ---------------------------------------------------------------------------
# Stage-layer tests: contrast_stage, saturation_stage, sharpen_stage
# ---------------------------------------------------------------------------


class TestContrastStageUnit:
    """Direct calls to contrast_stage()."""

    def test_no_image_returns_success(self):
        request = ImageGenerationRequest(backend=None, model=None, prompt="x", contrast_amount=1.3)
        artifacts = ImageWorkingArtifacts(image=None)
        assert contrast_stage(request, artifacts) is StageOutcome.success
        assert artifacts.image is None

    def test_modifies_image(self):
        img = Image.new("RGB", (100, 100), (128, 64, 192))
        request = ImageGenerationRequest(backend=None, model=None, prompt="x", contrast_amount=1.3)
        artifacts = ImageWorkingArtifacts(image=img)
        outcome = contrast_stage(request, artifacts)
        assert outcome is StageOutcome.success
        assert artifacts.image is not img
        assert artifacts.image.size == img.size


class TestSaturationStageUnit:
    """Direct calls to saturation_stage()."""

    def test_no_image_returns_success(self):
        request = ImageGenerationRequest(backend=None, model=None, prompt="x", saturation_amount=0.5)
        artifacts = ImageWorkingArtifacts(image=None)
        assert saturation_stage(request, artifacts) is StageOutcome.success
        assert artifacts.image is None

    def test_modifies_image(self):
        img = Image.new("RGB", (100, 100), (128, 64, 192))
        request = ImageGenerationRequest(backend=None, model=None, prompt="x", saturation_amount=0.5)
        artifacts = ImageWorkingArtifacts(image=img)
        outcome = saturation_stage(request, artifacts)
        assert outcome is StageOutcome.success
        assert artifacts.image is not img
        assert artifacts.image.size == img.size


class TestSharpenStageUnit:
    """Direct calls to sharpen_stage()."""

    def test_no_image_returns_success(self):
        request = ImageGenerationRequest(backend=None, model=None, prompt="x")
        artifacts = ImageWorkingArtifacts(image=None)
        assert sharpen_stage(request, artifacts) is StageOutcome.success
        assert artifacts.image is None

    def test_override_uses_override_amount(self):
        from unittest.mock import patch

        img = Image.new("RGB", (100, 100), (128, 64, 192))
        request = ImageGenerationRequest(
            backend=None,
            model=None,
            prompt="x",
            sharpen_amount_override=0.5,
            sharpen_amount_normal=0.8,
            sharpen_amount_upscaled=1.2,
        )
        artifacts = ImageWorkingArtifacts(image=img)
        with patch(
            "zvisiongenerator.processing.sharpen.contrast_adaptive_sharpening",
            return_value=img,
        ) as mock_cas:
            outcome = sharpen_stage(request, artifacts)
        assert outcome is StageOutcome.success
        mock_cas.assert_called_once_with(img, amount=0.5)
