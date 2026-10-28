"""Tests for dynamic workflow composition via build_workflow()."""

from __future__ import annotations

import os
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._images import is_bluish, is_reddish, write_sideways_photo
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.workflows import build_workflow
from zvisiongenerator.workflows.image_stages import (
    resolve_prompt_stage,
    suppress_negative_stage,
    load_reference_stage,
    load_references_stage,
    text_to_image_stage,
    upscale_stage,
    contrast_stage,
    saturation_stage,
    sharpen_stage,
    save_image_stage,
)
from zvisiongenerator.utils.provenance import IMAGE_CONFIG_SCHEMA, read_png_config


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

    _HEAD = [resolve_prompt_stage, suppress_negative_stage, load_reference_stage, load_references_stage, text_to_image_stage]

    @pytest.mark.parametrize(
        ("args", "post_stages"),
        [
            pytest.param({}, [sharpen_stage], id="default: sharpen only"),
            pytest.param({"sharpen": False}, [], id="all off generates only"),
            pytest.param(
                {"sharpen": True, "contrast": True, "saturation": True, "upscale": 2}, [upscale_stage, contrast_stage, saturation_stage, sharpen_stage], id="all on with upscale, in fixed order"
            ),
            pytest.param({"sharpen": False, "upscale": 4}, [upscale_stage], id="upscale without post-processing"),
            pytest.param({"sharpen": 0.0, "contrast": 0.0, "saturation": 0.0}, [contrast_stage, saturation_stage, sharpen_stage], id="0.0 is not False"),
        ],
    )
    def test_stage_list(self, args, post_stages):
        assert build_workflow(_default_args(**args)).stages == [*self._HEAD, *post_stages, save_image_stage]


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

    def test_text_to_image_stage_forwards_first_sigma_to_backend(self):
        mock_backend = _make_mock_backend()
        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_family="ideogram4",
            supports_negative_prompt=False,
            prompt="a cat",
            seed=42,
            width=64,
            height=64,
            steps=10,
            guidance=3.5,
            first_sigma=1.005,
        )
        artifacts = ImageWorkingArtifacts(resolved_prompt="a cat")

        outcome = text_to_image_stage(request, artifacts)

        assert outcome is StageOutcome.success
        assert mock_backend.text_to_image.call_args.kwargs["first_sigma"] == 1.005

    def test_default_workflow_embeds_config_in_png_and_writes_no_sidecar(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args())

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_name="zit",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a cat",
            ratio="1:1",
            size="s",
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

        # No sidecar written (neither asset.ext.json nor asset.json)
        from pathlib import Path as _Path

        fp = _Path(artifacts.filepath)
        assert not fp.with_name(f"{fp.name}.json").exists()
        assert not fp.with_suffix(".json").exists()

        # Config embedded in PNG
        payload = read_png_config(artifacts.filepath)
        assert payload is not None
        assert payload["schema"] == IMAGE_CONFIG_SCHEMA
        assert payload["workflow"] == "txt2img"
        assert payload["prompt"] == "a cat"
        assert payload["model"] == "zit"
        assert payload["seed"] == 42
        assert payload["steps"] == 10
        assert payload["guidance"] == 3.5
        assert payload["width"] == 64
        assert payload["height"] == 64
        assert payload["ratio"] == "1:1"
        assert payload["size"] == "s"

    def test_default_workflow_keeps_description_and_exif_metadata(self, tmp_path):
        mock_backend = _make_mock_backend()
        workflow = build_workflow(_default_args())

        request = ImageGenerationRequest(
            backend=mock_backend,
            model="fake",
            model_name="zit",
            model_family="zimage",
            supports_negative_prompt=True,
            prompt="a glowing forest",
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
        with Image.open(artifacts.filepath) as saved_image:
            exif = saved_image.getexif()
            assert saved_image.info.get("Description") == "a glowing forest"
            assert exif.get(0x010E) == "a glowing forest"

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


class TestReviewStage:
    """review_stage sits between the small render and the upscale, and only exists with both upscale and review."""

    @staticmethod
    def _request(review):
        return ImageGenerationRequest(backend=MagicMock(), model="fake", prompt="a cat", review=review)

    def test_build_workflow_adds_review_between_render_and_upscale(self):
        from zvisiongenerator.workflows.image_stages import review_stage

        wf = build_workflow(_default_args(upscale=2, review=True, sharpen=False))
        assert wf.stages.index(text_to_image_stage) + 1 == wf.stages.index(review_stage)
        assert wf.stages.index(review_stage) + 1 == wf.stages.index(upscale_stage)

    @pytest.mark.parametrize("args", [dict(upscale=None, review=True), dict(upscale=2, review=False), dict(upscale=2)])
    def test_build_workflow_omits_review_otherwise(self, args):
        from zvisiongenerator.workflows.image_stages import review_stage

        assert review_stage not in build_workflow(_default_args(**args)).stages

    def test_discard_skips_without_upscaling_or_saving(self, tmp_path):
        backend = _make_mock_backend()
        request = ImageGenerationRequest(
            backend=backend,
            model="fake",
            prompt="a cat",
            width=64,
            height=64,
            upscale_factor=2,
            upscale_denoise=0.3,
            output_dir=str(tmp_path),
            filename_base="test",
            review=MagicMock(return_value=False),
        )
        artifacts = ImageWorkingArtifacts(filename="test")

        outcome = build_workflow(_default_args(upscale=2, review=True)).run(request, artifacts)

        assert outcome is StageOutcome.skipped
        backend.image_to_image.assert_not_called()
        assert artifacts.was_upscaled is False
        assert list(tmp_path.iterdir()) == []

    def test_stage_is_a_noop_without_a_review_callable(self):
        from zvisiongenerator.workflows.image_stages import review_stage

        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (8, 8)))
        assert review_stage(self._request(None), artifacts) is StageOutcome.success


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _reference_request(paths, *, cap=4, backend=None, **overrides):
    return ImageGenerationRequest(
        backend=backend or _make_mock_backend(),
        model="fake",
        model_family="flux2_klein",
        max_reference_images=cap,
        prompt="the woman from image 1",
        width=64,
        height=64,
        reference_paths=tuple(str(path) for path in paths),
        **overrides,
    )


def _save_images(tmp_path, sizes):
    paths = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"ref{index}.png"
        Image.new("RGBA", size, color=(index * 40, 10, 10, 255)).save(path)
        paths.append(path)
    return paths


class TestReferences:
    def test_loads_each_reference_rgb_at_its_own_size(self, tmp_path):
        request = _reference_request(_save_images(tmp_path, [(100, 50), (30, 90)]))
        artifacts = ImageWorkingArtifacts()

        assert load_references_stage(request, artifacts) is StageOutcome.success

        assert [(image.mode, image.size) for image in artifacts.reference_images] == [("RGB", (100, 50)), ("RGB", (30, 90))]
        assert artifacts.image is None

    def test_no_references_is_a_noop(self):
        artifacts = ImageWorkingArtifacts()

        assert load_references_stage(_reference_request([]), artifacts) is StageOutcome.success
        assert artifacts.reference_images == []

    def test_more_than_the_cap_is_refused_and_none_is_loaded(self, tmp_path):
        paths = _save_images(tmp_path, [(16, 16)] * 3)
        artifacts = ImageWorkingArtifacts()

        assert load_references_stage(_reference_request(paths[:2], cap=2), ImageWorkingArtifacts()) is StageOutcome.success
        with pytest.raises(ValueError):
            load_references_stage(_reference_request(paths, cap=2), artifacts)

        assert artifacts.reference_images == []

    def test_text_to_image_passes_references_to_the_backend(self):
        backend = _make_mock_backend()
        refs = [Image.new("RGB", (32, 32))]
        artifacts = ImageWorkingArtifacts(resolved_prompt="p", reference_images=refs)

        text_to_image_stage(_reference_request(["a.png"], backend=backend), artifacts)

        assert backend.text_to_image.call_args.kwargs["reference_images"] == refs
        backend.image_to_image.assert_not_called()

    def test_start_image_and_references_go_to_image_to_image_together(self):
        backend = _make_mock_backend()
        refs = [Image.new("RGB", (32, 32))]
        artifacts = ImageWorkingArtifacts(resolved_prompt="p", image=Image.new("RGB", (64, 64)), reference_images=refs)

        text_to_image_stage(_reference_request(["a.png"], backend=backend, image_path="start.png"), artifacts)

        assert backend.image_to_image.call_args.kwargs["reference_images"] == refs

    def test_upscale_refine_passes_the_references(self):
        backend = _make_mock_backend()
        refs = [Image.new("RGB", (32, 32))]
        artifacts = ImageWorkingArtifacts(resolved_prompt="p", image=Image.new("RGB", (64, 64)), reference_images=refs)
        request = _reference_request(["a.png"], backend=backend, upscale_factor=2, upscale_denoise=0.3, upscale_sharpen=False)

        upscale_stage(request, artifacts)

        assert backend.image_to_image.call_args.kwargs["reference_images"] == refs


# ---------------------------------------------------------------------------
# Start image
# ---------------------------------------------------------------------------


def _start_request(image_path, *, width, height):
    return ImageGenerationRequest(backend=_make_mock_backend(), model="fake", prompt="p", width=width, height=height, image_path=str(image_path))


class TestStartImage:
    def test_a_sideways_photo_is_loaded_upright_before_it_is_filled(self, tmp_path):
        request = _start_request(write_sideways_photo(tmp_path / "phone.jpg"), width=40, height=80)
        artifacts = ImageWorkingArtifacts()

        assert load_reference_stage(request, artifacts) is StageOutcome.success

        assert artifacts.image.size == (40, 80)
        assert is_reddish(artifacts.image.getpixel((20, 4))) and is_bluish(artifacts.image.getpixel((20, 75)))

    def test_a_missing_start_image_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_reference_stage(_start_request(tmp_path / "gone.png", width=8, height=4), ImageWorkingArtifacts())
