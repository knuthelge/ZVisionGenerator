"""Test provenance payload construction and serializability."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.provenance import (
    IMAGE_CONFIG_SCHEMA,
    PROVENANCE_SCHEMA,
    VIDEO_CONFIG_SCHEMA,
    build_image_config_payload,
    build_image_provenance,
    build_video_config_payload,
    build_video_provenance,
    embed_mp4_config,
    embed_png_config,
    read_mp4_config,
    read_png_config,
)


def test_build_image_provenance_is_json_serializable_and_keeps_expected_fields(tmp_path):
    asset_path = tmp_path / "image.png"
    request = ImageGenerationRequest(
        backend=None,
        model=None,
        prompt="A cinematic skyline",
        model_name="zit",
        model_family="zimage",
        negative_prompt="low detail",
        ratio="16:9",
        size="m",
        width=1344,
        height=768,
        seed=1234,
        steps=12,
        guidance=2.5,
        scheduler="karras",
        lora_paths=["/tmp/style.safetensors"],
        lora_weights=[0.8],
        upscale_factor=2,
        upscale_denoise=0.3,
        upscale_steps=6,
        upscale_guidance=1.2,
        sharpen=True,
        contrast=True,
        saturation=False,
    )
    artifacts = ImageWorkingArtifacts(
        image=Image.new("RGB", (1344, 768), color="red"),
        resolved_prompt="A cinematic skyline at dusk",
        generation_time=4.2,
        was_upscaled=True,
    )

    payload = build_image_provenance(asset_path, request, artifacts)

    assert payload["schema"] == PROVENANCE_SCHEMA
    assert payload["media_type"] == "image"
    assert payload["workflow"] == "txt2img"
    assert payload["prompt"] == "A cinematic skyline"
    assert payload["resolved_prompt"] == "A cinematic skyline at dusk"
    assert payload["model_name"] == "zit"
    assert payload["model_family"] == "zimage"
    assert payload["seed"] == 1234
    assert payload["steps"] == 12
    assert payload["guidance"] == 2.5
    assert payload["width"] == 1344
    assert payload["height"] == 768
    assert payload["ratio"] == "16:9"
    assert payload["size"] == "m"
    assert payload["loras"] == [{"name": "style", "path": "/tmp/style.safetensors", "weight": 0.8}]
    assert payload["generation"]["was_upscaled"] is True
    assert payload["output"]["filename"] == "image.png"
    json.dumps(payload)


def test_build_video_provenance_is_json_serializable_and_keeps_expected_fields(tmp_path):
    asset_path = tmp_path / "clip.mp4"
    request = VideoGenerationRequest(
        backend=None,
        model=None,
        prompt="Camera pushes through a neon alley",
        model_name="ltx-8",
        model_family="ltx",
        width=704,
        height=448,
        num_frames=49,
        seed=77,
        steps=8,
        image_path="/tmp/ref.png",
        lora_paths=["/tmp/motion.safetensors"],
        lora_weights=[0.75],
        upscale=2,
        upscale_steps=3,
        no_audio=True,
        output_format="mp4",
    )
    artifacts = VideoWorkingArtifacts(
        resolved_prompt="Camera pushes through a neon alley with rain",
        generation_time=12.5,
        video_path=Path(asset_path),
        filename="clip.mp4",
    )

    payload = build_video_provenance(asset_path, request, artifacts)

    assert payload["schema"] == PROVENANCE_SCHEMA
    assert payload["media_type"] == "video"
    assert payload["workflow"] == "img2vid"
    assert payload["prompt"] == "Camera pushes through a neon alley"
    assert payload["resolved_prompt"] == "Camera pushes through a neon alley with rain"
    assert payload["model_name"] == "ltx-8"
    assert payload["model_family"] == "ltx"
    assert payload["seed"] == 77
    assert payload["steps"] == 8
    assert payload["width"] == 704
    assert payload["height"] == 448
    assert payload["frame_count"] == 49
    assert payload["image_path"] == "/tmp/ref.png"
    assert payload["loras"] == [{"name": "motion", "path": "/tmp/motion.safetensors", "weight": 0.75}]
    assert payload["generation"]["audio"] is False
    assert payload["output"]["filename"] == "clip.mp4"
    json.dumps(payload)


def _make_image_request(**overrides):
    defaults = dict(
        backend=None,
        model=None,
        prompt="A red barn",
        model_name="zit",
        model_family="zimage",
        ratio="4:3",
        size="s",
        width=512,
        height=384,
        seed=7,
        steps=10,
        guidance=3.0,
    )
    defaults.update(overrides)
    return ImageGenerationRequest(**defaults)


class TestBuildImageConfigPayload:
    def test_contains_all_minimal_reusable_fields(self):
        request = _make_image_request(
            lora_paths=["/models/style.safetensors"],
            lora_weights=[0.7],
        )
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)))

        payload = build_image_config_payload(request, artifacts)

        assert payload["schema"] == IMAGE_CONFIG_SCHEMA
        assert payload["workflow"] == "txt2img"
        assert payload["prompt"] == "A red barn"
        assert payload["model"] == "zit"
        assert payload["seed"] == 7
        assert payload["steps"] == 10
        assert payload["guidance"] == 3.0
        assert payload["width"] == 512
        assert payload["height"] == 384
        assert payload["ratio"] == "4:3"
        assert payload["size"] == "s"
        assert payload["image_path"] is None
        assert payload["lora"] == "/models/style.safetensors:0.7"
        json.dumps(payload)

    def test_excludes_non_reusable_fields(self):
        request = _make_image_request()
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)), generation_time=3.1)

        payload = build_image_config_payload(request, artifacts)

        assert "model_name" not in payload
        assert "model_family" not in payload
        assert "resolved_prompt" not in payload
        assert "generation_time" not in payload
        assert "media_type" not in payload
        assert "output" not in payload
        assert "generation" not in payload

    def test_workflow_is_img2img_when_image_path_set(self):
        request = _make_image_request(image_path="/tmp/ref.png")
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)))

        payload = build_image_config_payload(request, artifacts)

        assert payload["workflow"] == "img2img"
        assert payload["image_path"] == "/tmp/ref.png"

    def test_dimensions_from_artifact_image_take_priority(self):
        request = _make_image_request(width=512, height=384)
        # Artifact image has different actual dimensions (e.g. after upscale)
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (1024, 768)))

        payload = build_image_config_payload(request, artifacts)

        assert payload["width"] == 1024
        assert payload["height"] == 768

    def test_is_json_serializable(self):
        request = _make_image_request()
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)))
        payload = build_image_config_payload(request, artifacts)
        json.dumps(payload)


class TestEmbedAndReadPngConfig:
    def test_round_trip_embed_and_read(self, tmp_path):
        request = _make_image_request()
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)))
        payload = build_image_config_payload(request, artifacts)

        png_path = tmp_path / "test.png"
        img = Image.new("RGB", (512, 384), color="blue")
        pnginfo = PngInfo()
        embed_png_config(pnginfo, payload)
        img.save(str(png_path), pnginfo=pnginfo)

        result = read_png_config(png_path)

        assert result is not None
        assert result["schema"] == IMAGE_CONFIG_SCHEMA
        assert result["prompt"] == "A red barn"
        assert result["model"] == "zit"
        assert result["seed"] == 7

    def test_read_png_config_returns_none_for_missing_key(self, tmp_path):
        png_path = tmp_path / "plain.png"
        Image.new("RGB", (64, 64), color="gray").save(str(png_path))

        result = read_png_config(png_path)

        assert result is None

    def test_embed_does_not_clobber_description(self, tmp_path):
        png_path = tmp_path / "meta.png"
        img = Image.new("RGB", (64, 64), color="green")
        pnginfo = PngInfo()
        pnginfo.add_text("Description", "original prompt")
        request = _make_image_request()
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (64, 64)))
        embed_png_config(pnginfo, build_image_config_payload(request, artifacts))
        img.save(str(png_path), pnginfo=pnginfo)

        with Image.open(png_path) as saved:
            assert saved.info.get("Description") == "original prompt"
            assert saved.info.get("zvisiongenerator.config") is not None


def _make_video_request(**overrides):
    defaults = dict(
        backend=None,
        model=None,
        prompt="A sweeping landscape",
        model_name="ltx-8",
        model_family="ltx",
        width=704,
        height=448,
        num_frames=49,
        seed=42,
        steps=8,
    )
    defaults.update(overrides)
    return VideoGenerationRequest(**defaults)


class TestBuildVideoConfigPayload:
    def test_contains_all_minimal_reusable_fields(self):
        request = _make_video_request(
            lora_paths=["/models/motion.safetensors"],
            lora_weights=[0.7],
            image_path="/tmp/ref.png",
        )
        artifacts = VideoWorkingArtifacts()

        payload = build_video_config_payload(request, artifacts)

        assert payload["schema"] == VIDEO_CONFIG_SCHEMA
        assert payload["workflow"] == "img2vid"
        assert payload["prompt"] == "A sweeping landscape"
        assert payload["model"] == "ltx-8"
        assert payload["seed"] == 42
        assert payload["steps"] == 8
        assert payload["width"] == 704
        assert payload["height"] == 448
        assert payload["frame_count"] == 49
        assert payload["image_path"] == "/tmp/ref.png"
        assert payload["lora"] == "/models/motion.safetensors:0.7"
        json.dumps(payload)

    def test_excludes_non_reusable_fields(self):
        request = _make_video_request()
        artifacts = VideoWorkingArtifacts(generation_time=12.5, resolved_prompt="resolved version")

        payload = build_video_config_payload(request, artifacts)

        assert "model_name" not in payload
        assert "model_family" not in payload
        assert "resolved_prompt" not in payload
        assert "generation_time" not in payload
        assert "media_type" not in payload
        assert "output" not in payload
        assert "generation" not in payload
        assert "no_audio" not in payload
        assert "audio" not in payload
        assert "output_format" not in payload

    def test_workflow_is_txt2vid_without_image_path(self):
        request = _make_video_request()
        payload = build_video_config_payload(request, VideoWorkingArtifacts())
        assert payload["workflow"] == "txt2vid"

    def test_is_json_serializable(self):
        request = _make_video_request()
        payload = build_video_config_payload(request, VideoWorkingArtifacts())
        json.dumps(payload)


class TestEmbedMp4Config:
    def test_calls_ffmpeg_with_metadata_arg(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")
        payload = {"schema": VIDEO_CONFIG_SCHEMA, "prompt": "test prompt"}

        def _side_effect(cmd, **kwargs):
            tmp = video_path.with_suffix(".tmp.mp4")
            tmp.write_bytes(b"processed")
            return MagicMock(returncode=0)

        with patch("zvisiongenerator.utils.provenance.subprocess.run", side_effect=_side_effect) as mock_run:
            embed_mp4_config(video_path, payload)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-metadata" in cmd
        meta_idx = cmd.index("-metadata")
        meta_val = cmd[meta_idx + 1]
        assert meta_val.startswith("zvisiongenerator.config=")
        embedded = json.loads(meta_val[len("zvisiongenerator.config=") :])
        assert embedded["prompt"] == "test prompt"

    def test_does_not_create_adjacent_sidecars(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")
        payload = {"schema": VIDEO_CONFIG_SCHEMA, "prompt": "test prompt"}

        def _side_effect(cmd, **kwargs):
            tmp = video_path.with_suffix(".tmp.mp4")
            tmp.write_bytes(b"processed")
            return MagicMock(returncode=0)

        with patch("zvisiongenerator.utils.provenance.subprocess.run", side_effect=_side_effect):
            embed_mp4_config(video_path, payload)

        assert not video_path.with_suffix(".json").exists()
        assert not video_path.with_name(f"{video_path.name}.json").exists()

    def test_raises_on_ffmpeg_failure_and_cleans_tmp(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")
        payload = {"schema": VIDEO_CONFIG_SCHEMA}

        with patch("zvisiongenerator.utils.provenance.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"some error")
            import pytest

            with pytest.raises(subprocess.CalledProcessError):
                embed_mp4_config(video_path, payload)

        assert video_path.exists()
        assert not video_path.with_suffix(".tmp.mp4").exists()


class TestReadMp4Config:
    def test_returns_none_when_config_tag_is_missing(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")

        with patch("zvisiongenerator.utils.provenance.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps({"format": {"tags": {"title": "demo"}}}))

            result = read_mp4_config(video_path)

        assert result is None

    def test_reads_json_payload_from_mp4_metadata(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")
        payload = {
            "schema": VIDEO_CONFIG_SCHEMA,
            "workflow": "txt2vid",
            "prompt": "A sweeping landscape",
            "model": "ltx-8",
            "seed": 42,
            "steps": 8,
            "frame_count": 49,
        }
        ffprobe_output = {
            "format": {
                "tags": {
                    "zvisiongenerator.config": json.dumps(payload),
                }
            }
        }

        with patch("zvisiongenerator.utils.provenance.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(ffprobe_output))

            result = read_mp4_config(video_path)

        assert result == payload
