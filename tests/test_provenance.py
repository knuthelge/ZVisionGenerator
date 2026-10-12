"""Test provenance payload construction and serializability."""

from __future__ import annotations

import json
import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_extend_source
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.video_types import ExtendSource, ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.provenance import (
    IMAGE_CONFIG_SCHEMA,
    VIDEO_CONFIG_SCHEMA,
    build_image_config_payload,
    build_video_config_payload,
    embed_mp4_config,
    embed_png_config,
    image_prompt_text,
    optional_float,
    optional_int,
    optional_text,
    read_mp4_config,
    read_png_config,
    recorded_settings,
)


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

    def test_lora_paths_are_preserved(self):
        request = _make_image_request(
            lora_paths=[r"C:\loras\style.SAFETENSORS", "owner/detail.v1.safetensors"],
            lora_weights=[0.8, 0.5],
        )
        payload = build_image_config_payload(request, ImageWorkingArtifacts(image=Image.new("RGB", (32, 32))))

        assert payload["lora"] == r"C:\loras\style.SAFETENSORS:0.8,owner/detail.v1.safetensors:0.5"

    def test_excludes_runtime_and_output_fields(self):
        request = _make_image_request()
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)), generation_time=3.1)

        payload = build_image_config_payload(request, artifacts)

        assert "model_name" not in payload
        assert "resolved_prompt" not in payload
        assert "media_type" not in payload
        assert "output" not in payload
        assert "source" not in payload
        assert "image_strength" not in payload

    def test_records_negative_prompt_scheduler_and_family(self):
        request = _make_image_request(negative_prompt="blurry", scheduler="pkg.BetaScheduler", scheduler_name="beta")
        payload = build_image_config_payload(request, ImageWorkingArtifacts(image=Image.new("RGB", (512, 384))))

        assert payload["negative_prompt"] == "blurry"
        assert payload["scheduler"] == "beta"
        assert payload["model_family"] == "zimage"

    def test_suppressed_negative_prompt_is_not_recorded(self):
        request = _make_image_request(negative_prompt="blurry")
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (512, 384)), metadata={"negative_suppressed": True})

        payload = build_image_config_payload(request, artifacts)

        assert payload["negative_prompt"] is None

    def test_generation_records_time_upscale_and_postprocessing(self):
        request = _make_image_request(
            upscale_factor=2,
            upscale_denoise=0.4,
            upscale_steps=3,
            sharpen=True,
            sharpen_amount_upscaled=1.2,
            contrast=True,
            contrast_amount=1.1,
            saturation=False,
        )
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (1024, 768)), generation_time=12.345, was_upscaled=True)

        payload = build_image_config_payload(request, artifacts)

        assert payload["generation"] == {
            "time": 12.3,
            "upscale": {"factor": 2, "denoise": 0.4, "steps": 3, "pre_sharpen": 0.4},
            "sharpen": 1.2,
            "contrast": 1.1,
        }

    def test_generation_omits_upscale_that_did_not_run(self):
        request = _make_image_request(upscale_factor=2, upscale_denoise=0.4, sharpen=False)
        payload = build_image_config_payload(request, ImageWorkingArtifacts(image=Image.new("RGB", (512, 384))))

        assert payload["generation"] == {}

    def test_img2img_records_strength(self):
        request = _make_image_request(image_path="/tmp/ref.png", image_strength=0.6)
        payload = build_image_config_payload(request, ImageWorkingArtifacts(image=Image.new("RGB", (512, 384))))

        assert payload["image_strength"] == 0.6

    def test_upscale_of_an_img2img_image_keeps_its_reference_and_source_workflow(self):
        request = _make_image_request(upscale_source="/out/p.png", upscale_factor=2, upscale_denoise=0.4, image_path="/out/ref.png", image_strength=0.6)
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (1024, 768)), metadata={"upscale_source": {"path": "/out/p.png", "width": 512, "height": 384}})

        payload = build_image_config_payload(request, artifacts)

        assert payload["workflow"] == "upscale"
        assert payload["image_path"] == "/out/ref.png"
        assert payload["image_strength"] == 0.6
        assert payload["source"]["workflow"] == "img2img"

    def test_upscale_workflow_records_source(self):
        request = _make_image_request(upscale_source="/out/portrait.png", upscale_factor=2, upscale_denoise=0.4)
        source = {"path": "/out/portrait.png", "width": 512, "height": 384}
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (1024, 768)), was_upscaled=True, metadata={"upscale_source": source})

        payload = build_image_config_payload(request, artifacts)

        assert payload["workflow"] == "upscale"
        assert payload["source"] == {**source, "workflow": "txt2img"}
        assert payload["generation"]["upscale"]["factor"] == 2

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

    def test_excludes_runtime_and_output_fields(self):
        request = _make_video_request()
        artifacts = VideoWorkingArtifacts(generation_time=12.5, resolved_prompt="resolved version")

        payload = build_video_config_payload(request, artifacts)

        assert "model_name" not in payload
        assert "resolved_prompt" not in payload
        assert "media_type" not in payload
        assert "output" not in payload
        assert "no_audio" not in payload

    def test_generation_records_time_upscale_audio_and_format(self):
        request = _make_video_request(upscale=2, upscale_steps=3, no_audio=True)
        payload = build_video_config_payload(request, VideoWorkingArtifacts(generation_time=12.54))

        assert payload["model_family"] == "ltx"
        assert payload["generation"] == {"time": 12.5, "upscale": {"factor": 2, "steps": 3}, "audio": False, "output_format": "mp4"}

    def test_workflow_is_txt2vid_without_image_path(self):
        request = _make_video_request()
        payload = build_video_config_payload(request, VideoWorkingArtifacts())
        assert payload["workflow"] == "txt2vid"

    def test_is_json_serializable(self):
        request = _make_video_request()
        payload = build_video_config_payload(request, VideoWorkingArtifacts())
        json.dumps(payload)

    def test_plain_txt2vid_payload_is_unchanged_by_scene_support(self):
        """Golden: a video without a scene or enhancement embeds exactly the keys it did before scenes existed."""
        payload = build_video_config_payload(_make_video_request(), VideoWorkingArtifacts(generation_time=12.5))
        assert payload == {
            "schema": VIDEO_CONFIG_SCHEMA,
            "workflow": "txt2vid",
            "prompt": "A sweeping landscape",
            "model": "ltx-8",
            "model_family": "ltx",
            "seed": 42,
            "steps": 8,
            "width": 704,
            "height": 448,
            "ratio": None,
            "size": None,
            "frame_count": 49,
            "image_path": None,
            "lora": None,
            "generation": {"time": 12.5, "audio": True, "output_format": "mp4"},
        }

    def test_plain_img2vid_payload_is_unchanged_by_scene_support(self):
        """Golden: the same holds for image-to-video."""
        payload = build_video_config_payload(_make_video_request(image_path="/tmp/ref.png"), VideoWorkingArtifacts())
        assert payload["workflow"] == "img2vid" and payload["image_path"] == "/tmp/ref.png"
        assert set(payload) == {"schema", "workflow", "prompt", "model", "model_family", "seed", "steps", "width", "height", "ratio", "size", "frame_count", "image_path", "lora", "generation"}

    def test_enhanced_pair_payload_records_inputs_and_the_scene_apart_from_the_prompt(self):
        """Changed on purpose: an enhanced video records the user's input beside the enhanced scene so Reuse can offer both."""
        request = _make_video_request(prompt="a knight rides in", scene="A castle.")
        artifacts = VideoWorkingArtifacts(metadata={"scene": "A castle.", "enhanced_prompt": "A knight rides in at a gallop.", "enhanced_scene": "A grey castle at dusk."})
        payload = build_video_config_payload(request, artifacts)
        assert payload["prompt"] == "A knight rides in at a gallop."
        assert payload["scene"] == "A grey castle at dusk."
        assert payload["prompt_input"] == "a knight rides in"
        assert payload["scene_input"] == "A castle."
        assert "castle" not in payload["prompt"]

    def test_scene_key_present_only_with_a_scene(self):
        request = _make_video_request()
        assert "scene" not in build_video_config_payload(request, VideoWorkingArtifacts(metadata={"scene": "   "}))
        assert build_video_config_payload(request, VideoWorkingArtifacts(metadata={"scene": "A castle."}))["scene"] == "A castle."

    def test_plain_scene_run_records_no_inputs(self):
        request = _make_video_request(scene="A castle.")
        payload = build_video_config_payload(request, VideoWorkingArtifacts(metadata={"scene": "A castle."}))
        assert payload["scene"] == "A castle."
        assert "scene_input" not in payload and "prompt_input" not in payload

    def test_enhanced_without_input_text_omits_blank_inputs(self):
        """A scene-only run that is enhanced has no prompt input, and an unscened one has no scene input."""
        scene_only = build_video_config_payload(
            _make_video_request(prompt="", scene="A castle."),
            VideoWorkingArtifacts(metadata={"enhanced_prompt": "A knight rides in.", "enhanced_scene": "A grey castle."}),
        )
        assert scene_only["scene_input"] == "A castle." and "prompt_input" not in scene_only
        unscened = build_video_config_payload(
            _make_video_request(prompt="a knight rides in a castle"),
            VideoWorkingArtifacts(metadata={"enhanced_prompt": "A knight rides in.", "enhanced_scene": "A grey castle."}),
        )
        assert unscened["prompt_input"] == "a knight rides in a castle" and "scene_input" not in unscened
        assert unscened["scene"] == "A grey castle."

    def test_inputs_given_with_the_request_are_recorded_without_enhancement(self):
        """The Web UI enhances before submit: prompt and scene already hold the rewrite, the request carries the user's text."""
        request = _make_video_request(prompt="A knight rides in.", scene="A grey castle.", scene_input="A castle.", prompt_input="a knight rides in")
        payload = build_video_config_payload(request, VideoWorkingArtifacts(metadata={"scene": "A grey castle."}))
        assert payload["prompt"] == "A knight rides in." and payload["scene"] == "A grey castle."
        assert payload["scene_input"] == "A castle." and payload["prompt_input"] == "a knight rides in"

    def test_recorded_settings_round_trip(self):
        request = _make_video_request(prompt="a knight rides in", scene="A castle.")
        artifacts = VideoWorkingArtifacts(metadata={"enhanced_prompt": "A knight rides in at a gallop.", "enhanced_scene": "A grey castle at dusk."})
        settings = recorded_settings(json.loads(json.dumps(build_video_config_payload(request, artifacts))))
        assert (settings.prompt, settings.scene) == ("A knight rides in at a gallop.", "A grey castle at dusk.")
        assert (settings.prompt_input, settings.scene_input) == ("a knight rides in", "A castle.")

    def test_recorded_settings_without_scene_keys_are_none(self):
        settings = recorded_settings({"schema": VIDEO_CONFIG_SCHEMA, "prompt": "a sunset"})
        assert settings.scene is None and settings.scene_input is None and settings.prompt_input is None


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

    def test_keeps_custom_metadata_key_in_mp4_header(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-mp4")

        def _side_effect(cmd, **kwargs):
            video_path.with_suffix(".tmp.mp4").write_bytes(b"processed")
            return MagicMock(returncode=0)

        with patch("zvisiongenerator.utils.provenance.subprocess.run", side_effect=_side_effect) as mock_run:
            embed_mp4_config(video_path, {"schema": VIDEO_CONFIG_SCHEMA})

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-movflags") + 1] == "use_metadata_tags"

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


class TestRecordedValueHelpers:
    def test_optional_values_reject_missing_blank_bool_and_garbage(self):
        for value in (None, "", True, "abc"):
            assert optional_int(value) is None
            assert optional_float(value) is None
        assert optional_int("42") == 42
        assert optional_float("0.4") == 0.4
        assert optional_text("  beta ") == "beta"
        assert optional_text("   ") is None

    def test_image_prompt_text_reads_png_description(self, tmp_path):
        path = tmp_path / "p.png"
        info = PngInfo()
        info.add_text("Description", "a red barn")
        Image.new("RGB", (8, 8)).save(path, pnginfo=info)

        with Image.open(path) as image:
            assert image_prompt_text(image) == "a red barn"

    def test_image_prompt_text_is_none_without_a_prompt(self, tmp_path):
        path = tmp_path / "plain.png"
        Image.new("RGB", (8, 8)).save(path)

        with Image.open(path) as image:
            assert image_prompt_text(image) is None


class TestRecordedSettings:
    def test_reads_top_level_settings_only(self):
        # A missing top-level value must not be filled from a nested block, e.g. the upscale's own steps.
        settings = recorded_settings({"prompt": "p", "seed": "7", "generation": {"upscale": {"factor": 2, "steps": 3}}})

        assert settings.prompt == "p"
        assert settings.seed == 7
        assert settings.steps is None
        assert settings.generation == {"upscale": {"factor": 2, "steps": 3}}

    def test_reads_legacy_lora_lists(self):
        settings = recorded_settings({"lora": [{"name": "style", "weight": 0.8}, "detail"]})

        assert settings.lora == "style:0.8,detail"

    def test_drops_malformed_blocks(self):
        settings = recorded_settings({"generation": "oops", "source": {"width": 3}})

        assert settings.generation == {}
        assert settings.source is None


class TestHomePathsArePortable:
    @staticmethod
    def _home(monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/me")))

    def test_image_payload_records_home_paths_with_tilde(self, monkeypatch):
        self._home(monkeypatch)
        request = _make_image_request(
            model_name="/home/me/.ziv/models/zit",
            image_path="/home/me/out/ref.png",
            lora_paths=["/home/me/.ziv/loras/style.safetensors"],
            lora_weights=[0.7],
            upscale_source="/home/me/out/p.png",
            upscale_factor=2,
        )
        artifacts = ImageWorkingArtifacts(image=Image.new("RGB", (8, 8)), metadata={"upscale_source": {"path": "/home/me/out/p.png", "width": 8, "height": 8}})

        payload = build_image_config_payload(request, artifacts)

        assert "/home/me" not in json.dumps(payload)
        assert payload["model"] == "~/.ziv/models/zit"
        assert payload["image_path"] == "~/out/ref.png"
        assert payload["lora"] == "~/.ziv/loras/style.safetensors:0.7"
        assert payload["source"]["path"] == "~/out/p.png"

    def test_video_payload_records_home_paths_with_tilde(self, monkeypatch):
        self._home(monkeypatch)
        request = VideoGenerationRequest(
            backend=MagicMock(),
            model=MagicMock(),
            prompt="p",
            model_name="/home/me/.ziv/models/ltx",
            image_path="/home/me/ref.png",
            lora_paths=["/home/me/.ziv/loras/motion.safetensors"],
        )

        payload = build_video_config_payload(request, VideoWorkingArtifacts())

        assert "/home/me" not in json.dumps(payload)

    def test_recorded_settings_expand_tilde_paths(self, monkeypatch):
        self._home(monkeypatch)
        settings = recorded_settings(
            {
                "model": "~/.ziv/models/zit",
                "image_path": "~/out/ref.png",
                "lora": "~/.ziv/loras/a.safetensors:0.7,~/.ziv/loras/b.safetensors:1",
                "source": {"path": "~/out/p.png"},
            }
        )

        assert settings.model == str(Path("/home/me")) + "/.ziv/models/zit"
        assert settings.image_path == str(Path("/home/me")) + "/out/ref.png"
        assert settings.lora == f"{Path('/home/me')}/.ziv/loras/a.safetensors:0.7,{Path('/home/me')}/.ziv/loras/b.safetensors:1"
        assert settings.source is not None and settings.source["path"] == str(Path("/home/me")) + "/out/p.png"


_make_extend_source = partial(make_extend_source, scene="A woman in a red raincoat on a rainy street at night.")


def _make_extend_request(source: ExtendSource | None = None, frames: int = 48, **overrides) -> VideoGenerationRequest:
    source = source or _make_extend_source()
    defaults: dict = dict(width=source.width, height=source.height, num_frames=25 + frames, extend=ExtendSpec(source, frames), prompt="")
    defaults.update(overrides)
    return _make_video_request(**defaults)


class TestBuildExtendConfigPayload:
    def test_plain_extend_payload_keys_and_values(self):
        request = _make_extend_request(prompt="She walks away.", scene="A woman in a red raincoat on a rainy street at night.", lora_paths=["/models/motion.safetensors"], lora_weights=[0.7])
        artifacts = VideoWorkingArtifacts(
            generation_time=30.04,
            metadata={"scene": "A woman in a red raincoat on a rainy street at night.", "prompt_sent": "She walks away."},
        )
        assert build_video_config_payload(request, artifacts) == {
            "schema": VIDEO_CONFIG_SCHEMA,
            "workflow": "extend",
            "scene": "A woman in a red raincoat on a rainy street at night.",
            "prompt": "She walks away.",
            "model": "ltx-8",
            "model_family": "ltx",
            "seed": 42,
            "steps": 8,
            "width": 512,
            "height": 256,
            "ratio": None,
            "size": None,
            "frame_count": 97,
            "image_path": None,
            "lora": "/models/motion.safetensors:0.7",
            "generation": {"time": 30.0, "audio": True, "output_format": "mp4"},
            "source_video": "/videos/clip.mp4",
            "extend_frames": 48,
        }

    def test_frame_count_is_the_whole_clip_not_the_window(self):
        request = _make_extend_request(frames=16)
        payload = build_video_config_payload(request, VideoWorkingArtifacts())
        assert request.num_frames == 41
        assert payload["frame_count"] == 49 + 16 and payload["extend_frames"] == 16

    def test_size_is_the_displayed_size_of_a_rotated_source(self):
        source = _make_extend_source(width=256, height=512)
        payload = build_video_config_payload(_make_extend_request(source, width=999, height=999), VideoWorkingArtifacts())
        assert (payload["width"], payload["height"]) == (256, 512)

    def test_enhanced_extend_records_the_action_alone_and_the_inputs(self):
        request = _make_extend_request(prompt="she leaves", scene="A street at night.")
        artifacts = VideoWorkingArtifacts(
            metadata={
                "scene": "A street at night.",
                "prompt_sent": "she leaves",
                "enhanced_prompt": "She turns and walks into the rain.",
                "enhanced_scene": "A street at night.",
            }
        )
        payload = build_video_config_payload(request, artifacts)
        assert payload["prompt"] == "She turns and walks into the rain."
        assert payload["scene"] == "A street at night."
        assert payload["prompt_input"] == "she leaves" and payload["scene_input"] == "A street at night."
        assert "street" not in payload["prompt"]

    @pytest.mark.parametrize("extends", [True, False], ids=["extend", "text to video"])
    def test_an_action_only_rewrite_records_the_users_scene_and_no_enhanced_scene(self, extends):
        """No ``enhanced_scene`` metadata: ``scene`` is the user's own and ``scene_input`` repeats it, which tells it from a rewrite of a blank scene."""
        make_request = _make_extend_request if extends else _make_video_request
        artifacts = VideoWorkingArtifacts(metadata={"scene": "A street at night.", "prompt_sent": "she leaves", "enhanced_prompt": "She turns and walks into the rain."})
        payload = build_video_config_payload(make_request(prompt="she leaves", scene="A street at night."), artifacts)
        assert payload["prompt"] == "She turns and walks into the rain."
        assert payload["scene"] == "A street at night." == payload["scene_input"]
        assert payload["prompt_input"] == "she leaves"

    def test_enhanced_empty_continuation_records_no_prompt_input(self):
        request = _make_extend_request(scene="A street at night.")
        artifacts = VideoWorkingArtifacts(metadata={"enhanced_prompt": "She keeps walking.", "enhanced_scene": "A street at night."})
        payload = build_video_config_payload(request, artifacts)
        assert payload["prompt"] == "She keeps walking." and "prompt_input" not in payload

    def test_legacy_stripped_action_is_recorded_as_sent(self):
        """The old dialogue was dropped before generation: the record holds what the model got, not the template."""
        request = _make_extend_request(_make_extend_source(scene=None, prompt='She says "go." Then she sighs.'), prompt='She says "go." Then she sighs.')
        artifacts = VideoWorkingArtifacts(metadata={"prompt_sent": "Then she sighs."})
        payload = build_video_config_payload(request, artifacts)
        assert payload["prompt"] == "Then she sighs." and "scene" not in payload

    def test_empty_continuation_records_an_empty_prompt(self):
        payload = build_video_config_payload(_make_extend_request(scene="A street."), VideoWorkingArtifacts(metadata={"scene": "A street.", "prompt_sent": ""}))
        assert payload["prompt"] == "" and payload["scene"] == "A street."

    @pytest.mark.parametrize(("has_audio", "no_audio", "expected"), [(True, True, False), (False, False, False)])
    def test_audio_is_what_the_joined_clip_has(self, has_audio, no_audio, expected):
        request = _make_extend_request(_make_extend_source(has_audio=has_audio), no_audio=no_audio)
        assert build_video_config_payload(request, VideoWorkingArtifacts())["generation"]["audio"] is expected

    def test_source_video_is_recorded_with_the_home_directory_as_tilde(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        request = _make_extend_request(_make_extend_source(path=tmp_path / "videos" / "clip.mp4"))
        assert build_video_config_payload(request, VideoWorkingArtifacts())["source_video"] == "~/videos/clip.mp4"
        settings = recorded_settings(build_video_config_payload(request, VideoWorkingArtifacts()))
        assert settings.source_video == str(tmp_path / "videos" / "clip.mp4")

    def test_recorded_settings_round_trip(self):
        request = _make_extend_request(prompt="She walks away.", scene="A street.")
        artifacts = VideoWorkingArtifacts(metadata={"scene": "A street.", "prompt_sent": "She walks away."})
        settings = recorded_settings(json.loads(json.dumps(build_video_config_payload(request, artifacts))))
        assert (settings.workflow, settings.scene, settings.prompt) == ("extend", "A street.", "She walks away.")
        assert (settings.source_video, settings.extend_frames, settings.frame_count) == ("/videos/clip.mp4", 48, 97)

    def test_recorded_settings_of_other_videos_have_no_extend_fields(self):
        settings = recorded_settings(build_video_config_payload(_make_video_request(), VideoWorkingArtifacts()))
        assert settings.source_video is None and settings.extend_frames is None

    def test_extend_of_extend_copies_the_scene_unchanged_and_uses_the_prompt_as_what_just_happened(self):
        first = _make_extend_request(prompt="She walks away.", scene="A street at night.")
        first_payload = build_video_config_payload(first, VideoWorkingArtifacts(metadata={"scene": "A street at night.", "prompt_sent": "She walks away."}))
        recorded = recorded_settings(first_payload)
        # The next link starts from the joined clip, pre-filled with the recorded scene, and sends it as it is.
        second_source = _make_extend_source(path=Path("/videos/clip-extended.mp4"), frame_count=first_payload["frame_count"], scene=recorded.scene, prompt=recorded.prompt)
        second = _make_extend_request(second_source, prompt="A taxi stops.", scene=recorded.scene)
        second_payload = build_video_config_payload(second, VideoWorkingArtifacts(metadata={"scene": recorded.scene, "prompt_sent": "A taxi stops."}))
        assert second_payload["scene"] == first_payload["scene"] == "A street at night."
        assert second_source.prompt == "She walks away."
        assert second_payload["frame_count"] == 97 + 48 and second_payload["source_video"] == "/videos/clip-extended.mp4"
