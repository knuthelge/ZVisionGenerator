"""Tests for video workflow stage functions."""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_extend_source

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource, ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.workflows.video_stages import (
    enhance_prompt_stage,
    extend_video_stage,
    generate_filename_stage,
    image_to_video_stage,
    join_clips_stage,
    log_video_stage,
    resolve_prompt_stage,
    text_to_video_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(**overrides) -> VideoGenerationRequest:
    """Build a minimal VideoGenerationRequest for stage testing."""
    defaults: dict = dict(
        backend=MagicMock(),
        model=MagicMock(),
        prompt="a beautiful sunset",
    )
    defaults.update(overrides)
    return VideoGenerationRequest(**defaults)


def _source(tmp_path: Path | None = None, **overrides) -> ExtendSource:
    """Build a checked extend source: a 24 fps clip with sound that was recorded with a scene."""
    return make_extend_source(**{"path": (tmp_path or Path("/videos")) / "clip.mp4", "prompt": 'She says "I told you." Then she sighs.', **overrides})


def _extend_req(source: ExtendSource | None = None, frames: int = 48, **overrides) -> VideoGenerationRequest:
    """Build the request of an extension: ``num_frames`` is the generation window (25 + the new frames)."""
    defaults: dict = dict(prompt="", width=512, height=256, num_frames=25 + frames, extend=ExtendSpec(source or _source(), frames))
    defaults.update(overrides)
    return _req(**defaults)


# ---------------------------------------------------------------------------
# resolve_prompt_stage
# ---------------------------------------------------------------------------


class TestResolvePromptStage:
    """Verify random-choice block expansion in video prompts."""

    def test_plain_prompt_passthrough(self):
        req = _req(prompt="a simple prompt")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt == "a simple prompt"

    def test_choice_block_resolves(self):
        req = _req(prompt="{red|green|blue} sky")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt in ("red sky", "green sky", "blue sky")

    def test_nested_choice_blocks(self):
        req = _req(prompt="{a|{b|c}}")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt in ("a", "b", "c")

    def test_whitespace_stripped(self):
        req = _req(prompt="  hello world  ")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "hello world"

    def test_blank_scene_sends_the_prompt_byte_identical(self):
        for scene in (None, "", "   "):
            req = _req(prompt="a sunset", resolved_prompt="a sunset", scene=scene, resolved_scene=None if scene is None else scene)
            arts = VideoWorkingArtifacts()
            resolve_prompt_stage(req, arts)
            assert arts.resolved_prompt == "a sunset"
            assert "scene" not in arts.metadata

    def test_scene_comes_before_the_prompt(self):
        req = _req(prompt="a knight rides in", resolved_prompt="a knight rides in", scene="A castle.", resolved_scene="A castle.")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "A castle. a knight rides in"
        assert arts.metadata["scene"] == "A castle."

    def test_scene_only_run_sends_the_scene(self):
        req = _req(prompt="", resolved_prompt="", scene="A castle at dusk.", resolved_scene="A castle at dusk.")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "A castle at dusk."

    def test_choices_expand_in_each_part_without_a_plan(self):
        req = _req(prompt="a {fox|fox} runs", scene="A {snowy|snowy} field.")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "A snowy field. a fox runs"
        assert arts.metadata["scene"] == "A snowy field."

    def test_planned_resolved_scene_wins_over_the_template(self):
        req = _req(prompt="p", resolved_prompt="p", scene="{a|b}", resolved_scene="b")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "b p" and arts.metadata["scene"] == "b"


class TestResolvePromptStageExtend:
    """Verify the action an extension sends: the scene first, and a legacy prompt's old dialogue stripped only while unchanged."""

    _OLD = 'A woman stands in the rain. She says "I told you." Then she sighs.'

    def _resolve(self, source: ExtendSource, prompt: str, **overrides) -> VideoWorkingArtifacts:
        arts = VideoWorkingArtifacts()
        assert resolve_prompt_stage(_extend_req(source, prompt=prompt, **overrides), arts) is StageOutcome.success
        return arts

    def test_scene_bearing_source_sends_scene_then_action_and_never_strips(self):
        arts = self._resolve(_source(), 'She whispers "go now."', scene="A street at night.")
        assert arts.resolved_prompt == 'A street at night. She whispers "go now."'
        assert arts.metadata["prompt_sent"] == 'She whispers "go now."'

    def test_empty_prompt_sends_the_scene_only(self):
        arts = self._resolve(_source(), "", scene="A street at night.")
        assert arts.resolved_prompt == "A street at night."
        assert arts.metadata["prompt_sent"] == ""

    def test_scene_bearing_source_with_its_own_prompt_is_not_stripped(self):
        arts = self._resolve(_source(prompt=self._OLD), self._OLD)
        assert arts.resolved_prompt == self._OLD

    def test_legacy_source_with_the_unchanged_prompt_loses_its_quoted_sentences(self):
        arts = self._resolve(_source(scene=None, prompt=self._OLD), self._OLD)
        assert arts.resolved_prompt == "A woman stands in the rain. Then she sighs."
        assert arts.metadata["prompt_sent"] == "A woman stands in the rain. Then she sighs."

    def test_legacy_prompt_differing_only_in_whitespace_still_counts_as_unchanged(self):
        arts = self._resolve(_source(scene=None, prompt=self._OLD), "  " + self._OLD.replace(". ", ".\n"))
        assert '"' not in arts.resolved_prompt

    def test_legacy_source_with_an_edited_prompt_is_sent_as_written(self):
        edited = 'She says "run."'
        arts = self._resolve(_source(scene=None, prompt=self._OLD), edited)
        assert arts.resolved_prompt == edited

    def test_legacy_comparison_uses_the_template_before_choice_expansion(self):
        template = 'She says "{hi|hi}." Then {she|she} sighs.'
        arts = self._resolve(_source(scene=None, prompt=template), template)
        assert arts.resolved_prompt == "Then she sighs."

    def test_legacy_source_with_a_scene_typed_in_still_strips_an_unchanged_prompt(self):
        arts = self._resolve(_source(scene=None, prompt=self._OLD), self._OLD, scene="A street at night.")
        assert arts.resolved_prompt == "A street at night. A woman stands in the rain. Then she sighs."

    def test_other_workflows_do_not_record_a_prompt_sent(self):
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(_req(prompt="a sunset"), arts)
        assert "prompt_sent" not in arts.metadata


# ---------------------------------------------------------------------------
# enhance_prompt_stage
# ---------------------------------------------------------------------------


class TestEnhancePromptStage:
    """Verify the planned scene and action rewrite is applied without leaking the scene into the action."""

    def test_not_enhanced_is_a_no_op(self):
        seen: list[str] = []
        arts = VideoWorkingArtifacts(resolved_prompt="A castle. a knight")
        assert enhance_prompt_stage(_req(on_prompt_enhanced=seen.append), arts) is StageOutcome.success
        assert arts.resolved_prompt == "A castle. a knight" and "enhanced_prompt" not in arts.metadata and seen == []

    def test_pair_is_composed_into_resolved_prompt_only(self):
        seen: list[str] = []
        req = _req(enhanced_prompt="A knight rides in.", enhanced_scene="A castle at dusk.", on_prompt_enhanced=seen.append)
        arts = VideoWorkingArtifacts(resolved_prompt="A castle. a knight")
        enhance_prompt_stage(req, arts)
        assert arts.resolved_prompt == "A castle at dusk. A knight rides in."
        assert arts.metadata["enhanced_prompt"] == "A knight rides in."
        assert arts.metadata["enhanced_scene"] == "A castle at dusk."
        assert seen == ["A castle at dusk. A knight rides in."]

    def test_action_without_a_scene_behaves_like_a_plain_rewrite(self):
        req = _req(enhanced_prompt="A fox sprints.")
        arts = VideoWorkingArtifacts()
        enhance_prompt_stage(req, arts)
        assert arts.resolved_prompt == "A fox sprints." and arts.metadata == {"enhanced_prompt": "A fox sprints."}


# ---------------------------------------------------------------------------
# generate_filename_stage
# ---------------------------------------------------------------------------


class TestGenerateFilenameStage:
    """Verify the set-name + timestamp filename and its collision counter."""

    def test_set_name_and_timestamp(self, tmp_path):
        req = _req(filename_base="sunset", output_dir=str(tmp_path), output_format="mp4")
        arts = VideoWorkingArtifacts()
        outcome = generate_filename_stage(req, arts)
        assert outcome is StageOutcome.success
        assert re.fullmatch(r"sunset_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4", arts.filename)

    def test_settings_are_not_in_the_name(self, tmp_path):
        req = _req(filename_base="sunset", model_name="dgrauet/ltx-2.3-mlx-q4", seed=42, output_dir=str(tmp_path))
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert "ltx" not in arts.filename
        assert "seed" not in arts.filename

    def test_taken_name_gets_counter(self, tmp_path):
        req = _req(filename_base="sunset", output_dir=str(tmp_path), output_format="mp4")
        with patch("zvisiongenerator.utils.filename.generate_filename", return_value="sunset_2026-10-04_14-03-22"):
            (tmp_path / "sunset_2026-10-04_14-03-22.mp4").write_bytes(b"")
            arts = VideoWorkingArtifacts()
            generate_filename_stage(req, arts)
        assert arts.filename == "sunset_2026-10-04_14-03-22_2.mp4"


# ---------------------------------------------------------------------------
# text_to_video_stage
# ---------------------------------------------------------------------------


class TestTextToVideoStage:
    """Verify text_to_video_stage delegates to backend correctly."""

    def test_success_sets_artifacts(self):
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/out.mp4")
        req = _req(backend=backend, prompt="test", output_dir="/tmp", output_format="mp4", seed=42, width=704, height=480, num_frames=49, steps=30)
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test"

        outcome = text_to_video_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/out.mp4")
        assert arts.generation_time > 0
        backend.text_to_video.assert_called_once()

    def test_backend_returns_none_fails(self):
        backend = MagicMock()
        backend.text_to_video.return_value = None
        req = _req(backend=backend)
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test"

        outcome = text_to_video_stage(req, arts)
        assert outcome is StageOutcome.failed
        assert arts.video_path is None

    def test_uses_resolved_prompt(self):
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/x.mp4")
        req = _req(backend=backend, prompt="original")
        arts = VideoWorkingArtifacts()
        arts.filename = "f.mp4"
        arts.resolved_prompt = "resolved version"

        text_to_video_stage(req, arts)
        call_kwargs = backend.text_to_video.call_args
        assert call_kwargs.kwargs.get("prompt") == "resolved version" or call_kwargs[1].get("prompt") == "resolved version"


# ---------------------------------------------------------------------------
# image_to_video_stage
# ---------------------------------------------------------------------------


class TestImageToVideoStage:
    """Verify image_to_video_stage delegates to backend correctly."""

    def test_success_sets_artifacts(self):
        backend = MagicMock()
        backend.image_to_video.return_value = Path("/tmp/i2v.mp4")
        req = _req(backend=backend, image_path="/tmp/input.png", output_dir="/tmp", output_format="mp4")
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"
        arts.resolved_prompt = "describe this"

        outcome = image_to_video_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/i2v.mp4")
        backend.image_to_video.assert_called_once()

    def test_backend_returns_none_fails(self):
        backend = MagicMock()
        backend.image_to_video.return_value = None
        req = _req(backend=backend, image_path="/tmp/input.png")
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"

        outcome = image_to_video_stage(req, arts)
        assert outcome is StageOutcome.failed


# ---------------------------------------------------------------------------
# log_video_stage
# ---------------------------------------------------------------------------


class TestLogVideoStage:
    """Verify log_video_stage embeds config metadata and logs correctly."""

    def test_returns_success_without_mutating_artifacts(self):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = Path("/tmp/out.mp4")
        arts.generation_time = 12.5

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config"):
            outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/out.mp4")
        assert arts.generation_time == 12.5

    def test_calls_embed_mp4_config_with_video_path(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-video")
        req = _req(
            prompt="a moving scene",
            model_name="ltx-8",
            model_family="ltx",
            width=704,
            height=448,
            num_frames=49,
            seed=77,
            steps=8,
            image_path="/tmp/ref.png",
        )
        arts = VideoWorkingArtifacts(video_path=video_path, generation_time=12.5, filename="out.mp4")

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as mock_embed:
            outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        mock_embed.assert_called_once()
        call_path, call_payload = mock_embed.call_args[0]
        assert call_path == video_path
        assert call_payload["prompt"] == "a moving scene"
        assert call_payload["model"] == "ltx-8"
        assert call_payload["seed"] == 77
        assert call_payload["frame_count"] == 49
        assert call_payload["workflow"] == "img2vid"

    def test_warns_and_succeeds_on_ffmpeg_embed_failure(self):
        req = _req(prompt="test", model_name="ltx")
        arts = VideoWorkingArtifacts(video_path=Path("/tmp/out.mp4"), generation_time=5.0)

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as mock_embed:
            mock_embed.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"mux error")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert any("ffmpeg metadata embed failed" in str(w.message) for w in caught)

    def test_none_path_still_succeeds(self):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = None
        arts.generation_time = 0.0

        outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert arts.video_path is None
        assert arts.generation_time == 0.0


# ---------------------------------------------------------------------------
# extend_video_stage
# ---------------------------------------------------------------------------


class TestExtendVideoStage:
    """Verify the extension is generated through backend.extend_video() and its segment directory is cleaned up on failure."""

    def _run(self, tmp_path, *, source=None, result="segment", backend=None, **overrides):
        source = source or _source(tmp_path)
        backend = backend if backend is not None else MagicMock()
        segment = ExtendSegment(tmp_path / "seg.mp4", tmp_path / "seg.wav")
        if result == "segment":
            backend.extend_video.return_value = segment
        elif result is None:
            backend.extend_video.return_value = None
        req = _extend_req(source, backend=backend, output_dir=str(tmp_path), seed=7, steps=6, **overrides)
        arts = VideoWorkingArtifacts(resolved_prompt="A street at night. She walks away.")
        return req, arts, segment

    @pytest.fixture(autouse=True)
    def _scratch(self, tmp_path, monkeypatch):
        """Point the system temp directory at a folder of its own, so the segment directory can be found and counted."""
        scratch = tmp_path / "system-temp"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        return scratch

    def _leftovers(self, tmp_path):
        return sorted(p.name for p in (tmp_path / "system-temp").iterdir())

    def test_passes_the_request_to_the_backend(self, tmp_path):
        callback = MagicMock()
        req, arts, segment = self._run(tmp_path, step_callback=callback)
        assert extend_video_stage(req, arts) is StageOutcome.success
        kwargs = req.backend.extend_video.call_args.kwargs
        assert kwargs["model"] is req.model
        assert kwargs["source"] is req.extend.source
        assert kwargs["prompt"] == "A street at night. She walks away."
        assert kwargs["extend_frames"] == 48
        assert (kwargs["seed"], kwargs["steps"]) == (7, 6)
        assert kwargs["step_callback"] is callback
        assert kwargs["audio_output"] is True
        segment_dir = Path(kwargs["segment_dir"])
        assert segment_dir.parent == tmp_path / "system-temp" and segment_dir.name.startswith("ziv-extend-")
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
        assert arts.metadata["extend_segment"] is segment
        assert arts.metadata["extend_segment_dir"] == str(segment_dir)
        assert segment_dir.is_dir()

    def test_generation_time_covers_the_backend_call(self, tmp_path):
        req, arts, _ = self._run(tmp_path)
        with patch("zvisiongenerator.workflows.video_stages.time.monotonic", side_effect=[100.0, 112.5]):
            extend_video_stage(req, arts)
        assert arts.generation_time == 12.5

    @pytest.mark.parametrize(("has_audio", "no_audio", "expected"), [(True, True, False), (False, False, False)])
    def test_audio_is_requested_only_for_a_source_with_sound_and_no_no_audio(self, tmp_path, has_audio, no_audio, expected):
        req, arts, _ = self._run(tmp_path, source=_source(tmp_path, has_audio=has_audio), no_audio=no_audio)
        extend_video_stage(req, arts)
        assert req.backend.extend_video.call_args.kwargs["audio_output"] is expected

    def test_no_segment_fails_and_removes_the_directory(self, tmp_path):
        req, arts, _ = self._run(tmp_path, result=None)
        assert extend_video_stage(req, arts) is StageOutcome.failed
        assert self._leftovers(tmp_path) == []
        assert "extend_segment" not in arts.metadata and "extend_segment_dir" not in arts.metadata

    def test_backend_error_removes_the_directory_and_propagates(self, tmp_path):
        req, arts, _ = self._run(tmp_path)
        req.backend.extend_video.side_effect = ValueError("upscaled models cannot extend")
        with pytest.raises(ValueError, match="upscaled"):
            extend_video_stage(req, arts)
        assert self._leftovers(tmp_path) == []

    def test_cancellation_removes_the_directory(self, tmp_path):
        req, arts, _ = self._run(tmp_path)
        req.backend.extend_video.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            extend_video_stage(req, arts)
        assert self._leftovers(tmp_path) == []

    def test_backend_without_extend_video_is_a_runtime_error(self, tmp_path):
        req, arts, _ = self._run(tmp_path, backend=MagicMock(spec=["text_to_video", "image_to_video"]), result="none")
        with pytest.raises(RuntimeError, match="cannot extend"):
            extend_video_stage(req, arts)
        assert self._leftovers(tmp_path) == []

    def test_requires_an_extend_spec(self):
        with pytest.raises(ValueError, match="extend"):
            extend_video_stage(_req(), VideoWorkingArtifacts())

    def test_falls_back_to_the_request_prompt_without_a_resolved_one(self, tmp_path):
        req, _, _ = self._run(tmp_path)
        req = replace(req, prompt="walk away")
        extend_video_stage(req, VideoWorkingArtifacts())
        assert req.backend.extend_video.call_args.kwargs["prompt"] == "walk away"


# ---------------------------------------------------------------------------
# join_clips_stage
# ---------------------------------------------------------------------------


class TestJoinClipsStage:
    """Verify the segment is spliced onto the source into the output file and the segment directory is always removed."""

    def _arts(self, tmp_path):
        segment_dir = tmp_path / "ziv-extend-abc"
        segment_dir.mkdir()
        segment = ExtendSegment(segment_dir / "seg.mp4", segment_dir / "seg.wav")
        arts = VideoWorkingArtifacts(filename="out.mp4", metadata={"extend_segment": segment, "extend_segment_dir": str(segment_dir)})
        return arts, segment, segment_dir

    def test_splices_into_the_output_file_and_removes_the_directory(self, tmp_path):
        source = _source(tmp_path)
        req = _extend_req(source, output_dir=str(tmp_path))
        arts, segment, segment_dir = self._arts(tmp_path)
        with patch("zvisiongenerator.workflows.video_stages.splice_clips", return_value=tmp_path / "out.mp4") as splice:
            assert join_clips_stage(req, arts) is StageOutcome.success
        splice.assert_called_once_with(source, segment, tmp_path / "out.mp4", extend_frames=48, include_audio=True)
        assert arts.video_path == tmp_path / "out.mp4"
        assert not segment_dir.exists()
        assert "extend_segment" not in arts.metadata

    @pytest.mark.parametrize(("has_audio", "no_audio", "expected"), [(True, True, False), (False, False, False)])
    def test_include_audio_needs_source_sound_and_no_no_audio(self, tmp_path, has_audio, no_audio, expected):
        req = _extend_req(_source(tmp_path, has_audio=has_audio), no_audio=no_audio, output_dir=str(tmp_path))
        arts, _, _ = self._arts(tmp_path)
        with patch("zvisiongenerator.workflows.video_stages.splice_clips") as splice:
            join_clips_stage(req, arts)
        assert splice.call_args.kwargs["include_audio"] is expected

    @pytest.mark.parametrize("error", [RuntimeError("ffmpeg could not join the clips"), subprocess.CalledProcessError(1, "ffprobe", stderr=b"bad file")])
    def test_failed_splice_warns_fails_and_removes_the_directory(self, tmp_path, error):
        req = _extend_req(output_dir=str(tmp_path))
        arts, _, segment_dir = self._arts(tmp_path)
        with patch("zvisiongenerator.workflows.video_stages.splice_clips", side_effect=error):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outcome = join_clips_stage(req, arts)
        assert outcome is StageOutcome.failed
        assert any("Joining the clips failed" in str(w.message) for w in caught)
        assert arts.video_path is None
        assert not segment_dir.exists()

    def test_unexpected_error_still_removes_the_directory(self, tmp_path):
        req = _extend_req(output_dir=str(tmp_path))
        arts, _, segment_dir = self._arts(tmp_path)
        with patch("zvisiongenerator.workflows.video_stages.splice_clips", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                join_clips_stage(req, arts)
        assert not segment_dir.exists()

    def test_without_a_segment_it_is_a_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="segment"):
            join_clips_stage(_extend_req(output_dir=str(tmp_path)), VideoWorkingArtifacts(filename="out.mp4"))
