"""Run the review wrapper of ``video_mac`` through the vendored ``generate_and_save`` / ``generate_two_stage``.

The heavy steps (stage 1, upsampling, stage 2, decoder loading, the muxing decode) are replaced; the orchestration in
``DistilledPipeline`` and the ``_decode_and_save_video`` low-memory frees are the real vendored code. The point is the
claim of the review: the final output and the pipeline state after the run do not depend on whether a review ran.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytest.importorskip("ltx_pipelines_mlx")

from ltx_pipelines_mlx import DistilledPipeline  # noqa: E402
from ltx_pipelines_mlx.utils import _orchestration  # noqa: E402

from zvisiongenerator.backends.video_mac import LtxVideoBackend, _LtxModel  # noqa: E402

_STAGE1 = SimpleNamespace(video_tokens="video_tokens", audio_tokens="audio_tokens", latent_dims=(3, 7, 11))


class _Discard(Exception):
    """Stands for the runner's private discard exception."""


class _Run:
    """One generation on a pipeline whose heavy steps are replaced, and what it observed."""

    def __init__(self, monkeypatch, tmp_path: Path, *, low_memory: bool) -> None:
        self.tmp_path = tmp_path
        self.stage1_calls = 0
        self.stage1_entry: list[tuple[object, bool, object]] = []
        self.loads: list[str] = []
        self.stage2_saw: tuple[object, bool, object, object] | None = None
        self.decodes: list[tuple[object, object, str]] = []
        self.review_saw: tuple[object, bool] | None = None
        self.pipe = self._make_pipeline(low_memory)
        self.pipe.video_decoder_block.run = self

        # The stubs sit on the class, so each finds its run through the pipeline it is called on.
        self.pipe._run = self

        def _stage1(self, *args, **kwargs):
            run = self._run
            run.stage1_calls += 1
            # What is resident while the text encoder would load, and what the real load() then brings back.
            run.stage1_entry.append((self.dit, self._loaded, self.upsampler))
            if self.low_memory:
                self.prompt_encoder.free()
            self.load()
            return _STAGE1, 49, 480, 704

        def _stage2(self, stage1, video_upscaled, **kwargs):
            self._run.stage2_saw = (self.dit, self._loaded, stage1, video_upscaled)
            return ("video_latent", "audio_latent")

        def _load_decoders(self):
            self.video_decoder_block._decoder = "vae"
            self.audio_decoder_block._audio_decoder = "audio"
            self.audio_decoder_block._vocoder = "vocoder"

        def _decode(video_block, audio_block, video_latent, audio_latent, output_path, **kwargs):
            video_block.run.decodes.append((video_latent, audio_latent, output_path))
            Path(output_path).write_bytes(b"mp4")
            return output_path

        def _load_transformer(self, path):
            self._run.loads.append("dit")
            return "dit"

        def _load_vae_encoder(self):
            self._run.loads.append("vae_encoder")

        def _load_upsampler(self):
            self._run.loads.append("upsampler")
            self.upsampler = "upsampler"

        (tmp_path / "model").mkdir(parents=True, exist_ok=True)
        (tmp_path / "model" / "transformer.safetensors").write_bytes(b"")
        self.pipe.model_dir = tmp_path / "model"
        monkeypatch.setattr(DistilledPipeline, "_load_transformer_with_optional_streaming", _load_transformer)
        monkeypatch.setattr(DistilledPipeline, "_load_vae_encoder", _load_vae_encoder)
        monkeypatch.setattr(DistilledPipeline, "_load_upsampler", _load_upsampler)
        monkeypatch.setattr(DistilledPipeline, "_stage1", _stage1)
        monkeypatch.setattr(DistilledPipeline, "_upsample_latent", lambda self, video_half: ("upscaled", video_half))
        monkeypatch.setattr(DistilledPipeline, "_stage2", _stage2)
        monkeypatch.setattr(DistilledPipeline, "_load_decoders", _load_decoders)
        monkeypatch.setattr(_orchestration, "decode_and_save_video", _decode)

    @staticmethod
    def _make_pipeline(low_memory: bool) -> DistilledPipeline:
        pipe = object.__new__(DistilledPipeline)
        pipe.low_memory = low_memory
        pipe.verbose = False
        pipe.stepwise = None
        pipe.dit = "dit"
        pipe._loaded = True
        pipe.upsampler = "upsampler"
        pipe.prompt_encoder = MagicMock()
        pipe.image_conditioner = MagicMock()
        pipe.video_decoder_block = SimpleNamespace(_decoder=None, run=None)
        pipe.audio_decoder_block = SimpleNamespace(_audio_decoder=None, _vocoder=None)
        pipe.video_patchifier = MagicMock()
        pipe.video_patchifier.unpatchify.side_effect = lambda tokens, dims: ("video", tokens, dims)
        pipe.audio_patchifier = MagicMock()
        pipe.audio_patchifier.unpatchify.side_effect = lambda tokens: ("audio", tokens)
        return pipe

    def generate(self, *, review: bool, review_error: Exception | None = None) -> None:
        def _review(path: Path) -> None:
            self.review_saw = (self.pipe.dit, self.pipe._loaded)
            assert path.is_file()
            if review_error is not None:
                raise review_error

        self.tmp_path.mkdir(parents=True, exist_ok=True)
        model = _LtxModel(pipeline=self.pipe, frame_rate=24.0, upscale=True, low_ram_streaming=False)
        kwargs = {"review": _review} if review else {}
        LtxVideoBackend().text_to_video(model, "p", 704, 480, 49, 7, 8, str(self.tmp_path / "o.mp4"), stage1_steps=6, **kwargs)

    def state(self) -> dict[str, object]:
        pipe = self.pipe
        return {
            "dit": pipe.dit,
            "_loaded": pipe._loaded,
            "upsampler": pipe.upsampler,
            "stepwise": pipe.stepwise,
            "decoder": pipe.video_decoder_block._decoder,
            "audio_decoder": pipe.audio_decoder_block._audio_decoder,
            "vocoder": pipe.audio_decoder_block._vocoder,
            "generated_keyframes": getattr(pipe, "generated_keyframes", None),
            "instance_attributes": sorted(vars(pipe)),
        }


@pytest.fixture(params=[True, False], ids=["low_memory", "plenty_of_memory"])
def low_memory(request) -> bool:
    return request.param


class TestReviewAgainstTheVendoredOrchestration:
    def test_stage_two_and_the_final_decode_get_the_same_inputs_with_and_without_review(self, monkeypatch, tmp_path, low_memory):
        plain = _Run(monkeypatch, tmp_path / "plain", low_memory=low_memory)
        reviewed = _Run(monkeypatch, tmp_path / "reviewed", low_memory=low_memory)

        plain.generate(review=False)
        reviewed.generate(review=True)

        assert reviewed.stage1_calls == plain.stage1_calls == 1
        assert reviewed.stage2_saw == plain.stage2_saw
        assert [(v, a) for v, a, _ in reviewed.decodes[-1:]] == [(v, a) for v, a, _ in plain.decodes]
        assert (len(plain.decodes), len(reviewed.decodes)) == (1, 2)

    def test_the_preview_decodes_the_half_size_latents_and_the_final_decode_the_stage_two_output(self, monkeypatch, tmp_path, low_memory):
        run = _Run(monkeypatch, tmp_path, low_memory=low_memory)

        run.generate(review=True)

        (preview_video, preview_audio, preview_path), (final_video, final_audio, final_path) = run.decodes
        assert preview_video == ("video", "video_tokens", (3, 7, 11))
        assert preview_audio == ("audio", "audio_tokens")
        assert (final_video, final_audio) == ("video_latent", "audio_latent")
        assert Path(preview_path).parent != Path(final_path).parent
        assert not Path(preview_path).exists()
        assert Path(final_path).exists()

    def test_the_pipeline_ends_in_the_same_state_with_and_without_review(self, monkeypatch, tmp_path, low_memory):
        plain = _Run(monkeypatch, tmp_path / "plain", low_memory=low_memory)
        reviewed = _Run(monkeypatch, tmp_path / "reviewed", low_memory=low_memory)

        plain.generate(review=False)
        reviewed.generate(review=True)

        assert reviewed.state() == plain.state()
        assert "_stage1" not in vars(reviewed.pipe)

    def test_the_dit_slot_is_intact_for_the_review_and_for_stage_two(self, monkeypatch, tmp_path, low_memory):
        run = _Run(monkeypatch, tmp_path, low_memory=low_memory)

        run.generate(review=True)

        assert run.review_saw == ("dit", True)
        assert run.stage2_saw is not None
        assert run.stage2_saw[:2] == ("dit", True)

    def test_a_discard_never_reaches_stage_two_and_leaves_no_shadow_or_decoders_under_low_memory(self, monkeypatch, tmp_path, low_memory):
        run = _Run(monkeypatch, tmp_path, low_memory=low_memory)

        with pytest.raises(_Discard):
            run.generate(review=True, review_error=_Discard())

        assert run.stage2_saw is None
        assert len(run.decodes) == 1
        assert "_stage1" not in vars(run.pipe)
        if low_memory:
            assert run.state()["decoder"] is None
            assert run.state()["audio_decoder"] is None
            assert run.state()["vocoder"] is None
        assert not list(run.tmp_path.glob("*.mp4"))


class TestDiscardedReviewReleasesTheModel:
    """A review that raises ends the generation between the stages, where the vendored frees never run."""

    @pytest.mark.parametrize("error", [_Discard(), KeyboardInterrupt()], ids=["discard", "quit"])
    def test_low_memory_frees_what_stage_two_and_generate_and_save_would_have(self, monkeypatch, tmp_path, error):
        run = _Run(monkeypatch, tmp_path, low_memory=True)

        with pytest.raises(type(error)):
            run.generate(review=True, review_error=error)

        assert run.stage2_saw is None
        assert (run.pipe.dit, run.pipe._loaded, run.pipe.upsampler) == (None, False, None)
        run.pipe.image_conditioner.free.assert_called_once_with()
        run.pipe.prompt_encoder.free.assert_called()
        assert "_stage1" not in vars(run.pipe)

    def test_without_low_memory_the_state_is_left_as_stage_one_left_it(self, monkeypatch, tmp_path):
        run = _Run(monkeypatch, tmp_path, low_memory=False)

        with pytest.raises(_Discard):
            run.generate(review=True, review_error=_Discard())

        assert (run.pipe.dit, run.pipe._loaded, run.pipe.upsampler) == ("dit", True, "upsampler")
        run.pipe.image_conditioner.free.assert_not_called()
        run.pipe.prompt_encoder.free.assert_not_called()
        assert "_stage1" not in vars(run.pipe)

    def test_the_next_generation_starts_without_a_resident_dit_and_reloads_everything_under_low_memory(self, monkeypatch, tmp_path):
        run = _Run(monkeypatch, tmp_path, low_memory=True)
        with pytest.raises(_Discard):
            run.generate(review=True, review_error=_Discard())

        run.generate(review=True)

        assert run.stage1_calls == 2
        # Stage 1 of the second video begins (the text encoder loads) with no DiT and no upsampler resident.
        assert run.stage1_entry[1] == (None, False, None)
        assert sorted(run.loads) == ["dit", "upsampler", "vae_encoder"]
        assert run.stage2_saw == ("dit", True, _STAGE1, ("upscaled", ("video", "video_tokens", (3, 7, 11))))
        assert Path(run.decodes[-1][2]).exists()

    def test_the_next_generation_after_a_discard_matches_one_on_a_fresh_pipeline_without_low_memory(self, monkeypatch, tmp_path):
        run = _Run(monkeypatch, tmp_path, low_memory=False)
        with pytest.raises(_Discard):
            run.generate(review=True, review_error=_Discard())

        run.generate(review=True)

        assert run.stage1_entry[1] == ("dit", True, "upsampler")
        assert run.loads == []
        assert run.stage2_saw is not None


class TestCacheIsClearedBeforeAReviewedGeneration:
    """What a discarded video left in the MLX cache is freed once its traceback is gone, before stage 1 loads anything."""

    @pytest.mark.parametrize(
        ("low_memory", "review", "expected"),
        [(True, True, ["cleanup", "generate"]), (True, False, ["generate"]), (False, True, ["generate"])],
        ids=["review_low_memory", "no_review", "no_low_memory"],
    )
    def test_cleanup_runs_before_generate_and_save_only_for_a_reviewed_low_memory_run(self, monkeypatch, tmp_path, low_memory, review, expected):
        run = _Run(monkeypatch, tmp_path, low_memory=low_memory)
        events: list[str] = []
        monkeypatch.setattr("ltx_core_mlx.utils.memory.aggressive_cleanup", lambda: events.append("cleanup"))
        monkeypatch.setattr(DistilledPipeline, "generate_and_save", lambda self, **kwargs: events.append("generate"))

        run.generate(review=review)

        assert events == expected
