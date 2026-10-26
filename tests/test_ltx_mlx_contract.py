"""Pin the vendored LTX MLX internals that ``backends/mac/video.py`` relies on.

Introspection, plus a few tiny-array shape checks: no pipeline is constructed and no weights are loaded. A failure here after
``make update-ltx`` means upstream changed a name or signature the macOS video backend calls.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytest.importorskip("ltx_pipelines_mlx")

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, compute_video_latent_shape, snap_output_dimensions  # noqa: E402
from ltx_core_mlx.conditioning.types import VideoConditionByKeyframeIndex, VideoConditionByLatentIndex  # noqa: E402
from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep  # noqa: E402
from ltx_core_mlx.model.audio_vae import encode_audio  # noqa: E402
from ltx_core_mlx.text_encoders.gemma.encoders import encoder_configurator  # noqa: E402
from ltx_core_mlx.model.transformer.model import X0Model  # noqa: E402
from ltx_core_mlx.model.video_vae.video_vae import (  # noqa: E402
    VideoDecoder as VaeVideoDecoder,
    _compute_decode_tiling,
    _ffmpeg_sink,
    decode_cache_limit,
    stream_chunks_to_ffmpeg,
)
from ltx_core_mlx.utils.audio import AudioData, load_audio  # noqa: E402
from ltx_core_mlx.utils.memory import aggressive_cleanup  # noqa: E402
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions  # noqa: E402
from ltx_pipelines_mlx import BasePipeline, DistilledPipeline  # noqa: E402
from ltx_pipelines_mlx.distilled import ANCESTRAL_ETA, ANCESTRAL_NOISE_SEED_OFFSET, ANCESTRAL_S_NOISE, Stage1Result  # noqa: E402
from ltx_pipelines_mlx.extend_distilled import carry_audio_tokens, fit_audio_latent  # noqa: E402
from ltx_pipelines_mlx.retake import encode_video_tensor  # noqa: E402
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, LTX_2_5_DISTILLED_SIGMAS, LTX_2_5_STAGE_2_DISTILLED_SIGMAS, STAGE_2_SIGMAS, shorten_schedule  # noqa: E402
from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings  # noqa: E402
from ltx_pipelines_mlx.utils.args import ImageConditioningInput, resolve_frame_indices  # noqa: E402
from ltx_pipelines_mlx.utils.blocks import AudioConditioner, AudioDecoder, ImageConditioner, PromptEncoder, VideoDecoder as VideoDecoderBlock, _video_vae_names  # noqa: E402
from ltx_pipelines_mlx.utils.generation import is_ltx25_pack  # noqa: E402
from ltx_pipelines_mlx.utils.helpers import create_noised_state  # noqa: E402
from ltx_pipelines_mlx.utils.media_io import video_preprocess  # noqa: E402
from ltx_pipelines_mlx.utils.samplers import denoise_loop, euler_ancestral_denoising_loop  # noqa: E402
from ltx_pipelines_mlx.utils.stepwise import StepwisePreview  # noqa: E402


from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: E402
from zvisiongenerator.utils.config import load_config, ltx_mlx_packs, ltx_mlx_text_encoder_repo_id  # noqa: E402
from zvisiongenerator.utils.model_files import ltx_mlx_bundled_text_encoder, ltx_mlx_transformer_file, ltx_mlx_upsampler_file, ltx_mlx_video_decoder_file  # noqa: E402


def _params(fn) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(fn).parameters)


def _assert_params(fn, *names: str) -> None:
    missing = [name for name in names if name not in _params(fn)]
    assert not missing, f"{fn.__qualname__} does not accept {missing}"


def _assert_keyword_only(fn, *names: str) -> None:
    params = _params(fn)
    for name in names:
        assert name in params, f"{fn.__qualname__} does not accept {name}"
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{fn.__qualname__} takes {name} positionally"


class TestBasePipelineContract:
    def test_constructor_arguments(self):
        _assert_params(BasePipeline.__init__, "model_dir", "low_memory", "low_ram_streaming")

    def test_distilled_pipeline_shares_the_base_constructor(self):
        _assert_params(DistilledPipeline.__init__, "model_dir", "low_memory", "low_ram_streaming")

    def test_pre_denoise_flush_takes_both_states(self):
        params = list(_params(BasePipeline._pre_denoise_flush))
        assert params == ["video_state", "audio_state"]

    def test_decode_and_save_video_keywords(self):
        _assert_keyword_only(BasePipeline._decode_and_save_video, "frame_rate", "seed")
        assert list(_params(BasePipeline._decode_and_save_video))[1:4] == ["video_latent", "audio_latent", "output_path"]

    def test_stepwise_hook_forwards_stage(self):
        _assert_keyword_only(BasePipeline._stepwise_hook, "stage")

    def test_load_takes_no_arguments(self):
        assert list(_params(BasePipeline.load)) == ["self"]

    def test_instance_slots_are_assigned_by_the_constructor(self):
        source = inspect.getsource(BasePipeline.__init__)
        for attr in ("self.stepwise", "self.dit", "self.video_patchifier", "self.audio_patchifier", "self.prompt_encoder", "self.image_conditioner"):
            assert attr in source, f"BasePipeline.__init__ does not assign {attr}"

    def test_decoder_properties_are_settable(self):
        for name in ("vae_decoder", "audio_decoder", "vocoder"):
            prop = getattr(BasePipeline, name, None)
            assert isinstance(prop, property) and prop.fset is not None, f"BasePipeline.{name} is not a settable property"

    def test_pending_loras_is_read_when_the_transformer_loads(self):
        source = inspect.getsource(BasePipeline._load_transformer_with_optional_streaming)
        assert "_pending_loras" in source

    def test_distilled_generate_and_save_keywords(self):
        _assert_params(DistilledPipeline.generate_and_save, "prompt", "output_path", "height", "width", "num_frames", "stage1_steps", "image", "images")
        _assert_keyword_only(DistilledPipeline.generate_and_save, "frame_rate", "seed", "stage1_steps", "image", "images")


class TestReviewContract:
    """What the review before upscaling relies on: it shadows ``_stage1`` and decodes its half-size result."""

    def test_two_stage_generation_calls_stage1_through_the_instance(self):
        source = inspect.getsource(DistilledPipeline.generate_two_stage)
        assert "self._stage1(" in source

    def test_generate_and_save_runs_generate_two_stage(self):
        source = inspect.getsource(DistilledPipeline.generate_and_save)
        assert "self.generate_two_stage(" in source

    def test_stage1_returns_the_result_first_and_takes_the_generation_arguments(self):
        _assert_params(DistilledPipeline._stage1, "prompt", "height", "width", "num_frames", "frame_rate", "seed", "stage1_steps", "image")
        assert "Stage1Result" in str(inspect.signature(DistilledPipeline._stage1).return_annotation)

    def test_stage1_result_fields_used_for_the_preview(self):
        fields = {field.name for field in dataclasses.fields(Stage1Result)}
        assert {"video_tokens", "audio_tokens", "latent_dims"} <= fields

    def test_load_decoders_takes_no_arguments(self):
        assert list(_params(BasePipeline._load_decoders)) == ["self"]

    def test_patchifiers_unpatchify_the_stage1_tokens(self):
        source = inspect.getsource(DistilledPipeline.generate_two_stage)
        assert "self.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)" in source
        assert "audio_patchifier" in inspect.getsource(BasePipeline.__init__)

    def test_stage1_result_comes_first_in_the_returned_tuple(self):
        """The wrapper returns ``_stage1``'s tuple unchanged and reads the result as ``[0]``."""
        assert "stage1, num_frames, height, width = self._stage1(" in inspect.getsource(DistilledPipeline.generate_two_stage)
        assert str(inspect.signature(DistilledPipeline._stage1).return_annotation).startswith("tuple[Stage1Result")

    def test_audio_tokens_unpatchify_without_dims(self):
        assert "self.audio_patchifier.unpatchify(output_2.audio_latent)" in inspect.getsource(DistilledPipeline._stage2)

    def test_stage_two_reads_the_dit_slot(self):
        """Why ``dit`` is restored after the preview decode: sparse attention and tiling read it in stage 2."""
        source = inspect.getsource(DistilledPipeline._stage2) + inspect.getsource(DistilledPipeline._start_stage2_sparse_attention)
        assert "self.dit" in source

    def test_generate_and_save_loads_the_decoders_itself(self):
        """The preview may leave the decoders freed: the final decode loads them on demand."""
        assert "self._load_decoders()" in inspect.getsource(DistilledPipeline.generate_and_save)

    def test_low_memory_decode_empties_the_dit_slot(self):
        """The wrapper restores ``dit`` and ``_loaded`` because the decode clears them."""
        source = inspect.getsource(BasePipeline._decode_and_save_video)
        assert "self.dit = None" in source and "self._loaded = False" in source


class TestBlockContract:
    def test_prompt_encoder_call(self):
        _assert_keyword_only(PromptEncoder.__call__, "free_after")

    def test_image_conditioner_call_and_crf(self):
        _assert_keyword_only(ImageConditioner.__call__, "free_after")
        _assert_params(ImageConditioner.resolve_crf, "images")

    def test_image_conditioning_input_fields(self):
        assert {"path", "frame_idx", "strength"} <= set(ImageConditioningInput._fields)

    def test_a_last_frame_index_of_minus_one_resolves_to_the_last_pixel_frame(self):
        resolved = resolve_frame_indices([ImageConditioningInput(path="a", frame_idx=0, strength=1.0), ImageConditioningInput(path="b", frame_idx=-1, strength=1.0)], 49)
        assert [image.frame_idx for image in resolved] == [0, 48]

    def test_a_keyframe_conditioning_holds_its_latent_as_keyframe_latent(self):
        keyframe = VideoConditionByKeyframeIndex(frame_idx=48, keyframe_latent="latent", spatial_dims=(7, 15, 22), frame_rate=24.0)
        assert keyframe.keyframe_latent == "latent"
        assert not hasattr(keyframe, "clean_latent")


class TestLoadTimeContract:
    """The vendored behaviour ``load_model`` and ``utils/model_files`` mirror."""

    def test_pack_detection_takes_the_model_dir(self):
        assert list(_params(is_ltx25_pack)) == ["model_dir"]

    def test_base_pipeline_assigns_model_dir_in_its_constructor(self):
        source = inspect.getsource(BasePipeline.__init__)
        assert "self.model_dir" in source

    def test_text_encoder_file_names_match_the_bundled_encoder_rule(self):
        assert encoder_configurator._TEXT_ENCODER_WEIGHTS == "text_encoder.safetensors"
        assert encoder_configurator._TEXT_ENCODER_CONFIG == "text_encoder_config.json"

    @pytest.mark.parametrize(("weights", "config"), [(True, True), (True, False), (False, True), (False, False)])
    def test_bundled_encoder_rule_agrees_with_select_text_encoder(self, tmp_path, weights, config):
        if weights:
            (tmp_path / "text_encoder.safetensors").write_bytes(b"")
        if config:
            (tmp_path / "text_encoder_config.json").write_text("{}")

        bundled = ltx_mlx_bundled_text_encoder(tmp_path) is not None
        assert (encoder_configurator.select_text_encoder(tmp_path) == "gemma4") is bundled

    @pytest.mark.parametrize("conv", [True, False])
    def test_decoder_rule_agrees_with_the_video_vae_names(self, tmp_path, conv):
        (tmp_path / "vae_decoder.safetensors").write_bytes(b"")
        if conv:
            (tmp_path / "vae_decoder_conv.safetensors").write_bytes(b"")

        decoder_name, _ = _video_vae_names(tmp_path)
        chosen = ltx_mlx_video_decoder_file(tmp_path)
        assert chosen is not None
        assert chosen.name == f"{decoder_name}.safetensors"


class TestStepwiseContract:
    def test_bind_accepts_stage(self):
        _assert_keyword_only(StepwisePreview.bind, "stage")


class TestSamplingContract:
    def test_denoise_loop_arguments(self):
        _assert_params(denoise_loop, "model", "video_state", "audio_state", "video_text_embeds", "audio_text_embeds", "sigmas", "on_step")

    def test_create_noised_state_arguments(self):
        _assert_params(create_noised_state, "base_shape", "conditionings", "spatial_dims", "positions", "seed", "sigma", "initial_latent", "legacy_scalar_blend")

    def test_combined_image_conditionings_arguments(self):
        _assert_keyword_only(combined_image_conditionings, "enc_h", "enc_w", "spatial_dims", "video_encoder", "frame_rate")

    def test_shorten_schedule_keep(self):
        _assert_keyword_only(shorten_schedule, "keep")
        assert shorten_schedule(DISTILLED_SIGMAS, 3, keep="start")[-1] == DISTILLED_SIGMAS[-1]

    def test_sigma_tables_are_sequences(self):
        assert len(DISTILLED_SIGMAS) > 1
        assert len(STAGE_2_SIGMAS) > 1

    def test_ancestral_loop_arguments(self):
        _assert_params(euler_ancestral_denoising_loop, "sigmas", "video_state", "audio_state", "stepper", "transformer", "video_text_embeds", "audio_text_embeds")
        _assert_keyword_only(euler_ancestral_denoising_loop, "noise_seed", "on_step")

    def test_ancestral_stepper_takes_eta_and_s_noise(self):
        stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
        assert (stepper.eta, stepper.s_noise) == (1.0, 1.0)

    def test_ancestral_constants(self):
        assert (ANCESTRAL_ETA, ANCESTRAL_S_NOISE, ANCESTRAL_NOISE_SEED_OFFSET) == (1.0, 1.0, 10000)

    def test_ltx_2_5_tables_have_the_lengths_of_the_2_3_tables(self):
        # The progress totals (_TwoStageProgress, the one-stage step count) are derived from the 2.3 tables.
        assert len(LTX_2_5_DISTILLED_SIGMAS) == len(DISTILLED_SIGMAS)
        assert len(LTX_2_5_STAGE_2_DISTILLED_SIGMAS) == len(STAGE_2_SIGMAS)

    def test_x0_model_wraps_a_transformer_positionally(self):
        assert len(_params(X0Model.__init__)) == 2


class TestGeometryHelpers:
    def test_latent_shape_and_snap_arguments(self):
        _assert_params(compute_video_latent_shape, "num_frames", "height", "width")
        _assert_keyword_only(snap_output_dimensions, "two_stage")

    def test_position_helpers(self):
        _assert_params(compute_video_positions, "num_frames", "height", "width", "frame_rate")
        _assert_params(compute_audio_token_count, "num_video_frames", "frame_rate")
        _assert_params(compute_audio_positions, "num_tokens")

    def test_aggressive_cleanup_takes_no_arguments(self):
        assert not _params(aggressive_cleanup)


class TestExtendContract:
    """Internals the macOS backend uses to continue a clip from its tail."""

    def test_video_pin_condition_arguments(self):
        assert list(_params(VideoConditionByLatentIndex.__init__))[1:4] == ["frame_indices", "clean_latent", "strength"]

    def test_video_tail_encoding_arguments(self):
        assert list(_params(encode_video_tensor)) == ["encoder", "video_tensor"]
        assert list(_params(video_preprocess))[:3] == ["frames", "height", "width"]

    def test_tail_audio_loading_arguments(self):
        _assert_params(load_audio, "path", "target_sample_rate", "start_time", "max_duration", "mono")

    def test_audio_encoding_arguments(self):
        assert list(_params(encode_audio))[:4] == ["waveform", "sample_rate", "encoder", "processor"]

    def test_audio_carry_helpers(self):
        assert list(_params(fit_audio_latent)) == ["audio_latent", "audio_tokens"]
        assert list(_params(carry_audio_tokens))[:2] == ["frame_rate", "carry_pixel_frames"]

    def test_audio_conditioner_frees_after_the_call(self):
        _assert_keyword_only(AudioConditioner.__call__, "free_after")

    def test_pipeline_assigns_the_blocks_and_flags_the_extension_relies_on(self):
        source = inspect.getsource(BasePipeline.__init__)
        for attr in ("self._loaded", "self.audio_conditioner", "self.video_decoder_block", "self.audio_decoder_block", "self.audio_patchifier"):
            assert attr in source, f"BasePipeline.__init__ does not assign {attr}"

    def test_freeing_the_transformer_follows_the_upstream_decode_wrapper(self):
        source = inspect.getsource(BasePipeline._decode_and_save_video)
        assert "self.dit = None" in source
        assert "self._loaded = False" in source

    def test_video_decoder_block_loads_without_arguments_and_names_its_decoder(self):
        assert list(_params(VideoDecoderBlock.load)) == ["self"]
        assert "self.video_decoder" in inspect.getsource(VideoDecoderBlock.__init__)

    def test_audio_decoder_block_decodes_a_latent(self):
        assert list(_params(AudioDecoder.__call__)) == ["self", "audio_latent"]

    def test_audio_patchifier_patchifies_a_latent(self):
        assert list(_params(AudioPatchifier.patchify)) == ["self", "latent"]

    def test_tiled_decode_arguments(self):
        assert list(_params(VaeVideoDecoder.tiled_decode)) == ["self", "latent", "tiling_config"]

    def test_decode_tiling_and_streaming_helpers(self):
        assert list(_params(_compute_decode_tiling)) == ["latent_shape", "frame_rate", "budget_bytes"]
        assert not _params(decode_cache_limit)
        assert list(_params(stream_chunks_to_ffmpeg)) == ["chunks", "proc"]
        assert list(_params(_ffmpeg_sink)) == ["cmd"]


class TestExtendShapes:
    """Tail-carry shapes the backend builds, checked on tiny arrays (no weights are loaded)."""

    def test_loaded_audio_exposes_its_waveform_and_rate(self):
        assert {"waveform", "sample_rate"} <= set(inspect.signature(AudioData).parameters)

    def test_audio_tail_pin_is_one_token_per_carry_step(self):
        import mlx.core as mx

        tokens = carry_audio_tokens(24.0, 25)
        fitted = fit_audio_latent(mx.zeros((1, 8, tokens + 3, 16)), tokens)
        padded = fit_audio_latent(mx.zeros((1, 8, tokens - 2, 16)), tokens)

        assert tokens == 26
        assert AudioPatchifier().patchify(fitted)[0].shape == (1, 26, 128)
        assert AudioPatchifier().patchify(padded)[0].shape == (1, 26, 128)

    def test_video_tail_pin_patchifies_to_one_token_per_latent_cell(self):
        import mlx.core as mx

        tokens, dims = VideoLatentPatchifier().patchify(mx.zeros((1, 128, 4, 14, 22)))

        assert tokens.shape == (1, 4 * 14 * 22, 128)
        assert dims == (4, 14, 22)


class TestPinnedPackFilesContract:
    """Every file-selection decision of the vendored loader, taken on a folder of exactly a pack's listed files, lands on a listed file.

    The packs are downloaded as just those files, so a name the loader picks that is not listed would be a missing file at run time.
    """

    @pytest.fixture
    def packs(self):
        return ltx_mlx_packs(load_config())

    @staticmethod
    def _folder(root, files):
        for name in files:
            (root / name).write_text("{}" if name.endswith(".json") else "")
        return root

    @pytest.fixture(params=["ltx-4", "ltx-8", "ltx25-4"])
    def pack(self, request, tmp_path, packs):
        config = load_config()
        reference = config["model_aliases"][request.param]["darwin"]
        files = packs[reference].files
        return SimpleNamespace(alias=request.param, files=files, folder=self._folder(tmp_path, files), bundled=request.param == "ltx25-4")

    def test_the_transformer_is_a_listed_file(self, pack):
        folder = pack.folder
        plain = folder / "transformer.safetensors"
        chosen = plain if plain.exists() else BasePipeline._resolve_safetensors(folder, "transformer-distilled")  # the choice _base.py's load makes

        assert chosen.name in pack.files
        assert ltx_mlx_transformer_file(folder) == chosen
        assert sum(name.startswith("transformer") for name in pack.files) == 1

    def test_the_video_decoder_and_encoder_are_listed_files(self, pack):
        decoder, encoder = _video_vae_names(pack.folder)

        assert {f"{decoder}.safetensors", f"{encoder}.safetensors"} <= set(pack.files)
        assert ltx_mlx_video_decoder_file(pack.folder).name == f"{decoder}.safetensors"
        assert ImageConditioner(pack.folder).encoder_path.name == f"{encoder}.safetensors"

    def test_the_audio_and_connector_files_are_listed(self, pack):
        assert {"audio_vae.safetensors", "vocoder.safetensors", "connector.safetensors"} <= set(pack.files)

    def test_the_spatial_upsampler_and_its_config_are_listed_files(self, pack):
        pipeline = SimpleNamespace(model_dir=pack.folder, _is_25=pack.bundled, _resolve_safetensors=BasePipeline._resolve_safetensors)

        weights = TI2VidTwoStagesPipeline._resolve_upsampler_path(pipeline)

        assert weights.name in pack.files
        assert f"{weights.stem}_config.json" in pack.files
        assert ltx_mlx_upsampler_file(pack.folder) == weights

    def test_the_text_encoder_is_bundled_exactly_for_the_ltx25_pack(self, pack):
        kind = encoder_configurator.select_text_encoder(pack.folder)

        assert kind == ("gemma4" if pack.bundled else "gemma3")
        assert (ltx_mlx_bundled_text_encoder(pack.folder) is not None) is pack.bundled
        if pack.bundled:
            assert {"text_encoder.safetensors", "text_encoder_config.json", "tokenizer.json", "tokenizer_config.json"} <= set(pack.files)

    def test_the_model_config_is_a_listed_file(self, pack):
        assert "embedded_config.json" in pack.files  # LTXModelConfig reads it before config.json

    def test_the_ltx25_pack_lists_its_duration_head(self, pack):
        assert ("duration_head.safetensors" in pack.files) is pack.bundled

    def test_the_configured_gemma_3_is_the_pipelines_default(self):
        default = inspect.signature(BasePipeline.__init__).parameters["gemma_model_id"].default

        assert ltx_mlx_text_encoder_repo_id(load_config()) == default
        assert inspect.signature(DistilledPipeline.__init__).parameters["gemma_model_id"].default == default

    def test_both_pipelines_take_the_text_encoder_folder(self):
        _assert_params(BasePipeline.__init__, "gemma_model_id")
        _assert_params(DistilledPipeline.__init__, "gemma_model_id")
