"""Pin the vendored LTX MLX internals that ``backends/video_mac.py`` relies on.

Introspection only: no pipeline is constructed and no weights are loaded. A failure here after
``make update-ltx`` means upstream changed a name or signature the macOS video backend calls.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

pytest.importorskip("mlx")
pytest.importorskip("ltx_pipelines_mlx")

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions  # noqa: E402
from ltx_core_mlx.model.transformer.model import X0Model  # noqa: E402
from ltx_core_mlx.utils.memory import aggressive_cleanup  # noqa: E402
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions  # noqa: E402
from ltx_pipelines_mlx import BasePipeline, DistilledPipeline  # noqa: E402
from ltx_pipelines_mlx.distilled import Stage1Result  # noqa: E402
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS, shorten_schedule  # noqa: E402
from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings  # noqa: E402
from ltx_pipelines_mlx.utils.args import ImageConditioningInput  # noqa: E402
from ltx_pipelines_mlx.utils.blocks import ImageConditioner, PromptEncoder  # noqa: E402
from ltx_pipelines_mlx.utils.helpers import create_noised_state  # noqa: E402
from ltx_pipelines_mlx.utils.samplers import denoise_loop  # noqa: E402
from ltx_pipelines_mlx.utils.stepwise import StepwisePreview  # noqa: E402


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
        _assert_params(DistilledPipeline.generate_and_save, "prompt", "output_path", "height", "width", "num_frames", "stage1_steps", "image")
        _assert_keyword_only(DistilledPipeline.generate_and_save, "frame_rate", "seed", "stage1_steps", "image")


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
