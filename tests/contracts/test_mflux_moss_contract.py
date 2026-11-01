"""Pin the mflux MOSS-Image internals that ``backends/mac/moss.py`` and ``moss_lora.py`` drive directly.

Introspection only: no model is constructed and no weights are loaded. A failure here after an mflux bump means upstream
changed a name, signature or layout the staged load, the denoising loop or the LoRA mapping depends on.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mflux")

from mflux.callbacks.callback_registry import CallbackRegistry  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.lora.mapping.lora_loader import LoRALoader  # noqa: E402
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget  # noqa: E402
from mflux.models.common.resolution.model_revisions import ModelRevisions  # noqa: E402
from mflux.models.common.vae.vae_util import VAEUtil  # noqa: E402
from mflux.models.common.weights.loading.weight_applier import WeightApplier  # noqa: E402
from mflux.models.common.weights.loading.weight_loader import WeightLoader  # noqa: E402
from mflux.models.common.weights.saving.model_saver import ModelSaver  # noqa: E402
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE  # noqa: E402
from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator  # noqa: E402
from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import MOSSImageScheduler  # noqa: E402
from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_prompt_encoder import MOSSImagePromptEncoder  # noqa: E402
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer  # noqa: E402
from mflux.models.moss_image.moss_image_initializer import MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS, MOSSImageInitializer  # noqa: E402
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping  # noqa: E402
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition  # noqa: E402

from zvisiongenerator.backends.mac import moss  # noqa: E402
from zvisiongenerator.backends.mac.moss_lora import moss_lora_mapping  # noqa: E402
from zvisiongenerator.utils.config import load_config  # noqa: E402
from zvisiongenerator.utils.moss_build import moss_source  # noqa: E402


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


class TestWeightsContract:
    def test_the_stages_cover_the_components_mflux_loads(self):
        names = {component.name for component in MOSSImageWeightDefinition.get_components()}

        assert names == {*moss._TEXT_ENCODER, *moss._DENOISER}
        assert {component.name for component in moss._stage_definition(moss._TEXT_ENCODER).get_components()} == {"text_encoder"}
        assert {component.name for component in moss._stage_definition(moss._DENOISER).get_components()} == {"transformer", "vae"}
        assert moss._stage_definition(moss._DENOISER).get_tokenizers() == MOSSImageWeightDefinition.get_tokenizers()

    def test_components_read_the_built_folders_subfolders(self):
        subdirs = {component.name: component.hf_subdir for component in MOSSImageWeightDefinition.get_components()}

        assert subdirs == {"vae": "vae", "transformer": "transformer", "text_encoder": "text_encoder"}

    def test_the_text_encoder_loads_only_the_language_model(self):
        text_encoder = next(component for component in MOSSImageWeightDefinition.get_components() if component.name == "text_encoder")

        assert text_encoder.weight_prefix_filters == ["language_model."]

    def test_the_tokenizer_is_the_one_prompts_are_encoded_with(self):
        assert [tokenizer.name for tokenizer in MOSSImageWeightDefinition.get_tokenizers()] == [moss._TOKENIZER]

    def test_mflux_pins_the_sft_revision_the_builder_downloads(self):
        source = moss_source(load_config())

        assert ModelRevisions.PINNED[source.sft_repo] == source.sft_revision

    def test_loader_applier_and_saver_take_a_weight_definition(self):
        assert _params(WeightLoader.load)[:2] == ["weight_definition", "model_path"]
        assert _params(WeightApplier.apply_and_quantize) == ["weights", "models", "quantize_arg", "weight_definition"]
        assert _params(ModelSaver.save_model)[:4] == ["model", "bits", "base_path", "weight_definition"]

    def test_the_text_encoder_drops_the_vision_tower_by_its_prefix(self):
        from zvisiongenerator.utils import model_memory

        text_encoder = next(component for component in MOSSImageWeightDefinition.get_components() if component.name == "text_encoder")
        # The pinned checkpoint names the language model language_model.* and the vision tower visual.*.
        assert all(not prefix.startswith("visual.") for prefix in text_encoder.weight_prefix_filters)
        assert "visual." in model_memory._MOSS_MLX_TEXT_ENCODER_SKIPPED_PREFIXES

    def test_the_initializer_builds_the_parts_the_stages_build(self):
        config = ModelConfig.moss_image_sft()

        assert isinstance(config.text_encoder_overrides, dict)
        assert isinstance(config.transformer_overrides, dict)
        assert MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS == (96, 192, 384, 384)
        assert "decoder_block_out_channels" in _params(Flux2VAE.__init__)
        assert hasattr(MOSSImageInitializer, "_init_models")


class TestLoopContract:
    def test_transformer_call(self):
        assert _params(MOSSImageTransformer.__call__)[1:] == ["hidden_states", "timestep", "encoder_hidden_states", "skip_layers"]

    def test_scheduler_reads_only_the_step_count(self):
        scheduler = MOSSImageScheduler(moss._RunConfig(width=64, height=64, num_inference_steps=4, init_time_step=0, time_steps=range(4)))

        assert scheduler.sigmas.shape == (5,)
        assert scheduler.timesteps.shape == (4,)
        assert _params(MOSSImageScheduler.step)[1:4] == ["noise", "timestep", "latents"]

    def test_latents_and_prompts(self):
        assert _params(MOSSImageLatentCreator.create_noise) == ["seed", "height", "width"]
        assert _params(MOSSImageLatentCreator.check_size) == ["height", "width"]
        assert _params(MOSSImagePromptEncoder.encode) == ["prompt", "tokenizer", "text_encoder"]

    def test_vae(self):
        assert _params(Flux2VAE.decode_packed_latents)[1:] == ["packed_latents", "tiling_config"]
        assert _params(VAEUtil.encode) == ["vae", "image", "tiling_config"]
        assert Flux2VAE.DECODE_LOW_MEMORY != Flux2VAE.DECODE_LEGACY

    def test_callbacks(self):
        assert _params(CallbackRegistry.start)[1:] == ["seed", "prompt", "config"]


class TestLoraContract:
    def test_loader_keeps_loras_unmerged_on_request(self):
        assert {"lora_mapping", "transformer", "lora_paths", "lora_scales", "bake_lora"} <= set(_params(LoRALoader.load_and_apply_lora))

    def test_the_target_fields_the_mapping_sets(self):
        fields = {field.name for field in dataclasses.fields(LoRATarget)}

        assert {"model_path", "possible_up_patterns", "possible_down_patterns", "possible_alpha_patterns", "possible_lokr_w1_patterns", "possible_lokr_w2_patterns", "up_transform"} <= fields

    def test_our_mapping_covers_every_pattern_of_mfluxs(self):
        ours = {pattern for target in moss_lora_mapping() for pattern in target.possible_up_patterns + target.possible_down_patterns + target.possible_alpha_patterns}
        theirs = {pattern for target in MOSSImageLoRAMapping.get_mapping() for pattern in target.possible_up_patterns + target.possible_down_patterns + target.possible_alpha_patterns}

        assert theirs <= ours
