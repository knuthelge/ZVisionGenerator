"""Pin the diffusers MiniMax H3 Ref2VA internals the reference path relies on.

Introspection only: no pipeline is constructed and no weights are loaded. A failure here after a diffusers bump means a
name, signature or field the Ref2VA layout or the prompt presentation uses has changed.
"""

from __future__ import annotations

import dataclasses
import inspect

import diffusers
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep


def _parameters(callable_obj) -> list[str]:
    return [name for name in inspect.signature(callable_obj).parameters if name != "self"]


def test_the_diffusers_release_is_the_pinned_one():
    assert diffusers.__version__ == "0.42.0"


def test_the_packed_sequence_builder_takes_the_arguments_the_layout_passes():
    assert _parameters(MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence) == [
        "text_token_tags",
        "references",
        "condition_latents",
        "audio_condition_latents",
        "num_latent_frames",
        "latent_height",
        "latent_width",
        "num_audio_latents",
        "patch_size",
        "audio_channels",
        "audio_tag",
        "video_tag",
    ]


def test_the_vision_gather_takes_a_processor_references_and_a_frame_rate():
    assert _parameters(MiniMaxH3Ref2VATextEncoderStep._gather_vision_features) == ["processor", "references", "fps"]


def test_the_presentation_builder_takes_the_token_counts_and_the_tags():
    assert _parameters(MiniMaxH3Ref2VATextEncoderStep._build_presentation) == [
        "tokenizer",
        "prompt",
        "references",
        "image_token_counts",
        "video_block_token_counts",
        "video_block_timestamps",
        "text_tag",
        "video_tag",
    ]


def test_the_video_condition_sampler_takes_frames_and_the_sampling_rates():
    assert _parameters(MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames) == ["frames", "fps", "sample_fps", "temporal_patch"]


def test_an_image_reference_holds_one_image():
    assert [field.name for field in dataclasses.fields(MiniMaxH3ImageReference)] == ["image"]
    assert MiniMaxH3ImageReference.kind == "image"


def test_a_video_reference_holds_frames_a_rate_and_its_soundtrack():
    assert [field.name for field in dataclasses.fields(MiniMaxH3VideoReference)] == ["frames", "fps", "audio", "sample_rate"]
    assert MiniMaxH3VideoReference.kind == "video"
