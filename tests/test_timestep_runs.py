"""Tests for running the LTX-2 transformer on per-token timesteps by runs of equal timesteps."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import LTX2VideoTransformer3DModel  # noqa: E402

from zvisiongenerator.backends.timestep_runs import modulate_by_runs, token_runs  # noqa: E402

_FRAMES, _HEIGHT, _WIDTH = 3, 2, 2
_TOKENS = _FRAMES * _HEIGHT * _WIDTH


def _tiny_transformer():
    """An LTX-2.3-shaped transformer small enough to run on the CPU, with random weights."""
    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        num_layers=2,
        caption_channels=16,
        use_prompt_embeddings=False,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        gated_attn=True,
        audio_gated_attn=True,
    ).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.3)
    return model


def _inputs(video_timestep, **overrides):
    """Inputs like diffusers' LTX-2 image-to-video pipeline gives the transformer."""
    torch.manual_seed(1)
    sigma = torch.tensor([0.7])
    inputs = {
        "hidden_states": torch.randn(1, _TOKENS, 8),
        "audio_hidden_states": torch.randn(1, 5, 8),
        "encoder_hidden_states": torch.randn(1, 4, 16),
        "audio_encoder_hidden_states": torch.randn(1, 4, 16),
        "timestep": video_timestep,
        "audio_timestep": sigma,
        "sigma": sigma,
        "num_frames": _FRAMES,
        "height": _HEIGHT,
        "width": _WIDTH,
        "fps": 24.0,
        "audio_num_frames": 5,
        "use_cross_timestep": True,
        "return_dict": False,
    }
    return {**inputs, **overrides}


def _first_frame_conditioned():
    """Per-token timesteps with the first latent frame clean and the rest at 0.7."""
    mask = torch.zeros(1, _TOKENS)
    mask[:, : _HEIGHT * _WIDTH] = 1
    return 0.7 * (1 - mask)


@pytest.mark.parametrize(
    "timestep",
    [_first_frame_conditioned(), torch.full((1, _TOKENS), 0.4), (torch.arange(_TOKENS, dtype=torch.float32)[None] // 5) / 4],
    ids=["first-frame", "uniform", "three-runs"],
)
def test_outputs_match_the_per_token_path(timestep):
    reference = _tiny_transformer()
    by_runs = copy.deepcopy(reference)
    modulate_by_runs(by_runs)
    inputs = _inputs(timestep)

    with torch.no_grad():
        expected = reference(**inputs)
        actual = by_runs(**inputs)
        uniform = reference(**{**inputs, "timestep": timestep[:, -1:]})

    for want, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    if timestep.unique().numel() > 1:
        assert not torch.allclose(uniform[0], expected[0], atol=1e-3)  # the runs really do differ


def test_without_per_sample_cross_timesteps_the_pass_is_untouched():
    model = _tiny_transformer()
    modulate_by_runs(model)
    seen: list[object] = []
    model.time_embed.register_forward_pre_hook(lambda _module, args: seen.append(tuple(args[0].shape)))

    with torch.no_grad():
        model(**_inputs(_first_frame_conditioned(), use_cross_timestep=False))

    assert seen[0] == (_TOKENS,)  # the timestep embedding still ran per token


def test_token_runs_finds_equal_stretches():
    assert token_runs(torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5]])) == [(0, 2), (2, 5)]
    assert token_runs(torch.tensor([[0.3, 0.3]])) == [(0, 2)]
    assert token_runs(torch.tensor([[0.0, 0.5], [0.5, 0.5]])) is None  # samples differ
    assert token_runs(torch.arange(40.0)[None]) is None  # too many runs to be worth it
