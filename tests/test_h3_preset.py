"""Tests for the MiniMax H3 preset value types."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from zvisiongenerator.utils import h3_preset
from zvisiongenerator.utils.h3_preset import H3Files, H3MlxFiles, H3MlxPreset, H3Preset, H3Schedule, HubFiles


def _hub(name: str = "a/b") -> HubFiles:
    return HubFiles(name, "rev", ("x.json",))


def _preset(**overrides) -> H3Preset:
    fields = dict(
        repo="o/r",
        revision="rev",
        repo_allow=("a/*",),
        repo_ignore=("b/*",),
        text_encoder=_hub("te/r"),
        text_encoder_layers=50,
        lora=_hub("lo/ra"),
        lora_scale=0.0625,
        schedules={8: H3Schedule(8, 6.0, 3.0), 4: H3Schedule(4, 6.0, 3.0)},
    )
    fields.update(overrides)
    return H3Preset(**fields)


@pytest.mark.parametrize("forwards", [4, 8])
def test_num_inference_steps_is_forwards_plus_the_terminal_zero(forwards):
    assert H3Schedule(forwards, 6.0, 3.0).num_inference_steps == forwards + 1


def test_preset_defaults():
    preset = _preset()
    assert preset.attention == "sdpa"
    assert (preset.ffn_chunk_tokens, preset.qk_norm_chunk_tokens, preset.query_chunk) == (None, None, None)
    assert preset.schedules[4].forwards == 4


@pytest.mark.parametrize(
    "value",
    [
        _hub(),
        H3Schedule(8, 6.0, 3.0),
        _preset(),
        H3MlxPreset(_hub(), _hub(), 4, 64, 50),
        H3Files(Path("r"), Path("t"), (Path("s"),), Path("l")),
        H3MlxFiles(Path("f"), Path("t"), Path("s"), Path("l")),
    ],
)
def test_value_types_are_frozen(value):
    field = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)


def test_public_names_are_exported():
    assert set(h3_preset.__all__) == {"H3Files", "H3MlxFiles", "H3MlxPreset", "H3Preset", "H3Schedule", "HubFiles"}
    assert all(hasattr(h3_preset, name) for name in h3_preset.__all__)
