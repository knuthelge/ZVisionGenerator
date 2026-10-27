"""Tests for converting MiniMax H3 fine-tunes from the CLI: flags, names, the turbo choice and the folder that results."""

from __future__ import annotations

import pytest
from safetensors.torch import save_file

from tests._mac_helpers import _TINY, _diffusers_state, _native
from zvisiongenerator.converters import h3_checkpoint
from zvisiongenerator.converters.convert_checkpoint import _build_model_parser, _cmd_model
from zvisiongenerator.utils.h3_converted import read_converted_h3, write_converted_h3
from zvisiongenerator.utils.video_model_detect import detect_video_model


def _run_cli(argv: list[str]) -> None:
    _cmd_model(_build_model_parser().parse_args(["model", *argv]))


@pytest.fixture
def h3_cli(monkeypatch, tmp_path):
    """A data dir, a tiny H3 checkpoint and a recorded platform conversion that writes a complete folder."""
    import torch

    monkeypatch.setattr(h3_checkpoint, "STOCK_CONFIG", _TINY)
    data_dir = tmp_path / "data"
    monkeypatch.setattr("zvisiongenerator.converters.convert_checkpoint.get_ziv_data_dir", lambda: data_dir)
    monkeypatch.setattr("zvisiongenerator.utils.config.load_config", lambda: {"model_aliases": {"h3": "x"}})
    calls: list[dict] = []

    def _convert(checkpoint, target, **kwargs):
        calls.append({"checkpoint": checkpoint, "target": target, **kwargs})
        write_converted_h3(target, {"storage": "nf4-folded", "turbo": "distilled" if kwargs["turbo"] else "merged"})

    monkeypatch.setattr("zvisiongenerator.backends.convert_h3_checkpoint", _convert)
    checkpoint = tmp_path / "ft.safetensors"
    save_file(_native(_diffusers_state(torch.Generator().manual_seed(0)), interleave=False), str(checkpoint))
    return data_dir / "models", checkpoint, calls


@pytest.mark.parametrize(("flags", "turbo"), [([], False), (["--turbo"], True)])
def test_h3_converts_with_the_turbo_choice_given(h3_cli, flags, turbo):
    models, checkpoint, calls = h3_cli

    _run_cli(["-i", str(checkpoint), "--model-type", "h3", "--name", "myh3", *flags])

    assert calls[0]["turbo"] is turbo
    assert read_converted_h3(models / "myh3")["turbo"] == ("distilled" if turbo else "merged")
    assert sorted(path.name for path in models.iterdir()) == ["myh3"]  # no partial left behind
    assert detect_video_model(str(models / "myh3")).family == "h3"


@pytest.mark.parametrize("flags", [["--quantize", "4"], ["--distill"]])
def test_h3_refuses_the_ltx_flags(h3_cli, flags):
    _models, checkpoint, calls = h3_cli

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "h3", *flags])

    assert calls == []


def test_turbo_is_refused_for_ltx(h3_cli, monkeypatch):
    _models, checkpoint, _calls = h3_cli
    monkeypatch.setattr("zvisiongenerator.converters.ltx_convert.convert_ltx", lambda *a, **k: pytest.fail("LTX takes no --turbo"))

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.5", "--quantize", "4", "--turbo"])


def test_a_symlinked_input_keeps_its_own_name(h3_cli, tmp_path):
    models, checkpoint, calls = h3_cli
    link = tmp_path / "linked-finetune.safetensors"
    link.symlink_to(checkpoint)

    _run_cli(["-i", str(link), "--model-type", "h3"])

    assert (models / "linked-finetune").is_dir()
    assert calls[0]["checkpoint"].path == link
