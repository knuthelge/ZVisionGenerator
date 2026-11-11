"""Tests for ``ziv-image --edit``, ``--mask`` and ``--extend``: what reaches the batch and what is refused before a model loads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests._factories import _patch_image_main_dependencies
from zvisiongenerator import image_cli

_KLEIN = {"steps": 4, "guidance": 1.0, "scheduler": None, "supports_mask_edit": True, "max_reference_images": 4}
_ZIMAGE = {"steps": 8, "guidance": 0.0, "scheduler": None, "supports_mask_edit": True, "max_reference_images": 0}
_IDEOGRAM = {"steps": 20, "guidance": 7.0, "scheduler": None, "supports_mask_edit": False, "max_reference_images": 0}
_SIZE = (64, 48)


def _save_source(path: Path, *, cutout: bool = False, size: tuple[int, int] = _SIZE) -> str:
    """Write a source picture, with a transparent corner when *cutout*, and return its path."""
    image = Image.new("RGBA" if cutout else "RGB", size, (120, 80, 40, 255) if cutout else (120, 80, 40))
    if cutout:
        image.putpixel((0, 0), (0, 0, 0, 0))
    image.save(path)
    return str(path)


def _save_mask(path: Path, *, size: tuple[int, int] = _SIZE, painted: bool = True) -> str:
    """Write a mask PNG that is white in its middle when *painted*, else all black, and return its path."""
    data = np.zeros((size[1], size[0]), dtype=np.uint8)
    if painted:
        data[size[1] // 4 : size[1] // 2, size[0] // 4 : size[0] // 2] = 255
    Image.fromarray(data).save(path)
    return str(path)


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return ``run(argv, defaults=_KLEIN)``: run ``main()`` on an edit command and return what the batch received.

    The returned namespace has ``args`` (None when the batch never ran), ``batch`` and ``backend``.
    """

    def start(argv: list[str], *, defaults: dict | None = None) -> SimpleNamespace:
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(image_cli, "resolve_defaults", lambda _info, _config, overrides, _backend: {**(defaults or _KLEIN), **overrides})
        monkeypatch.setattr(image_cli, "ensure_built_model", lambda *a, **kw: True)
        monkeypatch.setattr(sys, "argv", ["ziv-image", *argv])
        image_cli.main()
        batch = image_cli.run_batch
        return SimpleNamespace(args=batch.call_args.args[4] if batch.called else None, batch=batch, backend=backend)

    return start


@pytest.fixture
def source(tmp_path) -> str:
    return _save_source(tmp_path / "a.png")


def _refused(run, argv: list[str], capsys, *, defaults: dict | None = None) -> str:
    """Run *argv*, assert it exits with a usage error before the batch, and return the error text."""
    with pytest.raises(SystemExit) as excinfo:
        run(argv, defaults=defaults)
    assert excinfo.value.code == 2
    assert image_cli.run_batch.called is False
    return capsys.readouterr().err


class TestEditArguments:
    def test_extend_builds_the_side_and_amount_without_a_mask(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--extend", "top=256"])

        assert result.args.edit_source == source
        assert result.args.edit_extend == ("top", 256)
        assert result.args.edit_mask is None

    def test_paint_passes_the_mask_path_and_the_decoded_paint(self, run, source, tmp_path):
        mask = _save_mask(tmp_path / "m.png")

        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_mask == mask
        assert result.args.edit_extend is None
        assert result.args.edit_paint.dtype == bool and result.args.edit_paint.shape == (_SIZE[1], _SIZE[0])
        assert result.args.edit_paint[_SIZE[1] // 4 : _SIZE[1] // 2, _SIZE[0] // 4 : _SIZE[0] // 2].all() and result.args.edit_paint.sum() == (_SIZE[1] // 4) * (_SIZE[0] // 4)

    def test_paint_with_extend_reaches_the_batch(self, run, source, tmp_path):
        mask = _save_mask(tmp_path / "m.png")

        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask, "--extend", "left=128"], defaults=_ZIMAGE)

        assert (result.args.edit_mask, result.args.edit_extend) == (mask, ("left", 128))

    def test_whole_picture_on_klein_runs_without_paint(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z"])

        assert (result.args.edit_mask, result.args.edit_paint, result.args.edit_extend) == (None, None, None)

    def test_a_plain_generation_carries_no_edit(self, run):
        result = run(["--prompt", "x", "-m", "z"])

        assert (result.args.edit_source, result.args.edit_mask, result.args.edit_extend) == (None, None, None)
        assert result.args.output == "."

    def test_an_edit_turns_off_what_belongs_to_a_generation(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--sharpen"])

        args = result.args
        assert args.sharpen is False
        assert args.no_enhance is True
        assert args.enhance is None
        assert args.upscale is None
        assert args.image_path is None
        assert args.reference_paths == []
        assert args.transparent is False

    @pytest.mark.parametrize("flag", ["--no-sharpen", "--no-contrast", "--no-saturation", "--no-enhance"])
    def test_switching_an_effect_off_is_allowed(self, run, source, flag):
        result = run(["--edit", source, "--prompt", "x", "-m", "z", flag])

        assert result.args.edit_source == source

    def test_runs_steps_seed_and_quantize_are_allowed(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z", "-r", "2", "--steps", "6", "--seed", "7", "-q", "8"])

        assert (result.args.runs, result.args.steps, result.args.seed, result.args.quantize) == (2, 6, 7, 8)


class TestEditOutput:
    def test_default_output_is_the_sources_folder(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z"])

        assert result.args.output == str(Path(source).parent)

    def test_output_option_overrides_it(self, run, source, tmp_path):
        out = tmp_path / "elsewhere"

        result = run(["--edit", source, "--prompt", "x", "-m", "z", "-o", str(out)])

        assert result.args.output == str(out)


class TestEditRefusals:
    @pytest.mark.parametrize(
        "extra",
        [
            ["--image", "start.png"],
            ["--reference", "r.png"],
            ["--upscale", "2"],
            ["--review"],
            ["--width", "512"],
            ["--height", "512"],
            ["--ratio", "1:1"],
            ["--size", "m"],
            ["--guidance", "2"],
            ["--scheduler", "beta"],
            ["--enhance"],
            ["--enhance", "style=noir"],
            ["--enhance-note", "at night"],
            ["--enhance-format", "json"],
            ["--transparent"],
            ["--contrast"],
            ["--contrast", "1.2"],
            ["--saturation"],
            ["--saturation", "1.2"],
            ["--sharpen", "0.5"],
        ],
        ids=lambda extra: " ".join(extra),
    )
    def test_a_flag_that_does_not_apply_is_refused_and_named(self, run, source, capsys, extra):
        error = _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", *extra], capsys)

        assert extra[0] in error

    def test_json_prompt_is_refused(self, run, source, capsys):
        error = _refused(run, ["--edit", source, "--json-prompt", "{}", "-m", "z"], capsys)

        assert "--json-prompt" in error

    def test_an_edit_without_a_prompt_is_refused(self, run, source, capsys):
        error = _refused(run, ["--edit", source, "-m", "z"], capsys)

        assert "--prompt" in error

    def test_an_edit_without_a_model_is_refused(self, run, source, capsys):
        error = _refused(run, ["--edit", source, "--prompt", "x"], capsys)

        assert "--model" in error

    @pytest.mark.parametrize("flag", [["--mask", "m.png"], ["--extend", "top=256"]])
    def test_mask_and_extend_need_edit(self, run, capsys, flag):
        error = _refused(run, ["--prompt", "x", "-m", "z", *flag], capsys)

        assert flag[0] in error

    @pytest.mark.parametrize("value", ["top", "up=256", "top=100", "top=130", "top=640", "top=abc", "=256"])
    def test_a_bad_extend_is_refused(self, run, source, capsys, value):
        error = _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--extend", value], capsys)

        assert "--extend" in error

    def test_a_model_that_cannot_edit_is_refused_before_it_loads(self, run, source, capsys):
        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--extend", "top=256"], capsys, defaults=_IDEOGRAM)

        assert image_cli.get_backend().load_model.called is False

    def test_a_model_without_references_cannot_restyle_the_whole_picture(self, run, source, capsys):
        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z"], capsys, defaults=_ZIMAGE)

    def test_a_cutout_needs_paint(self, run, tmp_path, capsys):
        cutout = _save_source(tmp_path / "c.png", cutout=True)

        _refused(run, ["--edit", cutout, "--prompt", "x", "-m", "z"], capsys)

    def test_a_cutout_cannot_be_extended(self, run, tmp_path, capsys):
        cutout = _save_source(tmp_path / "c.png", cutout=True)
        mask = _save_mask(tmp_path / "m.png")

        _refused(run, ["--edit", cutout, "--prompt", "x", "-m", "z", "--mask", mask, "--extend", "top=256"], capsys)

    def test_a_cutout_can_be_painted(self, run, tmp_path):
        cutout = _save_source(tmp_path / "c.png", cutout=True)
        mask = _save_mask(tmp_path / "m.png")

        result = run(["--edit", cutout, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_mask == mask

    def test_a_missing_source_is_refused(self, run, tmp_path, capsys):
        _refused(run, ["--edit", str(tmp_path / "nope.png"), "--prompt", "x", "-m", "z"], capsys)

    def test_a_source_that_is_not_a_picture_is_refused(self, run, tmp_path, capsys):
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")

        _refused(run, ["--edit", str(bad), "--prompt", "x", "-m", "z"], capsys)

    def test_a_missing_mask_is_refused(self, run, source, tmp_path, capsys):
        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--mask", str(tmp_path / "nope.png")], capsys)

    def test_a_mask_of_another_shape_is_refused(self, run, source, tmp_path, capsys):
        mask = _save_mask(tmp_path / "m.png", size=(_SIZE[0] * 2, _SIZE[1]))

        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask], capsys)

    def test_a_scaled_mask_is_read_at_the_pictures_size(self, run, source, tmp_path):
        mask = _save_mask(tmp_path / "m.png", size=(_SIZE[0] * 2, _SIZE[1] * 2))

        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_paint.shape == (_SIZE[1], _SIZE[0])

    def test_an_empty_mask_is_refused(self, run, source, tmp_path, capsys):
        mask = _save_mask(tmp_path / "m.png", painted=False)

        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask], capsys)

    def test_the_mask_is_compared_with_the_upright_source(self, run, tmp_path):
        sideways = tmp_path / "s.jpg"
        image = Image.new("RGB", _SIZE, (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotated a quarter turn, so it displays as 48 wide and 64 tall
        image.save(sideways, exif=exif)
        mask = _save_mask(tmp_path / "m.png", size=(_SIZE[1], _SIZE[0]))

        result = run(["--edit", str(sideways), "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_mask == mask


def test_the_image_flags_are_registered_with_the_edit_destinations() -> None:
    parsed = image_cli._build_parser().parse_args(["--edit", "a.png", "--mask", "m.png", "--extend", "top=256"])

    assert (parsed.edit_source, parsed.edit_mask, parsed.edit_extend) == ("a.png", "m.png", "top=256")
