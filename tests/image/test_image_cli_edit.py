"""Tests for ``ziv-image --edit``, ``--mask`` and ``--extend``: what reaches the batch and what is refused before a model loads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tests._factories import KLEIN_EDIT_DEFAULTS, _patch_image_main_dependencies
from tests._images import SIDEWAYS_UPRIGHT_SIZE, painted_block, write_cutout, write_mask_png, write_sideways_photo
from zvisiongenerator import image_cli

_IDEOGRAM = {"steps": 20, "guidance": 7.0, "scheduler": None, "supports_mask_edit": False, "max_reference_images": 0}
_SIZE = (64, 48)


def _save_source(path: Path) -> str:
    """Write a source picture and return its path."""
    Image.new("RGB", _SIZE, (120, 80, 40)).save(path)
    return str(path)


def _save_mask(path: Path, size: tuple[int, int] = _SIZE) -> str:
    """Write a mask PNG of *size* that is white in its middle and return its path."""
    return write_mask_png(path, painted_block(size[1], size[0]))


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return ``run(argv, defaults=KLEIN_EDIT_DEFAULTS)``: run ``main()`` on an edit command and return what the batch received.

    The returned namespace has ``args`` (None when the batch never ran), ``batch`` and ``backend``.
    """

    def start(argv: list[str], *, defaults: dict | None = None) -> SimpleNamespace:
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(image_cli, "resolve_defaults", lambda _info, _config, overrides, _backend: {**(defaults or KLEIN_EDIT_DEFAULTS), **overrides})
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
    def test_relative_paths_are_made_absolute_so_the_saved_file_records_the_full_path(self, run, source, tmp_path, monkeypatch):
        _save_mask(tmp_path / "m.png")
        monkeypatch.chdir(tmp_path)

        result = run(["--edit", "a.png", "--prompt", "x", "-m", "z", "--mask", "m.png"])

        assert result.args.edit_source == str(tmp_path / "a.png")
        assert result.args.edit_mask == str(tmp_path / "m.png")

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

    def test_switching_an_effect_off_is_allowed(self, run, source):
        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--no-contrast"])

        assert result.batch.called and result.args.contrast is False

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
        [["--image", "start.png"], ["--width", "512"], ["--transparent"], ["--sharpen", "0.5"], ["--enhance", "style=noir"]],
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

    @pytest.mark.parametrize("value", ["up=256", "top=100"])
    def test_a_bad_extend_is_refused(self, run, source, capsys, value):
        error = _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--extend", value], capsys)

        assert "--extend" in error

    def test_a_model_that_cannot_edit_is_refused_before_it_loads(self, run, source, capsys):
        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--extend", "top=256"], capsys, defaults=_IDEOGRAM)

        assert image_cli.get_backend().load_model.called is False

    def test_a_cutout_needs_paint(self, run, tmp_path, capsys):
        cutout = write_cutout(tmp_path / "c.png", _SIZE)

        _refused(run, ["--edit", cutout, "--prompt", "x", "-m", "z"], capsys)

    def test_a_cutout_cannot_be_extended(self, run, tmp_path, capsys):
        cutout = write_cutout(tmp_path / "c.png", _SIZE)

        error = _refused(run, ["--edit", cutout, "--prompt", "x", "-m", "z", "--extend", "top=256"], capsys)

        assert "cannot be extended" in error

    def test_a_cutout_can_be_painted(self, run, tmp_path):
        cutout = write_cutout(tmp_path / "c.png", _SIZE)
        mask = _save_mask(tmp_path / "m.png")

        result = run(["--edit", cutout, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_mask == mask

    def test_a_missing_source_is_refused(self, run, tmp_path, capsys):
        _refused(run, ["--edit", str(tmp_path / "nope.png"), "--prompt", "x", "-m", "z"], capsys)

    def test_a_source_that_is_not_a_picture_is_refused(self, run, tmp_path, capsys):
        bad = tmp_path / "bad.png"
        bad.write_text("not an image")

        _refused(run, ["--edit", str(bad), "--prompt", "x", "-m", "z"], capsys)

    def test_a_mask_of_another_shape_is_refused(self, run, source, tmp_path, capsys):
        mask = _save_mask(tmp_path / "m.png", size=(_SIZE[0] * 2, _SIZE[1]))

        _refused(run, ["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask], capsys)

    def test_a_scaled_mask_is_read_at_the_pictures_size(self, run, source, tmp_path):
        mask = _save_mask(tmp_path / "m.png", size=(_SIZE[0] * 2, _SIZE[1] * 2))

        result = run(["--edit", source, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_paint.shape == (_SIZE[1], _SIZE[0])

    def test_the_mask_is_compared_with_the_upright_source(self, run, tmp_path):
        sideways = write_sideways_photo(tmp_path / "s.jpg")  # stored sideways, shown upright
        mask = _save_mask(tmp_path / "m.png", SIDEWAYS_UPRIGHT_SIZE)

        result = run(["--edit", sideways, "--prompt", "x", "-m", "z", "--mask", mask])

        assert result.args.edit_mask == mask
