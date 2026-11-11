"""The commands Copy CLI builds for an edit (``frontend/src/lib/state/cliCommand.ts``, same fixture in its Vitest) run through ``ziv-image``."""

from __future__ import annotations

import json
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests._factories import _patch_image_main_dependencies
from tests._paths import REPO_ROOT
from zvisiongenerator import image_cli

CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "edit_cli_cases.json").read_text(encoding="utf-8"))
_KLEIN = {"steps": 4, "guidance": 1.0, "scheduler": None, "supports_mask_edit": True, "max_reference_images": 4}
_SIZE = (64, 48)


@pytest.fixture
def pictures(tmp_path):
    """Return the picture a case edits (``/pics/a.png`` in the fixture) and a mask of its size (the fixture's ``mask.png``)."""
    source = tmp_path / "a.png"
    Image.new("RGB", _SIZE, (120, 80, 40)).save(source)
    mask = tmp_path / "mask.png"
    painted = np.zeros((_SIZE[1], _SIZE[0]), dtype=np.uint8)
    painted[10:30, 10:30] = 255
    Image.fromarray(painted).save(mask)
    return SimpleNamespace(source=str(source), mask=str(mask))


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_copied_command_asks_for_the_edit_of_the_form(monkeypatch, pictures, case):
    command = case["command"].replace("/pics/a.png", pictures.source).replace("mask.png", pictures.mask)
    argv = shlex.split(command)
    assert argv[:2] == ["ziv", "image"]
    _patch_image_main_dependencies(monkeypatch)
    monkeypatch.setattr(image_cli, "resolve_defaults", lambda _info, _config, overrides, _backend: {**_KLEIN, **overrides})
    monkeypatch.setattr(image_cli, "ensure_built_model", lambda *a, **kw: True)
    monkeypatch.setattr(sys, "argv", ["ziv-image", *argv[2:]])

    image_cli.main()

    expected = case["args"]
    args = image_cli.run_batch.call_args.args[4]
    assert args.edit_source == pictures.source
    assert args.edit_mask == (pictures.mask if expected["edit_mask"] else None)
    assert args.edit_extend == (tuple(expected["edit_extend"]) if expected["edit_extend"] else None)
    assert (args.steps, args.seed, args.runs, args.quantize) == (expected["steps"], expected["seed"], expected["runs"], expected["quantize"])
