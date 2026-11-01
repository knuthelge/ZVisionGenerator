"""Pin the vendored BiRefNet model code that the background removal chain builds and loads.

The vendored file is read as a syntax tree, so a failure after ``make update-birefnet`` names the class, signature or
attribute that changed. One import check at the end proves the file loads against the installed torch, timm, kornia,
einops, torchvision and transformers.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re

from tests._paths import REPO_ROOT
from zvisiongenerator.utils.matte_models import BIREFNET

ROOT = REPO_ROOT
PACKAGE = ROOT / "packages" / "birefnet"
MAKEFILE = ROOT / "Makefile"
MODEL = PACKAGE / "birefnet.py"
CONFIG = PACKAGE / "BiRefNet_config.py"

CLASSES = ("BiRefNet", "DeformableConv2d", "Decoder")


def _tree(path=MODEL) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _classes(name: str, path=MODEL) -> list[ast.ClassDef]:
    return [node for node in _tree(path).body if isinstance(node, ast.ClassDef) and node.name == name]


def _class(name: str, path=MODEL) -> ast.ClassDef:
    (definition,) = _classes(name, path)
    return definition


def _function(scope: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _parameters(function: ast.FunctionDef) -> list[str]:
    """Return every parameter name of *function*, ``self`` excluded."""
    args = function.args
    return [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs) if arg.arg != "self"]


def _makefile_revision() -> str:
    match = re.search(r"^BIREFNET_REVISION\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "Makefile defines no BIREFNET_REVISION"
    return match.group(1)


def test_the_classes_the_chain_builds_exist() -> None:
    assert _classes("BiRefNet") and _classes("DeformableConv2d") and _classes("Decoder")
    assert _classes("BiRefNetConfig", CONFIG)


def test_birefnet_is_defined_once_and_is_a_pretrained_model() -> None:
    model = _class("BiRefNet")
    assert [ast.unparse(base) for base in model.bases] == ["PreTrainedModel"]


def test_decoder_is_defined_twice_and_the_last_definition_is_the_one_in_use() -> None:
    """The module rebinds ``Decoder`` after the refiner's first one, so ``BiRefNet`` builds the later class."""
    definitions = _classes("Decoder")
    assert len(definitions) == 2
    module = importlib.import_module("birefnet.birefnet")
    assert inspect.getsourcelines(module.Decoder)[1] == definitions[-1].lineno


def test_deformable_conv_constructor_parameters() -> None:
    init = _function(_class("DeformableConv2d"), "__init__")
    assert _parameters(init) == ["in_channels", "out_channels", "kernel_size", "stride", "padding", "bias"]


def test_deformable_conv_layers_stride_padding_and_modulator_the_replacement_relies_on() -> None:
    """The ``grid_sample`` stand-in reads ``offset_conv``, ``modulator_conv`` and ``regular_conv``, an int ``padding`` and the 2x sigmoid modulator."""
    scope = _class("DeformableConv2d")
    init = _function(scope, "__init__")
    stored = {node.attr for node in ast.walk(init) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == "self"}
    assert {"offset_conv", "modulator_conv", "regular_conv", "padding", "stride"} <= stored
    assignments = [ast.unparse(node.value) for node in ast.walk(init) if isinstance(node, ast.Assign) and any(ast.unparse(target) == "self.padding" for target in node.targets)]
    assert assignments == ["padding"]
    forward = ast.unparse(_function(scope, "forward"))
    assert "modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))" in forward
    assert "deform_conv2d(" in forward


def test_config_takes_bb_pretrained_and_keeps_its_model_type() -> None:
    config = _class("BiRefNetConfig", CONFIG)
    assert "bb_pretrained" in _parameters(_function(config, "__init__"))
    model_type = next(node.value for node in config.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "model_type")
    assert ast.literal_eval(model_type) == "SegformerForSemanticSegmentation"


def test_birefnet_constructor_takes_the_config() -> None:
    assert _parameters(_function(_class("BiRefNet"), "__init__")) == ["bb_pretrained", "config"]


def test_makefile_revision_is_the_one_named_in_the_package_and_pinned_for_the_weights() -> None:
    init = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert _makefile_revision() in init
    assert _makefile_revision() == BIREFNET.revision


def test_the_vendored_model_imports() -> None:
    module = importlib.import_module("birefnet.birefnet")
    for name in CLASSES:
        assert hasattr(module, name)
