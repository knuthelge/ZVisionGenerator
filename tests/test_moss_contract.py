"""Pin the vendored MOSS-Image pipeline that the CUDA backend and the MOSS builder call.

The vendored file is read as a syntax tree, so a failure after ``make update-moss`` names the class, signature or
attribute that changed. One import check at the end proves the file loads against the installed diffusers and
transformers.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "moss_image"
MAKEFILE = ROOT / "Makefile"
PIPELINE = PACKAGE / "pipeline_moss_image.py"

CLASSES = ("MOSSImageTransformer", "MOSSImageScheduler", "MOSSImagePipeline", "_Block")


def _tree() -> ast.Module:
    return ast.parse(PIPELINE.read_text(encoding="utf-8"))


def _class(name: str) -> ast.ClassDef:
    return next(node for node in _tree().body if isinstance(node, ast.ClassDef) and node.name == name)


def _function(scope: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _parameters(function: ast.FunctionDef) -> dict[str, ast.expr | None]:
    """Map every parameter name (``self`` excluded) to its default node, or ``None`` when it has none."""
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    result = {arg.arg: default for arg, default in zip(positional, defaults, strict=True)}
    result.update({arg.arg: default for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)})
    result.pop("self", None)
    return result


def _defaults(function: ast.FunctionDef) -> dict[str, object]:
    return {name: ast.literal_eval(node) for name, node in _parameters(function).items() if node is not None}


def _self_reads(function: ast.FunctionDef) -> set[str]:
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Load)}


def _block_projection_keys() -> set[str]:
    """Return the keys ``_Block.__init__`` puts into ``self.proj``: the dict comprehension's names and the added ones."""
    init = _function(_class("_Block"), "__init__")
    keys: set[str] = set()
    for node in ast.walk(init):
        if isinstance(node, ast.DictComp):
            for iterable in ast.walk(node.generators[0].iter):
                if isinstance(iterable, ast.Constant) and isinstance(iterable.value, str):
                    keys.add(iterable.value)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and ast.unparse(node.value) == "self.proj" and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
    return keys


def _makefile_revision() -> str:
    match = re.search(r"^MOSS_REVISION\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "Makefile defines no MOSS_REVISION"
    return match.group(1)


def test_the_classes_the_backend_builds_exist() -> None:
    names = {node.name for node in _tree().body if isinstance(node, ast.ClassDef)}
    assert set(CLASSES) <= names


def test_transformer_constructor_parameters_and_defaults() -> None:
    init = _function(_class("MOSSImageTransformer"), "__init__")
    assert list(_parameters(init)) == ["dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim", "latent_channels", "rope_base"]
    assert _defaults(init) == {
        "dim": 3072,
        "ffn_dim": 8192,
        "num_heads": 24,
        "num_layers": 42,
        "text_dim": 4096,
        "freq_dim": 256,
        "latent_channels": 128,
        "rope_base": 10000.0,
    }


def test_transformer_forward_signature() -> None:
    forward = _function(_class("MOSSImageTransformer"), "forward")
    assert list(_parameters(forward)) == ["hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask", "return_dict", "skip_layers"]
    assert _defaults(forward) == {"encoder_attention_mask": None, "return_dict": True, "skip_layers": None}


def test_block_projection_keys() -> None:
    assert _block_projection_keys() == {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}


def test_scheduler_set_timesteps_signature() -> None:
    set_timesteps = _function(_class("MOSSImageScheduler"), "set_timesteps")
    assert list(_parameters(set_timesteps)) == ["num_inference_steps", "device"]


def test_pipeline_constructor_parameters() -> None:
    init = _function(_class("MOSSImagePipeline"), "__init__")
    assert list(_parameters(init)) == ["transformer", "scheduler", "text_encoder", "tokenizer", "vae"]


def test_encode_prompt_signature_and_attributes_read() -> None:
    """``encode_prompt`` runs on a namespace that supplies exactly the attributes it reads from ``self``."""
    encode_prompt = _function(_class("MOSSImagePipeline"), "encode_prompt")
    assert list(_parameters(encode_prompt)) == ["prompt", "max_sequence_length"]
    assert _self_reads(encode_prompt) == {"tokenizer", "text_encoder", "_execution_device"}


def test_call_takes_the_arguments_the_backend_passes() -> None:
    call = _function(_class("MOSSImagePipeline"), "__call__")
    parameters = _parameters(call)
    expected = {
        "prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_embeds",
        "negative_prompt_attention_mask",
        "guidance_scale",
        "uncond_skip_layers",
        "num_inference_steps",
        "generator",
        "output_type",
        "height",
        "width",
    }
    assert expected <= set(parameters)
    assert expected <= {arg.arg for arg in call.args.kwonlyargs}


def test_makefile_revision_is_the_one_named_in_the_package() -> None:
    init = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert _makefile_revision() in init


def test_the_vendored_pipeline_imports() -> None:
    module = importlib.import_module("moss_image.pipeline_moss_image")
    for name in CLASSES:
        assert hasattr(module, name)
