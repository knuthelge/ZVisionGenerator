"""Pin the vendored Stable Audio 3 package that the audio backend calls.

The package is read as a syntax tree, so a failure after ``make update-sa3`` names the function, signature or call
site that moved. The backend builds the model through ``load_diffusion_cond``'s factory, passes ``sample_size`` to
``generate`` and wraps ``pretransform.decode`` to keep the waveform from before the library's clamp. One import check at
the end proves the inference path still loads after the excluded folders were removed.
"""

from __future__ import annotations

import ast
import importlib
import re

from tests._paths import REPO_ROOT

ROOT = REPO_ROOT
PACKAGE = ROOT / "packages" / "stable_audio_3"
MAKEFILE = ROOT / "Makefile"

EXCLUDED = ("training", "interface", "cli.py", "data/dataset.py")


def _tree(name: str) -> ast.Module:
    return ast.parse((PACKAGE / name).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _function(scope: ast.Module | ast.ClassDef, name: str) -> ast.FunctionDef:
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


def _default(function: ast.FunctionDef, name: str) -> object:
    node = _parameters(function)[name]
    assert node is not None, f"{name} has no default"
    return ast.literal_eval(node)


def _calls_to(scope: ast.AST, callee: str) -> list[ast.Call]:
    return [node for node in ast.walk(scope) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]


def _makefile_commit() -> str:
    match = re.search(r"^SA3_COMMIT\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, "Makefile defines no SA3_COMMIT"
    return match.group(1)


# ---- The vendored tree -----------------------------------------------------------------------------------------------


def test_pinned_commit_matches_the_makefile() -> None:
    commit = _makefile_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert (PACKAGE / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip() == commit


def test_training_interface_and_command_line_are_not_vendored() -> None:
    for name in EXCLUDED:
        assert not (PACKAGE / name).exists(), name


def test_licence_is_vendored() -> None:
    assert "MIT License" in (PACKAGE / "LICENSE").read_text(encoding="utf-8")


def test_nothing_imports_an_excluded_folder() -> None:
    """The inference path must not reach ``training``, ``interface``, ``cli`` or the dataset module."""
    pattern = re.compile(r"^\s*(?:from|import)\s+(?:stable_audio_3\.|\.+)(training|interface|cli|data\.dataset)\b", re.MULTILINE)
    offenders = [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_inference_path_keeps_the_data_helpers_it_imports() -> None:
    """``inference`` imports ``data.utils`` (the one module of ``data/`` that is kept), which is why only the dataset is dropped."""
    imported = {node.module for name in ("inference/sampling.py", "inference/audio_utils.py") for node in ast.walk(_tree(name)) if isinstance(node, ast.ImportFrom) and node.level == 2}
    assert imported == {"data.utils"}
    utils = _tree("data/utils.py")
    defined = {node.name for node in utils.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert {"PadCrop", "create_padding_mask_from_lengths", "compute_effective_seq_len_from_conditioning"} <= defined


# ---- Loading ---------------------------------------------------------------------------------------------------------


def test_load_diffusion_cond_signature() -> None:
    load = _function(_tree("loading_utils.py"), "load_diffusion_cond")
    assert list(_parameters(load)) == ["model_config", "ckpt_path", "device", "model_half"]
    assert _default(load, "device") == "cuda"
    assert _default(load, "model_half") is False


def test_load_diffusion_cond_builds_through_the_factory() -> None:
    """The backend builds the same model with the factory, then fills it tensor by tensor."""
    load = _function(_tree("loading_utils.py"), "load_diffusion_cond")
    factory = _calls_to(load, "create_diffusion_cond_from_config")
    assert [ast.unparse(arg) for arg in factory[0].args] == ["model_config"]
    assert [ast.unparse(arg) for arg in _calls_to(load, "copy_state_dict")[0].args] == ["model", "load_file(ckpt_path)"]
    moves = sorted(_calls_to(load, "model.to"), key=lambda call: call.lineno)
    assert [ast.unparse(call.args[0]) for call in moves] == ["device", "torch.float16"]


def test_factory_takes_the_model_config() -> None:
    factory = _function(_tree("factory.py"), "create_diffusion_cond_from_config")
    assert list(_parameters(factory)) == ["config"]
    reads = {ast.unparse(node.slice) for node in ast.walk(factory) if isinstance(node, ast.Subscript)}
    assert "'model'" in reads


# ---- StableAudioModel ------------------------------------------------------------------------------------------------


def test_stable_audio_model_constructor_and_attributes() -> None:
    model = _class(_tree("model.py"), "StableAudioModel")
    init = _function(model, "__init__")
    assert list(_parameters(init)) == ["model", "model_config", "device", "model_half"]
    stored = {ast.unparse(target) for node in ast.walk(init) if isinstance(node, ast.Assign) for target in node.targets}
    assert {"self.model", "self.model_config", "self.device", "self.model_half"} <= stored


def test_generate_signature() -> None:
    generate = _function(_class(_tree("model.py"), "StableAudioModel"), "generate")
    parameters = list(_parameters(generate))
    assert parameters[:6] == ["prompt", "negative_prompt", "duration", "steps", "cfg_scale", "batch_size"]
    assert {"sample_size", "seed", "return_latents", "truncate_output_to_duration", "chunked_decode"} <= set(parameters)
    assert _default(generate, "seed") == -1
    assert _default(generate, "sample_size") == 5292032, "the default caps a take at 120 s; the backend must pass the config's sample_size"
    assert _default(generate, "return_latents") is False


def test_generate_clamps_the_decoded_waveform() -> None:
    """The library clamps after decoding; the backend keeps the waveform from before it by wrapping ``pretransform.decode``."""
    generate = _function(_class(_tree("model.py"), "StableAudioModel"), "generate")
    clamps = [ast.unparse(node.value) for node in ast.walk(generate) if isinstance(node, ast.Assign) and "clamp" in ast.unparse(node.value)]
    assert clamps == ["result.to(torch.float32).clamp(-1, 1)"]
    sample = _calls_to(generate, "sample_diffusion")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in sample[0].keywords}
    assert keywords["pretransform"] == "self.model.pretransform"
    assert keywords["decode"] == "not return_latents"


def test_generate_passes_the_sample_size_into_the_length_adaptation() -> None:
    generate = _function(_class(_tree("model.py"), "StableAudioModel"), "generate")
    adapt = _calls_to(generate, "self._adapt_sample_size")
    assert adapt and "sample_size" in [ast.unparse(arg) for arg in adapt[0].args]
    helper = _function(_class(_tree("model.py"), "StableAudioModel"), "_adapt_sample_size")
    assert list(_parameters(helper)) == ["conditioning", "sample_size", "duration_padding_sec"]
    assert any(isinstance(node, ast.Return) and ast.unparse(node.value).startswith("min(") and "sample_size" in ast.unparse(node.value) for node in ast.walk(helper))


# ---- The decode call site the clipping capture wraps -----------------------------------------------------------------


def test_sample_diffusion_decodes_through_the_pretransform_instance() -> None:
    """``sample_diffusion`` calls ``pretransform.decode(sampled, chunked=chunked_decode)`` on the instance it is handed."""
    sample = _function(_tree("inference/sampling.py"), "sample_diffusion")
    parameters = _parameters(sample)
    assert {"pretransform", "decode", "chunked_decode", "steps", "cfg_scale"} <= set(parameters)
    decode = _calls_to(sample, "pretransform.decode")
    assert len(decode) == 1
    assert [ast.unparse(arg) for arg in decode[0].args] == ["sampled"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in decode[0].keywords} == {"chunked": "chunked_decode"}
    guarded = [node for node in ast.walk(sample) if isinstance(node, ast.If) and ast.unparse(node.test) == "decode and pretransform is not None"]
    assert guarded and any(isinstance(inner, ast.Call) and ast.unparse(inner.func) == "pretransform.decode" for inner in ast.walk(guarded[0]))


def test_the_pretransform_is_an_attribute_of_the_diffusion_model() -> None:
    init = _function(_class(_tree("model.py"), "StableAudioModel"), "__init__")
    assigned = {ast.unparse(target): ast.unparse(node.value) for node in ast.walk(init) if isinstance(node, ast.Assign) for target in node.targets}
    assert assigned["self.same"] == "self.model.pretransform"


def test_the_diffusion_model_holds_the_parts_the_backend_converts_by_name() -> None:
    init = _function(_class(_tree("models/diffusion.py"), "ConditionedDiffusionModelWrapper"), "__init__")
    assigned = {ast.unparse(target) for node in ast.walk(init) if isinstance(node, ast.Assign) for target in node.targets}
    assert {"self.model", "self.conditioner", "self.pretransform"} <= assigned


def test_the_text_encoder_is_not_a_registered_submodule_of_its_conditioner() -> None:
    """The backend relies on this: converting the model to fp16 never reaches the bf16-trained T5Gemma encoder."""
    init = _function(_class(_tree("models/conditioners.py"), "T5GemmaConditioner"), "__init__")
    assigned = {ast.unparse(target) for node in ast.walk(init) if isinstance(node, ast.Assign) for target in node.targets}
    assert "self.__dict__['model']" in assigned


# ---- Import check ----------------------------------------------------------------------------------------------------


def test_the_inference_path_imports_after_the_exclusions() -> None:
    package = importlib.import_module("stable_audio_3")
    model = importlib.import_module("stable_audio_3.model")
    loading = importlib.import_module("stable_audio_3.loading_utils")
    sampling = importlib.import_module("stable_audio_3.inference.sampling")
    assert package.StableAudioModel is model.StableAudioModel
    assert callable(loading.load_diffusion_cond) and callable(sampling.sample_diffusion)
